/// <reference lib="webworker" />

import { AutoTokenizer, env, pipeline } from "@huggingface/transformers";
import { SessionRanker, type Catalogue } from "./ranker";
import { transformQueryMetric } from "./metric";
import { decodeFloat16 } from "./float16";
import { sha256 } from "./sha256";
import type { MatchInput, ModelManifest, MultiVectorQuery, SessionRecord } from "./types";

type Extractor = ((
  texts: string | string[],
  options: { pooling: "mean"; normalize: true },
) => Promise<{ tolist(): number[][] }>);

let manifest: ModelManifest;
let ranker: SessionRanker;
let extractor: Extractor;
let tokenizer: any;
let backend = "wasm";
let queryMetric: Float32Array | null = null;
let cachedInputKey: string | null = null;
let cachedQuery: Float32Array | MultiVectorQuery | null = null;
const appBase = new URL("../", self.location.href);

function status(message: string): void {
  self.postMessage({ type: "status", message });
}

async function checkedBuffer(response: Response, expected?: string): Promise<ArrayBuffer> {
  if (!response.ok) throw new Error(`Static asset is unavailable: ${response.url}`);
  const buffer = await response.arrayBuffer();
  if (expected && (await sha256(buffer)) !== expected) {
    throw new Error(`Static asset checksum failed: ${response.url}`);
  }
  return buffer;
}

async function fetchCatalogue(): Promise<Catalogue> {
  const manifestResponse = await fetch(new URL("catalogue/model-manifest.json", appBase));
  if (!manifestResponse.ok) throw new Error("Catalogue manifest is not available. Run the export command.");
  manifest = (await manifestResponse.json()) as ModelManifest;
  const [sessionsResponse, vectorsResponse] = await Promise.all([
    fetch(new URL(`catalogue/${manifest.assets.sessions.path}`, appBase)),
    fetch(new URL(`catalogue/${manifest.assets.vectors.path}`, appBase)),
  ]);
  const [sessionsBuffer, vectorsBuffer] = await Promise.all([
    checkedBuffer(sessionsResponse, manifest.assets.sessions.sha256),
    checkedBuffer(vectorsResponse, manifest.assets.vectors.sha256),
  ]);
  const sessions = JSON.parse(new TextDecoder().decode(sessionsBuffer)) as SessionRecord[];
  const vectors = manifest.vectorDtype === "float16-le"
    ? decodeFloat16(vectorsBuffer)
    : new Float32Array(vectorsBuffer);
  if (manifest.metric) {
    const asset = manifest.assets.queryMetric;
    if (!asset) throw new Error("The learned query metric asset is missing.");
    const response = await fetch(new URL(`catalogue/${asset.path}`, appBase));
    queryMetric = new Float32Array(await checkedBuffer(response, asset.sha256));
  }
  return { manifest, sessions, vectors };
}

async function loadExtractor(): Promise<void> {
  const localName = manifest.browserModelPath?.replace(/^\/?models\//, "").replace(/\/$/, "");
  if (localName) {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL("models/", appBase).pathname;
  }
  const model = localName || manifest.browserModelId;
  if (manifest.assets.model?.length) {
    status("Verifying the local embedding model…");
    await Promise.all(
      manifest.assets.model.map(async (asset) => {
        const response = await fetch(new URL(asset.path, appBase));
        await checkedBuffer(response, asset.sha256);
      }),
    );
  }
  tokenizer = await AutoTokenizer.from_pretrained(model);
  if ("gpu" in navigator) {
    try {
      status("Loading the local model with WebGPU…");
      extractor = (await pipeline("feature-extraction", model, {
        device: "webgpu",
        dtype: "q8",
      })) as unknown as Extractor;
      backend = "webgpu";
      return;
    } catch (error) {
      console.warn("WebGPU model loading failed; using WASM.", error);
    }
  }
  status("Loading the local model with WebAssembly…");
  extractor = (await pipeline("feature-extraction", model, {
    device: "wasm",
    dtype: "q8",
  })) as unknown as Extractor;
  backend = "wasm";
}

function inputIds(encoded: any): number[] {
  const value = encoded.input_ids;
  if (value?.tolist) {
    const rows = value.tolist();
    return Array.isArray(rows[0]) ? rows[0].map(Number) : rows.map(Number);
  }
  return Array.from(value?.data ?? [], Number);
}

async function embed(input: MatchInput): Promise<Float32Array> {
  const body = input.abstract.trim();
  const text = `${manifest.queryPrefix}${input.title?.trim() ? `Title: ${input.title.trim()}\n` : ""}Abstract: ${body}`;
  const encoded = await (typeof tokenizer === "function" ? tokenizer : tokenizer._call.bind(tokenizer))(
    text,
    {
    add_special_tokens: false,
    truncation: false,
    },
  );
  const ids = inputIds(encoded);
  const capacity = Math.max(8, manifest.maxTokens - 2);
  const step = Math.max(1, capacity - manifest.overlapTokens);
  const chunks: string[] = [];
  const weights: number[] = [];
  if (ids.length <= capacity) {
    chunks.push(text);
    weights.push(Math.max(1, ids.length));
  } else {
    for (let start = 0; start < ids.length; start += step) {
      const chunk = ids.slice(start, start + capacity);
      if (!chunk.length) break;
      chunks.push(tokenizer.decode(chunk, { skip_special_tokens: true }));
      weights.push(chunk.length);
      if (start + capacity >= ids.length) break;
    }
  }
  const rows = (await extractor(chunks, { pooling: "mean", normalize: true })).tolist();
  const combined = new Float32Array(manifest.dimension);
  let totalWeight = 0;
  rows.forEach((row, rowIndex) => {
    const weight = weights[rowIndex];
    totalWeight += weight;
    row.forEach((value, column) => {
      if (column < combined.length) combined[column] += value * weight;
    });
  });
  let norm = 0;
  for (let index = 0; index < combined.length; index += 1) {
    combined[index] /= Math.max(totalWeight, 1);
    norm += combined[index] * combined[index];
  }
  norm = Math.sqrt(norm);
  for (let index = 0; index < combined.length; index += 1) combined[index] /= Math.max(norm, 1e-12);
  return combined;
}

async function tokenAwareParts(text: string, limit: number): Promise<string[]> {
  if (!text.trim()) return [];
  const encoded = await (typeof tokenizer === "function" ? tokenizer : tokenizer._call.bind(tokenizer))(
    text,
    { add_special_tokens: false, truncation: false },
  );
  const ids = inputIds(encoded);
  const capacity = Math.max(8, manifest.maxTokens - 2);
  if (ids.length <= capacity) return [text];
  const step = Math.max(1, capacity - manifest.overlapTokens);
  const chunks: string[] = [];
  for (let start = 0; start < ids.length && chunks.length < limit; start += step) {
    const chunk = ids.slice(start, start + capacity);
    if (!chunk.length) break;
    chunks.push(tokenizer.decode(chunk, { skip_special_tokens: true }));
    if (start + capacity >= ids.length) break;
  }
  return chunks;
}

async function embedParts(texts: string[]): Promise<Float32Array[]> {
  if (!texts.length) return [];
  const rows = (await extractor(texts, { pooling: "mean", normalize: true })).tolist();
  return rows.map((row) => Float32Array.from(row.slice(0, manifest.dimension)));
}

async function embedMultiVector(input: MatchInput): Promise<MultiVectorQuery> {
  const config = manifest.scoring.multiVector;
  if (!config) throw new Error("Multi-vector scoring configuration is missing.");
  status("Reading the abstract as a complete scientific summary…");
  const joint = await embed(input);
  status("Comparing its title and specialist passages…");
  const titleText = input.title?.trim() ?? "";
  const titleRows = await embedParts(titleText ? [titleText] : []);
  const title = titleRows[0] ?? new Float32Array(manifest.dimension);
  const chunkTexts = await tokenAwareParts(input.abstract.trim(), config.queryChunks);
  const chunks = await embedParts(chunkTexts);
  return { joint, title, chunks };
}

async function initialize(): Promise<void> {
  status("Loading the EGU session catalogue…");
  ranker = new SessionRanker(await fetchCatalogue());
  await loadExtractor();
  self.postMessage({
    type: "ready",
    backend,
    year: manifest.catalogueYear,
    sessionCount: manifest.sessionCount,
    scoring: manifest.scoring.kind ?? "dense",
  });
}

async function inputKey(input: MatchInput): Promise<string> {
  const text = `${input.title?.trim() ?? ""}\u0000${input.abstract.trim()}`;
  return sha256(new TextEncoder().encode(text).buffer as ArrayBuffer);
}

async function queryFor(input: MatchInput): Promise<Float32Array | MultiVectorQuery> {
  const key = await inputKey(input);
  if (cachedInputKey === key && cachedQuery) {
    status("Reusing the current on-device match…");
    return cachedQuery;
  }
  let query: Float32Array | MultiVectorQuery;
  if (manifest.scoring.kind === "multivector") {
    const values = await embedMultiVector(input);
    if (manifest.metric) {
      if (!queryMetric) throw new Error("The learned query metric is unavailable.");
      values.joint = transformQueryMetric(values.joint, queryMetric, manifest.metric);
      values.title = transformQueryMetric(values.title, queryMetric, manifest.metric);
      values.chunks = values.chunks.map((chunk) =>
        transformQueryMetric(chunk, queryMetric!, manifest.metric!),
      );
    }
    query = values;
  } else {
    let vector = await embed(input);
    if (manifest.metric) {
      if (!queryMetric) throw new Error("The learned query metric is unavailable.");
      vector = transformQueryMetric(vector, queryMetric, manifest.metric);
    }
    query = vector;
  }
  cachedInputKey = key;
  cachedQuery = query;
  return query;
}

const initialization = initialize();

self.addEventListener("message", async (event: MessageEvent) => {
  const message = event.data;
  if (message.type !== "rank") return;
  try {
    await initialization;
    const input = message.input as MatchInput;
    const query = await queryFor(input);
    status("Ranking the current programme entirely on this device…");
    const response = ranker.rank(input, query, message.topK);
    self.postMessage({ type: "result", id: message.id, response });
  } catch (error) {
    self.postMessage({
      type: "error",
      id: message.id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

initialization.catch((error) => {
  self.postMessage({
    type: "fatal",
    message: error instanceof Error ? error.message : String(error),
  });
});
