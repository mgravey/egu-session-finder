import type { MetricManifest } from "./types";

function layoutSlice(
  values: Float32Array,
  manifest: MetricManifest,
  name: string,
): Float32Array {
  const item = manifest.queryLayout.find((entry) => entry.name === name);
  if (!item) throw new Error(`Metric array is missing: ${name}`);
  return values.subarray(item.offsetFloats, item.offsetFloats + item.lengthFloats);
}

function normalize(values: Float32Array): Float32Array {
  let squaredNorm = 0;
  for (const value of values) squaredNorm += value * value;
  const norm = Math.max(Math.sqrt(squaredNorm), 1e-12);
  for (let index = 0; index < values.length; index += 1) values[index] /= norm;
  return values;
}

export function transformQueryMetric(
  query: Float32Array,
  weights: Float32Array,
  manifest: MetricManifest,
): Float32Array {
  if (query.length !== manifest.dimension) {
    throw new Error("Query and learned metric dimensions differ.");
  }
  const logWeight = layoutSlice(weights, manifest, "logWeight");
  if (logWeight.length !== query.length) throw new Error("Invalid metric log-weight length.");
  const base = new Float32Array(query.length);
  for (let index = 0; index < query.length; index += 1) {
    base[index] = query[index] * Math.exp(0.5 * logWeight[index]);
  }
  if (manifest.kind === "diagonal") return normalize(base);
  if (manifest.kind !== "lowrank") throw new Error(`Unsupported metric kind: ${manifest.kind}`);

  const down = layoutSlice(weights, manifest, "queryDown");
  const up = layoutSlice(weights, manifest, "queryUp");
  if (down.length !== manifest.dimension * manifest.rank || up.length !== manifest.rank * manifest.dimension) {
    throw new Error("Invalid low-rank metric dimensions.");
  }
  const latent = new Float32Array(manifest.rank);
  for (let column = 0; column < manifest.rank; column += 1) {
    let value = 0;
    for (let row = 0; row < manifest.dimension; row += 1) {
      value += base[row] * down[row * manifest.rank + column];
    }
    latent[column] = value;
  }
  const output = base.slice();
  for (let column = 0; column < manifest.dimension; column += 1) {
    for (let row = 0; row < manifest.rank; row += 1) {
      output[column] += latent[row] * up[row * manifest.dimension + column];
    }
  }
  return normalize(output);
}
