import { BM25Index, minmax } from "./bm25";
import type {
  Confidence,
  MatchInput,
  MatchResponse,
  ModelManifest,
  MultiVectorQuery,
  SessionRecord,
} from "./types";

export interface Catalogue {
  manifest: ModelManifest;
  sessions: SessionRecord[];
  vectors: Float32Array;
}

function confidence(top: number, margin: number, manifest: ModelManifest): Confidence {
  const bands = manifest.confidenceBands;
  if (top >= bands.top_score_high && margin >= bands.margin_high) return "high";
  if (top < bands.top_score_low) return "low";
  return "medium";
}

function dotRows(query: Float32Array, matrix: Float32Array, dimension: number): Float32Array {
  if (query.length !== dimension || matrix.length % dimension !== 0) {
    throw new Error("Embedding dimensions do not match the catalogue manifest.");
  }
  const rows = matrix.length / dimension;
  const scores = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let score = 0;
    const offset = row * dimension;
    for (let column = 0; column < dimension; column += 1) {
      score += query[column] * matrix[offset + column];
    }
    scores[row] = score;
  }
  return scores;
}

function dotVector(
  query: Float32Array,
  matrix: Float32Array,
  vectorIndex: number,
  dimension: number,
): number {
  let score = 0;
  const offset = vectorIndex * dimension;
  for (let column = 0; column < dimension; column += 1) {
    score += query[column] * matrix[offset + column];
  }
  return score;
}

export class SessionRanker {
  private readonly bm25: BM25Index;

  constructor(private readonly catalogue: Catalogue) {
    if (catalogue.sessions.length !== catalogue.manifest.sessionCount) {
      throw new Error("Session metadata count does not match the manifest.");
    }
    const stride = catalogue.manifest.scoring.multiVector?.sessionVectorStride ?? 1;
    if (catalogue.vectors.length !== catalogue.sessions.length * stride * catalogue.manifest.dimension) {
      throw new Error("Session vector count does not match the catalogue manifest.");
    }
    this.bm25 = new BM25Index(
      catalogue.sessions.map((session) => session.searchText ?? ""),
      catalogue.manifest.scoring.bm25.k1,
      catalogue.manifest.scoring.bm25.b,
    );
  }

  private multiVectorScores(query: MultiVectorQuery): Float32Array {
    const config = this.catalogue.manifest.scoring.multiVector;
    if (!config) throw new Error("Multi-vector scoring configuration is missing.");
    const dimension = this.catalogue.manifest.dimension;
    for (const vector of [query.joint, query.title, ...query.chunks]) {
      if (vector.length !== dimension) throw new Error("Query vector dimensions do not match.");
    }
    const scores = new Float32Array(this.catalogue.sessions.length);
    const weights = config.weights;
    this.catalogue.sessions.forEach((session, row) => {
      const base = row * config.sessionVectorStride;
      const joint = dotVector(query.joint, this.catalogue.vectors, base, dimension);
      const titleTitle = dotVector(query.title, this.catalogue.vectors, base + 1, dimension);
      let titleDescription = titleTitle;
      let chunksTitle = titleTitle;
      let chunksDescription = titleTitle;
      const descriptionCount = Math.min(
        session.descriptionVectorCount ?? 0,
        config.sessionChunks,
      );
      if (descriptionCount > 0) {
        titleDescription = -Infinity;
        for (let part = 0; part < descriptionCount; part += 1) {
          titleDescription = Math.max(
            titleDescription,
            dotVector(query.title, this.catalogue.vectors, base + 2 + part, dimension),
          );
        }
      }
      if (query.chunks.length > 0) {
        chunksTitle = -Infinity;
        for (const chunk of query.chunks) {
          chunksTitle = Math.max(
            chunksTitle,
            dotVector(chunk, this.catalogue.vectors, base + 1, dimension),
          );
        }
        chunksDescription = chunksTitle;
        if (descriptionCount > 0) {
          chunksDescription = -Infinity;
          for (const chunk of query.chunks) {
            for (let part = 0; part < descriptionCount; part += 1) {
              chunksDescription = Math.max(
                chunksDescription,
                dotVector(chunk, this.catalogue.vectors, base + 2 + part, dimension),
              );
            }
          }
        }
      }
      scores[row] =
        weights.joint * joint +
        weights.title_to_title * titleTitle +
        weights.title_to_description * titleDescription +
        weights.chunks_to_title * chunksTitle +
        weights.chunks_to_description * chunksDescription;
    });
    return scores;
  }

  rank(input: MatchInput, query: Float32Array | MultiVectorQuery, topK?: number): MatchResponse {
    const abstract = input.abstract.trim();
    if (!abstract) throw new Error("An abstract is required.");
    const isMultiVector = this.catalogue.manifest.scoring.kind === "multivector";
    const dense = isMultiVector
      ? this.multiVectorScores(query as MultiVectorQuery)
      : dotRows(query as Float32Array, this.catalogue.vectors, this.catalogue.manifest.dimension);
    let scores = dense;
    const denseWeight = this.catalogue.manifest.scoring.denseWeight;
    if (this.catalogue.manifest.scoring.bm25.enabled && denseWeight < 1) {
      const text = `${input.title?.trim() ? `Title: ${input.title.trim()}\n` : ""}Abstract: ${abstract}`;
      const normalizedDense = minmax(dense);
      const lexical = minmax(this.bm25.scores(text));
      scores = new Float32Array(dense.length);
      for (let index = 0; index < scores.length; index += 1) {
        scores[index] = denseWeight * normalizedDense[index] + (1 - denseWeight) * lexical[index];
      }
    }
    const order = Array.from(scores.keys()).sort(
      (left, right) => scores[right] - scores[left] || left - right,
    );
    const top = scores[order[0]] ?? 0;
    const margin = top - (scores[order[1]] ?? top);
    const calibration = this.catalogue.manifest.predictionSet;
    const withinGap = calibration
      ? order.filter((index) => scores[index] >= top - calibration.maxScoreGap - 1e-7).length
      : 10;
    const predictionSetSize = calibration
      ? Math.min(order.length, Math.max(calibration.minResults, withinGap))
      : undefined;
    const requestedCount = topK === undefined
      ? Math.min(predictionSetSize ?? 10, calibration?.maxResults ?? 10)
      : Math.max(1, topK);
    const shownCount = Math.min(requestedCount, order.length);
    const truncated = predictionSetSize !== undefined && shownCount < predictionSetSize;
    const confidenceLabel: Confidence = calibration
      ? truncated
        ? "low"
        : (predictionSetSize ?? 10) <= 5
          ? "high"
          : "medium"
      : confidence(top, margin, this.catalogue.manifest);
    return {
      modelVersion: [
        `${this.catalogue.manifest.browserModelId}@${this.catalogue.manifest.modelRevision}`,
        this.catalogue.manifest.metric
          ? `metric-${this.catalogue.manifest.metric.kind}-${this.catalogue.manifest.assets.queryMetric?.sha256?.slice(0, 12) ?? "unversioned"}`
          : null,
      ].filter(Boolean).join("+"),
      catalogueYear: this.catalogue.manifest.catalogueYear,
      selection: {
        mode: topK === undefined && calibration ? "dynamic" : "fixed",
        shownCount,
        candidateCount: order.length,
        predictionSetSize,
        targetCoverage: calibration?.targetCoverage,
        calibrationYear: calibration?.calibrationYear,
        truncated,
      },
      results: order.slice(0, shownCount).map((index, rank) => {
        const session = this.catalogue.sessions[index];
        return {
          rank: rank + 1,
          sessionId: session.sessionId,
          code: session.code ?? "Uncoded session",
          title: session.title ?? "Untitled session",
          description: session.description ?? undefined,
          divisions: session.divisions,
          primaryDivisionCode: session.primaryDivisionCode ?? undefined,
          coOrganizerLabel: session.coOrganizerLabel ?? undefined,
          visualDivisionCodes: session.visualDivisionCodes ?? session.divisionCodes,
          divisionColors: session.divisionColors ?? (session.color ? [session.color] : []),
          color: session.color ?? undefined,
          url: session.url,
          similarity: scores[index],
          confidence: rank === 0 ? confidenceLabel : "medium",
        };
      }),
    };
  }
}
