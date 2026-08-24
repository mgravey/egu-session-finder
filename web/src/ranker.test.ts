import { describe, expect, it } from "vitest";
import { SessionRanker, type Catalogue } from "./ranker";

function catalogue(denseWeight = 1): Catalogue {
  return {
    manifest: {
      schemaVersion: 1,
      catalogueYear: 2026,
      catalogueVersion: "test",
      modelId: "test",
      browserModelId: "test",
      modelRevision: "abc",
      dimension: 2,
      maxTokens: 10,
      overlapTokens: 2,
      pooling: "mean",
      normalized: true,
      queryPrefix: "",
      vectorDtype: "float32-le",
      sessionCount: 2,
      scoring: { denseWeight, bm25: { enabled: denseWeight < 1, k1: 1.2, b: 0.75 } },
      confidenceBands: { top_score_high: 0.8, margin_high: 0.1, top_score_low: 0.3 },
      assets: { sessions: { path: "" }, vectors: { path: "" } },
    },
    sessions: [
      {
        vectorRow: 0,
        eventCode: "EGU26",
        eventYear: 2026,
        sessionId: "1",
        code: "AS1",
        title: "Clouds",
        description: "Aerosols",
        descriptionMissing: false,
        divisions: ["Atmospheric Sciences (AS)"],
        divisionCodes: ["AS"],
        color: null,
        url: "https://example.test/1",
        searchText: "cloud aerosol",
      },
      {
        vectorRow: 1,
        eventCode: "EGU26",
        eventYear: 2026,
        sessionId: "2",
        code: "HS1",
        title: "Floods",
        description: "Hydrology",
        descriptionMissing: false,
        divisions: ["Hydrological Sciences (HS)"],
        divisionCodes: ["HS"],
        color: null,
        url: "https://example.test/2",
        searchText: "river flood hydrology",
      },
    ],
    vectors: new Float32Array([1, 0, 0, 1]),
  };
}

describe("SessionRanker", () => {
  it("returns deterministic dense top-k results", () => {
    const response = new SessionRanker(catalogue()).rank(
      { abstract: "Cloud research" },
      new Float32Array([0.9, 0.1]),
      2,
    );
    expect(response.results.map((item) => item.sessionId)).toEqual(["1", "2"]);
    expect(response.results[0].confidence).toBe("high");
    expect(response.selection.mode).toBe("fixed");
  });

  it("uses lexical terms when hybrid scoring is enabled", () => {
    const response = new SessionRanker(catalogue(0.4)).rank(
      { abstract: "river flood hydrology" },
      new Float32Array([0.6, 0.4]),
      1,
    );
    expect(response.results[0].sessionId).toBe("2");
  });

  it("rejects empty abstracts", () => {
    expect(() =>
      new SessionRanker(catalogue()).rank({ abstract: " " }, new Float32Array([1, 0])),
    ).toThrow("required");
  });

  it("uses a calibrated score gap when topK is omitted", () => {
    const input = catalogue();
    input.manifest.predictionSet = {
      method: "top-score-gap",
      score: "fixture",
      candidatePolicy: "typical-scientific-only",
      calibrationYear: 2024,
      calibrationExamples: 100,
      targetCoverage: 0.8,
      maxScoreGap: 0.15,
      minResults: 1,
      maxResults: 1,
    };
    const response = new SessionRanker(input).rank(
      { abstract: "Cloud research" },
      new Float32Array([0.9, 0.8]),
    );
    expect(response.selection.mode).toBe("dynamic");
    expect(response.selection.predictionSetSize).toBe(2);
    expect(response.selection.shownCount).toBe(1);
    expect(response.selection.truncated).toBe(true);
    expect(response.results).toHaveLength(1);
  });

  it("ranks packed multi-vector session fields", () => {
    const input = catalogue();
    input.manifest.scoring = {
      kind: "multivector",
      denseWeight: 1,
      bm25: { enabled: false, k1: 1.2, b: 0.75 },
      multiVector: {
        weights: {
          joint: 1,
          title_to_title: 0,
          title_to_description: 0,
          chunks_to_title: 0,
          chunks_to_description: 0,
        },
        queryChunks: 2,
        sessionChunks: 2,
        sessionVectorStride: 4,
      },
    };
    input.sessions[0].descriptionVectorCount = 1;
    input.sessions[1].descriptionVectorCount = 1;
    input.vectors = new Float32Array([
      1, 0, 0, 1, 1, 0, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0,
    ]);
    const response = new SessionRanker(input).rank(
      { title: "Clouds", abstract: "Cloud research" },
      {
        joint: new Float32Array([0.9, 0.1]),
        title: new Float32Array([0, 1]),
        chunks: [new Float32Array([1, 0])],
      },
      2,
    );
    expect(response.results.map((item) => item.sessionId)).toEqual(["1", "2"]);
  });
});
