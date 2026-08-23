export type Confidence = "high" | "medium" | "low";

export interface MatchInput {
  title?: string;
  abstract: string;
}

export interface RankedSession {
  rank: number;
  sessionId: string;
  code: string;
  title: string;
  description?: string;
  divisions: string[];
  color?: string;
  url: string;
  similarity: number;
  confidence: Confidence;
}

export interface MatchResponse {
  modelVersion: string;
  catalogueYear: number;
  selection: {
    mode: "dynamic" | "fixed";
    shownCount: number;
    predictionSetSize?: number;
    targetCoverage?: number;
    calibrationYear?: number;
    truncated: boolean;
  };
  results: RankedSession[];
}

export interface PredictionSetManifest {
  method: "top-score-gap";
  score: string;
  candidatePolicy: "typical-scientific-only" | "all-sessions";
  calibrationYear: number;
  calibrationExamples: number;
  targetCoverage: number;
  maxScoreGap: number;
  minResults: number;
  maxResults: number;
}

export interface SessionRecord {
  vectorRow: number;
  eventCode: string;
  eventYear: number;
  sessionId: string;
  code: string | null;
  title: string | null;
  description: string | null;
  descriptionMissing: boolean;
  divisions: string[];
  divisionCodes: string[];
  color: string | null;
  url: string;
  searchText?: string;
}

export interface MetricLayoutEntry {
  name: "logWeight" | "queryDown" | "queryUp";
  shape: number[];
  offsetFloats: number;
  lengthFloats: number;
}

export interface MetricManifest {
  kind: "diagonal" | "lowrank";
  dimension: number;
  rank: number;
  sessionTransformApplied: true;
  queryLayout: MetricLayoutEntry[];
}

export interface ModelManifest {
  schemaVersion: number;
  catalogueYear: number;
  catalogueVersion: string;
  modelId: string;
  browserModelId: string;
  browserModelPath?: string | null;
  modelRevision: string;
  dimension: number;
  maxTokens: number;
  overlapTokens: number;
  pooling: "mean";
  normalized: true;
  queryPrefix: string;
  vectorDtype: "float32-le";
  sessionCount: number;
  scoring: {
    denseWeight: number;
    bm25: { enabled: boolean; k1: number; b: number };
  };
  confidenceBands: {
    top_score_high: number;
    margin_high: number;
    top_score_low: number;
  };
  predictionSet?: PredictionSetManifest;
  metric?: MetricManifest;
  assets: {
    sessions: { path: string; sha256?: string };
    vectors: { path: string; sha256?: string };
    queryMetric?: { path: string; sha256?: string };
    model?: Array<{ path: string; sha256: string }>;
  };
}
