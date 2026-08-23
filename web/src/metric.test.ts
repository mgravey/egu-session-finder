import { describe, expect, it } from "vitest";
import { transformQueryMetric } from "./metric";
import type { MetricManifest } from "./types";

describe("learned query metric", () => {
  it("applies a normalized diagonal weighting", () => {
    const manifest: MetricManifest = {
      kind: "diagonal",
      dimension: 2,
      rank: 0,
      sessionTransformApplied: true,
      queryLayout: [{ name: "logWeight", shape: [2], offsetFloats: 0, lengthFloats: 2 }],
    };
    const result = transformQueryMetric(
      new Float32Array([1, 1]),
      new Float32Array([Math.log(4), 0]),
      manifest,
    );
    expect(Array.from(result)).toEqual(
      expect.arrayContaining([expect.closeTo(2 / Math.sqrt(5), 6), expect.closeTo(1 / Math.sqrt(5), 6)]),
    );
  });

  it("applies the exported low-rank query residual", () => {
    const manifest: MetricManifest = {
      kind: "lowrank",
      dimension: 2,
      rank: 1,
      sessionTransformApplied: true,
      queryLayout: [
        { name: "logWeight", shape: [2], offsetFloats: 0, lengthFloats: 2 },
        { name: "queryDown", shape: [2, 1], offsetFloats: 2, lengthFloats: 2 },
        { name: "queryUp", shape: [1, 2], offsetFloats: 4, lengthFloats: 2 },
      ],
    };
    const result = transformQueryMetric(
      new Float32Array([1, 0]),
      new Float32Array([0, 0, 1, 0, 0, 1]),
      manifest,
    );
    expect(result[0]).toBeCloseTo(2 ** -0.5, 6);
    expect(result[1]).toBeCloseTo(2 ** -0.5, 6);
  });
});
