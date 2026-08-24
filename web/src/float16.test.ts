import { describe, expect, it } from "vitest";
import { decodeFloat16 } from "./float16";

describe("float16 catalogue decoding", () => {
  it("decodes finite values, subnormal zero, and infinities", () => {
    const bits = new Uint16Array([0x0000, 0x3c00, 0xc000, 0x7c00]);
    expect(Array.from(decodeFloat16(bits.buffer))).toEqual([0, 1, -2, Infinity]);
  });

  it("rejects incomplete words", () => {
    expect(() => decodeFloat16(new ArrayBuffer(3))).toThrow("length");
  });
});
