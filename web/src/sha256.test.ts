import { describe, expect, it } from "vitest";
import { portableSha256 } from "./sha256";

const encode = (value: string): Uint8Array => new TextEncoder().encode(value);

describe("portable SHA-256", () => {
  it("matches standard empty and short-message vectors", () => {
    expect(portableSha256(encode(""))).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(portableSha256(encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("handles input spanning more than one block", () => {
    expect(portableSha256(encode("a".repeat(1000)))).toBe(
      "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
    );
  });
});
