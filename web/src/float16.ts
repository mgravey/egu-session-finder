export function decodeFloat16(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % 2 !== 0) throw new Error("Invalid float16 asset length.");
  const source = new Uint16Array(buffer);
  const output = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const bits = source[index];
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) {
      output[index] = sign * 2 ** -14 * (fraction / 1024);
    } else if (exponent === 0x1f) {
      output[index] = fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    } else {
      output[index] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
    }
  }
  return output;
}
