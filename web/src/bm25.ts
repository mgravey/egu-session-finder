const TOKEN = /[a-z0-9]+(?:[.-][a-z0-9]+)*/gi;

export function tokenize(text: string): string[] {
  return Array.from(text.matchAll(TOKEN), (match) => match[0].toLowerCase());
}

export class BM25Index {
  private readonly documents: string[][];
  private readonly frequencies: Map<string, number>[];
  private readonly documentFrequency: Map<string, number>;
  private readonly averageLength: number;

  constructor(
    texts: string[],
    private readonly k1 = 1.2,
    private readonly b = 0.75,
  ) {
    this.documents = texts.map(tokenize);
    this.frequencies = this.documents.map((document) => {
      const frequency = new Map<string, number>();
      for (const term of document) frequency.set(term, (frequency.get(term) ?? 0) + 1);
      return frequency;
    });
    this.documentFrequency = new Map();
    for (const frequency of this.frequencies) {
      for (const term of frequency.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
    }
    this.averageLength = this.documents.length
      ? this.documents.reduce((sum, document) => sum + document.length, 0) / this.documents.length
      : 0;
  }

  scores(query: string): Float32Array {
    const result = new Float32Array(this.documents.length);
    if (!this.averageLength) return result;
    for (const term of new Set(tokenize(query))) {
      const frequency = this.documentFrequency.get(term) ?? 0;
      if (!frequency) continue;
      const inverseFrequency = Math.log(
        1 + (this.documents.length - frequency + 0.5) / (frequency + 0.5),
      );
      this.frequencies.forEach((document, index) => {
        const occurrences = document.get(term) ?? 0;
        if (!occurrences) return;
        const denominator =
          occurrences +
          this.k1 *
            (1 - this.b + (this.b * this.documents[index].length) / this.averageLength);
        result[index] += (inverseFrequency * occurrences * (this.k1 + 1)) / denominator;
      });
    }
    return result;
  }
}

export function minmax(values: Float32Array): Float32Array {
  if (!values.length) return new Float32Array();
  let lower = Number.POSITIVE_INFINITY;
  let upper = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    lower = Math.min(lower, value);
    upper = Math.max(upper, value);
  }
  const result = new Float32Array(values.length);
  if (upper <= lower) return result;
  for (let index = 0; index < values.length; index += 1) {
    result[index] = (values[index] - lower) / (upper - lower);
  }
  return result;
}
