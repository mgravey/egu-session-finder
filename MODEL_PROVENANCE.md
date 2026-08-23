# Embedding model provenance

Verified on 2026-08-23. Immutable revisions are recorded when browser artifacts
are downloaded and are copied into the annual `model-manifest.json`.

| Candidate | Python source | Browser ONNX source | License | Dimension | Evaluation limit | Quantized browser artifact |
|---|---|---|---|---:|---:|---:|
| MiniLM L6 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) | Apache-2.0 | 384 | 256 word pieces | q8, about 23 MB |
| BGE small v1.5 | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | [Xenova/bge-small-en-v1.5](https://huggingface.co/Xenova/bge-small-en-v1.5) | MIT | 384 | 512 word pieces | q8, about 34 MB |
| MxBai xsmall v1 | [mixedbread-ai/mxbai-embed-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1) | same repository's [`onnx/`](https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/tree/main/onnx) | Apache-2.0 | 384 | 512 word pieces for this application | q8, about 24.4 MB |

MiniLM's model card documents 384-dimensional normalized mean-pooled vectors
and a 256-word-piece truncation limit. BGE documents its retrieval query prefix;
the prefix is applied only to abstract queries. MxBai supports longer contexts,
but this benchmark fixes its application limit at 512 so browser latency and
chunking remain comparable to BGE.

All candidate snapshots load successfully through Sentence Transformers on
Apple MPS. The chosen ONNX snapshot must additionally pass the browser parity,
quantization-regression, and WebGPU/WASM smoke tests before annual export.

The local MiniLM browser artifact is pinned to revision
`751bff37182d3f1213fa05d7196b954e230abad9`.
