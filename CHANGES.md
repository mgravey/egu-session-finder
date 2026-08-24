# Changes

- Made incremental result expansion available through the complete ranked
  catalogue, with a clear boundary between calibrated and lower-ranked results.
- Refined the static matcher layout and added incremental result expansion that
  reuses only an in-memory browser-worker embedding.
- Added frozen MiniLM multi-vector field/chunk embeddings and a five-feature
  positive-weight listwise ranker with temporal validation and sealed testing.
- Added composition with an existing learned query/session metric so the
  multi-vector experiment can be compared against the deployed learned head.
- Added complete multi-vector score-matrix export for exact typical-session rank
  distributions with the shared filtering and tie-breaking implementation.
- Added finite-sample top-score-gap calibration for dynamic prediction sets,
  including temporal coverage/set-size evaluation and explicit display-cap
  accounting.
- Added optional dynamic result counts to the browser API and UI, with fixed
  `topK` override, truncation metadata, and catalogue-export validation.
- Added a complete static multi-vector browser export with float16 catalogue
  vectors, float32 browser decoding, packed title/description fields, and the
  selected five-feature learned scorer.
- Rebuilt the static website around private on-device inference, calibrated
  variable-length shortlists, responsive session results, methodology, privacy,
  and annual catalogue export.
- Bundled the EGU26 catalogue and quantized MiniLM model with a static asset-only
  worker entry and Sites packaging metadata for owner-only publication.

- Prepared ignored local Git worktrees for isolated full-encoder, multi-vector,
  and CLS-pooling retrieval experiments.

- Added a fast, reusable dense-cosine rank-distribution export showing how many
  historical abstracts recover an assigned session within each rank threshold.
- Added persistent, complete same-year abstract-by-session cosine matrices with
  ordered row and column manifests, so rank analyses can be reproduced without
  recomputing embeddings or similarity scores.
- Added listwise metric learning on cached embeddings, comparing learned
  positive dimension weights and an asymmetric low-rank projection while
  preserving the temporal train/validation/sealed-test split.
- Integrated the selected learned metric into annual browser exports: session
  vectors are transformed offline and only a compact query-side transformation
  runs after embedding a new abstract in the Web Worker.
- Confidence bands are recalibrated on EGU24 after metric selection so the
  browser does not reuse thresholds fitted to untransformed cosine scores.
- Reduced dense-only catalogue size by omitting the unused duplicate BM25 text,
  versioned learned metrics in API responses, and made annual catalogue assets
  network-first with an offline cache fallback.
- The selected rank-32 low-rank metric improved full-year Recall@10 from 70.23%
  to 75.81% on EGU24, from 71.27% to 76.24% on sealed EGU25, and from 68.87%
  to 74.68% on forward EGU26.
- Added an explicit typical-scientific-session evaluation filter that removes
  medal/lecture, short-course, splinter, town-hall, debate, press, networking,
  administrative, union-special, and virtual-poster programme categories.

- Marked the sparse public EGU 2020–2021 assignment data as unlabeled
  auxiliary catalogues and moved supervised training to 2022–2023.
- Made limited embedding pilots deterministic and balanced across requested
  evaluation years.
- Added browser-side SHA-256 verification for catalogue and local model assets,
  and made immediate API calls wait safely for worker initialization.
- Replaced BM25's full document scan for every query term with an equivalent
  inverted postings index for scalable full-year evaluation.
- Added explicit model provenance and browser-artifact validation; corrected
  MiniLM to the monolithic q8 Transformers.js artifact.
- Added a same-origin, GET-only service worker cache for offline inference
  without persisting submitted text.
- Added ONNX-backend benchmarking with fixed hybrid weights and an explicit
  validation/test quantization Recall@10 acceptance gate.
- Excluded sparse auxiliary years from the stratified human-review sample.
- Vectorized evaluation dot products in blocks and reused BM25 scores across
  score weights and candidate models.

## 2026-08-23

- Added cache-only parsing support for legacy and modern EGU session layouts.
- Added canonical abstract/session joins, temporal splits, coverage reporting,
  and stratified human-review sampling.
- Added reproducible browser-compatible embedding benchmarks, BM25 hybrid
  evaluation, confidence calibration, hard-negative fine-tuning, and annual
  catalogue export tooling.
- Added a static TypeScript matcher using Transformers.js in a Web Worker with
  WebGPU/WASM fallback and no abstract transmission.
- Added Python and browser tests plus model/data provenance documentation.
