# EGU session matcher data preparation

This repository contains the first data-preparation utility for the EGU
session-matching project: a parallel scanner for locating Copernicus Crossref
records in a large gzip-compressed Crossref JSONL dump.

## Crossref DOI-prefix scan

The worker reads one gzip file, selects records whose normalized DOI starts
with `10.5194/`, and records the source file, physical decompressed line number,
item index, and DOI. The GNU Parallel runner processes files independently and
combines the results into one gzip-compressed JSONL file.

Requirements on the remote machine:

- Python 3.10 or newer
- GNU Parallel
- `gzip` and `find`
- Optional: `orjson` for faster JSON parsing

Create and activate the Python environment, then install all Python
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Debian or Ubuntu, install GNU Parallel separately if it is unavailable:

```bash
sudo apt-get update
sudo apt-get install parallel
```

Run the complete scan:

```bash
chmod +x scripts/run_crossref_prefix_scan.sh
scripts/run_crossref_prefix_scan.sh \
  ~/data/doiInfo/crossref_dump_2026 \
  ./copernicus_crossref_preview.jsonl.gz \
  50% \
  preview
```

The second argument is the output path. The example above stores the result in
the current working directory. The runner accepts both `*.jsonl.gz` and
`*.json.gz`, follows symbolic links, prints the number of matching input files
before starting, and fails instead of creating an empty result when no input
files are found.

`50%` tells GNU Parallel to start with half of the available CPU threads. This
is a conservative default for a large shared or network filesystem. Increase
it only if storage throughput and CPU usage show spare capacity.

The runner refuses to overwrite an existing result and creates:

- `copernicus_crossref_preview.jsonl.gz`: matching records
- `copernicus_crossref_preview.jsonl.gz.joblog.tsv`: GNU Parallel job log
- `copernicus_crossref_preview.jsonl.gz.summaries.jsonl`: counts and throughput
  for every source file

Each output row in `preview` mode resembles:

```json
{"source_file":"123.json.gz","source_line":417,"item_index":0,"doi":"10.5194/egusphere-egu26-3285","type":"proceedings-article","subtype":null,"publisher":"Copernicus GmbH","title":"Ionospheric Slant TEC Modelling Based on GNSS Data with Machine Learning","container_title":"EGU General Assembly 2026","published":{"date-parts":[[2026]]}}
```

Inspect a sample without uncompressing the result to disk:

```bash
gzip -cd ~/data/doiInfo/copernicus_crossref_preview.jsonl.gz | head
```

### Output modes

- `locator`: source file, source line, item index, and normalized DOI only.
- `preview`: locator plus lightweight fields useful for identifying record
  families. This is the recommended first pass.
- `full`: locator, preview, and the complete original Crossref record. Use this
  if result storage is acceptable; it avoids a second pass through matching
  source files.

A gzip line number is not a byte offset, so it does not provide true random
access. Later selective extraction should group requested locators by
`source_file`, decompress each relevant file once, and select the desired line
and item indices. Since each source file has only about 5,000 lines, this is
still inexpensive compared with rescanning the complete dump.

### Run one worker manually

```bash
python3 scripts/crossref_prefix_scan.py \
  ~/data/doiInfo/crossref_dump_2026/123.jsonl.gz \
  --dump-root ~/data/doiInfo/crossref_dump_2026 \
  --mode preview \
  > one-file-preview.jsonl
```

By default, malformed JSON causes that worker to fail and GNU Parallel stops
starting new work. Use `--on-json-error skip` only after deciding that a partial
scan is acceptable; skipped-line counts are retained in the summary log.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Extract complete historical EGU records

After creating the Copernicus preview index, reread only dump files containing
EGU records and extract complete EGU2020–EGU2025 Crossref metadata:

```bash
scripts/run_extract_egu_records.sh \
  ~/data/doiInfo/crossref_dump_2026 \
  ./copernicus_crossref_preview.jsonl.gz \
  ./egu_crossref_full_2020_2025.jsonl.gz \
  75%
```

Each output record contains normalized event identifiers, title, raw abstract,
plain-text abstract, authors, provenance, and the complete original Crossref
record. The runner reports abstract coverage and duplicates after extraction.

## Retrieve EGU26 from the Crossref REST API

The annual March 2026 dump predates most EGU26 deposits. Retrieve current 2026
Copernicus `posted-content` metadata through approximately 30 cursor-paginated
requests and filter exact `10.5194/egusphere-egu26-<number>` DOIs locally:

```bash
python scripts/fetch_crossref_egu_api.py \
  --year 2026 \
  --email YOUR_EMAIL@example.org \
  --cache-dir ./crossref_api_cache/2026 \
  --output ./egu_crossref_full_2026.jsonl.gz
```

Every raw API page is cached as a compressed JSON file. An interrupted download
resumes from the last cached cursor, and rerunning after completion makes no API
requests. The email identifies the client to Crossref's polite pool; it is sent
to Crossref as a request parameter and user-agent contact.

## Cache and parse EGU sessions

Discover and cache all EGU2020–EGU2026 session pages, then extract session
descriptions and abstract-to-session identifiers:

```bash
scripts/run_egu_session_crawl.sh
```

The crawler is sequential and capped at five requests per second. It follows
only programme and session links exposed by EGU pages, never probes numeric
identifiers, backs off on temporary failures, and stores every HTML response in
`data/sessions/raw/`. Repeated or resumed runs use the cache and make no request
for an existing page. The combined result is written to
`data/sessions/processed/egu_sessions_2020_2026.jsonl.gz`.

## Prepare the matcher dataset

After the crawl completes, parse the cache and join programme assignments to
the Crossref abstracts:

```bash
python scripts/crawl_egu_sessions.py parse \
  --years 2020-2026 \
  --data-dir data/sessions

python scripts/prepare_matcher_data.py \
  --crossref data/crossref/egu_crossref_full_2020_2025.jsonl.gz \
  --crossref data/crossref/egu_crossref_full_2026.jsonl.gz \
  --sessions data/sessions/processed/egu_sessions_2020_2026.jsonl.gz \
  --output-dir data/matcher/prepared
```

This emits normalized abstracts and sessions, one row per assignment, a
per-year coverage report, and a deterministic 100-record review sample. Splits
are temporal: 2022–2023 training, 2024 validation, 2025 sealed test, and 2026
forward confirmation. The public post-conference pages for 2020–2021 expose
too few assignment labels to be representative, so those catalogues are kept
as `auxiliary_unlabeled` data and excluded from supervised training and
evaluation. Session representations use programme metadata only; accepted
abstracts are never included in session text.
The human-review sample is stratified by year and division across labeled
2022–2026 records; sparse auxiliary 2020–2021 assignments are excluded.

## Embedding experiments

Use a Python 3.12 environment for broad PyTorch/ONNX wheel compatibility:

```bash
uv venv --python 3.12 .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements-ml.txt
```

Run a timing pilot, then the three configured candidates:

```bash
python scripts/benchmark_embeddings.py \
  --model bge-small-en-v1.5 \
  --limit 1000 \
  --device mps

python scripts/benchmark_embeddings.py --model all --device mps

python scripts/select_embedding_model.py \
  --benchmark data/matcher/benchmark.json
```

Embeddings are tokenizer-aware, overlap long inputs, use length-weighted mean
pooling, and are L2-normalized. Caches are keyed by model, preprocessing, row
order, and input SHA-256. Reports include Recall@1/3/5/10, MRR, division recall,
description availability, dense/BM25 variants, and validation-derived
confidence bands. `--limit` creates a deterministic sample balanced across the
requested evaluation years rather than taking a chronological prefix.

To export the full dense-cosine rank curve—how many historical abstracts have
an assigned session at or above each rank—using an existing embedding cache:

```bash
python scripts/export_dense_score_matrices.py \
  --embedding-dir data/matcher/embeddings/minilm-l6 \
  --years 2020 2021 2022 2023 2024 2025 2026 \
  --output-dir data/matcher/evaluation/minilm-dense-matrices

python scripts/export_dense_rank_distribution.py \
  --matrix-dir data/matcher/evaluation/minilm-dense-matrices \
  --years 2024 2025 2026 \
  --output data/matcher/evaluation/minilm-dense-ranks.json
```

The first command stores every same-year abstract-by-session cosine score in a
float32 NumPy matrix, including rows without historical labels. The manifest
records the ordered abstract and session keys. The second command derives exact
and cumulative rank counts from those matrices. Neither command recomputes
embeddings.

Add `--metric-head data/matcher/metric-learning/metric-head.npz` to the matrix
export command to persist the corresponding learned-similarity matrices for a
direct baseline-versus-selected rank-curve comparison.

Add `--typical-sessions-only` to the rank-distribution export to remove medal
and named lectures, short courses, splinter and town-hall meetings, great
debates, press/networking/administrative events, union special symposia, and
broad virtual-poster buckets from both the candidate set and evaluation rows.
The excluded code prefixes are explicit in `scripts/session_categories.py`;
ordinary division, `ITS`, and `EOS` scientific sessions remain included.

### Learn the ranking metric

The cached embeddings can be adapted without fine-tuning the full encoder. The
metric trainer compares every training abstract with every same-year session,
supports multiple valid session labels, and optimizes a listwise retrieval loss.
It trains on 2022–2023, selects the metric by 2024 Recall@10 and MRR, and opens
the sealed 2025 test only after selecting between diagonal and low-rank metrics:

```bash
python scripts/train_embedding_metric.py \
  --embedding-dir data/matcher/embeddings/minilm-l6 \
  --output-dir data/matcher/metric-learning
```

`metric-head.npz` is the compact learned transformation. Annual session vectors
can be transformed offline; browser inference applies only the query side of
the selected metric to a new abstract embedding.

The selected rank-32 low-rank metric produced these full-year results. Model
selection used EGU24 only; EGU25 remained the sealed test during selection.

| Year | Role | Cosine Recall@10 | Learned Recall@10 | Cosine MRR | Learned MRR |
|---|---|---:|---:|---:|---:|
| 2024 | validation | 70.23% | 75.81% | 0.4137 | 0.4574 |
| 2025 | sealed test | 71.27% | 76.24% | 0.4247 | 0.4592 |
| 2026 | forward confirmation | 68.87% | 74.68% | 0.4016 | 0.4389 |

The selected head has 49,536 trainable parameters. Its browser query asset is
about 100 KB; the corresponding session transformation is applied during the
annual export.

The browser manifest versions the learned metric by checksum. Catalogue assets
use network-first service-worker caching so an annual EGU27 export replaces the
EGU26 catalogue when online while retaining the last verified copy offline.

Pass the learned head to the annual catalogue export to transform session
vectors offline and emit the compact browser-side query transformation:

```bash
python scripts/export_browser_catalogue.py \
  --year 2026 \
  --model minilm-l6 \
  --vectors data/matcher/embeddings/minilm-l6/sessions_with_divisions.npy \
  --metric-head data/matcher/metric-learning/metric-head.npz \
  --confidence-bands data/matcher/metric-learning/confidence-bands.json
```
Exact model sources, licenses, limits, and ONNX artifact choices are recorded
in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

### Isolated ranking experiments

Larger ranking experiments are developed on dedicated Git worktree branches so
full-encoder fine-tuning, multi-vector retrieval, and alternative pooling do
not interfere with the deployable baseline. Local worktrees live under the
ignored `.worktrees/` directory; generated data and model checkpoints remain
ignored and are documented by each experiment branch.

On `experiment/multi-vector`, build title/chunk/description vectors and learn a
five-feature listwise scorer:

```bash
python scripts/build_multivector_embeddings.py \
  --prepared-dir /path/to/shared/data/matcher/prepared \
  --output-dir /path/to/shared/data/matcher/experiments/multi-vector/embeddings

python scripts/train_multivector_ranker.py \
  --prepared-dir /path/to/shared/data/matcher/prepared \
  --baseline-embedding-dir /path/to/shared/data/matcher/embeddings/minilm-l6 \
  --multivector-dir /path/to/shared/data/matcher/experiments/multi-vector/embeddings \
  --output-dir /path/to/shared/data/matcher/experiments/multi-vector/ranker
```

Pass `--metric-head /path/to/metric-head.npz` to apply the existing asymmetric
query/session metric to every field vector before fitting the mixture. This
tests whether multi-vector evidence adds value beyond the current learned
single-vector matcher rather than only beyond raw cosine.

`export_multivector_score_matrices.py` writes complete same-year score matrices
from the selected browser weights. They can be passed to the common
`export_dense_rank_distribution.py --typical-sessions-only` command for exact,
directly comparable cumulative-rank curves.

Calibrate a dynamic prediction set on EGU24 after the scorer is fixed:

```bash
python scripts/calibrate_prediction_sets.py \
  --prepared-dir data/matcher/prepared \
  --matrix-dir data/matcher/experiments/multi-vector/matrices \
  --calibration-year 2024 \
  --evaluation-years 2025 2026 \
  --target-coverage 0.80 \
  --min-results 3 \
  --max-results 20 \
  --typical-sessions-only \
  --output data/matcher/experiments/multi-vector/prediction-set.json
```

The set includes every session within a validation-calibrated score gap of the
top session. This makes the result count query-dependent. The uncapped set is
reported separately from the 20-card display cap, because truncation changes
the calibrated-coverage interpretation. On the completed scorer, the
80% calibration produced median set sizes of 10, 10, and 11 and uncapped
historical-assignment coverage of 80.3%, 81.4%, and 80.8% on EGU24–26.
Coverage means that at least one recorded assignment appears in the set; it is
not a probability that the assignment or any suggested session is uniquely
correct.

The learned browser scorer combines joint cosine, title-to-title,
title-to-description, chunk-to-title, and chunk-to-description maxima. Missing
descriptions fall back deterministically to the corresponding title feature.

Fine-tune only the selected candidate. If BGE wins the baseline comparison:

```bash
python scripts/finetune_embedder.py \
  --prepared-dir data/matcher/prepared \
  --embedding-dir data/matcher/embeddings/bge-small-en-v1.5 \
  --base-model BAAI/bge-small-en-v1.5 \
  --output-dir data/matcher/models/bge-small-egu \
  --device mps \
  --epochs 1 \
  --query-prefix "Represent this sentence for searching relevant passages: "

python scripts/benchmark_embeddings.py \
  --model bge-small-en-v1.5 \
  --model-path data/matcher/models/bge-small-egu \
  --cache-dir data/matcher/embeddings/bge-small-egu \
  --device mps
```

Keep the tuned model only if validation Recall@10 improves by at least two
percentage points and sealed-test Recall@10 falls by no more than one point.
Pass the baseline and tuned benchmark files to `select_embedding_model.py` to
apply that acceptance rule automatically. An accepted local model can be
exported with `scripts/export_finetuned_onnx.py`.

Evaluate a downloaded q8 ONNX artifact with the same fixed hybrid weight before
deployment (replace model name, path, and weight with the selected values):

```bash
python scripts/benchmark_embeddings.py \
  --model minilm-l6 \
  --model-path web/public/models/minilm-l6 \
  --backend onnx \
  --model-file onnx/model_quantized.onnx \
  --device cpu \
  --hybrid-dense-weight 0.85 \
  --cache-dir data/matcher/embeddings/minilm-l6-q8 \
  --output data/matcher/benchmark-minilm-l6-q8.json

python scripts/select_embedding_model.py \
  --benchmark data/matcher/benchmark.json \
  --quantized-benchmark data/matcher/benchmark-minilm-l6-q8.json \
  --variant hybrid
```

The q8 model is accepted only when Recall@10 drops by no more than one
percentage point on both 2024 validation and 2025 sealed test data.

## Export and run the private browser matcher

Download the selected quantized ONNX model. The command records the immutable
Hugging Face revision it resolved:

```bash
python scripts/download_browser_model.py \
  --model-id Xenova/bge-small-en-v1.5 \
  --name bge-small-en-v1.5
```

Use that revision, the validation-selected dense weight, and confidence bands
to produce the annual static catalogue:

```bash
python scripts/export_browser_catalogue.py \
  --year 2026 \
  --model bge-small-en-v1.5 \
  --model-revision THE_REPORTED_COMMIT_SHA \
  --vectors data/matcher/embeddings/bge-small-en-v1.5/sessions_with_divisions.npy \
  --confidence-bands data/matcher/embeddings/bge-small-en-v1.5/confidence_bands.json \
  --dense-weight 1.0 \
  --output-dir web/public/catalogue
```

Build or serve the TypeScript application:

```bash
cd web
npm install
npm test
npm run build
npm run dev
```

The app verifies SHA-256 checksums for the catalogue and locally bundled model,
embeds text in a Web Worker, prefers WebGPU, and falls back to WASM.
The title and abstract are never sent to an API, telemetry, or persistent
storage. Model and catalogue traffic consists only of static GET requests. The
same-origin service worker caches only those GET assets, enabling subsequent
offline inference without storing submitted text. The
public API is exported by `web/src/api.ts`:

```ts
rankSessions(
  { title?: string, abstract: string },
  { topK?: number }
): Promise<MatchResponse>
```

When the exported manifest contains a calibrated `predictionSet`, omitting
`topK` selects between 3 and 20 displayed sessions dynamically. The response
reports the uncapped prediction-set size and whether display truncation was
necessary. Supplying `topK` explicitly retains deterministic fixed-count
behavior. A calibration artifact must only be bundled with the exact scorer
that generated its score matrices.

The supplied division-color URL currently returns HTTP 404.
`config/division_colors.json` records this and uses a neutral accessible UI
fallback until official current values are recovered. Colors never influence
ranking.

## Matcher verification and known dependency finding

```bash
python -m unittest discover -s tests -v
cd web && npm test && npm run build
```

The npm production audit currently reports high-severity advisories in the
Node-only `sharp` and `onnxruntime-node` dependency branches pulled by
Transformers.js, with no upstream fix available. They are excluded from the
generated browser bundle, which uses `onnxruntime-web`. Re-run
`npm audit --omit=dev` whenever Transformers.js is upgraded.
