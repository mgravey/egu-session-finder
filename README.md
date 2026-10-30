# EGU27 Session Finder

**Live site: <https://mgravey.com/egu-session-finder/>**

Paste an abstract and get a calibrated shortlist of the best-matching sessions of
the EGU General Assembly 2027. Everything runs in the browser: a quantized
MiniLM model embeds the text locally and compares it with precomputed session
vectors, so the abstract is never sent to a server.

The EGU27 abstract deadline is **12 January 2027, 13:00 CET**.

## How well it works

Every past conference was replayed as if it were the next one: the model was
trained only on earlier years and calibrated on the year before, exactly as the
deployed EGU27 model was prepared (trained on 2022–2025, calibrated on 2026).
Candidates are typical scientific sessions only.

| Conference | Trained on | Top 1 | Top 2 | Top 5 | Top 10 | Top 20 | Right session in shortlist | Typical shortlist length |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EGU26 | 2022–24 | 32.2% | 46.0% | 64.8% | 77.7% | 87.2% | 79.5% | 9 sessions |
| EGU25 | 2022–23 | 31.6% | 45.5% | 63.2% | 75.2% | 84.3% | 81.4% | 10 sessions |
| EGU24 | 2022 | 30.1% | 43.2% | 61.3% | 74.1% | 84.6% | 78.8% | 11 sessions |

## Method in brief

- **Data.** EGU2020–2027 session programmes (titles, descriptions, divisions)
  crawled slowly from `meetingorganizer.copernicus.org` and cached; 121,971
  abstracts from Crossref; 2022–2026 abstract-to-session assignments from the
  post-conference programmes. Session text never includes accepted abstracts.
- **Scoring.** Five similarities (joint text, title↔title, title↔description,
  abstract passages↔title, abstract passages↔description) between frozen
  MiniLM embeddings, after a learned rank-32 query/session projection, combined
  with learned positive weights.
- **Shortlist.** A top-score-gap rule calibrated to include a historically
  chosen session 80% of the time, showing 3–20 sessions.

See `web/method.html` (or the live *Method* page) for details.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | Crawling, data preparation, training, evaluation, and export |
| `tests/` | Python unit tests (`python -m unittest discover -s tests`) |
| `config/` | Model pins and the (non-official) division colour palette |
| `web/` | Static TypeScript site (Vite), bundled model and EGU27 catalogue |
| `data/` | Local caches and intermediate results (ignored by git) |

## Refreshing the catalogue

When EGU publishes a new programme, or merges sessions after the abstract
deadline, rerun the pipeline. Crawling is cache-aware, so only new pages are
requested; keep the slow request rate.

```bash
# Python environments
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
uv venv --python 3.12 .venv-ml && .venv-ml/bin/pip install -r requirements-ml.txt

# 1. Crawl and parse the programme (add the new event to EVENTS first)
.venv/bin/python scripts/crawl_egu_sessions.py discover --years 2027 --data-dir data/sessions --delay 1.5 --jitter 1.0
.venv/bin/python scripts/crawl_egu_sessions.py fetch    --years 2027 --data-dir data/sessions --delay 1.5 --jitter 1.0
.venv/bin/python scripts/crawl_egu_sessions.py parse    --years 2020-2027 --data-dir data/sessions

# 2. Join sessions and abstracts
E=data/matcher/egu27
.venv-ml/bin/python scripts/prepare_matcher_data.py \
  --crossref data/crossref/egu_crossref_full_2020_2025.jsonl.gz \
  --crossref data/crossref/egu_crossref_full_2026.jsonl.gz \
  --sessions data/sessions/processed/egu_sessions_2020_2027.jsonl.gz \
  --output-dir $E/prepared

# 3. Embed (abstract caches are reused when inputs are unchanged)
.venv-ml/bin/python scripts/benchmark_embeddings.py --model minilm-l6 \
  --prepared-dir $E/prepared --cache-dir $E/embeddings --output $E/benchmark.json --device mps
.venv-ml/bin/python scripts/build_multivector_embeddings.py \
  --prepared-dir $E/prepared --output-dir $E/multivector --device mps

# 4. Train on every labeled year, select and calibrate on the most recent one
.venv-ml/bin/python scripts/train_embedding_metric.py --prepared-dir $E/prepared \
  --embedding-dir $E/embeddings/minilm-l6 --output-dir $E/metric-learning --metric-types lowrank \
  --train-years 2022 2023 2024 2025 --validation-year 2026 --test-year none --forward-years --device mps
.venv-ml/bin/python scripts/train_multivector_ranker.py --prepared-dir $E/prepared \
  --baseline-embedding-dir $E/embeddings/minilm-l6 --multivector-dir $E/multivector \
  --metric-head $E/metric-learning/metric-head.npz --output-dir $E/ranker \
  --train-years 2022 2023 2024 2025 --validation-year 2026 --report-years --device mps
.venv-ml/bin/python scripts/export_multivector_score_matrices.py --prepared-dir $E/prepared \
  --baseline-embedding-dir $E/embeddings/minilm-l6 --multivector-dir $E/multivector \
  --metric-head $E/metric-learning/metric-head.npz --weights $E/ranker/browser-weights.json \
  --output-dir $E/matrices --years 2026 --device mps
.venv-ml/bin/python scripts/calibrate_prediction_sets.py --prepared-dir $E/prepared \
  --matrix-dir $E/matrices --calibration-year 2026 --evaluation-years 2026 \
  --typical-sessions-only --output $E/prediction-set.json

# 5. Export the browser catalogue and build the site
scripts/build_static_site.sh --year 2027
```

`scripts/export_track_record.py` regenerates `web/src/track-record.json` (the
statistics shown on the site) from rolling backtests in `data/matcher/egu27/rolling/`.

## Website development

```bash
cd web
npm ci
npm test
npm run dev      # http://localhost:5173
npm run build    # static output in web/dist
```

Pushing to `main` deploys `web/dist` to GitHub Pages through
`.github/workflows/pages.yml`.

## Privacy

The site only downloads static files (application, model, catalogue). The
title and abstract are processed in a Web Worker (WebGPU, or WebAssembly when no
GPU adapter is available) and are never uploaded, logged, or stored. There is no
analytics.

## Disclaimer

An independent tool by [Mathieu Gravey](https://mgravey.com/). Not affiliated
with or endorsed by the European Geosciences Union. Session titles and
descriptions belong to their conveners and EGU; always check the
[official programme](https://meetingorganizer.copernicus.org/EGU27/sessionprogramme).
Model provenance is recorded in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

The code, data pipeline, and website were developed with the help of AI coding
tools.
