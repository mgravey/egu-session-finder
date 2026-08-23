#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"
CONTACT=${1:-egu-session-matcher-local-research}

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project environment: $PYTHON" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 2
fi

cd -- "$PROJECT_DIR"

echo "Step 1/3: discover programme pages (cached pages are reused)" >&2
"$PYTHON" scripts/crawl_egu_sessions.py discover \
  --years 2020-2026 \
  --data-dir data/sessions \
  --delay 0.2 \
  --jitter 0 \
  --max-requests 500 \
  --contact "$CONTACT"

echo "Step 2/3: fetch session pages at no more than five requests/second" >&2
"$PYTHON" scripts/crawl_egu_sessions.py fetch \
  --years 2020-2026 \
  --data-dir data/sessions \
  --delay 0.2 \
  --jitter 0 \
  --max-requests 7000 \
  --contact "$CONTACT"

echo "Step 3/3: parse all cached pages offline" >&2
"$PYTHON" scripts/crawl_egu_sessions.py parse \
  --years 2020-2026 \
  --data-dir data/sessions

echo "Complete: data/sessions/processed/egu_sessions_2020_2026.jsonl.gz" >&2
