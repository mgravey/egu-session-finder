#!/usr/bin/env bash
set -euo pipefail

SITE_YEAR=2027
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data/matcher/egu27"
WEB_DIR="$PROJECT_DIR/web"
PYTHON_BIN="$PROJECT_DIR/.venv-ml/bin/python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --year) SITE_YEAR="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --web-dir) WEB_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_multivector_browser_catalogue.py" \
    --prepared-dir "$DATA_DIR/prepared" \
    --baseline-embedding-dir "$DATA_DIR/embeddings/minilm-l6" \
    --multivector-dir "$DATA_DIR/multivector" \
    --metric-head "$DATA_DIR/metric-learning/metric-head.npz" \
    --weights "$DATA_DIR/ranker/browser-weights.json" \
    --prediction-set "$DATA_DIR/prediction-set.json" \
    --model-config "$PROJECT_DIR/config/models.json" \
    --division-colors "$PROJECT_DIR/config/division_colors.json" \
    --model minilm-l6 \
    --year "$SITE_YEAR" \
    --output-dir "$WEB_DIR/public/catalogue"

npm --prefix "$WEB_DIR" run build
echo "Static site complete: $WEB_DIR/dist"
