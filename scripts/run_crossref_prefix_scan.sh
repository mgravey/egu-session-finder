#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 DUMP_ROOT OUTPUT.jsonl.gz [JOBS] [MODE]" >&2
  echo "  JOBS defaults to 50%; MODE is locator, preview, or full (default preview)." >&2
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

DUMP_ROOT=$1
OUTPUT=$2
JOBS=${3:-50%}
MODE=${4:-preview}

case "$MODE" in
  locator|preview|full) ;;
  *)
    echo "Invalid mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -d "$DUMP_ROOT" ]]; then
  echo "Dump directory does not exist: $DUMP_ROOT" >&2
  exit 2
fi

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi

for command_name in python3 parallel gzip find; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCANNER="$SCRIPT_DIR/crossref_prefix_scan.py"
OUTPUT_DIR=$(dirname -- "$OUTPUT")
OUTPUT_NAME=$(basename -- "$OUTPUT")
TEMP_OUTPUT="$OUTPUT_DIR/.${OUTPUT_NAME}.tmp.$$"
INPUT_LIST="$OUTPUT_DIR/.${OUTPUT_NAME}.inputs.$$.nul"
JOBLOG="$OUTPUT.joblog.tsv"
SUMMARY_LOG="$OUTPUT.summaries.jsonl"

mkdir -p -- "$OUTPUT_DIR"
trap 'rm -f -- "$TEMP_OUTPUT" "$INPUT_LIST"' EXIT

if ! find -L "$DUMP_ROOT" -type f \
  \( -name '*.jsonl.gz' -o -name '*.json.gz' \) \
  -print0 >"$INPUT_LIST"; then
  echo "Failed while searching for compressed JSONL files under: $DUMP_ROOT" >&2
  exit 2
fi

FILE_COUNT=$(python3 -c \
  'import sys; print(open(sys.argv[1], "rb").read().count(b"\0"))' \
  "$INPUT_LIST")

if [[ "$FILE_COUNT" -eq 0 ]]; then
  echo "No files matching *.jsonl.gz or *.json.gz were found under: $DUMP_ROOT" >&2
  echo "Check the directory and filenames with: find -L '$DUMP_ROOT' -type f | head" >&2
  exit 2
fi

echo "Found $FILE_COUNT input files." >&2
echo "Scanning $DUMP_ROOT with GNU Parallel jobs=$JOBS mode=$MODE" >&2

parallel --will-cite -0 --jobs "$JOBS" --keep-order --halt soon,fail=1 \
  --joblog "$JOBLOG" \
  python3 "$SCANNER" {} \
  --dump-root "$DUMP_ROOT" \
  --prefix '10.5194/' \
  --mode "$MODE" \
  2>>"$SUMMARY_LOG" \
  :::: "$INPUT_LIST" \
  | gzip -c >"$TEMP_OUTPUT"

mv -- "$TEMP_OUTPUT" "$OUTPUT"
rm -f -- "$INPUT_LIST"
trap - EXIT

echo "Completed: $OUTPUT" >&2
echo "GNU Parallel job log: $JOBLOG" >&2
echo "Per-file summaries: $SUMMARY_LOG" >&2
