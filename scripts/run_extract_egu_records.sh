#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 DUMP_ROOT PREVIEW.jsonl.gz OUTPUT.jsonl.gz [JOBS]" >&2
  exit 2
fi

DUMP_ROOT=$1
PREVIEW=$2
OUTPUT=$3
JOBS=${4:-50%}

if [[ ! -d "$DUMP_ROOT" ]]; then
  echo "Dump directory does not exist: $DUMP_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PREVIEW" ]]; then
  echo "Preview file does not exist: $PREVIEW" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 2
fi

for command_name in python3 parallel gzip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is missing: $command_name" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXTRACTOR="$SCRIPT_DIR/extract_egu_records.py"
OUTPUT_DIR=$(dirname -- "$OUTPUT")
OUTPUT_NAME=$(basename -- "$OUTPUT")
INPUT_LIST="$OUTPUT_DIR/.${OUTPUT_NAME}.inputs.$$.nul"
TEMP_OUTPUT="$OUTPUT_DIR/.${OUTPUT_NAME}.tmp.$$"
JOBLOG="$OUTPUT.joblog.tsv"
SUMMARY_LOG="$OUTPUT.summaries.jsonl"

mkdir -p -- "$OUTPUT_DIR"
trap 'rm -f -- "$INPUT_LIST" "$TEMP_OUTPUT"' EXIT

python3 "$EXTRACTOR" list-files "$PREVIEW" \
  --dump-root "$DUMP_ROOT" >"$INPUT_LIST"

FILE_COUNT=$(python3 -c \
  'import sys; print(open(sys.argv[1], "rb").read().count(b"\0"))' \
  "$INPUT_LIST")
if [[ "$FILE_COUNT" -eq 0 ]]; then
  echo "No source files containing EGU records were identified." >&2
  exit 2
fi

echo "Extracting full EGU records from $FILE_COUNT source files with jobs=$JOBS" >&2

parallel --will-cite -0 --jobs "$JOBS" --keep-order --halt soon,fail=1 \
  --joblog "$JOBLOG" \
  python3 "$EXTRACTOR" extract-file {} \
  --dump-root "$DUMP_ROOT" \
  2>>"$SUMMARY_LOG" \
  :::: "$INPUT_LIST" \
  | gzip -c >"$TEMP_OUTPUT"

mv -- "$TEMP_OUTPUT" "$OUTPUT"
rm -f -- "$INPUT_LIST"
trap - EXIT

echo "Completed: $OUTPUT" >&2
python3 "$EXTRACTOR" summarize "$OUTPUT"
