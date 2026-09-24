#!/usr/bin/env bash
# Restore one audio file or every supported file below a directory.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${1:-data/historical_unpaired_test}"
OUTPUT_ROOT="${2:-output/samecfm40_fos}"
CHECKPOINT="${CHECKPOINT:-checkpoints/samecfm_40m_fos.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CFM_STEPS="${CFM_STEPS:-10}"
SEED="${SEED:-42}"
CHUNK_SEC="${CHUNK_SEC:-30}"
OVERLAP="${OVERLAP:-0.5}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Place the released file at checkpoints/samecfm_40m_fos.pt or set CHECKPOINT." >&2
    exit 1
fi
if [[ ! -e "$INPUT_PATH" ]]; then
    echo "Input not found: $INPUT_PATH" >&2
    exit 1
fi

restore_one() {
    local source="$1"
    local destination="$2"
    mkdir -p "$(dirname "$destination")"
    "$PYTHON_BIN" main.py infer \
        --checkpoint "$CHECKPOINT" \
        --input "$source" \
        --output "$destination" \
        --device "$DEVICE" \
        --cfm-steps "$CFM_STEPS" \
        --seed "$SEED" \
        --chunk-sec "$CHUNK_SEC" \
        --overlap "$OVERLAP"
}

if [[ -f "$INPUT_PATH" ]]; then
    mkdir -p "$OUTPUT_ROOT"
    stem="$(basename "${INPUT_PATH%.*}")"
    restore_one "$INPUT_PATH" "$OUTPUT_ROOT/${stem}_samecfm40_fos.wav"
    exit 0
fi

while IFS= read -r -d '' source; do
    relative="${source#"$INPUT_PATH"/}"
    destination="$OUTPUT_ROOT/${relative%.*}_samecfm40_fos.wav"
    restore_one "$source" "$destination"
done < <(
    find "$INPUT_PATH" -type f \
        \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.aif' \
           -o -iname '*.aiff' -o -iname '*.mp3' -o -iname '*.ogg' \) \
        -print0 | sort -z
)
