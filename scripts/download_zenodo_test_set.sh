#!/usr/bin/env bash
# Download and unpack the official 149-song unpaired historical test set.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-$ROOT_DIR/data/historical_unpaired_test}"
DOWNLOAD_DIR="$DATA_ROOT/downloads"
RECORD_API="https://zenodo.org/api/records/22737610/files"

mkdir -p "$DOWNLOAD_DIR" "$DATA_ROOT/audio/orchestra" "$DATA_ROOT/audio/light_orchestra"

download() {
    local name="$1"
    curl -fL --retry 5 --continue-at - \
        "$RECORD_API/$name/content" \
        --output "$DOWNLOAD_DIR/$name"
}

download audio_orchestra_70.zip
download audio_light_orchestra_79.zip
download metadata.csv
download rights.csv
download checksums.sha256
download README.md
download RIGHTS.md
download CITATION.cff

(
    cd "$DOWNLOAD_DIR"
    sha256sum -c --ignore-missing checksums.sha256
)

unzip -q -o "$DOWNLOAD_DIR/audio_orchestra_70.zip" \
    -d "$DATA_ROOT/audio/orchestra"
unzip -q -o "$DOWNLOAD_DIR/audio_light_orchestra_79.zip" \
    -d "$DATA_ROOT/audio/light_orchestra"

echo "Dataset ready at: $DATA_ROOT"
echo "Run: scripts/infer_samecfm40_fos.sh '$DATA_ROOT'"
