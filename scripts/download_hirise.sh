#!/usr/bin/env bash
# =============================================================================
# download_hirise.sh
# Downloads the NASA HiRISE DTM for Jezero Crater (Perseverance landing site).
#
# Run this ONCE on the GPU host before starting the simulation.
# The file is about 1.8 GB. If it is missing, the scene refuses to start with
# --terrain hirise rather than silently using a different terrain.
#
# Usage:
#   bash scripts/download_hirise.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
OUTPUT="$DATA_DIR/jezero_hirise.tif"

mkdir -p "$DATA_DIR"

if [ -f "$OUTPUT" ]; then
    echo "[download_hirise] Already exists: $OUTPUT"
    echo "[download_hirise] Delete it first if you want to re-download."
    exit 0
fi

echo "[download_hirise] Downloading Jezero Crater HiRISE DTM..."

# ── Primary: Mars 2020 HiRISE DTM mosaic (1 m/px, ~1.76 GB, verified) ────────
PRIMARY_URL="https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif"

download_url() {
    local url="$1"
    local label="$2"
    echo "[download_hirise] Trying: $label"
    if curl --connect-timeout 60 --retry 3 --progress-bar -L -o "$OUTPUT" "$url"; then
        echo "[download_hirise] SUCCESS: $(du -sh $OUTPUT | cut -f1) saved to $OUTPUT"
        return 0
    else
        rm -f "$OUTPUT"
        echo "[download_hirise] FAILED: $label"
        return 1
    fi
}

download_url "$PRIMARY_URL" "Mars 2020 HiRISE 1 m/px" || {
    echo ""
    echo "[download_hirise] Download failed."
    echo "Manual option: https://www.uahirise.org/dtm/"
    echo "Place the file at: $OUTPUT"
    exit 1
}

echo "[download_hirise] Done. It is checked when the Isaac container opens it."
