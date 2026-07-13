#!/usr/bin/env bash
# =============================================================================
# download_hirise.sh
# Downloads the NASA HiRISE DTM for Jezero Crater (Perseverance landing site).
#
# Run this ONCE on the Vast.ai server before starting the simulation.
# The file is ~50-248 MB depending on source.
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

# ── Fallback: CTX mosaic (20 m/px, ~9 MB, verified) ─────────────────────────
FALLBACK_CTX="https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif"

FALLBACK_MOLA=""  # no longer needed

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

download_url "$PRIMARY_URL"  "USGS HiRISE 1m/px" || \
download_url "$FALLBACK_CTX" "Murray Lab CTX 6m/px" || \
download_url "$FALLBACK_MOLA" "MOLA 463m/px (low-res fallback)" || \
{
    echo ""
    echo "[download_hirise] All downloads failed."
    echo "Manual download options:"
    echo "  1. HiRISE browser: https://www.uahirise.org/dtm/"
    echo "     Search 'Jezero', download any DTEEC_*.tif"
    echo "  2. NASA PDS: https://pds-geosciences.wustl.edu/"
    echo "  Place the file at: $OUTPUT"
    echo ""
    echo "IMPORTANT: If no file is downloaded, the sim falls back to"
    echo "procedural terrain automatically — it will still run fine."
    exit 1
}

# Verify it's a valid GeoTIFF
python3 -c "
import rasterio, sys
try:
    with rasterio.open('$OUTPUT') as src:
        print(f'[download_hirise] Verified: {src.width}x{src.height} px, '
              f'CRS={src.crs}, res={src.res[0]:.1f}m/px')
except Exception as e:
    print(f'[download_hirise] WARNING: File may be corrupt: {e}')
    sys.exit(1)
" 2>/dev/null || echo "[download_hirise] rasterio check skipped (install with: pip install rasterio)"

echo "[download_hirise] Done. Start Isaac Sim to use real HiRISE terrain."
