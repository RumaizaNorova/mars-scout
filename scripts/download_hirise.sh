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
DATA_DIR="$HOME/mars-rover-agent/data"
OUTPUT="$DATA_DIR/jezero_hirise.tif"

mkdir -p "$DATA_DIR"

if [ -f "$OUTPUT" ]; then
    echo "[download_hirise] Already exists: $OUTPUT"
    echo "[download_hirise] Delete it first if you want to re-download."
    exit 0
fi

echo "[download_hirise] Downloading Jezero Crater HiRISE DTM..."

# ── Primary source: USGS Astrogeology HiRISE mosaic (1 m/px, ~248 MB) ────────
PRIMARY_URL="https://planetarymaps.usgs.gov/mosaic/Mars_MRO_HiRISE_Jezero_Crater_DTM_1m.tif"

# ── Fallback 1: Murray Lab CTX mosaic (6 m/px, ~40 MB) ───────────────────────
# Good for terrain shape, lower resolution.
FALLBACK_CTX="https://murray-lab.caltech.edu/CTX/V01/tiles/Murray-Lab-CTX-Mosaic-V01_equirect_tiles_18N_282E_v01.tif"

# ── Fallback 2: MOLA PEDR interpolated (128 px/degree, ~15 MB patch) ─────────
# Lowest quality but always available.
FALLBACK_MOLA="https://astrogeology.usgs.gov/download/Mars/GlobalSurveyor/MOLA/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"

download_url() {
    local url="$1"
    local label="$2"
    echo "[download_hirise] Trying: $label"
    if wget --timeout=60 --tries=3 --show-progress -O "$OUTPUT" "$url"; then
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
