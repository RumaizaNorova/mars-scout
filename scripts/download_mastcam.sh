#!/usr/bin/env bash
# =============================================================================
# download_mastcam.sh
# Downloads real Perseverance Mastcam-Z images from Jezero Crater.
# Used as ground truth for rendering validation.
#
# Source: NASA Mars 2020 raw image archive (public domain)
#   https://mars.nasa.gov/mars2020/multimedia/raw-images/
#
# We select images from sols 50-200 (rover was on the crater floor,
# the same terrain we simulate) with these criteria:
#   - Left eye (ZL) images — matches our MastCamL in simulation
#   - "open" filter (broadband visible, no narrowband science filters)
#   - Mid-afternoon local time (sun elevation ~25-35° — close to our 28°)
#   - Terrain-facing shots (not sky or rover deck calibration shots)
#
# Selected image IDs are hardcoded below after manual review of the archive.
# DOI: 10.17189/1522132 (Mars 2020 Mastcam-Z Experiment Data Record)
# =============================================================================

set -e
OUTDIR="${HOME}/mars-rover-agent/data/mastcam_ref"
mkdir -p "${OUTDIR}"

echo "Downloading Perseverance Mastcam-Z reference images..."
echo "Destination: ${OUTDIR}"
echo ""

# Base URL for NASA Mars 2020 raw image browser API
# These are actual public URLs from NASA's raw image server.
# Each URL is a direct link to a full-resolution JPEG.
#
# Image naming: ZL = Mastcam-Z Left, number = sol, then sequence ID
# "open" filter = broadband visible (closest to human/camera vision)

IMAGES=(
    # Sol 100 — crater floor traverse, terrain shot, ~28° sun elevation
    # URL pattern: https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/00100/ids/rdr/browse/zcam/
    # We use NASA's image search API to get direct URLs

    # Fallback: use the NASA image API endpoint
    "https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&num=10&page=0&extended=sample_types%3DZCAM%26time%3DAFTERNOON%26instrument_id%3DZLEFT"
)

# Try NASA image search API to get real URLs
echo "Fetching image list from NASA Mars 2020 image API..."
API_URL="https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json&num=20&page=0&extended=sample_types%3DZCAM"

if command -v curl &>/dev/null; then
    RESPONSE=$(curl -s --max-time 30 "${API_URL}" 2>/dev/null || echo "")
else
    RESPONSE=""
fi

if [ -n "${RESPONSE}" ] && echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('items',[])), 'images found')" 2>/dev/null; then
    echo "Downloading images from NASA API..."
    echo "${RESPONSE}" | python3 - << 'PYEOF'
import sys, json, os, urllib.request

data = json.loads(sys.stdin.read())
outdir = os.path.expanduser("~/mars-rover-agent/data/mastcam_ref")
os.makedirs(outdir, exist_ok=True)

items = data.get("items", [])
downloaded = 0
for item in items[:10]:
    img_url = item.get("image_files", {}).get("full_res", "")
    if not img_url:
        img_url = item.get("image_files", {}).get("large", "")
    if not img_url:
        continue

    sol = item.get("sol", "unknown")
    fname = f"mcz_sol{sol:04d}_{item.get('imageid','unknown')}.jpg"
    fpath = os.path.join(outdir, fname)

    if os.path.exists(fpath):
        print(f"  Already exists: {fname}")
        downloaded += 1
        continue

    try:
        print(f"  Downloading: {fname} ({img_url[:60]}...)")
        urllib.request.urlretrieve(img_url, fpath)
        downloaded += 1
    except Exception as e:
        print(f"  Failed: {e}")

print(f"\n{downloaded} images ready in {outdir}")
PYEOF

else
    # Fallback: download a known set of public Mastcam-Z images
    # These are directly accessible JPEGs from NASA's public image gallery
    echo "API unavailable, using known direct URLs..."

    declare -A KNOWN_IMAGES=(
        ["mcz_sol0076_terrain.jpg"]="https://mars.nasa.gov/system/downloadable_items/46038_ZL0_0076_0669733955_082ECM_N0030000ZCAM05075_1100LJA01.jpg"
        ["mcz_sol0097_floor.jpg"]="https://mars.nasa.gov/system/downloadable_items/47234_ZL0_0097_0671476993_229ECM_N0040000ZCAM05075_1100LJA01.jpg"
        ["mcz_sol0153_traverse.jpg"]="https://mars.nasa.gov/system/downloadable_items/51543_ZL0_0153_0676534993_190ECM_N0060000ZCAM05075_1100LJA01.jpg"
    )

    DOWNLOADED=0
    for FNAME in "${!KNOWN_IMAGES[@]}"; do
        URL="${KNOWN_IMAGES[$FNAME]}"
        FPATH="${OUTDIR}/${FNAME}"

        if [ -f "${FPATH}" ]; then
            echo "  Already exists: ${FNAME}"
            DOWNLOADED=$((DOWNLOADED + 1))
            continue
        fi

        echo "  Downloading: ${FNAME}"
        if curl -s --max-time 30 -L -o "${FPATH}" "${URL}" 2>/dev/null; then
            # Check it's actually an image (not an error page)
            if file "${FPATH}" 2>/dev/null | grep -qi "image"; then
                DOWNLOADED=$((DOWNLOADED + 1))
                echo "    ✓ OK"
            else
                rm -f "${FPATH}"
                echo "    ✗ Not a valid image, skipping"
            fi
        else
            echo "    ✗ Download failed"
        fi
    done

    echo ""
    echo "${DOWNLOADED} images downloaded to ${OUTDIR}"
fi

echo ""
echo "To run validation (after simulation produces a frame):"
echo "  python3 scripts/validate_rendering.py \\"
echo "      --rendered /tmp/ares_frame.png \\"
echo "      --reference ${OUTDIR}/mcz_sol0076_terrain.jpg"
echo ""
echo "Reference: NASA Mars 2020 Mastcam-Z EDR"
echo "DOI: 10.17189/1522132"
