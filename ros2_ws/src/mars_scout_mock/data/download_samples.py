"""
Download real Mars surface imagery for the mock bridge.

Sources
-------
1. NASA Mars Exploration Rover (MER) Panoramic Camera — public domain
2. MSL Curiosity Mastcam panoramas — public domain
3. HiRISE (high-res orbital) — public domain

All NASA imagery is in the public domain (17 U.S.C. 105).

Usage
-----
    python3 data/download_samples.py                  # download everything
    python3 data/download_samples.py --panorama-only  # just the wide panorama
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── Image catalogue ───────────────────────────────────────────────────────────
# Each entry: (filename, url, sha256_prefix_8, description)
IMAGES = [
    (
        "mars_panorama.jpg",
        # Curiosity 360° panorama stitched by NASA/JPL-Caltech/MSSS (2019)
        "https://mars.nasa.gov/system/resources/detail_files/24584_PIA23623.jpg",
        "auto",
        "Curiosity 360° panorama — Glen Etive, Gale Crater (2019)",
    ),
    (
        "mars_surface_01.jpg",
        "https://mars.nasa.gov/system/resources/detail_files/7794_PIA19839.jpg",
        "auto",
        "Curiosity — rocky outcrop, Marias Pass",
    ),
    (
        "mars_surface_02.jpg",
        "https://mars.nasa.gov/system/resources/detail_files/6453_PIA18390.jpg",
        "auto",
        "Curiosity — 'Hidden Valley' sandy terrain",
    ),
    (
        "mars_surface_03.jpg",
        "https://mars.nasa.gov/system/resources/detail_files/3847_PIA16570.jpg",
        "auto",
        "Curiosity — Yellowknife Bay rock slabs",
    ),
]


def sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:n]


def download(url: str, dest: Path, desc: str) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    print(f"  Downloading {desc} …")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarsScoutBot/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := r.read(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:5.1f}%  {downloaded//1024} KB", end="", flush=True)
        print(f"\r  [ok] {dest.name} ({downloaded//1024} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panorama-only", action="store_true")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = IMAGES[:1] if args.panorama_only else IMAGES
    failed  = []

    print(f"Downloading {len(targets)} Mars image(s) to {data_dir}\n")
    for filename, url, _, desc in targets:
        dest = data_dir / filename
        ok = download(url, dest, desc)
        if not ok:
            failed.append(filename)

    print()
    if failed:
        print(f"⚠  Failed: {failed}")
        print("The mock will use procedural terrain instead.")
        sys.exit(1)
    else:
        print("✓  All images downloaded.")
        print("Restart mock_bridge_node to use real Mars imagery.")


if __name__ == "__main__":
    main()
