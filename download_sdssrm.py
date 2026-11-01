#!/usr/bin/env python
"""Download the public SDSS-RM final-result light curves and summary table into this directory.

Source: https://ariel.astro.illinois.edu/sdssrm/final_result/ (Shen et al. 2024, ApJS, 272, 26).

By default this fetches summary.fits plus the continuum (cont_lc.txt) and H-beta (hb_lc.txt) light curves of every
quasar with a detected H-beta lag, which is all rm_lag_profile.py needs. Existing files are not re-downloaded.

usage: python download_sdssrm.py [--all] [--lines hb ha mg2 c4]
  --all     every object (rm000-rm848), not only those with a detected H-beta lag
  --lines   which emission-line light curves to fetch (default: hb)
"""
import argparse
import urllib.error
import urllib.request
from pathlib import Path

from astropy.io import fits

BASE = "https://ariel.astro.illinois.edu/sdssrm/final_result"
ROOT = Path(__file__).resolve().parent


def fetch(url, path):
    """Download url to path unless it exists; return False if the file is not on the server."""
    if path.exists():
        return True
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--all", action="store_true")
parser.add_argument("--lines", nargs="+", default=["hb"], choices=["hb", "ha", "mg2", "c4"])
args = parser.parse_args()

fetch(f"{BASE}/summary.fits", ROOT / "summary.fits")
summary = fits.open(ROOT / "summary.fits")[1].data
rmids = [int(r) for r, lag in zip(summary["RMID"], summary["HB_LAG_DET"]) if args.all or lag > 0]
n_files = 0
for k, rmid in enumerate(rmids):
    obj = f"rm{rmid:03d}"
    for name in ["cont_lc.txt"] + [f"{line}_lc.txt" for line in args.lines]:
        n_files += fetch(f"{BASE}/{obj}/{name}", ROOT / obj / name)
    if (k + 1) % 25 == 0:
        print(f"{k + 1}/{len(rmids)} objects", flush=True)
print(f"done: {len(rmids)} objects, {n_files} light-curve files in {ROOT}")
