#!/usr/bin/env python
"""Write a job list ("<rmid> <case>" per line) for rm_lag_profile.py.

Selects SDSS-RM quasars with a detected Hbeta lag (HB_LAG_DET > 0) of at least --min-grade, and (for synthetic
cases) a synthetic photometric band in synthetic_hbeta_prm/.

usage: python make_jobs.py <jobs.txt> [--cases spec a1 a0p75 a0p5 a0p25] [--min-grade 4]
"""
import argparse
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
parser.add_argument("--cases", nargs="+", default=["a1", "a0p75", "a0p5", "a0p25"])
parser.add_argument("--min-grade", type=int, default=4)
args = parser.parse_args()

summary = fits.open(ROOT / "summary.fits")[1].data
rmids = [int(r) for r, lag, grade in zip(summary["RMID"], summary["HB_LAG_DET"], summary["HB_LAG_GRADE"])
         if lag > 0 and grade >= args.min_grade]
jobs = []
for rmid in rmids:
    for case in args.cases:
        if case == "spec":
            ok = (ROOT / f"rm{rmid:03d}" / "hb_lc.txt").exists()
        else:
            ok = (ROOT / "synthetic_hbeta_prm" / f"rm{rmid:03d}" / f"prm_hb_{case}_lc.txt").exists()
        if ok:
            jobs.append(f"{rmid} {case}")
args.out.write_text("\n".join(jobs) + "\n")
print(f"{len(jobs)} jobs ({len(set(j.split()[0] for j in jobs))} objects) -> {args.out}")
