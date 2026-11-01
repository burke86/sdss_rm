#!/usr/bin/env python
"""Summarize lag-profile results (rm_lag_profile.py): recovered vs SDSS-RM adopted Hbeta lag, per case.

usage: python analyze_lag_recovery.py <run_dir>   (reads <run_dir>/out/*.json; writes summary.csv and lag_recovery.pdf)
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = Path(sys.argv[1])
CASES = [("spec", "spectroscopic Hβ"), ("a1", "PRM, line/cont = 1.0"), ("a0p75", "PRM, line/cont = 0.75"),
         ("a0p5", "PRM, line/cont = 0.5"), ("a0p25", "PRM, line/cont = 0.25"), ("a0p1", "PRM, line/cont = 0.1")]
rows = []
for f in glob.glob(str(R / "out/*.json")):
    j = json.load(open(f)); opz = 1 + j["z"]
    lo, med, hi = (np.array(j["q_post_rest"]) * opz)
    plo, pmed, phi = (np.array(j["q_prior_divided_rest"]) * opz)
    rows.append(dict(rmid=j["rmid"], case=j["case"], z=j["z"], ref=j["ref_lag_obs"], ref_lo=j["ref_err"][0], ref_hi=j["ref_err"][-1],
                     grade=j["ref_grade"], lag=med, lo=lo, hi=hi, lag_pd=pmed, width=np.log10(hi / lo) / 2,
                     multimodal=j["multimodal"], edge=j["edge"], n_line=j["n_line"]))
d = pd.DataFrame(rows)
d["dex"] = np.log10(d.lag / d.ref)
d["clean"] = ~d.multimodal & ~d.edge
d["tight"] = d.width < 0.3
d["within2x"] = d.dex.abs() < np.log10(2)
d["sigma_off"] = d.dex / d.width  # offset in units of the quoted posterior half-width
d.to_csv(R / "summary.csv", index=False)
nmad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) > 2 else np.nan
out = []
for c, _ in CASES:
    s = d[d.case == c]
    if not len(s): continue
    for lab, m in [("all", np.ones(len(s), bool)), ("clean & tight", s.clean & s.tight)]:
        x = s[m]
        out.append(dict(case=c, subset=lab, N=len(x), median_width=x.width.median(), within2x=x.within2x.mean(),
                        offset=x.dex.median(), nmad=nmad(x.dex), rho=spearmanr(x.ref, x.lag)[0] if len(x) > 3 else np.nan,
                        multimodal=x.multimodal.mean(), edge=x.edge.mean(), off_gt3sigma=(x.sigma_off.abs() > 3).mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(out).round(2).to_string(index=False))

plt.style.use(Path(__file__).resolve().parent / "sdssrm.mplstyle")
cases_present = [c for c in CASES if (d.case == c[0]).any()]
fig, axes = plt.subplots(1, len(cases_present), figsize=(4.3 * len(cases_present), 4.4), sharex=True, sharey=True, squeeze=False)
axes = axes[0]
lim = (3, 1500)
for ax, (c, lab) in zip(axes, cases_present):
    s = d[d.case == c]
    for m, kw in [(s.clean, dict(fmt="o", color="C0", label="single peak")), (~s.clean, dict(fmt="x", color="C3", label="multimodal/edge"))]:
        x = s[m]
        ax.errorbar(x.ref, x.lag, xerr=[x.ref_lo, x.ref_hi], yerr=[x.lag - x.lo, x.hi - x.lag], ms=4, lw=0.6, alpha=0.8, **kw)
    ax.plot(lim, lim, "k-", lw=1)
    ax.fill_between(lim, np.array(lim) / 2, np.array(lim) * 2, color="0.92")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    ax.set_xlabel("SDSS-RM Hβ lag (observed days)")
    ax.text(0.04, 0.95, f"{lab}\nwithin 2×: {s.within2x.mean():.0%} (N={len(s)})", transform=ax.transAxes, va="top", fontsize=9)
axes[0].set_ylabel("lag-profile lag (observed days)")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(R / "lag_recovery.pdf")
print("plot:", R / "lag_recovery.pdf")
