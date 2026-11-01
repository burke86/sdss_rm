#!/usr/bin/env python
"""Hbeta lag profile for one SDSS-RM quasar with EzTaoX's MixedBLRKernel (see README.md).

Channel 0 is the continuum. Channel 1 is the released Hbeta light curve (case "spec") or a synthetic photometric
band "a<amp>" = continuum + amp x Hbeta (see generate_synthetic_hbeta_prm.py). The model is a DRW driver with an
Erlang BLR response (MixedBLRKernel), one jitter per (channel, facility) group, one mean per channel, a linear
trend, and (by default) a slow Matern-3/2 component seen only by channel 1. The lag posterior is a profile over a
fixed rest-frame lag grid with a Laplace correction for the other parameters.

Requires burke86/EzTaoX (branch mixed-blr-kernel) and the GitHub version of tinygp; see README.md.

usage: python rm_lag_profile.py <rmid> <case> <out_dir> [--no-slow] [--order K]
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import tinygp.kernels.quasisep as tkq
from astropy.io import fits
from scipy.optimize import minimize
from tinygp import GaussianProcess

from eztaox.kernels.quasisep import Exp, MixedBLRKernel
from sdssrm_data import RELFLUX_TO_MAG_SCALE, facility_groups, load_target

jax.config.update("jax_enable_x64", True)

LAG_GRID_REST = np.geomspace(2.0, 900.0, 24)
LAG_PRIOR_CENTER_REST, LAG_PRIOR_SIGMA_DEX = 100.0, 1.0
ORDER = 3
MULTIMODAL_MASS = 0.2
SLOW_SCALE_PRIOR = (np.log(200.0), 0.5 * np.log(10.0))  # observed days: median ~1.5 seasons, 0.5 dex
SLOW_AMP_PRIOR = (np.log(0.03), np.log(10.0))  # fractional flux: median 3%, 1 dex


class BandMasked(tkq.Wrapper):
    """A one-dimensional quasiseparable kernel seen only by the bands where mask is nonzero."""

    mask: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.mask[X[1]] * self.kernel.observation_model(X[0])


def main():
    rmid, case, out_dir = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    use_slow = "--no-slow" not in sys.argv
    erlang_order = int(sys.argv[sys.argv.index("--order") + 1]) if "--order" in sys.argv else ORDER
    out_dir.mkdir(parents=True, exist_ok=True)
    row = (lambda s: s[s["RMID"] == rmid][0])(fits.open(ROOT / "summary.fits")[1].data)
    z = float(row["ZSYS"])
    prm = case != "spec"
    data = load_target(ROOT, rmid, "hb", float(case[1:].replace("p", ".")) if prm else None)
    groups, gnames = facility_groups(rmid, data, line_from_continuum=prm)
    t_start = time.time()

    order = np.lexsort((data["band"], data["t"]))
    t = jnp.asarray(data["t"][order]); band = jnp.asarray(data["band"][order], jnp.int32)
    y = jnp.asarray(data["y"][order]); yerr = jnp.asarray(data["yerr"][order]); grp = jnp.asarray(groups[order])
    G = len(gnames)
    tc = 0.5 * (t.min() + t.max()); ts = jnp.maximum(jnp.std(t), 1e-6)
    jitter_loc = jnp.log(jnp.asarray([0.3 * np.median(data["yerr"][groups == g]) for g in range(G)]))
    onepz = 1.0 + z
    trend_scale = 1e-4 * ts / (onepz * RELFLUX_TO_MAG_SCALE)
    norm = jax.scipy.stats.norm.logpdf
    # theta: log_tau, log_sigma, log_amp_blr, log_amp_cont1, trend, mean0, mean1, logjit[G], (log_slow_amp, log_slow_scale)
    names = ["log_tau", "log_sigma", "log_amp_blr", "log_amp_cont1", "trend", "mean0", "mean1"]
    names += [f"logjit_{g}" for g in gnames]
    if use_slow:
        names += ["log_slow_amp", "log_slow_scale"]
    k0 = 7 + G

    def neg_log_post(theta, lag_obs):
        amp_cont1 = jnp.exp(theta[3]) if prm else 0.0
        kernel = MixedBLRKernel(params={"lag": lag_obs, "amp_cont": jnp.stack([jnp.asarray(1.0), jnp.asarray(amp_cont1)]),
                                        "amp_blr": jnp.stack([jnp.asarray(0.0), jnp.exp(theta[2])])},
                                kernel=Exp(scale=jnp.exp(theta[0]), sigma=jnp.exp(theta[1])), order=erlang_order)
        if use_slow:
            kernel = kernel + BandMasked(kernel=tkq.Matern32(scale=jnp.exp(theta[k0 + 1]), sigma=jnp.exp(theta[k0])),
                                         mask=jnp.array([0.0, 1.0]))
        trend, means, logjit = theta[4], theta[5:7], theta[7:7 + G]
        mean = means[band] + jnp.where((band == 0) | prm, trend * (t - tc) / ts, 0.0)
        gp = GaussianProcess(kernel, (t, band), diag=yerr ** 2 + jnp.exp(2 * logjit)[grp] + 1e-10, assume_sorted=True)
        lp = gp.log_probability(y - mean)
        lp += norm(theta[0], jnp.log(10 ** 2.5 * onepz), 1.2 * jnp.log(10.0))
        lp += norm(theta[1], -0.6 * jnp.log(10.0) - jnp.log(RELFLUX_TO_MAG_SCALE), jnp.log(10.0))
        lp += norm(theta[2], -1.0, 1.0)
        lp += norm(theta[3], 0.0, 0.5)
        lp += norm(theta[4], 0.0, trend_scale)
        lp += jnp.sum(norm(theta[5:7], 0.0, 0.2 / RELFLUX_TO_MAG_SCALE))
        lp += jnp.sum(norm(theta[7:7 + G], jitter_loc, 0.5))
        if use_slow:
            lp += norm(theta[k0], *SLOW_AMP_PRIOR) + norm(theta[k0 + 1], *SLOW_SCALE_PRIOR)
        return -lp

    vg = jax.jit(jax.value_and_grad(neg_log_post))
    hess = jax.jit(jax.hessian(neg_log_post))
    bounds = [(np.log(10 * onepz), np.log(1e4 * onepz)), (-8, 3), (-6, 3), (-3, 3), (-2, 2), (-2, 2), (-2, 2)] + [(-12, 0)] * G
    if use_slow:
        bounds += [(np.log(1e-4), np.log(0.5)), (np.log(90.0), np.log(5000.0))]
    free = np.ones(len(names), bool)
    if not prm:
        free[3] = False  # no continuum in the pure line channel

    def fit(lag_obs, starts):
        best = None
        for s in starts:
            def fun(x):
                v, g = vg(jnp.asarray(x), lag_obs)
                return float(v), np.asarray(g) * free
            r = minimize(fun, s, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800, "ftol": 1e-11})
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        return best

    def base(tau):
        v = [np.log(tau), np.log(max(data["continuum_std"], 0.02)), -1.0, 0.0, 0.0, 0.0, 0.0, *np.asarray(jitter_loc)]
        return np.array(v + ([np.log(0.01), np.log(200.0)] if use_slow else []))

    U, logdet, nonpos, maps, prev = [], [], [], [], None
    for lag_rest in LAG_GRID_REST:
        lag_obs = lag_rest * onepz
        r = fit(lag_obs, [base(300.0), base(3000.0)] if prev is None else [prev, base(500.0)])
        prev = r.x
        H = np.asarray(hess(jnp.asarray(r.x), lag_obs))[np.ix_(free, free)]
        ev = np.linalg.eigvalsh(0.5 * (H + H.T))
        nonpos.append(bool(np.any(ev <= 0)))
        logdet.append(float(np.sum(np.log(np.clip(ev, 1e-8, None)))))
        U.append(float(r.fun)); maps.append(r.x.tolist())
    U, logdet = np.array(U), np.array(logdet)
    loglag = np.log10(LAG_GRID_REST)
    log_prior = -0.5 * ((loglag - np.log10(LAG_PRIOR_CENTER_REST)) / LAG_PRIOR_SIGMA_DEX) ** 2
    logp = -U - 0.5 * logdet + log_prior
    post = np.exp(logp - logp.max()); post /= post.sum()
    lik = np.exp(logp - log_prior - (logp - log_prior).max()); lik /= lik.sum()

    def quantiles(w):
        c = np.cumsum(w) - 0.5 * w
        return [float(10 ** np.interp(q, c, loglag)) for q in (0.16, 0.5, 0.84)]

    def peaks(w):
        out = []
        for i in [i for i in range(len(w)) if (i == 0 or w[i] >= w[i - 1]) and (i == len(w) - 1 or w[i] > w[i + 1])]:
            lo = i
            while lo > 0 and w[lo - 1] <= w[lo]:
                lo -= 1
            hi = i
            while hi < len(w) - 1 and w[hi + 1] <= w[hi]:
                hi += 1
            out.append((float(LAG_GRID_REST[i]), float(w[lo:hi + 1].sum())))
        return sorted(out, key=lambda p: -p[1])

    pk = peaks(post)
    res = dict(rmid=rmid, case=case, z=z, slow=use_slow, erlang_order=erlang_order, n_cont=data["n_cont"], n_line=data["n_line"], groups=gnames,
               ref_lag_obs=float(row["HB_LAG_DET"]), ref_err=np.asarray(row["HB_LAG_DET_ERR"], float).ravel().tolist(),
               ref_grade=int(row["HB_LAG_GRADE"]), lag_rest_grid=LAG_GRID_REST.tolist(), U=U.tolist(), logdet=logdet.tolist(),
               hessian_nonpos=nonpos, posterior=post.tolist(), prior_divided=lik.tolist(),
               q_post_rest=quantiles(post), q_prior_divided_rest=quantiles(lik), peaks_post=pk,
               multimodal=bool(len(pk) > 1 and pk[1][1] >= MULTIMODAL_MASS),
               edge=bool(np.argmax(post) in (0, len(post) - 1)), map_theta=maps, theta_names=names,
               t_total=time.time() - t_start)
    tag = ("slow" if use_slow else "noslow") + ("" if erlang_order == ORDER else f"_K{erlang_order}")
    (out_dir / f"rm{rmid:03d}_{case}_{tag}.json").write_text(json.dumps(res))
    q = res["q_post_rest"]
    print(f"DONE rm{rmid:03d} {case} {tag}: lag {q[1]*onepz:.1f} [{q[0]*onepz:.1f},{q[2]*onepz:.1f}] obs d vs ref "
          f"{res['ref_lag_obs']:.1f} | multimodal {res['multimodal']} edge {res['edge']} | {res['t_total']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
