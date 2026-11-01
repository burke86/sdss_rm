"""SDSS-RM light-curve loading and facility grouping for rm_lag_profile.py."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation

ROOT = Path(__file__).resolve().parent

LINE_FILES = {"ha": "ha_lc.txt", "hb": "hb_lc.txt", "mg2": "mg2_lc.txt", "c4": "c4_lc.txt"}

RELFLUX_TO_MAG_SCALE = 2.5 / np.log(10.0)

MIN_GROUP = 3


SPEC_STREAM = "sdss_spec"  # continuum measured from the SDSS-RM spectra (same epochs as the line light curves)


def load_ascii_light_curve(path, continuum=False, exclude_stream=None):
    if continuum:
        data = np.genfromtxt(path, comments="#", dtype=None, encoding="utf-8")
        t = np.asarray(data["f0"], dtype=float)
        flux = np.asarray(data["f1"], dtype=float)
        error = np.asarray(data["f2"], dtype=float)
        stream = np.asarray(data["f3"], dtype=str)
    else:
        data = np.loadtxt(path, comments="#", usecols=(0, 1, 2))
        t, flux, error = data.T
        stream = None
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(error) & (error > 0.0)
    if exclude_stream is not None and stream is not None:
        good &= ~np.char.startswith(stream, exclude_stream)
    t, flux, error = t[good], flux[good], error[good]
    if stream is not None:
        stream = stream[good]
    if t.size < 5:
        raise ValueError(f"Too few valid observations in {path}")
    # A robust multiplicative normalization puts continuum and line channels
    # in fractional-flux units while preserving their reported uncertainties.
    # The released cont_lc.txt product is already calibration-merged.  Its
    # final string column records provenance only; independently normalizing
    # those labels destroys the calibrated long-timescale variability.
    scale = np.full(flux.shape, np.median(flux))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"Invalid median flux in {path}")
    y = flux / scale - 1.0
    yerr = error / scale
    # Within each channel, reject the center of a 13-point
    # running window when it differs from the window mean by >2.5 window MADs.
    keep = np.ones(t.size, dtype=bool)
    window_size = 6
    if t.size >= 2 * window_size + 1:
        order = np.argsort(t, kind="mergesort")
        ys = y[order]
        windows = sliding_window_view(ys, 2 * window_size + 1)
        centers = ys[window_size:-window_size]
        local_mean = np.nanmean(windows, axis=1)
        local_mad = median_abs_deviation(windows, axis=1, nan_policy="omit")
        rejected_sorted = np.zeros(t.size, dtype=bool)
        rejected_sorted[window_size:-window_size] = (
            np.abs(centers - local_mean) > 2.5 * local_mad
        )
        keep[order[rejected_sorted]] = False
    return t[keep], y[keep], yerr[keep], int(np.count_nonzero(~keep))


def load_target(data_dir, rmid, line, synthetic_prm_amplitude=None):
    obj_dir = data_dir / f"rm{rmid:03d}"
    # A synthetic band is built from the spectroscopic continuum points, so the continuum channel drops them:
    # the two channels then never share a measurement or its noise.
    tc, yc, ec, n_cont_rejected = load_ascii_light_curve(
        obj_dir / "cont_lc.txt", continuum=True,
        exclude_stream=None if synthetic_prm_amplitude is None else SPEC_STREAM,
    )
    if synthetic_prm_amplitude is None:
        line_path = obj_dir / LINE_FILES[line]
    else:
        if line != "hb":
            raise ValueError("Synthetic PRM validation currently supports Hbeta only")
        tag = f"{synthetic_prm_amplitude:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        line_path = data_dir / "synthetic_hbeta_prm" / obj_dir.name / f"prm_hb_a{tag}_lc.txt"
    tl, yl, el, n_line_rejected = load_ascii_light_curve(line_path)
    t0 = min(np.min(tc), np.min(tl))
    t = np.concatenate((tc - t0, tl - t0))
    band = np.concatenate((np.zeros(tc.size, dtype=np.int32), np.ones(tl.size, dtype=np.int32)))
    y = np.concatenate((yc, yl))
    yerr = np.concatenate((ec, el))
    order = np.lexsort((band, t))
    return {
        "t": t[order], "band": band[order], "y": y[order], "yerr": yerr[order],
        "continuum_std": max(float(np.std(yc)), 1e-3),
        "line_std": max(float(np.std(yl)), 1e-3),
        "baseline": float(max(np.max(tc), np.max(tl)) - min(np.min(tc), np.min(tl))),
        "n_cont": int(tc.size), "n_line": int(tl.size),
        "n_cont_rejected": n_cont_rejected, "n_line_rejected": n_line_rejected,
    }


def facility_groups(rmid, data, line_from_continuum=True):
    """Group index per point: (channel, facility) with facility = telescope + filter from cont_lc.txt.

    If line_from_continuum is False (released Hbeta light curve), channel 1 is one group of its own.
    """
    raw = np.genfromtxt(ROOT / f"rm{rmid:03d}" / "cont_lc.txt", comments="#", dtype=None, encoding="utf-8")
    raw_t = np.asarray(raw["f0"], float)
    raw_fac = np.array(["_".join(str(s).split("_")[:2]) for s in raw["f3"]])
    o = np.argsort(raw_t); raw_t, raw_fac = raw_t[o], raw_fac[o]
    t, band = np.asarray(data["t"]), np.asarray(data["band"])
    if not line_from_continuum:  # match continuum epochs only
        sub = {"t": t[band == 0], "band": band[band == 0]}
        g0, names0 = facility_groups(rmid, sub)
        groups = np.full(t.size, len(names0))
        groups[band == 0] = g0
        return groups, names0 + ["1:hbeta"]
    # load_target subtracts the earliest kept time; recover that offset from the continuum epochs
    tc = np.sort(t[band == 0])
    for cand in raw_t[:50] - tc[0]:
        j = np.clip(np.searchsorted(raw_t, tc + cand), 1, raw_t.size - 1)
        if np.all(np.minimum(np.abs(raw_t[j] - tc - cand), np.abs(raw_t[j - 1] - tc - cand)) < 1e-3):
            offset = cand
            break
    else:
        raise ValueError("could not match continuum epochs to cont_lc.txt")
    j = np.clip(np.searchsorted(raw_t, t + offset), 1, raw_t.size - 1)
    j = np.where(np.abs(raw_t[j] - t - offset) < np.abs(raw_t[j - 1] - t - offset), j, j - 1)
    if np.any(np.abs(raw_t[j] - t - offset) > 1e-3):
        raise ValueError("some epochs have no continuum facility match")
    keys = np.array([f"{b}:{f}" for b, f in zip(band, raw_fac[j])])
    for b in (0, 1):  # pool small groups per channel
        sel = band == b
        names_, counts = np.unique(keys[sel], return_counts=True)
        small = set(names_[counts < MIN_GROUP])
        keys[sel] = [k if k not in small else f"{b}:other" for k in keys[sel]]
    gnames = sorted(set(keys))
    return np.array([gnames.index(k) for k in keys]), gnames
