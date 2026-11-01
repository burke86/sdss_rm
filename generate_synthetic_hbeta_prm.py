#!/usr/bin/env python
"""Generate simple synthetic photometric-RM light curves from SDSS-RM data.

For every SDSS-RM object with both ``cont_lc.txt`` and ``hb_lc.txt``, this
script evaluates the H-beta light curve at real continuum epochs using linear
interpolation and writes synthetic bands

    F_a(t) = 1 + delta_C(t) + a * delta_Hbeta(t),

where each input light curve is divided by its own median before its fractional
variation is computed.  Thus ``a`` is the H-beta/continuum fractional-
variability mixing coefficient, not the mean H-beta flux fraction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation


ROOT = Path(__file__).resolve().parent
DEFAULT_AMPLITUDES = (0.1, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "synthetic_hbeta_prm"
    )
    parser.add_argument("--amplitudes", type=float, nargs="+", default=DEFAULT_AMPLITUDES)
    parser.add_argument(
        "--max-line-gap",
        type=float,
        default=30.0,
        help="Maximum observer-frame separation (days) between the two H-beta points bracketing an output epoch.",
    )
    return parser.parse_args()


def load_continuum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, comments="#", dtype=None, encoding="utf-8")
    t = np.atleast_1d(np.asarray(data["f0"], dtype=float))
    flux = np.atleast_1d(np.asarray(data["f1"], dtype=float))
    error = np.atleast_1d(np.asarray(data["f2"], dtype=float))
    return clean_and_sort(t, flux, error)


def load_line(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, comments="#", usecols=(0, 1, 2)))
    return clean_and_sort(data[:, 0], data[:, 1], data[:, 2])


def clean_and_sort(t, flux, error):
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    t, flux, error = t[good], flux[good], error[good]
    order = np.argsort(t, kind="mergesort")
    return t[order], flux[order], error[order]


def reject_local_outliers(t, flux, error, half_window=6, threshold=5.0):
    """Conservatively reject points extreme relative to scatter *and* error.

    The extra quoted-error criterion is intentional: the released continuum is
    assembled from heterogeneous facilities, so a plain 2.5-MAD rule is
    too aggressive when applied to this product as a single stream.
    """
    if flux.size < 2 * half_window + 1:
        return t, flux, error, 0
    windows = sliding_window_view(flux, 2 * half_window + 1)
    centers = flux[half_window:-half_window]
    local_center = np.nanmedian(windows, axis=1)
    local_sigma = 1.4826 * median_abs_deviation(
        windows, axis=1, nan_policy="omit"
    )
    residual = np.abs(centers - local_center)
    center_error = error[half_window:-half_window]
    rejected = np.zeros(flux.size, dtype=bool)
    rejected[half_window:-half_window] = (
        np.isfinite(local_sigma)
        & (local_sigma > 0)
        & (residual > threshold * local_sigma)
        & (residual > threshold * center_error)
    )
    return t[~rejected], flux[~rejected], error[~rejected], int(rejected.sum())


def fractionalize(flux, error):
    scale = float(np.median(flux))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Light-curve median must be finite and positive")
    return flux / scale - 1.0, error / scale, scale


def interpolate_bracketed(t_out, t_line, y_line, e_line, max_gap):
    """Linearly interpolate values and independently propagated errors."""
    right = np.searchsorted(t_line, t_out, side="right")
    valid = (right > 0) & (right < t_line.size)
    left = np.clip(right - 1, 0, t_line.size - 1)
    right = np.clip(right, 0, t_line.size - 1)
    gap = t_line[right] - t_line[left]
    valid &= (gap > 0) & (gap <= max_gap)
    weight_right = np.divide(
        t_out - t_line[left], gap, out=np.zeros_like(t_out), where=gap > 0
    )
    weight_left = 1.0 - weight_right
    y = weight_left * y_line[left] + weight_right * y_line[right]
    e = np.sqrt((weight_left * e_line[left]) ** 2 + (weight_right * e_line[right]) ** 2)
    return valid, y, e, gap


def amplitude_tag(amplitude: float) -> str:
    return f"{amplitude:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def generate_object(obj_dir: Path, output_dir: Path, amplitudes, max_line_gap):
    tc, fc, ec = load_continuum(obj_dir / "cont_lc.txt")
    tl, fl, el = load_line(obj_dir / "hb_lc.txt")
    tc, fc, ec, n_cont_rejected = reject_local_outliers(tc, fc, ec)
    tl, fl, el, n_line_rejected = reject_local_outliers(tl, fl, el)
    dc, dec, continuum_scale = fractionalize(fc, ec)
    dl, del_, line_scale = fractionalize(fl, el)
    valid, dl_interp, del_interp, gaps = interpolate_bracketed(
        tc, tl, dl, del_, max_line_gap
    )
    if np.count_nonzero(valid) < 5:
        raise ValueError("fewer than five safely interpolated epochs")

    obj_out = output_dir / obj_dir.name
    obj_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for amplitude in amplitudes:
        synthetic_flux = 1.0 + dc[valid] + amplitude * dl_interp[valid]
        synthetic_error = np.sqrt(dec[valid] ** 2 + (amplitude * del_interp[valid]) ** 2)
        path = obj_out / f"prm_hb_a{amplitude_tag(amplitude)}_lc.txt"
        header = (
            "Synthetic SDSS-RM photometric Hbeta band\n"
            f"F = 1 + delta_cont + {amplitude:g} * delta_Hbeta\n"
            f"input continuum median = {continuum_scale:.10g}; input Hbeta median = {line_scale:.10g}\n"
            "MJD normalized_flux normalized_flux_err line_bracketing_gap_days"
        )
        np.savetxt(
            path,
            np.column_stack((tc[valid], synthetic_flux, synthetic_error, gaps[valid])),
            fmt=("%.6f", "%.10e", "%.10e", "%.6f"),
            header=header,
        )
        rows.append(
            {
                "object": obj_dir.name,
                "amplitude": amplitude,
                "n_epochs": int(np.count_nonzero(valid)),
                "median_line_gap_days": float(np.median(gaps[valid])),
                "max_line_gap_days": float(np.max(gaps[valid])),
                "n_cont_rejected": n_cont_rejected,
                "n_hbeta_rejected": n_line_rejected,
                "path": str(path.relative_to(output_dir)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if any(a < 0 or not np.isfinite(a) for a in args.amplitudes):
        raise ValueError("All amplitudes must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only artifacts created by this generator, so rerunning with new
    # cleaning settings cannot leave stale light curves for skipped objects.
    for old_path in args.output_dir.glob("rm[0-9][0-9][0-9]/prm_hb_a*_lc.txt"):
        old_path.unlink()
    for old_dir in args.output_dir.glob("rm[0-9][0-9][0-9]"):
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()
    all_rows = []
    skipped = []
    for obj_dir in sorted(args.data_dir.glob("rm[0-9][0-9][0-9]")):
        if not (obj_dir / "cont_lc.txt").exists() or not (obj_dir / "hb_lc.txt").exists():
            continue
        try:
            all_rows.extend(
                generate_object(
                    obj_dir, args.output_dir, args.amplitudes, args.max_line_gap
                )
            )
        except (OSError, ValueError) as exc:
            skipped.append((obj_dir.name, str(exc)))

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        fieldnames = [
            "object", "amplitude", "n_epochs", "median_line_gap_days",
            "max_line_gap_days", "n_cont_rejected", "n_hbeta_rejected", "path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} light curves for {len(all_rows) // len(args.amplitudes)} objects")
    print(f"Manifest: {manifest}")
    if skipped:
        print(f"Skipped {len(skipped)} objects: " + "; ".join(f"{x}: {why}" for x, why in skipped))


if __name__ == "__main__":
    main()
