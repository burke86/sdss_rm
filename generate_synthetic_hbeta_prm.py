#!/usr/bin/env python
"""Generate synthetic photometric-RM bands from the SDSS-RM spectra, with no interpolation.

Each SDSS-RM spectrum gives a continuum flux density (the ``sdss_spec`` rows of ``cont_lc.txt``) and an H-beta flux
(``hb_lc.txt``) at the same epoch. At every such epoch this script writes a band containing both,

    F_a(t) = 1 + (delta_C(t) + a * delta_Hbeta(t)) / (1 + a),

where delta = flux / median(flux) - 1 for each light curve and ``a`` is the H-beta/continuum flux ratio in the
band. The errors combine the two measured errors the same way. rm_lag_profile.py drops the ``sdss_spec`` points
from the continuum channel for these cases, so the two channels share no measurements.

usage: python generate_synthetic_hbeta_prm.py --amplitudes 1.0 0.75 0.5 0.25
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC_STREAM = "sdss_spec"
MATCH_TOL = 0.01  # days between an H-beta epoch and its spectroscopic continuum point


def amplitude_tag(amplitude):
    return f"{amplitude:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def fractional(flux, error):
    scale = np.median(flux)
    return flux / scale - 1.0, error / scale


def generate_object(obj_dir, output_dir, amplitudes):
    raw = np.genfromtxt(obj_dir / "cont_lc.txt", comments="#", dtype=None, encoding="utf-8")
    spec = np.char.startswith(np.asarray(raw["f3"], dtype=str), SPEC_STREAM)
    tc, fc, ec = (np.asarray(raw[k], dtype=float)[spec] for k in ("f0", "f1", "f2"))
    tl, fl, el = np.loadtxt(obj_dir / "hb_lc.txt", comments="#", usecols=(0, 1, 2)).T
    good_c = np.isfinite(fc) & np.isfinite(ec) & (ec > 0)
    good_l = np.isfinite(fl) & np.isfinite(el) & (el > 0)
    tc, fc, ec = tc[good_c], fc[good_c], ec[good_c]
    tl, fl, el = tl[good_l], fl[good_l], el[good_l]
    if tc.size < 5:
        raise ValueError("fewer than five spectroscopic continuum points")

    o = np.argsort(tc)
    tc, fc, ec = tc[o], fc[o], ec[o]
    j = np.clip(np.searchsorted(tc, tl), 1, tc.size - 1)
    j = np.where(np.abs(tc[j] - tl) < np.abs(tc[j - 1] - tl), j, j - 1)
    matched = np.abs(tc[j] - tl) < MATCH_TOL
    if np.count_nonzero(matched) < 5:
        raise ValueError("fewer than five H-beta epochs with a spectroscopic continuum point")
    t = tl[matched]
    dc, dec = fractional(fc[j[matched]], ec[j[matched]])
    dl, del_ = fractional(fl[matched], el[matched])

    obj_out = output_dir / obj_dir.name
    obj_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in amplitudes:
        flux = 1.0 + (dc + a * dl) / (1.0 + a)
        err = np.sqrt(dec**2 + (a * del_) ** 2) / (1.0 + a)
        path = obj_out / f"prm_hb_a{amplitude_tag(a)}_lc.txt"
        header = (
            "Synthetic SDSS-RM photometric Hbeta band at the spectroscopic epochs\n"
            f"F = 1 + (delta_cont + {a:g} * delta_Hbeta) / (1 + {a:g})\n"
            "MJD normalized_flux normalized_flux_err"
        )
        np.savetxt(path, np.column_stack((t, flux, err)), fmt=("%.6f", "%.10e", "%.10e"), header=header)
        rows.append({"object": obj_dir.name, "amplitude": a, "n_epochs": int(t.size),
                     "n_hbeta_unmatched": int(np.count_nonzero(~matched)), "path": str(path.relative_to(output_dir))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "synthetic_hbeta_prm")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25))
    args = parser.parse_args()
    if any(a < 0 or not np.isfinite(a) for a in args.amplitudes):
        raise ValueError("All amplitudes must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only files this script wrote, so a rerun cannot leave stale bands behind.
    for old_path in args.output_dir.glob("rm[0-9][0-9][0-9]/prm_hb_a*_lc.txt"):
        old_path.unlink()
    for old_dir in args.output_dir.glob("rm[0-9][0-9][0-9]"):
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()

    all_rows, skipped = [], []
    for obj_dir in sorted(args.data_dir.glob("rm[0-9][0-9][0-9]")):
        if not (obj_dir / "cont_lc.txt").exists() or not (obj_dir / "hb_lc.txt").exists():
            continue
        try:
            all_rows.extend(generate_object(obj_dir, args.output_dir, args.amplitudes))
        except (OSError, ValueError) as exc:
            skipped.append((obj_dir.name, str(exc)))

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object", "amplitude", "n_epochs", "n_hbeta_unmatched", "path"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Generated {len(all_rows)} light curves for {len(all_rows) // len(args.amplitudes)} objects")
    print(f"Manifest: {manifest}")
    if skipped:
        print(f"Skipped {len(skipped)} objects: " + "; ".join(f"{x}: {why}" for x, why in skipped))


if __name__ == "__main__":
    main()
