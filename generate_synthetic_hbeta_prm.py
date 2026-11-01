#!/usr/bin/env python
"""Generate synthetic photometric-RM bands from SDSS-RM imaging and spectra, with no interpolation.

For each SDSS-RM spectrum (an H-beta flux in ``hb_lc.txt``) with a photometric continuum point taken within
MATCH_TOL days (any facility in ``cont_lc.txt`` except the spectroscopic ``sdss_spec`` rows), this script writes a
band containing both, at the photometric epoch:

    F_a(t) = 1 + (delta_C(t) + a * delta_Hbeta(t)) / (1 + a),

where delta = flux / median(flux) - 1 for each light curve and ``a`` is the H-beta/continuum flux ratio in the
band. The errors combine the two measured errors the same way. The continuum part has imaging calibration, as in a
real broad-band survey. rm_lag_profile.py drops these photometric points and the ``sdss_spec`` points from the
continuum channel, so the two channels share no measurements.

usage: python generate_synthetic_hbeta_prm.py --amplitudes 1.0 0.75 0.5 0.25
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC_STREAM = "sdss_spec"
MATCH_TOL = 1.0  # days between a spectrum and its photometric continuum point


def amplitude_tag(amplitude):
    return f"{amplitude:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def fractional(flux, error):
    scale = np.median(flux)
    return flux / scale - 1.0, error / scale


def generate_object(obj_dir, output_dir, amplitudes):
    raw = np.genfromtxt(obj_dir / "cont_lc.txt", comments="#", dtype=None, encoding="utf-8")
    phot = ~np.char.startswith(np.asarray(raw["f3"], dtype=str), SPEC_STREAM)
    tc, fc, ec = (np.asarray(raw[k], dtype=float)[phot] for k in ("f0", "f1", "f2"))
    tl, fl, el = np.loadtxt(obj_dir / "hb_lc.txt", comments="#", usecols=(0, 1, 2)).T
    good_c = np.isfinite(fc) & np.isfinite(ec) & (ec > 0)
    good_l = np.isfinite(fl) & np.isfinite(el) & (el > 0)
    tc, fc, ec = tc[good_c], fc[good_c], ec[good_c]
    tl, fl, el = tl[good_l], fl[good_l], el[good_l]
    if tc.size < 5:
        raise ValueError("fewer than five photometric continuum points")

    o = np.argsort(tc)
    tc, fc, ec = tc[o], fc[o], ec[o]
    j = np.clip(np.searchsorted(tc, tl), 1, tc.size - 1)
    j = np.where(np.abs(tc[j] - tl) < np.abs(tc[j - 1] - tl), j, j - 1)
    matched = np.abs(tc[j] - tl) < MATCH_TOL
    # two spectra matched to one photometric point: keep the closer spectrum
    for k in np.unique(j[matched]):
        same = np.flatnonzero(matched & (j == k))
        matched[same[np.argsort(np.abs(tc[k] - tl[same]))[1:]]] = False
    if np.count_nonzero(matched) < 5:
        raise ValueError("fewer than five H-beta epochs with a photometric continuum point")
    t = tc[j[matched]]
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
            f"Synthetic SDSS-RM photometric Hbeta band: photometric continuum + Hbeta from a spectrum within {MATCH_TOL:g} d\n"
            f"F = 1 + (delta_cont + {a:g} * delta_Hbeta) / (1 + {a:g})\n"
            "MJD(photometric) normalized_flux normalized_flux_err MJD(spectrum)"
        )
        np.savetxt(path, np.column_stack((t, flux, err, tl[matched])), fmt=("%.6f", "%.10e", "%.10e", "%.6f"),
                   header=header)
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
