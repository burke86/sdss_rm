# SDSS-RM Hβ lags with EzTaoX

These scripts measure Hβ reverberation lags from the public SDSS-RM light curves with a Gaussian-process model of
the continuum and the broad-line response. They also test how well the lag is recovered when the line is blended
with the continuum in one band, as in photometric reverberation mapping.

## Installation

The broad-line model, `MixedBLRKernel`, is not in a released EzTaoX yet. Install EzTaoX from the `mixed-blr-kernel`
branch ([LSST-AGN-Variability/EzTaoX#151](https://github.com/LSST-AGN-Variability/EzTaoX/pull/151)). The kernel
also needs the GitHub version of `tinygp`.

```bash
pip install "git+https://github.com/burke86/EzTaoX@mixed-blr-kernel"
pip install -U "git+https://github.com/dfm/tinygp"   # after EzTaoX, which may install a released tinygp
pip install "equinox<0.13" astropy scipy pandas matplotlib
```

- EzTaoX currently imports `equinox._module.BoundMethod`, which equinox 0.13 removed, so use `equinox<0.13`.
- Tested with jax 0.6.2, tinygp 0.3.2.dev17 and equinox 0.12.2. EzTaoX's `pyproject.toml` pins `jax==0.4.31`. If
  you already have a working jax installation, install EzTaoX with `--no-deps` to keep it.

## Data

The light curves are not included. Download them from the
[SDSS-RM final-result release](https://ariel.astro.illinois.edu/sdssrm/final_result/) (Shen et al. 2024, ApJS,
272, 26):

```bash
python download_sdssrm.py            # summary.fits + continuum and Hβ light curves of quasars with an Hβ lag
```

This writes `summary.fits` and `rmXXX/cont_lc.txt` and `rmXXX/hb_lc.txt`.
- `cont_lc.txt` is the merged continuum: MJD, flux density, error, and facility.
- `hb_lc.txt` is the Hβ light curve: MJD, flux, error.

Use `--all` for every object and `--lines` for other broad lines.

## Running

1. **Synthetic photometric bands.** Each one is the continuum plus a × Hβ, with `a` the Hβ/continuum flux ratio:

   ```bash
   python generate_synthetic_hbeta_prm.py --amplitudes 1.0 0.75 0.5 0.25
   ```

   This writes `synthetic_hbeta_prm/rmXXX/prm_hb_a<amp>_lc.txt`. It deletes and rewrites the files it created
   there, so list every amplitude you want in one call.

2. **Job list:** quasars with an SDSS-RM Hβ lag of grade ≥ 4, one line per object and case:

   ```bash
   python make_jobs.py jobs.txt --cases a1 a0p75 a0p5 a0p25      # add "spec" for the Hβ light curve itself
   ```

3. **Fits**, three at a time (finished fits are skipped, so the run can be restarted):

   ```bash
   sh run_lag_profiles.sh jobs.txt runs/run1 3
   ```

   A single fit is `python rm_lag_profile.py <rmid> <case> <out_dir> [--no-slow] [--order K]`.

4. **Recovery summary and plot:**

   ```bash
   python analyze_lag_recovery.py runs/run1     # summary.csv and lag_recovery.pdf
   ```

## Model

`rm_lag_profile.py` fits two channels:
- channel 0, the continuum;
- channel 1, the Hβ light curve (case `spec`) or a synthetic band (case `a<amp>`).

The model has these parts:

- **Kernel:** a damped-random-walk driver x(t) and an Erlang broad-line response ψ (`MixedBLRKernel`, order
  K = 3 by default). Each channel is `c x(t) + r (ψ * x)(t)`. The Hβ channel of `spec` has no continuum term.
- **Noise and means:** one jitter per (channel, facility) group, with the facility (telescope + filter) taken
  from the continuum file. There is one mean per channel and one linear trend.
- **Slow component (on by default; `--no-slow` to turn off):** a Matérn-3/2 process seen only by channel 1, with
  a timescale of about a season or longer. It absorbs slow structure in the line channel that the continuum
  doesn't drive. Without it, the broad-line term can absorb that structure and return confident, very long lags.
- **Lag posterior:** a profile over 24 rest-frame lags (2–900 days, log-normal prior centered at 100 days). At
  each lag the other parameters are maximized, and a Laplace correction is applied.

Each fit writes a JSON file to `<out_dir>/out/`. It contains:
- the lag grid, the posterior and the prior-divided likelihood;
- the 16/50/84% rest-frame lags (`q_post_rest`);
- posterior peaks and multimodal/edge flags;
- the SDSS-RM reference lag;
- the best-fit parameters at each lag.

A fit takes about 1–4 minutes and 0.1–1 GB of memory on a laptop CPU.

## Caveats

- **Synthetic bands are optimistic.** Each one reuses the continuum's epochs and noise, and linearly
  interpolates Hβ between epochs up to 30 days apart.
- **The test lines are strong.** In survey photometry the in-band line/continuum ratio is typically 0.05–0.1,
  below the weakest case here.
- **The slow component can absorb real echo.** It can absorb part of a slow echo when the line channel is pure
  Hβ, so compare `spec` fits with `--no-slow`.

## References

- SDSS-RM data: Shen et al. 2015, ApJS, 216, 4; Shen et al. 2024, ApJS, 272, 26
- EzTaoX: Yu et al. 2026, ApJ, 998, 144
