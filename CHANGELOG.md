# Changelog

## v0.2.0 (2026-09-16)

Packaging, tests and fixes. No change to the models, priors or sampler configuration of v0.1.0; a
v0.1.0 run reloads unchanged.

- `pyproject.toml`: `pip install` works; the four packages install from `app/` with `models.json` as package data.
- MIT licence.
- Tests (`pytest`) and a GitHub Actions workflow on Python 3.10 and 3.12.
- **Fix:** the six-mode equal-peak log-normal mixture prior failed under current JAX with
  `TracerBoolConversionError`, because its root finding ran inside the traced model. `Sampler` now builds
  every prior at construction, outside any trace.
- **Fix:** `BoxGridder` rebuilt its whole output table once per input point; gridding is now vectorised and
  linear in the number of points, with identical output (tested against the original algorithm).
- **Fix:** AIC was `2k - ln L`; now `2k - 2 ln L` in both the regression and Gaussian-process visualisers.
- **Fix:** the `numpyencoder` dependency broke on NumPy 2 (`np.float_` removed); replaced by
  `data_processing.json_encoder.NumpyEncoder`.
- **Change:** default `data_path` and results paths are `data/` and `results/` under the working directory,
  not `/data` and `/results` at the filesystem root.
- `Sampler.sample_all(resample=True)` forces a fresh run instead of reloading a matching saved instance.
- Removed `old_code/` and the committed Sphinx build from the tree (both remain in history).

## v0.1.0 (2026-03-16, commit db20200)

The framework as used for the thesis and for the marine cloud brightening dispersion paper.
