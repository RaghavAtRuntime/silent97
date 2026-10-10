# Phase 2: Core estimators

## Plan

### Tasks

1. **Phase 1 follow-ups.** Add a minimal labeler and fixed-n respondent draws (see
   the resolutions in `phase-1.md`).
2. **Estimators** (tests first): `classical`, `naive_respondent`, `naive_model` in
   `estimators/classical.py`; `ppi`, `ppi_pp`, `ppi_crossfit` (plus the `cross_fit`
   helper) in `estimators/ppi.py`; a registry in `estimators/__init__.py`.
3. **Reference cross-checks** (`tests/reference/`) against `ppi_py` 0.2.3: scalar
   means and the NPS contrast, 25 seeds × 4 (n, N) pairs, tolerance 1e-6.
4. **Harness and metrics.** `eval/harness.py` runs (estimator × scenario × n ×
   replicate) grids from YAML and writes Parquet. `eval/metrics.py` computes
   coverage with Wilson intervals, width, bias, RMSE, effective n and responses
   saved.
5. **E1.** `experiments/e01_coverage.py`, `configs/e01_coverage.yaml`, the figure
   in `viz/figures.py`, and the summary table.
6. **Coverage tests** (`tests/coverage/`, slow): valid estimators under MCAR;
   naive baselines failing under MAR/MNAR.

### Files touched

`pyproject.toml`/`uv.lock` (scikit-learn, matplotlib, joblib, polars; dev
`ppi-python`), `src/silent97/estimators/{__init__,_common,classical,ppi}.py`,
`src/silent97/labelers/{__init__,logistic}.py`, `src/silent97/eval/{__init__,harness,metrics}.py`,
`src/silent97/viz/{__init__,figures}.py`, `experiments/e01_coverage.py`,
`configs/e01_coverage.yaml`, `tests/{unit,reference,coverage}/…`, `Makefile`
(`reproduce` runs E1), `docs/results/` (E1 summary CSV and figure, committed),
`.gitignore`.

### Statistical design

All NPS estimators work on the indicator pair Z = (1[y ≥ 9], 1[y ≤ 6]) and the
prediction pair f = (P(promoter), P(detractor)), with the NPS gradient g = (100, −100).

- **classical:** θ̂ = mean(Z); Var = gᵀ cov₀(Z) g / n. ddof = 0, matching
  `ppi_py.classical_mean_ci`.
- **ppi (λ = 1) and the PPI core:** θ̂_λ = λ·mean(f_unlab) + mean(Z − λ f_lab);
  Σ̂ = cov₀(λ f_unlab)/N + cov₀(Z − λ f_lab)/n; Var(NPS) = gᵀ Σ̂ g. These are the
  same formulas as `ppi_py.ppi_mean_ci`.
- **ppi_pp:** a single scalar λ is shared by both coordinates and optimized for the
  variance of the NPS contrast. Because gᵀθ̂_λ = λ·mean(h_unlab) + mean(v − λ h_lab)
  with v = gᵀZ and h = gᵀf, this is exactly scalar PPI++ on (v, h). The plug-in
  optimum, in `ppi_py`'s estimator form, is
  λ̂ = gᵀ C g / (2 (1 + n/N) gᵀ V g), where C = (1/n)(Z̃ᵀ F̃ + F̃ᵀ Z̃) uses centered
  labeled data and V = cov₁ of the pooled labeled and unlabeled predictions.
  This reduces to Cov(v, h) / ((1 + n/N) Var(h)).
  **Clipping:** `ppi_py` clips λ to [0, 1] for means. The PPI++ paper says clipping
  is unnecessary for mean estimation, because the objective is convex for every
  λ ∈ ℝ (per search snippets of arXiv:2311.01453; arxiv.org itself is unreachable
  from here). CLAUDE.md rule 3 says to follow the paper, so `clip=False` is the
  default and `clip=True` reproduces `ppi_py`. Both are cross-checked. If the
  predictions are constant (Var(h) = 0), λ̂ is undefined; every λ then gives the
  classical estimate, so λ̂ = 0 is returned and flagged in the diagnostics.
- **ppi_crossfit:** labeled data split into K = 10 random folds. Model k is trained
  on the other folds, predicts its own fold and all unlabeled rows. The estimator
  is PPI with λ = 1, using the cross-fit labeled predictions and the K-averaged
  unlabeled predictions. This matches `ppi_py.crossppi_mean_ci` (Zrnic & Candès,
  PNAS 2024).
- **naive_respondent:** respondent NPS with the classical interval, computed
  identically to `classical`. It is a separate name because it is the "take
  respondents at face value" baseline used in MAR/MNAR scenarios (see Open
  questions).
- **naive_model:** NPS of the mean prediction over labeled + unlabeled customers,
  with se = sd(h)/√(n + N). It treats the model as truth and is deliberately
  invalid.

### Simulation design for E1

- Population: the Phase 1 simulator, seeded from config (N = 200,000, signal
  strength 0.6).
- Labeler: multinomial logistic regression on the labeler features (covariates,
  segment dummies via province and age band, and `signal`), trained on 5,000
  customers drawn uniformly. Those 5,000 are excluded from the inference pool
  (195,000). The truth is the pool's NPS.
- Each replicate draws n labeled customers from the pool with probability ∝ the
  scenario's response propensity: Efraimidis–Spirakis weighted sampling without
  replacement, which is uniform SRS under MCAR. The unlabeled set is the rest of
  the pool. The harness asserts the labeled draw is disjoint from the labeler's
  training indices.
- Each replicate's RNG is `default_rng([seed, 2, scenario, n, replicate])`, so
  results do not depend on how work is split across processes.

### Open questions

1. **λ clipping**: as above, I follow the paper (no clipping) and keep `ppi_py`'s
   clipped variant as an option. Please confirm.
2. **naive_respondent vs classical**: the spec lists both. Under MCAR they are the
   same estimator, so I implemented one computation under two names. If you meant
   a different naive interval (e.g. ignoring the promoter/detractor covariance),
   tell me, though CLAUDE.md rule 1 forbids that construction.
3. **naive_model coverage** is conditional on one trained labeler, so across
   replicates it is close to 0% or 100%, not a smooth rate. Its bias is reported
   alongside.
