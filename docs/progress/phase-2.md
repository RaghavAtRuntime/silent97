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

---

## Results (Gate G2)

### What was built

| File | Contents |
|---|---|
| `src/silent97/estimators/classical.py` | `classical` (also the naive respondent baseline under self-selection), `naive_model` |
| `src/silent97/estimators/ppi.py` | Generic PPI core (`ppi_mean_pointestimate`, `ppi_mean_cov`, `optimal_lambda`, `ppi_mean_ci`); NPS estimators `ppi`, `ppi_pp`, `ppi_crossfit`; the `cross_fit` helper |
| `src/silent97/estimators/_common.py` | `alpha` and input validation; Wald interval |
| `src/silent97/estimators/__init__.py` | `REGISTRY` (name → estimator) and `VALID_UNDER_MCAR` |
| `src/silent97/labelers/logistic.py` | Phase 2 labeler: standardized multinomial logistic regression (C = 1). Predictions use one stacked matrix product over all fold models (`predict_folds`), checked against scikit-learn's `predict_proba` to 1e-12 |
| `src/silent97/eval/harness.py` | YAML grid config, scenario preparation (simulate → train the labeler on a disjoint split → inference pool), respondent draws, leak guard, parallel replicates, Parquet output |
| `src/silent97/eval/metrics.py` | Wilson intervals; coverage, width, bias, RMSE, mean λ, effective n, responses saved |
| `src/silent97/viz/figures.py` | Shared style; coverage-and-width figure |
| `experiments/e01_coverage.py`, `configs/e01_coverage.yaml` | E1 |
| `docs/results/e01_coverage{.png,_summary.csv,_summary.md}` | E1 outputs (committed; raw replicates in git-ignored `data/results/`) |
| `tests/reference/test_ppi_py.py` | 901 cross-checks against `ppi_py` |
| `tests/coverage/test_coverage.py` | Slow coverage suite |

Naming: CLAUDE.md §5 says every estimator exposes `estimate(...)`. `ppi.py` holds three
estimators, so each function is named after its estimator (`ppi`, `ppi_pp`, …), all with
the `estimate` signature, and `estimators.REGISTRY` maps names to them.

### Gate G2 checks

| Criterion | Result |
|---|---|
| Reference cross-checks at 1e-6 (A2) | **Pass.** 901 tests across 25 seeds × 4 (n, N) pairs ((50, 500), (100, 5,000), (500, 2,000), (1,000, 100,000)). Scalar means: PPI (λ = 1), PPI++ (clipped, matching `ppi_py`'s default, and unclipped against `ppi_py`'s own λ formula), cross-PPI, classical. NPS contrast: `classical`, `ppi`, `ppi_pp` (both clip settings), `ppi_crossfit` against `ppi_py` applied to v = gᵀZ, h = gᵀf. Tolerance is an absolute 1e-6 (no relative slack); the largest observed difference on NPS bounds is 1.2e-12. |
| E1 coverage within criteria for every valid estimator at n ∈ {50, 100, 200, 500, 1000} (A1) | **Pass: 20/20** (4 valid estimators × 5 n). Table below. |
| Naive baselines fail | `naive_model`: 0.0% coverage at every n in E1. `classical` on self-selected respondents (the naive respondent baseline): 15.4% (MAR) and 31.2% (MNAR) at n = 1,000. |
| `make lint` | Clean. mypy `--strict` on `estimators/`. |
| `make test` | 1,143 tests in about 27 s (unit, property, probing, reference). |
| `make test-slow` | **15/15 passed** (4 min 52 s), after the review decisions below. (The first run, before the decisions, had 18 passed and 2 failed: the `naive_model` self-selection cases discussed below.) |
| Line coverage | `estimators/`: 100% of statements and branches (unit + reference tests). Package total 99%. |

### E1 (configs/e01_coverage.yaml; seed 20261011; 1,000 replicates; MCAR; signal 0.6)

Source: `docs/results/e01_coverage_summary.csv` (also `…_summary.md`, figure
`docs/results/e01_coverage.png`). Truth is the inference pool's NPS, 19.94 (195,000
customers). The labeler's predictions correlate 0.537 with the per-customer NPS value
on the pool.

| n | classical | PPI | PPI++ | cross-fit PPI | naive model |
|---|---|---|---|---|---|
| 50 | 0.889 [0.868, 0.907] | 0.900 [0.880, 0.917] | 0.888 [0.867, 0.906] | 0.889 [0.868, 0.907] | 0.000 |
| 100 | 0.906 [0.886, 0.923] | 0.904 [0.884, 0.921] | 0.904 [0.884, 0.921] | 0.908 [0.888, 0.924] | 0.000 |
| 200 | 0.900 [0.880, 0.917] | 0.886 [0.865, 0.904] | 0.884 [0.863, 0.902] | 0.890 [0.869, 0.908] | 0.000 |
| 500 | 0.903 [0.883, 0.920] | 0.914 [0.895, 0.930] | 0.916 [0.897, 0.932] | 0.902 [0.882, 0.919] | 0.000 |
| 1000 | 0.892 [0.871, 0.910] | 0.906 [0.886, 0.923] | 0.906 [0.886, 0.923] | 0.906 [0.886, 0.923] | 0.000 |

Mean width (NPS points) and effective n (classical width / method width, squared, × n):

| n | classical | PPI | PPI++ | cross-fit PPI |
|---|---|---|---|---|
| 50 | 38.15 (50) | 32.08 (71) | 31.97 (71) | 40.11 (45) |
| 100 | 27.22 (100) | 22.90 (141) | 22.87 (142) | 26.62 (105) |
| 200 | 19.30 (200) | 16.29 (281) | 16.28 (281) | 17.71 (237) |
| 500 | 12.23 (500) | 10.33 (701) | 10.33 (701) | 10.67 (657) |
| 1000 | 8.65 (1000) | 7.30 (1403) | 7.30 (1404) | 7.41 (1364) |

Every |bias| is ≤ 0.44 NPS points for the valid estimators, against RMSEs of 2.2 to 12.4.
PPI++'s mean λ̂ is 0.97 to 0.98. The labeler is calibrated, so Cov(v, h) ≈ Var(h)
and λ* ≈ N/(N + n) ≈ 1, and PPI and PPI++ nearly coincide. `naive_model` has
bias +0.44 with mean width 0.34, so it never covers. Its "effective n" of about
650,000 is the false precision the memo warns about.

Runtime: 7 min 40 s on 4 cores (`n_jobs: 4`), after the review's efficiency pass
(first run: 12 min 56 s). Almost all of it is cross-fitting: 10 logistic fits plus
predictions for 195,000 rows per replicate.

### Slow coverage suite (`tests/coverage/test_coverage.py`)

- **MCAR, seed 20261012 (not E1's seed), 1,000 replicates, n ∈ {50, 200, 1000}:**
  all 15 (estimator, n) checks for `classical`, `ppi`, `ppi_pp` and `ppi_crossfit`
  (plus the since-removed alias `naive_respondent`) pass (Wilson 95% interval overlaps [0.88, 0.92]).
  `naive_model` fails coverage under MCAR, as intended.
- **MAR and MNAR, default population (seed 20261010), n = 1,000, 1,000 replicates:**
  the naive respondent baseline (now `classical`) fails as intended. Coverage is 15.4% [13.3%, 17.8%] under MAR
  (bias +6.92, RMSE 7.41) and 31.2% [28.4%, 34.1%] under MNAR (bias +5.67).
- **FAILED: `naive_model` under MAR and MNAR**: 100% coverage [99.6%, 100%] in both,
  with bias +0.14 and mean width 0.35.

  *Why:* `naive_model` never uses the survey responses. Its labeler is trained on a
  *uniform* split (`labeler.train_mechanism: mcar`), so the response mechanism cannot
  affect it; MAR and MNAR give identical numbers. Within one experiment the labeler
  is fixed, so its error is the same in every replicate, and its coverage is all or
  nothing. On this population the error (+0.14) happens to fall inside its ±0.17
  half-width. On E1's population it is +0.44, giving 0% coverage. The test assumed
  `naive_model` would be hurt by self-selection, which this design never allows.
  That premise was mine, and it was wrong. The estimator behaves as specified.

  *Not changed* (CLAUDE.md §1.3: no editing a failing test without approval).
  Options:
  (a) under MAR/MNAR, train the labeler on respondents drawn under the scenario's own
  mechanism (`train_mechanism` = scenario mechanism). This is realistic, since past
  survey waves are self-selected too, and the model then inherits the bias;
  (b) drop the `naive_model` cases from the self-selection test and keep its MCAR
  failure test, which already shows the false precision;
  (c) assess `naive_model` across many independently trained labelers, so its
  coverage is a rate rather than 0/1.
  I recommend (a) for E4 plus (b) for this test, but I'll wait for your call.

### Observations and things that did not go as hoped

1. **Cross-fit PPI is worse than classical at n = 50** (width 40.1 vs 38.2;
   effective n 45). Each fold model is trained on 45 customers with 25 features, so
   its predictions are noisy, and λ = 1 has no power tuning to fall back on. Coverage
   is still within criteria. From n = 100 it gains, and it approaches PPI by
   n = 1,000. This is expected behaviour, not a bug. It is the reason the plan pairs
   cross-fitting with a separately trained labeler.
2. **A3 may be unreachable at exactly ρ = 0.5 as written.** E1's labeler has
   correlation 0.537 and PPI++ gains 1.40× effective n. That matches the theory: with
   N ≫ n, Var(PPI++)/Var(classical) ≈ 1 − ρ², so the gain is about 1/(1 − ρ²). That
   is 1.33× at ρ = 0.5 and 1.41× at ρ = 0.537, and reaching 1.5× needs ρ ≥ 0.577.
   A3 is measured in E2 (Phase 3), so nothing was tuned here. See the questions
   below.
3. **Performance work.** Cross-fit replicates first cost 1.4 s; folding the scaler
   into the coefficients and batching the K fold predictions brought this to
   0.57 s. Results were unchanged: the identical widths were checked before and
   after.

### Plain-English guide to each estimator (with the lines that implement it)

All estimators turn each surveyed customer's 0–10 score into two yes/no flags,
"promoter" and "detractor" (`nps.py:187`). NPS is 100 × (promoter share −
detractor share). Uncertainty is always computed on the *pair* of shares, including how
they move together, and then combined with the weights (+100, −100)
(`nps.py:299`: gᵀΣg). Every interval is point ± 1.645 × standard error for 90%
(`_common.py:29-30`).

- **classical** (`classical.py:49-90`). Average the two flags over respondents
  (`:72`, covariance with ddof = 0 at `nps.py:269`). NPS is the difference (`:73`),
  and the standard error is the delta-method one from the 2×2 covariance (`:74`).
  Valid when respondents are a random sample. On self-selected respondents it is
  the "trust whoever answered" baseline, which fails (slow suite).
- **naive_model** (`classical.py:93-124`). Ignores the survey answers. It averages
  the model's predicted NPS over everyone (`:110-111`), with a standard error that
  only reflects how many predictions were averaged (`:112`). Its interval is tiny and
  wrong whenever the model is even slightly biased. This is the "LLM-only" trap.
- **ppi** (`ppi.py:334-356`, shared engine `_nps_ppi` at `:282-297`). Start from
  the model's average prediction over the *unsurveyed* customers, then correct it by
  how wrong the model was, on average, on the *surveyed* customers:
  `λ·mean(f_unlab) + mean(Z − λ·f_lab)` (`:177`, called at `:292`). With λ = 1
  (`:356`) the model's bias cancels out in expectation, so the estimate is unbiased
  however bad the model is. The variance adds two independent pieces: the spread of
  predictions on the unsurveyed customers divided by N (`:293`), and the spread of
  the model's errors on the surveyed customers divided by n (`:294`), combined at
  `:295`. A good model has small errors, which makes the interval narrow.
- **ppi_pp** (`ppi.py:359-383`, λ at `:374`, `optimal_lambda` at `:189-238`). The
  same correction, but it chooses how much to trust the model. λ is the
  covariance between true and predicted NPS values, divided by (1 + n/N) times the
  prediction variance (`:217-222`, `:230`, `:234`). It is chosen for the NPS
  difference itself, using the contrast g, not per share. A useless model gives
  λ ≈ 0, which is the classical estimate; a good calibrated model gives λ ≈ 1. If
  predictions are constant, λ is set to 0 (`:232-233`). Clipping to [0, 1] is off
  by default (paper) and on with `clip=True` (`ppi_py`) (`:235-237`).
- **ppi_crossfit** (`ppi.py:386-411`; `cross_fit` at `:428-471`). For when there is
  no separate training data. Split the surveyed customers into 10 groups (`:456-458`).
  For each group, train on the other nine (`:464`) and predict that group
  (`:466`), so no customer is ever scored by a model that saw their answer. Each of
  the 10 models also scores every unsurveyed customer; those 10 predictions are
  averaged (`:410`). Then apply plain PPI (`:411`).
- **Disjointness** is enforced in the harness: the inference pool excludes the
  labeler's training customers (`harness.py:290`), and every draw is checked
  (`harness.py:310-311`, raising "training leak").

### Open questions for review

1. ~~λ clipping~~: resolved (below).
2. ~~naive_respondent~~: resolved, merged into `classical`.
3. ~~A3 wording~~: resolved, reworded.
4. ~~The failing `naive_model` self-selection test~~: resolved, (a) for E4 and (b) here.
5. ~~Runtime~~: efficiency pass done (below). E1 now takes 7 min 40 s on 4 cores.
   A7 (30 minutes for all of `make reproduce`) may still get tight once E2–E8
   exist; revisit then.

### Reviewer decision on the failing `naive_model` tests

The reviewer chose **(a) for E4** and **(b) for this test**.

- **(a):** `labeler.train_mechanism` now also accepts `"scenario"`, which trains the
  labeler on respondents drawn under the scenario's own mechanism. The default stays
  `"mcar"`, so E1 is unchanged. Phase 4's E4 config will set
  `train_mechanism: scenario`. Test: `test_labeler_trained_on_scenario_mechanism`.
  A one-replicate check on the default population (seed 20261010, informational
  only, nothing tuned): `naive_model` bias is +0.14 for every mechanism with uniform
  training. With `train_mechanism: scenario` it is +0.14 (MCAR), **+1.97 (MAR)**
  and **−2.66 (MNAR)**, against a half-width of about 0.2, so it fails under self-selection.
- **(b):** the self-selection coverage test now checks only the respondent baseline
  (`test_classical_fails_under_self_selection`, after the rename below). The
  `naive_model` MCAR failure test is unchanged.

### Reviewer decisions on the remaining open questions

1. **λ clipping: "take the best approach".** Kept as built. The default is no
   clipping: for means the variance objective is convex for every λ, so the plug-in
   is variance-optimal, and this is what the PPI++ paper and CLAUDE.md rule 3
   prescribe. `clip=True` reproduces `ppi_py`. Both are cross-checked.
2. **A3 reworded** in `docs/PLAN.md`: "within Monte Carlo error of 1/(1 − ρ²), and at
   least 1.5× when ρ ≥ 0.6". At ρ = 0.6 theory gives 1/(1 − 0.36) = 1.56×. E2
   (Phase 3) will test it.
3. **One name instead of `naive_respondent`.** Removed the alias. `classical`
   is now also the naive respondent baseline under self-selection. Updated:
   registry, `VALID_UNDER_MCAR`, E1 config, figure styles (palette slots 1–5 in
   fixed order), tests, `docs/PLAN.md` (estimator table and the baseline sentence)
   and the Phase 2 task line in `CLAUDE.md`. Two tests that only compared the two
   names were removed. In their place, `test_rows_match_estimators_on_the_replicate_draw`
   rebuilds a replicate's survey draw independently and checks the stored
   `classical` and `ppi` rows against it.
4. **Inefficiencies.** Profiling one cross-fit replicate (0.48 s) found two:
   - the softmax reduced over a length-3 class axis of a 195,000 × 10 × 3 array.
     Numpy is slow at that; it is now elementwise over the three class slices
     (`labelers/logistic.py`, `predict_folds`);
   - `validate_predictions` made separate NaN and finite passes and summed rows
     along a length-2 axis. It now makes one finite pass (NaN is identified only on
     failure) and adds the two columns. Same checks, same messages.

   Cross-fit replicates went from 0.48 s to 0.35 s, and the other estimators from
   0.059 s to 0.040 s. E1 went from 12 min 56 s to 7 min 40 s. Every E1 metric
   (coverage, width, bias, RMSE, λ, effective n) is identical to the previous run in
   all 25 shared rows. The logistic fits (about 0.12 s per replicate) are necessary
   work and were left alone. No statistical choice (K, replicates, n) was changed.

Phase 2 stops here, pending your review at Gate G2.
