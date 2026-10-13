# Phase 4: Active sampling, MRP, sensitivity

## Plan

Gate G4: criteria A4 to A6 in `docs/PLAN.md` are met, or each miss is explained with
numbers.

- **A4:** at an equal survey budget, active sampling gives narrower intervals than uniform
  sampling, with coverage still meeting A1.
- **A5:** for cells with fewer than 30 responses, MRP segment estimates have lower RMSE than
  raw segment means.
- **A6:** under MNAR, the bias is shown, not hidden, and the sensitivity bounds contain the
  truth at the stated assumption.

### Tasks (tests first for every module)

1. **`sampling/designs.py`.** Every design returns `Design(idx, pi, name)` with π for
   *every* pool customer, and asserts π > 0 (CLAUDE.md rule 5). Each takes a
   `numpy.random.Generator`.
   - `uniform(n, N, rng)`: simple random sample without replacement; π = n/N.
   - `stratified(n, strata, allocation, rng, sigma=None)`: fixed n_h per stratum, simple
     random sampling within each, π_i = n_h/N_h.
     - Proportional allocation: n_h ∝ N_h.
     - Neyman allocation: n_h ∝ N_h σ_h, with σ_h taken from the model's predictions
       (no survey data). Every stratum gets at least 2 responses.
   - `active(n, uncertainty, rng, floor=0.1)`: π ∝ √(predicted variance), mixed with the
     uniform rate, π = (1 − τ)·π_unc + τ·n/N with τ = `floor` = 0.1. That puts π at or
     above 0.1 × the uniform rate, which satisfies both PLAN ("mixed with a uniform floor")
     and CLAUDE.md ("floor 0.1 × the uniform rate"). π is capped at 1 and renormalized to
     Σπ = n. **Sampling scheme: see question Q1.**
   - For NPS, the predicted variance of v = 100·(Z₁ − Z₂) given x is
     100²·[p_P + p_D − (p_P − p_D)²]. This is the variance-optimal π shape for PPI with
     λ = 1 (Zrnic & Candès 2024).
2. **`estimators/active.py`, `ppi_active`.** Inverse-probability-weighted PPI with known
   π (Zrnic & Candès, ICML 2024).
   - Over every pool customer i, with ξ_i = 1 if surveyed:
     a_i = λ f_i + (ξ_i/π_i)(Z_i − λ f_i), and θ̂ = mean_i a_i.
   - NPS = gᵀθ̂, with a delta-method Wald interval from the covariance of a_i (2-D, as
     always).
   - λ = 1 by default, with an optional PPI++-style tuned λ for the contrast.
   - Probing tests: forgetting the 1/π weights must fail; π = n/N must reduce to the
     stream form of plain PPI.
3. **`estimators/stratified.py`, `ppi_stratified`.** PPI(++) within each stratum, combined
   with population stratum weights W_h = N_h/N: θ = Σ W_h θ_h,
   Var = Σ W_h² Var_h (Fisch et al., NeurIPS 2024). Each stratum gets its own λ_h,
   using the cross-fitted λ from Phase 3 at small n_h.
4. **`estimators/mrp.py`, `mrp` and `mrp_ppi`.**
   - Two hierarchical logistic models, promoter and detractor, poststratified to known
     population cell counts.
   - NUTS through Bambi/PyMC, with R-hat ≤ 1.01 and bulk ESS ≥ 400 enforced; it raises
     otherwise.
   - A Laplace approximation (`statsmodels` `BinomialBayesMixedGLM.fit_map`) for the
     1,000-replicate sweeps, validated against NUTS on a subset, with the agreement
     reported.
   - **Cells and model: see question Q2.**
5. **`estimators/sensitivity.py`.** δ-adjustment for nonresponse that covariates do not
   explain. It returns the estimate and interval as a function of δ, plus the breakpoint
   δ* where a stated conclusion flips. **Definition of δ: see question Q3.**
6. **Experiments (configs and figures):**
   - `e03_active.py`: uniform vs stratified vs active at equal n (A4).
   - `e04_nonresponse.py`: MAR and MNAR; naive vs PPI++ vs MRP vs MRP + PPI.
   - `e05_segments.py`: 60 segments, cell sizes 5 to 200; RMSE and coverage by cell
     size (A5).
   - `e07_sensitivity.py`: sensitivity curves and δ* on E4 and E6 (A6).
7. **Harness.** Scenarios gain a `design` (uniform / stratified / active) in place of the
   current respondent draw, plus covariate frames for MRP. Existing experiments keep
   their exact draws: uniform stays the default, with its own random streams.
8. **Dependencies.** `statsmodels`, `pymc` and `bambi` are already on the CLAUDE.md list,
   so no approval is needed. Added under a new `mrp` extra, which CI installs.

### Files

`src/silent97/sampling/{__init__,designs}.py`,
`src/silent97/estimators/{active,stratified,mrp,sensitivity}.py`,
`src/silent97/eval/harness.py`, `src/silent97/viz/figures.py`,
`experiments/e0{3,4,5,7}_*.py`, `configs/e0{3,4,5,7}_*.yaml`,
`tests/unit/test_{designs,active,stratified,mrp,sensitivity}.py`,
`tests/coverage/test_coverage.py` (active and stratified under their designs),
`pyproject.toml`/`uv.lock` (`mrp` extra), `Makefile`, the mypy strict override for
`silent97.sampling.*`.

### Open questions (statistical; asked before building those parts)

- **Q1. Active sampling scheme.**
  - Proposed: **Poisson sampling**, each customer surveyed independently with
    probability π_i, as in Zrnic & Candès. Realized n is random with mean n. The
    estimator and variance are exact and simple. A4 then compares at equal *expected*
    budget.
  - Alternative: fixed-size πps sampling (conditional Poisson). The π are only
    approximate there, which makes 1/π weighting less exact.
- **Q2. MRP cells and model.**
  - Proposed: poststratify over **province × age band × tenure band** (10 × 6 × tenure
    bands). The model has random intercepts for province, age band, tenure band and
    segment (province × age), plus a fixed intercept.
  - `mrp_ppi` adds the cell mean of the model's predicted logit as a cell-level fixed
    effect.
  - Note that the simulator's MAR mechanism also depends on digital engagement and
    complaints, which are not in those cells. So plain `mrp` will correct MAR only
    partly; `mrp_ppi` should recover more through the prediction. That is an honest
    result to report, not a flaw to tune away.
- **Q3. Sensitivity δ.**
  - Proposed: δ is a shift in **percentage points** applied to the non-respondents'
    promoter share (+δ) and detractor share (−δ), relative to the model-corrected
    respondent rates. Non-respondent NPS then moves by 2δ, and the overall NPS by
    2δ × (non-respondent share).
  - Reported for δ ∈ [−δ_max, δ_max], with the interval at each δ.
  - δ* is the smallest |δ| at which the stated conclusion (default: "NPS > threshold",
    or for a comparison "difference > 0") no longer holds at the interval's edge.
  - A6 is checked by computing the simulator's *true* δ under MNAR and verifying that
    the interval at that δ covers the truth.

### Decisions (reviewer, 2026-10-13)

- Q1: **Poisson sampling** for the active design.
- Q2: MRP cells **province × age band × tenure band**, with the proposed random
  intercepts. `mrp_ppi` adds the cell-mean predicted logit.
- Q3: **±δ percentage points** on the non-respondents' promoter and detractor shares.
