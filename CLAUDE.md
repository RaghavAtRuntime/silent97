# CLAUDE.md: Agent instructions for `silent97`

You are the implementation agent for **silent97**, an open-source Python toolkit that estimates customer satisfaction (NPS) for a whole customer base from a small survey plus model predictions on everyone else, with confidence intervals that are valid and verified by simulation.

The full specification is in `docs/PLAN.md`. This file tells you **how to work**. When they disagree, `docs/PLAN.md` wins on *what* to build, and this file wins on *how* to build it. If something is ambiguous, stop and ask; do not guess on anything statistical.

> Copy this file to `AGENTS.md` if you use a coding agent other than Claude Code.

---

## 1. Prime directives

1. **Correctness over progress.** A wrong confidence interval that looks right is the worst possible outcome of this project. Never trade statistical correctness for speed, coverage of features, or green tests.
2. **One phase at a time.** Only work on the phase you were asked to work on. Stop at its gate and wait for human review. Do not start the next phase.
3. **Never weaken a test to make it pass.** If a test fails, fix the code. If you believe the test itself is wrong, stop, explain why in the progress note, and wait. Do not edit, skip, loosen tolerances on, or `xfail` a test without explicit approval.
4. **Report failures honestly.** If an estimator misses coverage, say so, with numbers. Do not tune parameters on evaluation replicates to make a figure look better.
5. **No private data, ever.** See section 9.

---

## 2. Project in one paragraph

A bank surveys a small, self-selected slice of customers. We have model predictions f(x) for every customer. Estimators combine the small labeled set (y, f(x)) with predictions on the large unlabeled set to estimate NPS with tighter but still valid intervals (PPI, PPI++, cross-fit PPI). Sampling designs choose who to survey (uniform, stratified, active) and record inclusion probabilities π. MRP handles thin segments and nonresponse explained by covariates. A sensitivity analysis bounds bias from nonresponse that covariates cannot explain. A simulator with known ground truth proves the intervals keep their promise; a CFPB complaint benchmark shows the method on real text.

---

## 3. Environment and commands

- Python 3.11+, managed with `uv`. Source layout: `src/silent97/`.
- Install: `uv sync --all-extras`
- The `Makefile` must provide these targets, and you must keep them working:

| Command | Does |
|---|---|
| `make setup` | `uv sync --all-extras` and pre-commit install |
| `make test` | Fast tests only (`pytest -m "not slow"`), must finish in under 2 minutes |
| `make test-slow` | Simulation-based coverage tests (`pytest -m slow`) |
| `make lint` | `ruff check`, `ruff format --check`, `mypy src` |
| `make reproduce` | Rebuild every experiment result and figure from cached labels |
| `make demo` | Regenerate `demo/data/grid.json` and serve `demo/` locally |

- Before declaring any task done, run `make lint` and `make test`. Before a gate, also run `make test-slow`.
- Dependencies: numpy, scipy, polars (or pandas), pyarrow, scikit-learn, lightgbm, statsmodels, pymc, bambi, sentence-transformers, matplotlib, pyyaml, an LLM SDK. Dev: pytest, hypothesis, pytest-cov, ruff, mypy, `ppi-python` (reference implementation, **tests only**; never import it from `src/`).
- Ask before adding any dependency not listed here.

---

## 4. Repository layout

```text
silent97/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Makefile
├── configs/                  one YAML per experiment
├── docs/
│   ├── PLAN.md               the specification
│   ├── memo.md, talk.md
│   ├── ideas.md              parked out-of-scope ideas
│   └── progress/phase-N.md   your phase plans and results notes
├── src/silent97/
│   ├── types.py              Estimate, Design
│   ├── data/                 simulate.py, cfpb.py, amazon.py, schema.py
│   ├── labelers/             base.py, tabular.py, embedding.py, llm.py, cache.py
│   ├── sampling/             designs.py
│   ├── estimators/           nps.py, classical.py, ppi.py, active.py,
│   │                         stratified.py, mrp.py, sensitivity.py
│   ├── eval/                 harness.py, metrics.py, planning.py
│   └── viz/                  figures.py
├── experiments/              e01_coverage.py … e08_power.py
├── demo/                     index.html, app.js, data/grid.json
├── data/                     raw/, processed/          (git-ignored)
├── cache/                    LLM label cache            (git-ignored)
└── tests/
    ├── unit/
    ├── reference/            cross-checks against ppi_py
    └── coverage/             simulation-based, marked slow
```

Do not create top-level directories outside this layout without asking.

---

## 5. Core interfaces (do not change without approval)

```python
# src/silent97/types.py
from typing import NamedTuple
import numpy as np

class Estimate(NamedTuple):
    point: float            # NPS on the -100..100 scale
    lower: float
    upper: float
    alpha: float            # interval is (1 - alpha)
    se: float
    n_labeled: int
    n_unlabeled: int
    method: str
    diagnostics: dict       # lambda_hat, effective_n, weight summary, rhat, ...

class Design(NamedTuple):
    idx: np.ndarray         # indices of surveyed customers
    pi: np.ndarray          # inclusion probability for EVERY customer in the population
    name: str
```

Every estimator in `estimators/` exposes:

```python
def estimate(
    y_lab: np.ndarray,       # shape (n,), integer scores 0..10
    p_lab: np.ndarray,       # shape (n, 2), predicted [P(promoter), P(detractor)]
    p_unlab: np.ndarray,     # shape (N, 2)
    *,
    alpha: float = 0.1,
    **kwargs,
) -> Estimate: ...
```

Design-based estimators (`active`, `stratified`) also take `pi` and/or `strata`. MRP takes covariate frames. Keep signatures consistent and documented.

---

## 6. Statistical rules (read before touching `estimators/` or `sampling/`)

1. **NPS is a two-dimensional mean.** Work with the indicator vector Z = (1[y ≥ 9], 1[y ≤ 6]). NPS = 100 × (E[Z₁] − E[Z₂]). Compute the joint 2×2 covariance and use the gradient (100, −100) for the delta-method variance. Never estimate NPS as the mean of a single −100/0/100 variable *and* then treat promoters and detractors as independent elsewhere; be consistent and test that both routes agree.
2. **PPI rectifier.** For λ-weighted PPI:
   `theta_hat = lambda * mean(p_unlab) + mean(z_lab - lambda * p_lab)`.
   The first term uses the **unlabeled** set. The variance is the sum of the two independent terms' variances (divided by N and n respectively). λ = 1 is plain PPI.
3. **PPI++ λ.** Use the plug-in variance-optimal λ from the PPI++ paper for the mean case. For the 2-D NPS target, optimize λ for the variance of the NPS contrast, not separately per component, and document the derivation in a docstring. Clip λ to [0, 1] only if the paper's estimator does; check, don't assume.
4. **Disjointness.** The model that produced `p_lab` must not have been trained on the labeled inference set unless the estimator is `ppi_crossfit`, which handles this by K-fold cross-fitting. The harness must enforce this, not just document it.
5. **Design-based estimators need π for everyone.** Active and stratified PPI reweight by 1/π. Every sampling design must return π for the whole population, and every π must be strictly positive (enforce a floor; default 0.1 × the uniform rate, configurable). Assert this.
6. **Interval construction.** Default to normal (Wald) intervals from the delta method, matching the reference implementation. If you add any other interval type, it is a new, separately tested method.
7. **MRP.** Fit two hierarchical logistic models (promoter, detractor) with partial pooling over segment dimensions, then poststratify to known population cell counts. With NUTS, check R-hat ≤ 1.01 and bulk ESS ≥ 400; raise on failure. The Laplace-approximation fallback for large sweeps must be validated against NUTS on a subset, and the agreement reported.
8. **Sensitivity.** Delta-adjustment: shift non-respondents' promoter/detractor rates by δ relative to the model-corrected respondent rates. Return the estimate as a function of δ and the breakpoint δ* where a given conclusion flips.
9. **Randomness.** Every function that draws randomness takes a `numpy.random.Generator`, never global state. Every experiment is seeded from config.

If any of these rules seems wrong for a case you are implementing, stop and ask.

---

## 7. Testing requirements

Write the tests **before** the estimator. A phase is not done until all of these exist for the code it adds.

- **Unit tests:** shapes, dtypes, edge cases (n = 1 raises clearly; all promoters; all detractors; p exactly 0 or 1; NaNs rejected).
- **Reference cross-checks** (`tests/reference/`): for scalar means, `ppi`, `ppi_pp` and `ppi_crossfit` must match `ppi_py` to 1e-6 on point estimates and interval bounds across at least 20 random seeds and several (n, N) pairs.
- **Property tests** (hypothesis): if predictions are perfect (p equals the true indicator), PPI width is at most classical width; if predictions are constant, PPI++ reduces to classical within tolerance; estimates are invariant to permuting rows.
- **Probing tests** that would catch known mistakes: using the labeled mean of f instead of the unlabeled mean; forgetting the 1/π weights; computing NPS variance as if promoter and detractor shares were independent; training leakage between labeler and inference set.
- **Coverage tests** (`tests/coverage/`, `@pytest.mark.slow`): under MCAR, 1,000 replicates, nominal 90%, each valid estimator's empirical coverage must have a Wilson 95% interval that overlaps [0.88, 0.92]. Naive baselines are tested to show they *fail* under MAR/MNAR.
- Line coverage ≥ 90% for `estimators/` and `sampling/`.

---

## 8. Phases, tasks and gates

At the start of each phase, write `docs/progress/phase-N.md` with a short plan (tasks, files you will touch, open questions). At the end, append a results note: what was built, test and coverage numbers, what failed, open questions. Then **stop**.

### Phase 1: Foundations and simulator (weekends 1–2)
- Scaffold repo, `pyproject.toml`, `Makefile`, pre-commit, GitHub Actions running `make lint` and `make test`.
- `types.py`, `estimators/nps.py` (indicator helpers, NPS from counts, delta-method variance).
- `data/simulate.py`: population of 200,000 with covariates (age band, province, tenure, product holdings for credit card / chequing / wealth / insurance, digital engagement, complaints in past year, monthly transaction count), segment random effects, latent satisfaction, cumulative-logit 0–10 scores tuned to true NPS ≈ +20; response mechanisms `mcar`, `mar`, `mnar` (U-shaped in latent score); a knob for labeler signal strength. Output Parquet plus a manifest of parameters.
- Verify every citation in `docs/PLAN.md` (title, authors, venue, year, arXiv ID). Write findings to `docs/progress/phase-1.md`. Do not invent identifiers; mark anything you cannot confirm.
- **Gate G1:** simulator's realized NPS is within ±1 point of target across 20 seeds; identical seeds produce byte-identical Parquet; tests and lint pass.

### Phase 2: Core estimators (weekends 3–4)
- `classical`, `ppi`, `ppi_pp`, `ppi_crossfit`, plus the `naive_model` baseline (`classical` on self-selected respondents is the naive respondent baseline).
- `eval/harness.py`: run (estimator × scenario × n × replicate) grids from YAML, write Parquet. `eval/metrics.py`: coverage with Wilson intervals, width, bias, RMSE, effective n, responses saved.
- `experiments/e01_coverage.py` and its figure.
- **Gate G2 (most important):** reference cross-checks pass at 1e-6; E1 coverage within criteria for every valid estimator at n ∈ {50, 100, 200, 500, 1000}; the human reviews every line of `estimators/`. Write a plain-English explanation of each estimator, pointing to the exact lines that implement each formula.

### Phase 3: Labelers and real text (weekend 5)
- `labelers/base.py` (interface returning calibrated (n, 2) probabilities), `tabular.py` (LightGBM multiclass + isotonic calibration), `embedding.py` (sentence embeddings + logistic regression), `llm.py` (local open-weight model; zero-shot probabilities for detractor / passive / promoter from its answer-option token probabilities), `cache.py` (disk cache keyed by hash of model, prompt version and input).
- `data/cfpb.py`: download the public CFPB complaint export, keep 2015 to April 2017 complaints with a narrative and a "Consumer disputed?" value; outcome = disputed. Labelers see narrative, product, issue and state only.
- `data/amazon.py` (Amazon Reviews 2023, Gift Cards; score = 2 × stars), `eval/planning.py` (ρ from past responses → survey sizes).
- `experiments/e02_model_quality.py`, `experiments/e06_cfpb.py`, `experiments/e06_amazon.py`.
- LLM rules: dry-run mode that prints item count and projected compute before any scoring; hard cap on items from config; never re-score a cached key; only public CFPB, Amazon Reviews 2023 (owner-approved, research use) or synthetic text as input. The model runs locally; nothing is sent to an external API.
- **Gate G3:** all labels cached; LLM compute and item count reported; E6 coverage reported for each labeler.

### Phase 4: Active sampling, MRP, sensitivity (weekend 6)
- `sampling/designs.py`: `uniform`, `stratified` (proportional and Neyman), `active` (π ∝ sqrt of predicted variance, mixed with a uniform floor, scaled to budget).
- `estimators/active.py`, `estimators/stratified.py`, `estimators/mrp.py` (`mrp`, `mrp_ppi`), `estimators/sensitivity.py`.
- `experiments/e03_active.py`, `e04_nonresponse.py`, `e05_segments.py`, `e07_sensitivity.py`.
- **Gate G4:** criteria A4–A6 in `docs/PLAN.md` met, or each miss explained with numbers.

### Phase 5: Demo and memo (weekends 7–8)
- `experiments/e08_power.py`: simulated A/B test with a true +5 NPS lift; power vs n for classical vs PPI++ difference in means.
- Export `demo/data/grid.json`; build `demo/index.html` + `app.js` (plain HTML/JS, no framework, no server, no external calls at view time): survey-budget slider, model-quality slider, nonresponse toggle; chart of 50 replicate intervals against the true NPS with misses highlighted; readout of coverage, mean width, and "worth N classical responses." Must work on a phone screen.
- Draft `docs/memo.md` (2–3 pages, outline in `docs/PLAN.md`) and `docs/talk.md` (5-minute talk track). Every number in them must come from a results file; cite the file.
- README: one-paragraph pitch, one headline figure, quickstart, reproduce instructions, limits.
- **Gate G5:** `make reproduce` passes from a clean clone in under 30 minutes using cached labels; all acceptance criteria A1–A8 reported in `docs/progress/phase-5.md`.

---

## 9. Data governance (non-negotiable)

- **No TD Bank data, documents, screenshots, customer information or branding** in the repo, prompts, caches, commits or demo. Do not mention TD in code, comments, the README or the demo. The memo may describe "a large bank" generically.
- No scraping of app stores, social media, or anything behind a login or against a site's terms.
- Only public CFPB data or synthetic data may be sent to any external API. Amazon Reviews 2023 (research use, approved by the owner) is used locally only.
- Secrets come from `.env` (git-ignored). Never print, log or commit keys. Add a pre-commit secret scanner.
- `data/` and `cache/` are git-ignored. Do not commit raw data.

---

## 10. Coding standards

- Type hints everywhere in `src/`; `mypy --strict` on `estimators/` and `sampling/`.
- NumPy-style docstrings. Every estimator docstring includes the formula it implements, its validity assumptions, and a citation.
- Pure functions in `estimators/` and `sampling/`: no I/O, no global state, no printing.
- Validate inputs at public boundaries and raise `ValueError` with a message that says what was expected.
- Vectorize; no Python loops over customers in hot paths. Experiments may loop over replicates and use `joblib` for parallelism.
- Figures: matplotlib, consistent style module in `viz/`, always draw the nominal coverage line and Monte Carlo error bands, label axes with units.
- Small, focused commits with messages in the imperative ("Add PPI++ lambda estimator"). One branch per phase: `phase-N`.

---

## 11. When to stop and ask

Stop and write your question in `docs/progress/phase-N.md` (and in chat) when:

- a formula in `docs/PLAN.md` or this file seems wrong or underspecified;
- a test you did not write seems wrong;
- a coverage result is outside criteria and you cannot explain why after one investigation pass;
- you need a new dependency, a new top-level directory, or a change to a core interface;
- an LLM run would exceed the configured cap;
- anything touches the governance rules in section 9.

Do not stop for routine engineering decisions (file organization within a module, helper names, plotting details). Make a reasonable choice and note it in the progress file.

---

## 12. Definition of done (every task)

- [ ] Tests written first and passing (`make test`; `make test-slow` at gates)
- [ ] `make lint` clean
- [ ] Docstrings with formula, assumptions and citation for any statistical code
- [ ] No test weakened, skipped or loosened
- [ ] Progress note updated with what changed and any open questions
- [ ] Nothing from section 9 violated
