# Phase 1: Foundations and simulator

## Plan

### Tasks

1. **Scaffold.** `pyproject.toml` (uv, Python ≥ 3.11, src layout), `Makefile` with the six
   targets from CLAUDE.md §3, `.pre-commit-config.yaml` (ruff, ruff-format, a secret
   scanner, basic hygiene hooks), `.gitignore` (`data/`, `cache/`, `.env`, build output),
   and `.github/workflows/ci.yml` running `make lint` and `make test`.
2. **Core types.** `src/silent97/types.py` with `Estimate` and `Design` exactly as in
   CLAUDE.md §5.
3. **NPS helpers.** `src/silent97/estimators/nps.py`: score validation, the indicator
   vector Z = (1[y ≥ 9], 1[y ≤ 6]), NPS from shares and from counts, the joint 2×2
   covariance, and the delta-method variance with gradient (100, −100). Prediction
   validation for `(n, 2)` probability arrays also lives here because every estimator
   will need it.
4. **Simulator.** `src/silent97/data/simulate.py` plus `data/schema.py` (column
   groups, so later phases can enforce that ground-truth columns never reach a
   labeler). Writes Parquet and a JSON manifest. Config in `configs/simulate.yaml`.
5. **Citations.** Verify every reference in `docs/PLAN.md`; findings below.
6. **Gate G1 tests.** Realized NPS within ±1 point of +20 across 20 seeds;
   byte-identical Parquet for identical seeds.

### Files touched

`pyproject.toml`, `uv.lock`, `Makefile`, `.python-version`, `.gitignore`,
`.pre-commit-config.yaml`, `.secrets.baseline`, `.github/workflows/ci.yml`,
`README.md` (stub), `configs/simulate.yaml`,
`src/silent97/{__init__,types}.py`, `src/silent97/estimators/{__init__,nps}.py`,
`src/silent97/data/{__init__,schema,simulate}.py`, `tests/unit/*`,
`docs/progress/phase-1.md`.

### Simulator design (summary; details in the module docstring)

- **Covariates** (200,000 customers): age band (6 levels), province (10 Canadian
  provinces, roughly population-weighted), tenure in years and tenure band, four
  product flags (credit card, chequing, wealth, insurance), digital engagement in
  [0, 1], complaints in the past year (count), monthly transaction count. Products,
  engagement, transactions and complaints depend on age and on each other so the
  covariates are realistically correlated.
- **Segments.** segment = province × age band (60 cells, matching E5's "60 segments").
  Random intercepts for province, age band and segment.
- **Latent satisfaction** η = Xβ + u_province + u_age + u_segment + σ_e·e, e ~ N(0, 1).
  The e term is satisfaction that no covariate explains; it is what makes MNAR
  non-ignorable.
- **Scores** from a cumulative-logit model: P(Y ≤ k | η) = logistic(c_k + s − η). The
  cutpoint *shape* c is fixed in config (from a realistic base distribution at η = 0);
  the scalar shift s is solved by root-finding so that the population's *expected*
  NPS, (1/N) Σ_i [P(Y_i ≥ 9 | η_i) − P(Y_i ≤ 6 | η_i)] × 100, equals the target. Tuning
  uses the drawn η, never the drawn scores. Equivalently Y* = η + ε, ε ~ Logistic(0, 1),
  and Y counts the cutpoints below Y*.
- **Response mechanisms.** For each customer, a response propensity under each of
  `mcar` (constant), `mar` (logit-linear in covariates) and `mnar` (the MAR part plus
  κ·z(η)², U-shaped in standardized latent satisfaction). The intercept is solved so
  the mean propensity equals the configured response rate (default 3%). All three
  propensities are written, plus one realized Bernoulli response flag per mechanism.
- **Labeler signal knob.** A feature `signal = a·z(Y*) + √(1 − a²)·ξ`, ξ ~ N(0, 1),
  where a = `signal_strength` ∈ [0, 1]. a = 0 is pure noise; a = 1 reveals Y* and so
  the score. Realized correlations of `signal` with the score and with the per-customer
  NPS value are recorded in the manifest so E2 can map the knob to model–outcome
  correlation.
- **Independent random streams.** The caller's `Generator` is spawned into fixed
  child streams (covariates, effects, outcome, signal, response). Changing the signal
  knob or a response parameter therefore leaves every other column unchanged, which
  lets E2 and E4 compare settings on the same population.

### Decisions made (routine)

- **Branch.** CLAUDE.md asks for `phase-1`; this session is required to push to
  `claude/wonderful-mendel-wik46d`. Work goes there; rename on merge if you like.
- **Heavy dependencies deferred.** Phase 1 needs only numpy, scipy (root-finding),
  pyarrow and pyyaml. LightGBM, PyMC/Bambi, sentence-transformers, the LLM SDK,
  matplotlib and `ppi-python` will be added in the phases that use them, so CI stays
  fast and the lock file only carries what is used.
- **ddof.** `ppi_py` 0.2.3 computes the standard errors of its mean intervals with
  `ndarray.std(0)` (ddof = 0) and its λ estimates with `np.cov` (ddof = 1). The NPS
  covariance helper takes `ddof` as a keyword, defaulting to 0 to match the reference
  mean intervals. Phase 2 must recheck this against the version pinned for tests.
- **Parquet determinism.** Tables are built directly from numpy arrays with pyarrow
  (no pandas metadata), written with fixed compression and no timestamps. The
  manifest has no timestamps either, so it is also byte-identical across runs.

### Open questions

1. **Dev tooling not on the CLAUDE.md list.** `pre-commit` (needed by `make setup`)
   is added to the dev group. The secret scanner (`detect-secrets`) runs inside
   pre-commit's own environment, not as a project dependency. Please confirm both.
2. **Predictions before Phase 3.** Phase 2 estimators need `p_lab`/`p_unlab`, but the
   labelers arrive in Phase 3. Proposal for Phase 2: a minimal calibrated predictor
   (multinomial logistic regression on covariates + `signal`) trained on a split of
   the population that is disjoint from every inference sample. Not built in Phase 1.
3. **Fixed-n respondent draws.** Phase 1 writes propensities and one realized
   response flag per mechanism. Replicate experiments will need "draw n respondents
   with probability ∝ propensity"; I propose that belongs to the Phase 2 harness.

---

## Results (Gate G1)

### What was built

| File | Contents |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | uv project, Python 3.11, hatchling build, ruff/mypy/pytest config. mypy `strict` applies to `silent97.estimators.*` (add `sampling.*` when it exists in Phase 4; mypy warns about config sections for missing modules). |
| `Makefile` | `setup`, `test`, `test-slow`, `lint`, `reproduce`, `demo`, plus `format`, `coverage`, `simulate`. |
| `.pre-commit-config.yaml`, `.secrets.baseline` | Hygiene hooks, `detect-private-key`, ruff check/format (pinned to the locked ruff 0.17.0), `detect-secrets` 1.5.0. I checked that the scanner rejects a planted fake AWS key. |
| `.github/workflows/ci.yml` | `uv sync --locked --all-extras`, `make lint`, `make test` on every push and PR. |
| `src/silent97/types.py` | `Estimate`, `Design` as in CLAUDE.md §5. |
| `src/silent97/estimators/nps.py` | `validate_scores`, `validate_predictions`, `indicators`, `nps_values`, `nps_from_shares`, `nps_from_counts`, `indicator_mean_cov`, `nps_variance`, `delta_method_se`, `NPS_GRADIENT` (read-only). |
| `src/silent97/data/schema.py` | Column groups (ID, covariates, segment, signal, ground truth, response), Arrow schema, `assert_no_ground_truth` guard for labeler inputs. |
| `src/silent97/data/simulate.py` | `SimConfig`, `simulate_population`, `write_population`, `run`, `load_config`, CLI (`python -m silent97.data.simulate`). |
| `configs/simulate.yaml` | Default run: seed 20261010, N = 200,000, target NPS +20. |

### Gate G1 checks

| Criterion | Result |
|---|---|
| Realized NPS within ±1 of +20 across 20 seeds (N = 200,000) | **Pass.** Seeds 0–19: min 19.71, max 20.36, largest deviation 0.36 points. The expected NPS is exactly 20.00 (to 1e-6) for every seed by construction; the realized gap is Monte Carlo noise. Test: `test_gate_g1_realized_nps_within_one_point_across_20_seeds`. |
| Identical seeds give byte-identical Parquet | **Pass.** Two `run()` calls with seed 2026 give identical Parquet *and* manifest bytes (`test_byte_identical_parquet`). Running `make reproduce` twice gives the same SHA-256 (`6555d5a2…a2fc83` for the config seed). |
| `make lint` | Clean (ruff check, ruff format --check, mypy on `src`). |
| `make test` | 134 passed in about 10 s. |
| `make test-slow` | No slow tests exist yet; the target treats pytest's "no tests collected" (exit 5) as success. Phase 2 adds the coverage suite. |
| Line coverage | 100% of statements and branches for `silent97` (only `estimators/nps.py` counts toward the §7 rule this phase). |

### The default population (seed 20261010)

- True NPS **+19.67** (expected +20.00); promoters 46.4%, detractors 26.7%.
  Score shares 0→10: 2.7, 1.3, 2.4, 3.3, 4.0, 6.4, 6.7, 11.0, 15.9, 15.3, 31.1%.
- Latent satisfaction SD 1.14 (the logistic score noise has SD 1.81, so covariates
  explain only part of each score, which is realistic for NPS).
- Segment true NPS ranges from −4.9 to +52.6 (SD 13.5); segment sizes 82 to 16,589.
- Covariate means: chequing 88.6%, credit card 60.6%, wealth 33.3%, insurance 23.3%,
  digital engagement 0.59, complaints 0.13 per year, 46 transactions per month,
  tenure 9.6 years.
- Response at a 3% rate (about 6,000 respondents). The large-sample limit of the raw
  respondent NPS is: MCAR 19.67 (unbiased), **MAR 26.56 (+6.9)**,
  **MNAR 25.31 (+5.6)**. So MNAR adds −1.2 points that no covariate explains. Across
  seeds 0–5 the MAR bias ranges from +2.2 to +5.7, and the extra MNAR bias from −2 to −6.
- Signal knob → realized correlation with the per-customer NPS value (seed 1):
  a = 0: 0.00, 0.2: 0.17, 0.4: 0.34, 0.6: 0.51, 0.8: 0.69, 0.9: 0.77, 1.0: 0.86.
  Correlation 0.8 for E2 needs a ≈ 0.93. These are correlations of the raw feature;
  a trained labeler that also uses the covariates will do a little better.

### Changes from the plan above

- **MAR response index.** My first MAR index used only age, digital engagement,
  complaints and tenure. Its bias in the respondent NPS was small (+0.4 points) and
  changed sign across seeds, because the age random effects are redrawn each seed.
  I added a term γ_obs · z(η − σ_e·e), the standardized *covariate-explained* part of
  latent satisfaction (fixed effects plus segment intercepts). The default is γ_obs = 0.3.
  This keeps MAR a function of observed traits and segment only, and gives a
  stable positive bias. Please review: this is a simulation-design choice, not an
  estimator.
- **`Estimate.diagnostics` typed as `dict[str, Any]`** rather than bare `dict`, so
  mypy `--strict` accepts it. Same fields and runtime behaviour.
- **Ruff excludes `docs/` and `*.md`.** Ruff 0.17 formats Python blocks inside
  Markdown and rewrote snippets in `docs/PLAN.md` and `CLAUDE.md` on its first run.
  I reverted both files to the committed text and excluded Markdown from ruff.

### Citation verification

Checked by web search. arXiv, Crossref and consumerfinance.gov cannot be reached
directly from this environment, so the details come from search-indexed pages:
arXiv listings, publisher pages, PMLR/NeurIPS proceedings, Statistics Canada,
RePEc and NSF PAR.

| PLAN reference | Status | Verified details / corrections |
|---|---|---|
| Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. *Prediction-powered inference.* Science 2023 | ✅ Verified | *Science* 382(6671):669–674 (2023), DOI 10.1126/science.adi6000; arXiv:2301.09633. Volume/pages come from NSF PAR and citing papers' reference lists, not the Science site itself. |
| Angelopoulos, Duchi, Zrnic. *PPI++: Efficient prediction-powered inference.* arXiv 2023 | ✅ Verified | arXiv:2311.01453 (v1 Nov 2023, v2 Mar 2024). No journal version found. |
| Zrnic, Candès. *Cross-prediction-powered inference.* PNAS 2024 | ✅ Verified | *PNAS* 121(15):e2322083121 (2024), DOI 10.1073/pnas.2322083121; arXiv:2309.16598. |
| Zrnic, Candès. *Active statistical inference.* ICML 2024 | ✅ Verified | ICML 2024 (oral), PMLR 235:62993–63010; arXiv:2403.03208. |
| Fisch et al. *Stratified prediction-powered inference for hybrid language model evaluation.* NeurIPS 2024 | ⚠️ Title differs by venue | The PLAN title is the arXiv title (arXiv:2406.04291). The NeurIPS 2024 proceedings title is *Stratified Prediction-Powered Inference for **Effective Hybrid Evaluation of Language Models***. Authors: Fisch, Maynez, Hofer, Dhingra, Globerson, Cohen. The memo should use the NeurIPS title when citing NeurIPS. |
| Gelman, Little. *Poststratification into many categories using hierarchical logistic regression.* Survey Methodology 1997 | ✅ Verified | *Survey Methodology* 23(2):127–135 (Dec 1997), Statistics Canada cat. 12-001-X19970023616. StatCan's English page misspells the title as "Postratification". Pages come from a secondary source. |
| Wang, Rothschild, Goel, Gelman. *Forecasting elections with non-representative polls.* IJF 2015 | ✅ Verified | *International Journal of Forecasting* 31(3):980–991 (2015). I did not confirm the DOI, so it is not listed here. |
| `ppi_py` (GitHub aangelopoulos/ppi_py) | ✅ Verified | MIT licence; PyPI name `ppi-python`; latest release 0.2.3 (downloaded and inspected). |
| Bisbee et al. *Synthetic replacements for human survey data? The perils of large language models.* Political Analysis 2024 | ✅ Verified | Bisbee, Clinton, Dorff, Kenkel, Larson. *Political Analysis* 32:401–416 (2024), DOI 10.1017/pan.2024.5. Issue number not confirmed. |
| Park et al. *Generative agent simulations of 1,000 people.* arXiv 2024 | ✅ Verified | arXiv:2411.10109 (Nov 2024), J. S. Park and 8 co-authors. One aggregator lists a later revision; not confirmed. |
| Reichheld. *The one number you need to grow.* HBR 2003 | ✅ Verified (secondary) | *Harvard Business Review*, December 2003, pp. 46–54. Volume/issue not confirmed. |
| CFPB Consumer Complaint Database (URL; "Consumer disputed?" collected until April 2017) | ❓ Not confirmed | consumerfinance.gov is unreachable from this environment and search found no authoritative page for the April 2017 cutoff. Check against the data dictionary when the export is downloaded in Phase 3. |
| Employer press-release link ("Data and … context") | ⏭️ Not checked | Left alone on purpose (CLAUDE.md §9). It is context for the memo, not a method citation. |

Note: PLAN.md says citations are verified "in Phase 0". CLAUDE.md puts this in
Phase 1, which is where it was done.

### Open questions for review

Questions 1–3 under "Open questions" above are still open. Two more:

4. **MAR design** (see "Changes from the plan"): is γ_obs = 0.3, giving a raw
   respondent bias of +2 to +7 NPS points, the strength you want for E4?
5. **MNAR shape.** The U-shape is symmetric in standardized latent satisfaction, as
   specified. With the default κ = 0.5, MNAR moves the respondent NPS by −1 to −6
   points relative to MAR, depending on the seed. If E4/E7 need a larger or
   steadier MNAR effect, raise κ or add an asymmetric term. That would be a spec
   change, so I have not made it.

Phase 1 stops here, pending your review at Gate G1.

### Resolutions (recorded at the start of Phase 2)

The reviewer asked for the recommendations above to be implemented:

1. `pre-commit` stays in the dev group; `detect-secrets` runs inside pre-commit.
2. Phase 2 predictions come from a minimal multinomial logistic-regression labeler
   (`labelers/logistic.py`) trained on a uniform split of the population. That split
   is removed from every inference pool, and the harness enforces this.
3. Fixed-n respondent draws (probability ∝ propensity) live in `eval/harness.py`.
4. and 5. The MAR/MNAR defaults are kept: γ_obs = 0.3, κ = 0.5.
