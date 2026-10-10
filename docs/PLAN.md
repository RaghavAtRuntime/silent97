# Hearing the Silent 97% — Project Plan

Oct 10, 2026 · @Raghav

## Summary

Build an open-source Python toolkit, `silent97`, that estimates customer satisfaction (NPS) for an entire customer base from a few hundred survey responses plus model predictions on everyone else, with confidence intervals that are provably valid and verified by simulation.

Why it matters to the personalization CoE: TD already predicts customer needs at scale (TD AI Prism, launched June 2025). The bottleneck the team describes is measurement. Survey response rates are low, so segment-level NPS is noisy and nudge evaluations are underpowered. This project turns that pain point into a working method.

The toolkit combines four ideas from recent statistics research:

- **Prediction-powered inference (PPI / PPI++):** use a model's predictions on all customers, corrected by the small surveyed set, to get tighter but still valid intervals.
- **Active statistical inference:** choose *which* customers to survey so each response carries the most information.
- **MRP (multilevel regression and poststratification):** stable estimates for small segments, corrected for who chooses to respond.
- **Sensitivity analysis:** an honest account of how far conclusions move if responders differ in ways the data cannot see.

Headline deliverables: a tested Python package, a reproducible experiment suite, a 2–3 page memo written like an internal TD proposal, and a static interactive demo where shrinking the survey budget shows naive intervals blowing up while the corrected ones stay tight and honest.

Estimated effort: 8 weekends (about 80–100 hours), with an AI coding agent doing most of the implementation under the review gates in this plan. The agent instructions ship as a separate `CLAUDE.md` file for the repo root.

## Problem and research foundations

The core problem: a bank surveys a tiny, self-selected slice of customers and treats their answers as the voice of everyone. That fails in three ways.

1. **Too few responses.** A segment with 80 responses has an NPS margin of error around ±20 points, so month-over-month changes are mostly noise.
2. **The wrong responses.** Very happy and very unhappy customers respond more. A raw average over respondents is biased toward the extremes.
3. **Wasted responses.** Surveys go out uniformly, even though some customers' answers would tell you far more than others.

NPS itself is a difference of two proportions:

```latex
\mathrm{NPS} = 100 \left( P(Y \ge 9) - P(Y \le 6) \right), \quad Y \in \{0, \dots, 10\}
```

So every method below estimates a two-dimensional mean (promoter share, detractor share) and propagates uncertainty to their difference.

### Prediction-powered inference

PPI (Angelopoulos et al., *Science* 2023) combines a small labeled set of n surveyed customers with predictions f(X) on a large set of N unsurveyed customers. The model's average over everyone is corrected by its measured error on the surveyed set:

```latex
\hat\theta_{\lambda} = \lambda \cdot \frac{1}{N}\sum_{j=1}^{N} f(\tilde X_j) + \frac{1}{n}\sum_{i=1}^{n} \left( Y_i - \lambda f(X_i) \right)
```

With λ = 1 this is the original PPI. PPI++ (Angelopoulos, Duchi & Zrnic) picks λ to minimize variance, so the estimate is never worse than using the survey alone, and is much better when the model is good. The interval stays valid even if the model is biased; a bad model only costs width, not correctness.

### Active statistical inference

Zrnic & Candès (ICML 2024) extend PPI to choose who gets labeled. Customers are surveyed with probability π(x) that rises with model uncertainty, and the estimator reweights by 1/π to stay unbiased. Applied here, it answers "which 500 customers should we survey this month?"

### MRP for segments and nonresponse

Multilevel regression and poststratification (Gelman & Little 1997; Wang, Rothschild, Goel & Gelman 2015, the Xbox poll) fits a hierarchical model on respondents, then reweights predictions to the known population mix. It is how pollsters get state-level estimates from small, unrepresentative samples. Here it gives stable NPS for thin segments (province × product × tenure) and corrects for responders being unlike the base on observed traits.

### The assumption that must be stated out loud

PPI assumes the surveyed set is a random sample. Real survey response is not random, and may depend on satisfaction itself (missing not at random). Covariates and MRP fix only the part of the bias explained by observed traits. The project therefore includes a sensitivity analysis: how much would unobserved responder differences need to be to overturn a conclusion? Saying this clearly is what separates this from LLM-survey hype, where recent work (e.g. Bisbee et al., *Political Analysis* 2024) shows synthetic respondents are overconfident and too uniform.

## Scope and success criteria

The project succeeds when a reviewer can run one command, see that corrected intervals hit their nominal coverage, and see how many survey responses the method saves.

**In scope**

- Estimators: classical, PPI, PPI++, cross-fit PPI, active (IPW) PPI, stratified PPI, MRP, MRP with model predictions as a covariate.
- Targets: overall NPS, promoter share, detractor share, segment-level NPS, and the NPS difference between two arms of an experiment.
- A synthetic customer simulator with known ground truth and controllable nonresponse.
- A semi-synthetic benchmark on public CFPB complaint data.
- Three labelers: a tabular model, a text-embedding model, and an LLM zero-shot scorer.
- A sensitivity analysis for nonresponse not explained by covariates.
- A static interactive demo and a written memo.

**Out of scope**

- Any TD data, systems, branding or internal information.
- Production deployment, real-time serving, or a dashboard backend.
- Training or fine-tuning large models.
- Causal claims about what *drives* NPS (a natural follow-up, kept out to protect focus).

**Acceptance criteria**

| ID | Criterion | Target |
| --- | --- | --- |
| A1 | Coverage of nominal 90% intervals, missing-at-random scenarios, n from 50 to 1,000, 1,000 replicates each | Between 88% and 92% for every valid estimator |
| A2 | Our PPI and PPI++ match the reference `ppi_py` library on identical inputs | Point estimates and interval bounds agree to 1e-6 |
| A3 | Effective-sample-size gain of PPI++ over classical, as a function of model–outcome correlation ρ (theory: about 1/(1 − ρ²) when N ≫ n) | Within Monte Carlo error of 1/(1 − ρ²), and at least 1.5× when ρ ≥ 0.6 (reported honestly if lower) |
| A4 | Active sampling width at equal survey budget vs uniform sampling | Narrower, with coverage still meeting A1 |
| A5 | MRP segment estimates for cells with fewer than 30 responses | Lower RMSE than raw segment means across replicates |
| A6 | Missing-not-at-random scenarios | Bias is shown, not hidden; sensitivity bounds contain the truth at the stated assumption |
| A7 | Reproducibility | `make reproduce` rebuilds every figure and table from cached labels in under 30 minutes on a laptop |
| A8 | Code quality | Tests pass in CI; line coverage at least 90% on `estimators/` and `sampling/`; ruff and mypy clean |

Naive baselines (`classical` applied to self-selected respondents, and the LLM- or model-only mean) are included as baselines expected to *fail* A1 under self-selection. Showing that failure is part of the result.

## Data strategy

Validity claims rest on the simulator, because only simulated data has a known true NPS; the CFPB benchmark shows the method works on messy real text.

| Source | Role | Ground truth | Notes |
| --- | --- | --- | --- |
| Synthetic customer simulator | Primary benchmark for coverage, width and bias | Exact, by construction | Fully controllable: population size, segment mix, model quality, nonresponse mechanism |
| [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) | Semi-synthetic real-text benchmark | The "Consumer disputed?" field (collected until April 2017) as a binary dissatisfaction outcome | Public US government data; includes credit card, bank account and investment products |
| LLM-generated verbatims (optional) | Free text for simulated customers | Inherited from simulator | Generated conditional on latent satisfaction; flagged as synthetic everywhere |

**Simulator design**

- Population of 200,000 customers with covariates: age band, province, tenure, product holdings (credit card, chequing, wealth, insurance), digital engagement, complaints in the past year, monthly transaction count.
- Latent satisfaction is a linear function of covariates plus segment random effects plus noise; the 0–10 score comes from an ordinal (cumulative logit) link with cutpoints tuned so true NPS lands near +20.
- Response propensity has three switchable mechanisms: MCAR (random), MAR (depends on covariates only), and MNAR (also depends on the latent score, U-shaped so extremes respond more).
- Model quality is controlled by how much signal the labeler's features carry, sweeping model–outcome correlation from 0 to 0.8.
- Every run is seeded and written to Parquet with a manifest recording parameters.

**CFPB semi-synthetic design**

- Use complaints from 2015 to April 2017 that have a narrative and a disputed flag.
- Treat disputed = Yes as the "detractor" outcome. Repeatedly draw small labeled sets (n = 50 to 1,000), hide the rest, and check whether intervals cover the full-data dispute rate.
- Labelers see only the narrative, product, issue and state, never the outcome.
- Framing: dispute rate is a proxy for dissatisfaction, not NPS. The memo says so plainly.

**Governance rules (non-negotiable)**

- No TD data, internal documents, screenshots or customer information, in any form.
- No scraping of app stores, social media or anything behind a login or against a site's terms.
- Only public data is ever sent to an LLM API.
- No TD logos or branding in the repo or demo. Before publishing, check TD's code of conduct on outside work and on naming your employer.

## Architecture and repo layout

The system is a one-way pipeline: labelers predict for every customer, a sampling design picks who is surveyed, and estimators combine the two into an interval.

&#91;embedded content: silent97 architecture · data to estimates to outputs\]

The estimators layer is the only part where a subtle error invalidates everything, so it is the part you review by hand.

**Stack:** Python 3.11+, uv, numpy, scipy, polars or pandas, pyarrow, scikit-learn, LightGBM, statsmodels, PyMC with Bambi, sentence-transformers, `ppi-python` (reference only, used in tests), an LLM SDK, pytest with hypothesis, ruff, mypy, matplotlib, GitHub Actions.

```text
silent97/
├── CLAUDE.md                 agent instructions
├── README.md
├── pyproject.toml            uv-managed
├── Makefile                  setup, test, lint, reproduce, demo
├── configs/                  one YAML per experiment
├── docs/
│   ├── PLAN.md               this plan
│   ├── memo.md, talk.md
│   ├── ideas.md              parked scope
│   └── progress/phase-N.md   phase plans and results notes
├── src/silent97/
│   ├── types.py              Estimate, Design
│   ├── data/                 simulate.py, cfpb.py, schema.py
│   ├── labelers/             base.py, tabular.py, embedding.py, llm.py, cache.py
│   ├── sampling/             designs.py
│   ├── estimators/           nps.py, classical.py, ppi.py, active.py,
│   │                         stratified.py, mrp.py, sensitivity.py
│   ├── eval/                 harness.py, metrics.py
│   └── viz/                  figures.py
├── experiments/              e01_coverage.py … e08_power.py
├── demo/                     index.html, app.js, data/grid.json
├── data/                     raw/, processed/ (git-ignored)
├── cache/                    LLM label cache (git-ignored)
└── tests/                    unit, reference cross-checks, coverage (marked slow)
```

## Methods specification

Every estimator shares one interface, so the experiment harness can swap them freely and compare them on identical draws.

```python
class Estimate(NamedTuple):
    point: float          # NPS on the -100..100 scale
    lower: float
    upper: float
    alpha: float
    se: float
    n_labeled: int
    n_unlabeled: int
    method: str
    diagnostics: dict     # e.g. lambda_hat, effective_n, weights summary

def estimate(y_lab, f_lab, f_unlab, *, alpha=0.1, weights=None, **kw) -> Estimate: ...
```

**Estimators**

| Name | What it does | Valid under |
| --- | --- | --- |
| `classical` | Respondent NPS with a delta-method interval on the promoter/detractor pair. On self-selected respondents it is also the naive respondent baseline | MCAR (baseline expected to fail under MAR/MNAR) |
| `naive_model` | Average of predictions on everyone | Never (baseline expected to fail) |
| `ppi` | PPI with λ = 1 | MCAR labeled set |
| `ppi_pp` | PPI++ with variance-optimal λ, estimated by plug-in | MCAR labeled set |
| `ppi_crossfit` | Cross-prediction PPI: model trained on K−1 folds of labeled data, predicts the held-out fold | MCAR; no separate training data needed |
| `ppi_active` | IPW-corrected PPI with known sampling probabilities π(x) | Any design where π(x) > 0 and is recorded |
| `ppi_stratified` | PPI within strata, combined with population stratum weights | Random sampling within strata |
| `mrp` | Hierarchical logistic models for promoter and detractor, poststratified to population cells | MAR given cell covariates |
| `mrp_ppi` | MRP with the model prediction as a cell-level covariate | MAR given cells |

**Implementation rules**

- NPS is always handled as a two-dimensional mean. Intervals use the joint covariance of (promoter, detractor) and the gradient (100, −100).
- Prediction targets are calibrated probabilities p\_promoter(x) and p\_detractor(x), not a single 0–10 point score.
- `ppi`, `ppi_pp` and `ppi_crossfit` are cross-checked against `ppi_py` in the test suite.
- `mrp` uses Bambi or PyMC with NUTS; report R-hat and effective sample size and fail loudly if R-hat exceeds 1.01. A fast fallback (glmer-style Laplace approximation via `statsmodels`) is allowed for the large experiment sweeps, documented as such.
- Model training data must be disjoint from the inference labeled set unless the estimator is explicitly cross-fit.

**Sampling designs** (`sampling/`)

- `uniform(n)`, `stratified(n, strata, allocation="proportional" | "neyman")`.
- `active(n, uncertainty, floor=0.1)`: π(x) proportional to sqrt of predicted variance, mixed with a uniform floor so no customer has π near zero, then scaled to the budget.
- Every design returns the drawn indices *and* the π used, so estimators can reweight.

**Labelers** (`labelers/`)

- `TabularLabeler`: LightGBM multiclass (detractor / passive / promoter) with isotonic calibration.
- `EmbeddingLabeler`: sentence-embedding model plus logistic regression, for text.
- `LLMLabeler`: zero-shot prompt returning probabilities for the three classes as JSON. All calls are cached to disk keyed by a hash of (model, prompt version, input), so experiments never re-call the API. Hard cap on items and spend set in config.

**Sensitivity analysis** (`estimators/sensitivity.py`)

- Delta-adjustment: assume non-respondents' promoter and detractor rates differ from model-corrected respondents by up to δ percentage points. Report the NPS range as δ grows, and the breakpoint δ\* at which a conclusion (e.g. "NPS rose this quarter") flips.

**Uncertainty tool for the demo**

- Precompute the full experiment grid to JSON so the demo needs no Python at view time.

## Milestones and timeline

Five phases over eight weekends, each closed by a gate you review before the agent moves on.

&#91;embedded content: roadmap · 5 phases, 5 gates, 8 weekends\]

If a phase overruns, cut scope in this order: E5 segment study, then the embedding labeler, then stratified PPI. Never skip a gate review, and never cut E1 or E8, which carry the pitch.

## Experiments and evaluation protocol

Each experiment is a script in `experiments/` that reads a YAML config, runs seeded replicates, writes a results Parquet file, and renders its own figure.

**Metrics, computed per (estimator, scenario, n)**

- **Coverage:** share of replicates whose interval contains the true NPS. Report with a Wilson interval, since 1,000 replicates give about ±1.9 points of Monte Carlo error.
- **Mean interval width**, in NPS points.
- **Bias** and **RMSE** of the point estimate.
- **Effective sample size:** the classical-survey n that would give the same width, n\_eff = n × (width\_classical / width\_method)².
- **Responses saved:** n\_eff − n, the headline number for the memo.

**Experiment list**

| ID | Question | Design |
| --- | --- | --- |
| E1 | Do intervals keep their promise? | Simulator, MCAR, all estimators, n ∈ {50, 100, 200, 500, 1000}, 1,000 replicates |
| E2 | How much does model quality buy? | Simulator, sweep model–outcome correlation 0 to 0.8; plot n\_eff / n |
| E3 | Does active sampling help at a fixed budget? | Simulator, uniform vs stratified vs active at equal n |
| E4 | What happens under realistic nonresponse? | Simulator, MAR and MNAR; naive vs PPI++ vs MRP vs MRP+PPI |
| E5 | Can we report thin segments? | Simulator, 60 segments, cell sizes 5 to 200; RMSE and coverage by cell size |
| E6 | Does it work on real text? | CFPB dispute benchmark with tabular, embedding and LLM labelers |
| E7 | How fragile is a conclusion? | Sensitivity curves and breakpoint δ\* on E4 and E6 |
| E8 | Can we detect a nudge's effect sooner? | Simulator A/B test with a true +5 NPS lift; power vs n for classical vs PPI++ difference-in-means |

E8 is the bridge to the personalization team's day job: if PPI++ reaches 80% power with half the responses, that is the pitch.

**Reporting rules**

- Every figure shows the nominal coverage line and Monte Carlo error bands.
- Failures are reported, not tuned away. If an estimator misses coverage, the results note explains why.
- No hyperparameter is tuned on the replicates used for evaluation.

## Deliverables

Four artifacts, each aimed at a different reader on the CoE team.

| Deliverable | Reader | Contents |
| --- | --- | --- |
| GitHub repo `silent97` | Data scientists and ML engineers | Installable package, tests in CI, `make reproduce`, a README with one quickstart and one headline figure |
| Memo, 2–3 pages (`docs/memo.md` and PDF) | Team lead and manager | Problem, method in plain words, E1/E2/E8 results, "responses saved" number, assumptions, what a TD pilot would need |
| Interactive demo (static page on GitHub Pages) | Anyone in a 5-minute conversation | Survey-budget slider, model-quality slider, nonresponse toggle; live interval chart and coverage readout from precomputed results |
| 5-minute talk track (`docs/talk.md`) | You, in interviews or a team intro | The story in four beats: the pain, the trap (LLM-only), the fix (PPI + MRP), the honest limit (MNAR) |

**Demo spec**

- One page, plain HTML and JavaScript, no server. Data comes from `demo/data/grid.json` produced by the experiment harness.
- Main chart: intervals from 50 replicate surveys at the chosen settings, drawn as horizontal bars against the true NPS line, with misses in a warning colour.
- Side readout: coverage, mean width, and "this survey is worth N classical responses."
- A toggle that switches nonresponse from random to self-selected, so viewers see the naive estimate drift and MRP pull it back.

**Memo outline**

1. The problem in TD-agnostic terms: low response, self-selection, noisy segments.
2. The trap: model-only or LLM-only scores look precise and are biased.
3. The method, one paragraph each for PPI++, active sampling, MRP.
4. Results: three figures (coverage, responses saved, experiment power).
5. Assumptions and limits, including the MNAR sensitivity result.
6. A pilot sketch: what data a team would need, and what a first test would look like.

## Risks and mitigations

The biggest risk is not a bug that crashes but a subtle statistical error that looks right; most mitigations target that.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Agent writes plausible but wrong statistics | Invalid intervals presented as valid | Cross-check against `ppi_py`; simulation-based coverage tests in CI; human review gate on every estimator before it is used in experiments |
| Self-selection depends on satisfaction itself (MNAR) | No method fully removes the bias | Show it openly in E4; sensitivity analysis and breakpoint δ\*; memo states the assumption in the first page |
| LLM labeler is weak on complaint text | Small or no width gain | PPI++ falls back toward the classical estimate by design; report the gain honestly; tabular and embedding labelers as alternatives |
| CFPB dispute rate is not NPS | Reviewers question relevance | Frame as a real-text validity check; keep NPS claims on the simulator |
| MRP is slow inside 1,000-replicate sweeps | Experiments take days | Laplace-approximation fallback for sweeps; full NUTS on a subset to confirm agreement |
| LLM API cost runs away | Unexpected bill | Disk cache, hard item cap in config, dry-run mode that prints projected cost |
| Scope creep into causal drivers or dashboards | Project never ships | Out-of-scope list in this plan; new ideas go to `docs/ideas.md` |
| Employer policy on outside work | Reputational or HR problem | No TD data or branding; check code of conduct before publishing; frame as general banking research |

## How to run the AI agent

Run the agent one phase at a time, with you reviewing at every gate; never ask it to "build the whole project."

**Setup**

1. Create an empty GitHub repo `silent97` and put `CLAUDE.md` (the companion file) at its root. It works as-is for Claude Code; rename or copy to `AGENTS.md` for other coding agents.
2. Add this plan as `docs/PLAN.md` so the agent can read the spec.
3. Put any API key in a local `.env` that is git-ignored. Never paste keys into prompts.

**Per-phase loop**

1. Start a fresh session with: "Read CLAUDE.md and docs/PLAN.md. Execute Phase N only. Stop at its gate."
2. The agent writes a short plan in `docs/progress/phase-N.md` before coding. Skim it; correct it if it misread the spec.
3. The agent implements on a branch `phase-N`, with tests first for every estimator.
4. At the gate it writes a results note in the same file: what was built, test and coverage results, anything that failed, open questions.
5. You review the diff and the note, then merge. Only then start Phase N+1.

**What you personally review at each gate**

- Estimator code line by line against the formulas in this plan. This is the part where a plausible-looking mistake does the most damage.
- The coverage figure: does the line sit on 90%?
- Any test that was changed rather than the code it tests. Ask why.

**Useful prompts between phases**

- "Write a test that would fail if `ppi_pp` used the labeled-set mean of f instead of the unlabeled-set mean." (Probing tests catch the subtle bugs.)
- "Explain, in plain words, why the active estimator stays unbiased. Point to the line of code that does it."
- "Run E1 at 200 replicates for a quick check and show me the coverage table."

## References

Paper citations below are from memory and must be verified (titles, venues, arXiv IDs) in Phase 0 before they appear in the memo.

**Methods**

- Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. *Prediction-powered inference.* Science, 2023.
- Angelopoulos, Duchi, Zrnic. *PPI++: Efficient prediction-powered inference.* arXiv, 2023.
- Zrnic, Candès. *Cross-prediction-powered inference.* PNAS, 2024.
- Zrnic, Candès. *Active statistical inference.* ICML, 2024.
- Fisch et al. *Stratified prediction-powered inference for hybrid language model evaluation.* NeurIPS, 2024.
- Gelman, Little. *Poststratification into many categories using hierarchical logistic regression.* Survey Methodology, 1997.
- Wang, Rothschild, Goel, Gelman. *Forecasting elections with non-representative polls.* International Journal of Forecasting, 2015.
- Reference implementation: `ppi_py` (GitHub: aangelopoulos/ppi\_py).

**Context and critique**

- Bisbee et al. *Synthetic replacements for human survey data? The perils of large language models.* Political Analysis, 2024.
- Park et al. *Generative agent simulations of 1,000 people.* arXiv, 2024.
- Reichheld. *The one number you need to grow.* Harvard Business Review, 2003.

**Data and TD context**

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [TD announces launch of TD AI Prism, June 11, 2025](https://stories.td.com/ca/en/news/2025-06-11-td-announces-launch-of-groundbreaking-predictive-foundation)
