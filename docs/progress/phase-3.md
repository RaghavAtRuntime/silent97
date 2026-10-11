# Phase 3: Labelers and real text

## Plan

### Tasks

1. **Labelers** (tests first), all returning calibrated `(n, 2)` arrays of
   `[P(promoter), P(detractor)]`:
   - `labelers/base.py`: the `Labeler` protocol (`fit(x, y_scores)`, `predict(x)`),
     plus an adapter to the estimators' cross-fitting `FitFn`.
   - `labelers/cache.py`: disk cache keyed by `sha256(model, prompt_version, input)`,
     under git-ignored `cache/`.
   - `labelers/tabular.py`: LightGBM multiclass (detractor / passive / promoter) with
     isotonic calibration (`CalibratedClassifierCV(method="isotonic")`; each fold
     model is calibrated on its held-out fold).
   - `labelers/embedding.py`: sentence embeddings (sentence-transformers,
     `all-MiniLM-L6-v2` by default, set in config) plus logistic regression.
     Embeddings are cached on disk. The encoder can be injected, so tests run
     without PyTorch.
   - `labelers/llm.py`: zero-shot JSON probabilities for detractor / passive /
     promoter (structured outputs, a fixed JSON schema, low effort), with the
     CLAUDE.md LLM rules. Dry-run is the default and prints the item count and
     projected cost. There are hard caps on items and spend. Cached keys are never
     re-called. Only items tagged `cfpb` or `synthetic` can be sent. It can use the
     Batches API (50% price) or plain sync calls.
2. **CFPB data** (`data/cfpb.py`): download the public export once
   (`files.consumerfinance.gov/ccdb/complaints.csv.zip`, SHA-256 recorded). Keep
   complaints received 2015-01-01 to 2017-04-30 that have a non-empty narrative and
   "Consumer disputed?" ∈ {Yes, No}. Write `data/processed/cfpb.parquet` plus a
   manifest. Labelers see only narrative, product, issue and state; a guard
   enforces this, as for the simulator.
3. **Harness generalization:** the labeler kind (`logistic` | `tabular`) is set in
   config. The cross-fitting model travels with the prepared scenario, so
   non-simulator datasets (CFPB) reuse `run_replicate`.
4. **E2** (`experiments/e02_model_quality.py`): simulator, MCAR, signal strength
   swept so the realized model–outcome correlation ρ runs from 0 to about 0.8.
   Tabular labeler. Plots effective n / n against ρ, with the theory curve
   1/(1 − ρ²/(1 + n/N)) and bootstrap Monte Carlo intervals. Checks reworded A3.
5. **E6** (`experiments/e06_cfpb.py`): CFPB dispute-rate benchmark with the tabular,
   embedding and LLM labelers; coverage of the full-pool dispute rate for
   n ∈ {50, …, 1000}.

### Statistical design notes

- **Dispute rate through the NPS estimators.** Disputed is coded as score 0
  (detractor) and not disputed as score 7 (passive). Predictions are
  `(0, P(disputed))`. The promoter column is then identically 0, so
  NPS = −100 × dispute rate, and every estimator (including PPI++'s λ, whose
  contrast reduces to the detractor coordinate) is exactly the scalar estimator of
  the dispute rate, rescaled. E6 reports rate = −NPS / 100 with the bounds swapped.
  A test will check this against `ppi_py` on the scalar indicator.
- **E6 pools.** A labeler-training split (uniform, 20%) is removed first. Then a
  fixed-size inference pool is drawn uniformly from the rest, so the LLM's item
  count is bounded. All three labelers predict the same pool. The truth is the
  pool's dispute rate. Pool size is set in config.
- **LLM predictions** are a fixed function of each item, cached once, so PPI's
  validity argument (the model is fixed independently of the labeled sample) holds.
  An item whose response fails validation after retries gets the constant prediction
  (1/3, 1/3, 1/3). Such items are counted and reported. A constant is a fixed
  function of the item, so validity is unaffected.

### Blockers found at the start (need you)

| Need | Status here | What unblocks it |
|---|---|---|
| CFPB export (`files.consumerfinance.gov`) | **Blocked by the environment's network policy** (403 on CONNECT) | Add `files.consumerfinance.gov` (and `www.consumerfinance.gov`) to the environment's allowed domains |
| Embedding model weights (`huggingface.co`) | **Blocked** | Add `huggingface.co`, `cdn-lfs.huggingface.co` and `cas-bridge.xethub.hf.co` |
| CPU PyTorch wheels (`download.pytorch.org`) | **Blocked** (PyPI's default torch build pulls several GB of CUDA libraries) | Optional: allow `download.pytorch.org`; otherwise the PyPI build works, only larger |
| Claude API key | **Not present.** This session's own `ANTHROPIC_BASE_URL` must not be used for project spend | Add `ANTHROPIC_API_KEY` to `.env` (git-ignored) or as an environment secret. The client always targets `https://api.anthropic.com` explicitly |
| LLM spend cap | **Unset on purpose** (`max_spend_usd: null` blocks any call) | You choose the cap and the model in `configs/llm.yaml` |

### LLM cost projection (to inform the cap)

Assumptions: about 650 input tokens per item (300-token instructions + about
350-token narrative) and about 150 output tokens (JSON plus any thinking at low
effort). Prices are from the current price table: Opus 5.5 $4 / $20, Sonnet 5.5
$2 / $10, Haiku 5.5 $0.10 / $0.50 per million tokens. Batches are half price. The
dry-run recomputes the projection from the actual texts before any call.

| Pool size | Opus 5.5 (batch) | Sonnet 5.5 (batch) | Haiku 5.5 (batch) |
|---|---|---|---|
| 5,000 | ≈ $14 | ≈ $7 | ≈ $0.35 |
| 10,000 | ≈ $28 | ≈ $14 | ≈ $0.70 |
| 20,000 | ≈ $56 | ≈ $28 | ≈ $1.40 |

The configured default model is `claude-opus-5-5`, and choosing a cheaper one is
your call.

### Files touched

`pyproject.toml`/`uv.lock` (lightgbm; extras `llm` = anthropic, `text` =
sentence-transformers), `.github/workflows/ci.yml` (CI installs `--extra llm`
only; the `text` extra pulls PyTorch), `src/silent97/labelers/{base,cache,tabular,embedding,llm}.py`,
`src/silent97/data/cfpb.py`, `src/silent97/eval/harness.py`,
`src/silent97/viz/figures.py`, `experiments/e02_model_quality.py`,
`experiments/e06_cfpb.py`, `configs/{e02_model_quality,e06_cfpb,llm,cfpb}.yaml`,
`tests/unit/…`.

### Open questions

1. Model and spend cap for the LLM labeler (table above), and the E6 pool size
   (default 10,000).
2. The network and API-key items in "Blockers".

---

## Results so far (Gate G3: blocked on external access, see "What I need")

### What was built

| File | Contents |
|---|---|
| `labelers/base.py` | `Labeler` protocol; `score_classes`; `class_probs_to_pair` (classes absent from training get probability 0); `as_fit_fn` adapter for cross-fitting |
| `labelers/cache.py` | `make_key` = sha256 of `[model, prompt_version, payload]`; `DiskCache` for JSON and `.npy`, atomic writes, key validation |
| `labelers/tabular.py` | LightGBM multiclass + isotonic calibration of the two used columns (see "Changes") |
| `labelers/embedding.py` | Injectable encoder; on-disk encoding cache per (model, text list); logistic regression on embeddings; sentence-transformers loaded lazily (`text` extra) |
| `labelers/llm.py` | Zero-shot JSON labeler. Dry run by default; cumulative item and spend caps from a ledger; worst-case reservation before each call or batch, so the cap is a hard cap; failures cached and never re-called; only `cfpb` / `synthetic` sources; key from `.env` / environment, never logged; base URL pinned to `https://api.anthropic.com`; Batches API (half price) or sync with server-side refusal fallback |
| `data/cfpb.py` | Download once; lazy scan of the multi-GB export; window, narrative and dispute filters; deterministic Parquet + manifest with drop counts and SHA-256s; one-hot product/issue/state; dispute coding; guard on labeler columns |
| `eval/harness.py` | `labeler.kind: tabular`; scenario `labeler_features: signal_only`; `model_corr` per scenario; unlabeled set = pool minus draw; `run_grid(prepare=...)` for non-simulator datasets |
| `eval/metrics.py` | `effective_n_bootstrap` (paired over replicates); `ppi_pp_theory_ratio` |
| `experiments/e02_model_quality.py` | E2 (run, below) |
| `experiments/e06_cfpb.py` | E6 (built and smoke-tested; **not run**) |
| `configs/{llm,e02_model_quality,e06_cfpb}.yaml` | `llm.yaml` ships with `max_spend_usd: null` |

### Tests

`make test`: 1,307 fast tests pass (about 1 minute). `make lint` is clean. New tests:
- LLM labeler: 24 tests with a fake client. Dry run makes no calls; caps; worst-case
  reservation keeps actual spend ≤ cap; cached keys and failures are never re-called;
  keys change with model, prompt, effort or text; source restriction; batch path;
  pinned base URL even when `ANTHROPIC_BASE_URL` is set; no secret in errors; the
  committed config has no cap.
- Tabular calibration; embedding (stub encoder); CFPB loader on an export-shaped
  fixture (both date formats, blank narratives, N/A disputes, zip and plain CSV).
- E6 smoke tests (all three labelers, stub encoder and fake LLM; dry run skips the LLM).
- Reference: **the CFPB dispute-rate coding reproduces `ppi_py`'s scalar dispute-rate
  intervals at 1e-6** (100 cases, both clip settings, plus classical).

Line coverage: 95% overall. `estimators/` is at 100%; labelers are at 87–100%. The
remaining uncovered lines are the real-encoder and network paths.

### E2: model quality (`configs/e02_model_quality.yaml`, 1,000 replicates, seed 20261013)

Sources: `docs/results/e02_model_quality_summary.{csv,md}` and
`docs/results/e02_model_quality.png`. Tabular labeler, MCAR, n ∈ {100, 500}. 10 signal
levels give ρ from 0.003 (signal-only labeler, pure noise) up to 0.881. With
covariates, the labeler's floor is ρ = 0.146. 411 s on 4 cores.

PPI++ effective n / n:

| ρ | n = 100: widths [95%] | n = 100: MSE ratio [95%] | n = 500: widths [95%] | n = 500: MSE ratio [95%] | theory (n=100 / 500) |
|---|---|---|---|---|---|
| 0.003 | 1.007 [1.006, 1.008] | 0.983 [0.969, 0.995] | 1.002 [1.002, 1.002] | 1.001 [0.995, 1.006] | 1.000 / 1.000 |
| 0.146 | 1.029 [1.027, 1.031] | 1.000 [0.979, 1.022] | 1.023 [1.022, 1.024] | 1.023 [1.004, 1.041] | 1.022 / 1.022 |
| 0.274 | 1.090 [1.086, 1.093] | 1.052 [1.018, 1.086] | 1.081 [1.080, 1.083] | 1.084 [1.049, 1.121] | 1.081 / 1.081 |
| 0.513 | 1.367 [1.358, 1.376] | 1.324 [1.249, 1.409] | 1.356 [1.352, 1.360] | 1.392 [1.310, 1.476] | 1.357 / 1.356 |
| 0.604 | 1.574 [1.562, 1.586] | 1.645 [1.543, 1.755] | 1.575 [1.569, 1.581] | 1.560 [1.439, 1.686] | 1.573 / 1.571 |
| 0.703 | 1.979 [1.962, 1.997] | 1.867 [1.709, 2.049] | 1.971 [1.964, 1.979] | 2.078 [1.904, 2.266] | 1.974 / 1.970 |
| 0.814 | 2.943 [2.912, 2.974] | 2.909 [2.632, 3.216] | 2.934 [2.919, 2.948] | 2.979 [2.686, 3.326] | 2.960 / 2.949 |
| 0.881 | 4.398 [4.345, 4.455] | 4.259 [3.830, 4.763] | 4.436 [4.411, 4.460] | 3.966 [3.542, 4.416] | 4.465 / 4.434 |

(Selected rows. The full table, including PPI and ρ = 0.184 / 0.388, is in the
summary files.)

**A3 as reworded:**
- *"At least 1.5× when ρ ≥ 0.6"*: **met at all 8 points** (ρ = 0.604 to 0.881, both n);
  the minimum is 1.574.
- *"Within Monte Carlo error of 1/(1 − ρ²)"*: depends on which efficiency is meant.
  - **Widths** (the docs/PLAN.md n_eff definition, the precision a method *reports*):
    inside the paired-bootstrap interval at **8 of 20** points. Every miss is within
    1.5% of theory (largest gap: 4.398 vs 4.465 at ρ = 0.881, n = 100). The paired
    intervals are only about ±0.1–0.6% wide, so small systematic effects show up as
    misses.
  - **MSE** (the precision actually *achieved*): inside the interval at **18 of 20**.
    The misses are n = 100, ρ ≈ 0 (0.983 [0.969, 0.995], see below), and n = 500,
    ρ = 0.881, by a hair (upper bound 4.416 against theory 4.434).

**Investigation (one pass) of the width-based misses.** At low ρ, PPI++'s reported
precision is slightly *better* than theory (1.007 against 1.000 at ρ ≈ 0, n = 100),
and the excess shrinks with n (n = 500: 1.002). Its actual MSE is slightly *worse*
(0.983). Both point to the usual O(1/n) effect of estimating λ on the same data. The
plug-in variance at λ̂ is minimized in-sample, so it is a little optimistic, while
λ̂'s own noise adds a little true variance. This matches the theory being asymptotic.
Coverage is unaffected: **all 60 (estimator × ρ × n) coverage cells are inside the A1
band** (range 0.876–0.919). Practical consequence: with a useless model at n = 100,
PPI++ is about 1.7% less efficient than classical, so "never worse than the survey
alone" holds only asymptotically. PPI with λ = 1 tracks the same curve, because the
calibrated labeler makes λ* ≈ N/(N + n) ≈ 1.

### E6: CFPB benchmark: **not run**

Everything is built and tested on an export-shaped fixture, but the run needs the
three external resources below. No CFPB data, embedding model or API key was
available, and **$0 was spent**. Nothing is mocked in place of real results.

### Gate G3 status

| Criterion | Status |
|---|---|
| All labels cached | ✗ Blocked: no CFPB data (network), no embedding model (network), no API key |
| Spend under cap and reported | $0 spent. No cap is set (`max_spend_usd: null`), so no call is possible. The E6 run writes `docs/results/e06_cfpb_llm_report.json` |
| E6 coverage reported for each labeler | ✗ Not run |
| `make test`, `make lint`, `make test-slow` | Pass: 1,307 fast tests, lint clean, slow suite 15/15 (4 min 43 s) after the Phase 3 changes |

### Changes and judgment calls to review

1. **Tabular calibration method.** One-vs-rest isotonic calibration over all three
   classes followed by row renormalization (sklearn's `CalibratedClassifierCV`) left
   individual columns off by up to 0.05 per decile. Instead, the two columns the
   estimators use are calibrated directly, each with isotonic regression on
   out-of-fold predictions. P(passive) is the remainder, and rows are rescaled only
   if the pair exceeds 1. Expected calibration error on held-out data
   (0.05-wide bins): raw LightGBM 0.024 (promoter) / 0.017 (detractor); calibrated
   0.014 / 0.015; the logistic labeler 0.018 / 0.014. The sampling-noise floor is
   about 0.01.
2. **A test I rewrote before it ever passed.** My first calibration test bounded the
   max decile gap at 0.04 using quantile bins. Quantile bins misbehave on isotonic
   plateaus (one decile came out empty), and even the logistic labeler missed the
   bound (0.069). I replaced it with an ECE bound (< 0.02 per column, fixed-width
   bins) before committing. I'm flagging it because CLAUDE.md §1.3 asks for care
   with tests.
3. **The E2 MSE ratio** is an addition. The docs/PLAN.md n_eff (widths) stays the
   headline, and the MSE ratio shows what the widths do not.
4. **Signal-only scenario in E2**, so ρ actually reaches 0. With covariates, the
   labeler's floor is ρ = 0.15.
5. **Cross-fitting uses the logistic model** even when the labeler is tabular, so 10
   fits per replicate stay fast.
6. **CI** installs `--extra llm` only. The `text` extra pulls PyTorch, and its tests
   use a stub encoder. `make setup` still installs all extras.
7. **`make reproduce`** now runs simulate, E1 and E2. E6 joins once its labels are
   cached.

### What I need from you (to finish Gate G3)

1. **Network access.** In the environment settings, allow these hosts:
   `files.consumerfinance.gov` (CFPB export), and `huggingface.co`,
   `cdn-lfs.huggingface.co` and `cas-bridge.xethub.hf.co` (embedding model).
   Optionally add `download.pytorch.org`, for the CPU-only PyTorch build (much
   smaller than PyPI's default).
2. **API key.** Put `ANTHROPIC_API_KEY` in `.env` (git-ignored) or add it as an
   environment secret. Please don't paste it into chat.
3. **Decisions:** the LLM model (default `claude-opus-5-5`), the spend cap
   (`max_spend_usd`) and the E6 pool size (default 10,000). The cost table is at the
   top of this note: about $28 for 10,000 items with Opus 5.5 batched, about $0.70
   with Haiku 5.5.

Environment changes take effect in a new session. Then the sequence is
`make cfpb`, then E6 with tabular and embedding only, then the LLM dry run (which
prints the exact projected cost for your approval), then the approved run with
`--allow-llm-calls`, then the G3 note.

### Open questions

1. A3 "within Monte Carlo error": judge it by widths (strict, 8/20), by MSE (18/20),
   or by a relative tolerance on the width ratio (for example within 2% of theory,
   which all 20 meet)?
2. The decisions in "What I need from you".

Phase 3 stops here: Gate G3 cannot be completed until the items in "What I need from
you" are in place.
