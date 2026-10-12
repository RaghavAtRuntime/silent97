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
| `labelers/llm.py` | Local open-weight LLM labeler (see "Change: local LLM" below). The first version, an Anthropic API labeler, was replaced at the reviewer's request |
| `data/cfpb.py` | Download once; lazy scan of the multi-GB export; window, narrative and dispute filters; deterministic Parquet + manifest with drop counts and SHA-256s; one-hot product/issue/state; dispute coding; guard on labeler columns |
| `eval/harness.py` | `labeler.kind: tabular`; scenario `labeler_features: signal_only`; `model_corr` per scenario; unlabeled set = pool minus draw; `run_grid(prepare=...)` for non-simulator datasets |
| `eval/metrics.py` | `effective_n_bootstrap` (paired over replicates); `ppi_pp_theory_ratio` |
| `experiments/e02_model_quality.py` | E2 (run, below) |
| `experiments/e06_cfpb.py` | E6 (built and smoke-tested; **not run**) |
| `configs/{llm,e02_model_quality,e06_cfpb}.yaml` | `llm.yaml` ships with `max_spend_usd: null` |

### Tests

`make test`: 1,305 fast tests pass (about 1 minute; 1,307 before the local-LLM change). `make lint` is clean. New tests:
- LLM labeler: originally 24 tests against a fake Anthropic client; now 22 tests for
  the local model (see "Change: local LLM").
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
| LLM compute and items reported (was "spend under cap") | Local model, so $0 by construction. Item cap 5,000 (was 10,000; see the session 2 update). The E6 run writes `docs/results/e06_cfpb_llm_report.json` (items scored, invalid, seconds) |
| E6 coverage reported for each labeler | ✗ Not run |
| `make test`, `make lint`, `make test-slow` | Pass: 1,305 fast tests, lint clean, slow suite 15/15 (4 min 43 s; estimators unchanged since) |

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

### Change: local open-weight LLM instead of the Anthropic API (reviewer request)

The reviewer asked for a lightweight open-source model run locally instead of the
Claude API. `labelers/llm.py` was rewritten:

- **Model and runtime:** `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, 494M parameters)
  on CPU through Hugging Face `transformers`, set in `configs/llm.yaml` (a larger
  model is a one-line change; `revision` can pin a commit). `transformers` and PyTorch
  were already transitive dependencies of the approved `sentence-transformers`;
  `transformers` is now declared explicitly in the `text` extra. The `anthropic` SDK
  and the `llm` extra are removed.
- **Probability readout:** one forward pass per complaint, no generation. The prompt
  asks for a single letter A / B / C (detractor / passive / promoter). The class
  probabilities are the softmax of the next-token logits for those three tokens,
  computed in float64. A 0.5B model writes unreliable numeric JSON, so this replaces
  the "zero-shot JSON" readout in the original spec. `docs/PLAN.md` and `CLAUDE.md`
  Phase 3 were updated to match. The readout is deterministic, and PPI corrects any
  miscalibration.
- **Correctness details:** prompts use the model's chat template; batches are
  left-padded so the last position is every row's next-token position (tested:
  batched scores equal one-at-a-time scores); only the last position's logits are
  computed (`logits_to_keep=1`); the option letters must be single tokens or loading
  fails; item text is truncated so the prompt fits `max_input_tokens = 512`, and
  truncations are counted.
- **Rules kept:** dry run by default, which prints items and projected compute; a
  cumulative item cap from a ledger; the cache key covers model@revision, prompt
  version (instructions, options, truncation length) and text; invalid outputs are
  cached as (1/3, 1/3, 1/3); only `cfpb` / `synthetic` sources. Spend is zero and
  nothing leaves the machine.
- **Tests:** 18 logic tests use a stub scorer, and 4 tests run the real `transformers`
  path on a tiny random-weight Llama with a locally trained tokenizer and chat
  template: valid probabilities, batching invariance, determinism, truncation, and
  end-to-end labeling. The 4 real-path tests need the `text` extra (PyTorch); CI does
  not install it, so they are skipped there via `pytest.importorskip`. They pass here.
  Anthropic-specific tests (spend cap, batches, fallbacks, base URL, API key) were
  removed together with that code.
- **Speed:** a randomly initialized model with Qwen2.5-0.5B's exact architecture runs
  at about 450 prompt tokens/s on this container's 4 CPU cores (0.9 s per 400-token
  complaint, batch 8). So a 10,000-complaint pool takes about **2.5 hours** once, and
  then replays from cache. A 5,000 pool takes about 1.2 hours.
- **E6:** `--run-llm` (replacing `--allow-llm-calls`) scores the pool locally; without
  it the LLM prints its dry-run plan and is skipped.

### What I need from you (to finish Gate G3)

1. **Network access** (environment settings, then Network access, then Allowed
   domains): `files.consumerfinance.gov` for the CFPB export, and `huggingface.co`,
   `cdn-lfs.huggingface.co` and `cas-bridge.xethub.hf.co` for both the embedding
   model and the Qwen weights. Changes apply to a new session.
2. **Pool size:** 10,000 (about 2.5 h of local LLM scoring) or 5,000 (about 1.2 h).
   No API key and no spend cap are needed any more.

Then: `make cfpb`, then E6 with all three labelers (`--run-llm`, cached), then the G3
note.

### Open questions

1. A3 "within Monte Carlo error": judge it by widths (strict, 8/20), by MSE (18/20),
   or by a relative tolerance on the width ratio (for example within 2% of theory,
   which all 20 meet)?
2. Pool size (above).

Phase 3 stops here: Gate G3 cannot be completed until the network access is in place.

---

## Update, session 2 (2026-10-11): reviewer decisions applied; still blocked on network

### Decisions received

1. **A3 is judged on the MSE ratio** (MSE_classical / MSE_PPI++).
2. **E6 pool size: 5,000.**
3. The environment's network policy was updated.

### What changed

| Change | Files |
|---|---|
| A3 is judged on the MSE ratio. `docs/PLAN.md` A3 now says so; the width-based n_eff is still reported alongside. E2's summary leads with the MSE ratio. Its A3 columns (`a3_within_mc`, `a3_threshold_met`) use MSE, and the width-based theory check is kept as `theory_within_mc_widths` (information only). The headline E2 figure is now MSE-based (`e02_model_quality.png`, with the A3 line); the width-based figure moved to `e02_model_quality_widths.png` | `experiments/e02_model_quality.py`, `viz/figures.py` (`metric="mse" \| "widths"`), `tests/unit/test_e02.py` (new, 3 tests), `docs/PLAN.md` |
| E2 rerun (1,000 replicates, same seed, 489 s). Every value matches the previous run exactly; only the column names and order changed | `docs/results/e02_model_quality*` |
| E6 pool 5,000, and the LLM's cumulative item cap set to 5,000 to match | `configs/e06_cfpb.yaml`, `configs/llm.yaml` |
| The CFPB loader reads a Parquet mirror and cross-checks it against the official export (see "CFPB export changed") | `data/cfpb.py`, `tests/unit/test_cfpb.py` (5 new tests) |
| Fixed the stale Makefile comment on E6 (`--run-llm`, not `--allow-llm-calls`) | `Makefile` |

### A3 result (MSE-based), from `docs/results/e02_model_quality_summary.csv`

- **At least 1.5× when ρ ≥ 0.6: met at all 8 points.** The MSE ratios are 1.645, 1.867, 2.909 and
  4.259 (n = 100) and 1.560, 2.078, 2.979 and 3.966 (n = 500). The smallest, 1.560 at n = 500,
  ρ = 0.604, has a bootstrap interval of [1.439, 1.686], so that point clears 1.5× on its point
  estimate, but not with Monte Carlo margin.
- **Within Monte Carlo error of the theory: 18 of 20 points.** The two misses:
  - n = 100, ρ = 0.003: 0.983 [0.969, 0.995] against theory 1.000. With a useless model,
    PPI++ is about 1.7% *less* efficient than classical at n = 100, from the noise in
    estimating λ̂ on the same 100 responses. This is an O(1/n) effect that the
    asymptotic theory leaves out. At n = 500 the same point is 1.001 [0.995, 1.006].
  - n = 500, ρ = 0.881: 3.966 [3.542, 4.416] against theory 4.434. It misses by 0.018 on
    the upper bound. The n = 100 point at the same ρ is inside (4.259 [3.830, 4.763]).
    I read this as Monte Carlo chance (1 in 20 is expected at 95%), not a systematic gap.
- Coverage is unaffected: all 60 coverage cells are inside the A1 band.

### CFPB export changed: narrative and dispute columns are gone

With `files.consumerfinance.gov` now reachable, the official export
(`complaints.csv.zip`, 353 MB, last modified 2026-10-11) downloads. But **it no longer
has the "Consumer complaint narrative", "Consumer consent provided?" or "Consumer
disputed?" columns.** It has 15 columns and about 26 million rows. Without them the benchmark
cannot be built from the official file. The CFPB search API (`www.consumerfinance.gov`)
is still blocked here, so I could not check whether it still serves them.

Two public mirrors of the older export still carry both fields:

| Mirror | License | Rows | Columns |
|---|---|---|---|
| `BEE-spoke-data/consumer-finance-complaints`, `has-text` subset (3 Parquet shards, 925 MB) | CC0-1.0 | rows with a narrative | identical to the original export |
| `davidheineman/consumer-finance-complaints-large` | Apache-2.0 | 7.2 M | API field names |

The loader now defaults to the first one (the same column names, so no remapping). Its
three shards are pinned by the SHA-256s that Hugging Face publishes, and
`download` refuses a file that does not match. Because the mirror is a third-party copy, `build` records
`crosscheck` in the manifest. For every kept complaint, it looks the complaint id up in
the current official export and counts how many have identical date, product, issue and
state. Narrative and dispute cannot be checked against the official source any more.
Comparing them against the second mirror is a further check I can add if you want it.

**This changes the data source from the one in the spec, so it needs your OK.** It is still
public CFPB data, and nothing is sent to an external API, so it is within section 9.

### Still blocked: Hugging Face's file CDN

Hugging Face now redirects every weight and dataset download from `huggingface.co` to
**`us.aws.cdn.hf.co`**, and the network policy denies that host (403 on CONNECT). The old
hosts (`cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) are no longer used. This blocks
the MiniLM embedding model, the Qwen2.5-0.5B weights and the CFPB mirror. Small metadata
files from `huggingface.co` itself work.

### New statistical caveat: the finite-pool effect at pool 5,000

E6's truth is the dispute rate of the 5,000-complaint pool, and each replicate draws n
complaints from that pool *without replacement*. The intervals use the usual iid variance
with no finite-population correction. They are therefore conservative for the pool rate,
by a factor of about √(1 − n/5000) on the true standard error. For classical, the expected
coverage of a nominal 90% interval is 2Φ(1.645 / √(1 − n/N_pool)) − 1:

| n | pool 5,000 | pool 10,000 |
|---|---|---|
| 50 | 0.902 | 0.901 |
| 100 | 0.903 | 0.902 |
| 200 | 0.907 | 0.903 |
| 500 | 0.917 | 0.909 |
| 1,000 | **0.934** | 0.917 |

PPI and PPI++ are conservative in the same way: the error variance relative to the pool
rate is (1 − n/N_pool)·S²(r − (n/N)f)/n, which is below the reported S_f²/N + S_r²/n. So at pool
5,000, E6 should show coverage **above** the A1 band at n = 1,000, and borderline at
n = 500. This is over-coverage (valid but too wide), not a failure, and A1 formally
covers the simulator only. I did not account for it when I offered 5,000 against
10,000 as a pure runtime trade-off last session.

### Gate G3 status (session 2)

| Criterion | Status |
|---|---|
| All labels cached | ✗ Blocked: `us.aws.cdn.hf.co` (models and mirror), and the CFPB source decision |
| LLM compute and items reported | Not run. Pool and cap are both 5,000 (about 1.2 h of CPU scoring, projected) |
| E6 coverage reported for each labeler | ✗ Not run |
| `make test`, `make lint` | Pass: 1,313 fast tests, lint clean. `make test-slow` not rerun (no estimator changes) |

### What I need from you

1. **Network:** add `us.aws.cdn.hf.co` to Allowed domains. Optionally also add
   `www.consumerfinance.gov`, so I can check whether the official API still serves
   narratives and disputes.
2. **CFPB source:** OK to use the CC0 Hugging Face mirror, with the official-export
   cross-check in the manifest?
3. **Pool size and the finite-pool effect** (table above): keep 5,000 and report the
   n = 1,000 over-coverage as expected and explained; or go to 10,000 (about 2.5 h of
   local scoring); or keep 5,000 and drop n = 1,000 from E6.

Once those are settled, the next session runs `make cfpb`, E6 with all three labelers
(`--run-llm`), and writes the G3 note.

---

## Gate G3 results (session 2, continued, 2026-10-11)

Reviewer decisions: the CC0 mirror is approved; keep pool 5,000 and report the over-coverage;
`us.aws.cdn.hf.co` and `www.consumerfinance.gov` were added to the allowed domains.

### CFPB data (`data/processed/cfpb/cfpb_manifest.json`)

- Source: the three `has-text` shards of `BEE-spoke-data/consumer-finance-complaints`,
  SHA-256-verified against the pinned values. There are 1,689,573 rows with a narrative; 1,522,903
  are outside the window and 2,664 have no dispute value. **Kept: 164,006 complaints, dispute
  rate 21.83%.**
- **Cross-check against the current official export:** 163,950 of the 164,006 complaint ids
  are found there. 163,883 have identical date, product, issue and state. The 67
  differences are all `state`: the mirror says `PR` (Puerto Rico) where the official export
  now has the field blank. The 56 ids not found have presumably been removed from the official
  database since the mirror was made. Narrative and dispute cannot be cross-checked: the CFPB
  API (`www.consumerfinance.gov`, now reachable) has also dropped
  `complaint_what_happened` and `consumer_disputed`, and its `has_narrative` filter is
  ignored (totals are unchanged with or without it).
- Hugging Face's `xet` download client needs one more host, `cas-server.xethub.hf.co`, which is
  still blocked. With `HF_HUB_DISABLE_XET=1`, downloads fall back to plain HTTPS through
  `us.aws.cdn.hf.co`, which works. Both models are now in `~/.cache/huggingface`, and E6 ran
  with `HF_HUB_OFFLINE=1`.

### LLM labels (`docs/results/e06_cfpb_llm_scoring.json`, `docs/results/e06_cfpb_llm_report.json`)

- Model `Qwen/Qwen2.5-0.5B-Instruct`, now pinned to revision
  `7ae557604adf67be50417f59c2c2f167def9a775` in `configs/llm.yaml`. CPU, 4 cores, $0, nothing
  sent off the machine.
- The dry run reported 5,000 items, of which 4,992 needed scoring (8 duplicate complaint
  texts share a cache key), projected at 4,255 s.
- **Actual: 4,992 scored in 6,710 s (1.86 h, about 1.34 s per complaint), 0 invalid.** That is
  58% slower than projected: the 450 tokens/s figure in `configs/llm.yaml` came from a
  random-weight model of the same architecture. I left the config value unchanged, because it
  only drives the dry-run estimate.
- The E6 run replayed all 5,000 from cache (0 newly scored). The item cap is 5,000 and the
  ledger shows 4,992 used.

### E6 (`docs/results/e06_cfpb_summary.{csv,md}`, figures `e06_cfpb_{tabular,embedding,llm}.png`)

Pool of 5,000 complaints, true dispute rate 21.32%, 1,000 replicates per cell, seed 20261014,
16.5 min on 4 cores (including encoding about 38,000 narratives for the embedding labeler).

**Labelers are weak on this task.** Model–outcome correlation ρ on the pool: embedding 0.179,
tabular 0.074, LLM **−0.083**. The LLM's P(detractor) averages 73.8%, against a true rate of 21.3%
(the naive-model bias is +52.5 points). Nearly every complaint reads as a detractor to
it, and its ranking is slightly *anti*-correlated with disputing. As a result there are no
meaningful width gains:

| labeler (ρ) | effective n at n = 1,000: PPI / PPI++ / cross-fit | PPI++ mean λ̂ (n = 50 → 1,000) |
|---|---|---|
| embedding (0.179) | 1,018 / 1,027 / 853 | 0.78 → 0.64 |
| tabular (0.074) | 1,004 / 1,005 / 853 | 0.82 → 0.70 |
| LLM (−0.083) | 880 / 1,006 / 852 | −0.33 → −0.25 |

PPI++ handles the anti-correlated LLM as designed. It picks a negative λ̂ and stays at
classical efficiency (1,006), whereas plain PPI (λ = 1) loses 12% (880). Cross-fit PPI does
not use the labelers at all (it fits its own logistic model on one-hot product, issue and
state) and costs about 15% efficiency at every n. These gains are far below what E2 shows on
the simulator, because ρ here is below 0.2 (theory: 1/(1 − 0.179²) = 1.03).

**Coverage.** Valid estimators (classical, PPI, PPI++, cross-fit) × 3 labelers = 12 cells
per n. Within the A1 band (Wilson interval overlaps [0.88, 0.92]): **57 of 60**. The 3
misses are all over-coverage at n = 1,000: tabular classical 0.940 [0.924, 0.953], tabular
PPI 0.938 [0.921, 0.951] and LLM PPI++ 0.937 [0.920, 0.950]. All 15 naive-model cells cover 0%,
as expected.

Mean coverage across the 12 cells against the finite-pool prediction
2Φ(1.645/√(1 − n/5000)) − 1:

| n | all valid | classical, PPI, PPI++ | cross-fit | predicted |
|---|---|---|---|---|
| 50 | 0.881 | 0.883 | 0.877 | 0.902 |
| 100 | 0.888 | 0.890 | 0.883 | 0.903 |
| 200 | 0.896 | 0.899 | 0.888 | 0.907 |
| 500 | 0.913 | 0.915 | 0.906 | 0.917 |
| 1,000 | 0.931 | 0.935 | 0.918 | 0.934 |

- **n ≥ 500: over-coverage matches the finite-pool prediction** (0.915 vs 0.917, and 0.935 vs
  0.934). The three n = 1,000 band misses are that effect, as anticipated: the intervals are
  valid for the pool rate, only wider than needed. Per the reviewer's decision, pool and
  criteria are unchanged.
- **Small n runs about 2 points below the prediction** (0.883 at n = 50). This is the
  familiar small-sample undercoverage of Wald intervals for a proportion near 0.2 (skewed
  binomial). All those cells are still inside the A1 band. The 12 cells share the
  same kind of draws and are not independent, so I give the averages as a description, not a test.
- **Cross-fit runs about 1–1.7 points below the others at every n** (0.918 against 0.935 at
  n = 1,000), even though its intervals are about 8% wider. Its plug-in variance leaves out
  the variability from refitting the fold models, which is noticeable with a few dozen labeled
  complaints per fold and hundreds of sparse one-hot features. Every cross-fit cell is
  inside the A1 band, so I note this for the record and have not investigated it further.

### Gate G3 status

| Criterion | Status |
|---|---|
| All labels cached | ✓ Tabular and embedding (fit on the 20% training split; embeddings cached under `cache/embeddings`), LLM (5,000 items under `cache/llm`) |
| LLM compute and item count reported | ✓ 4,992 scored, 6,710 s on CPU, 0 invalid, $0; cap 5,000 |
| E6 coverage reported for each labeler | ✓ Above; 57/60 valid cells in the A1 band, with the 3 misses being the predicted finite-pool over-coverage at n = 1,000 |
| `make lint`, `make test`, `make test-slow` | ✓ lint clean; 1,313 fast tests (56 s); slow suite 15/15 (4 min 55 s) |

Also: `make e06` now passes `--run-llm`, so it replays the cached LLM labels, and `make
reproduce` includes E6. E6 needs `make cfpb` first; `data/` and `cache/` are git-ignored, so a
clean clone has to rebuild them (G5 will need to account for this).

### Open questions

1. **Reproducing from a clean clone (for G5).** E6's labels live in the git-ignored
   `cache/`. A clean clone either rescores (about 2 h of CPU) or needs the cache shipped
   another way, for example as a release asset. It is your call how "from cached labels"
   should work.
2. The labelers are weak on dispute prediction (ρ ≤ 0.18). A larger local LLM would be a
   one-line config change, at several times the CPU time. That is out of Phase 3 scope, and
   I note it for `docs/ideas.md` only if you want it.

Phase 3 stops here for review.

---

## Update, session 2 (continued, 2026-10-12): stronger LLM and release download

Reviewer decisions: the label cache is distributed as a GitHub release download, and
the LLM labeler should use the strongest model that runs comfortably on a Mac M1.

### Model choice: `Qwen/Qwen3.5-4B` (revision `851bf6e8…`, Apache-2.0)

- **Constraint.** A 16 GB M1 runs PyTorch on `mps` in bf16, so about 4–5B parameters is the
  comfortable limit (8–9 GB of weights). The 8–9B candidates (Qwen3.5-9B, Gemma-4-E4B at
  8B stored parameters, Granite-4.1-8B, Olmo-3-7B) need 16–19 GB in bf16. Gated or
  non-Apache models (Llama 3.2, the Gemma 3 family, Qwen2.5-3B) were excluded, because
  downloading them needs a login or they carry restrictive licenses.
- **Candidates checked** on 8 complaints from the labeler-training split. The evaluation
  pool was not used for selection.
  - Qwen3.5-4B (Feb 2026, the newest generation at this size) gave graded probabilities,
    with about 99% of next-token mass on A/B/C once thinking is off.
  - Qwen3-4B-Instruct-2507 put probability 1.000 on "detractor" for all 8 complaints. That
    saturates the readout, so PPI would get no ranking information from it.
- **Code changes** (`labelers/llm.py`, tests first):
  1. `render_prompt` passes `enable_thinking=False`; otherwise Qwen3.5's next token is
     `<think>`.
  2. `weights_dtype`: linear weights are stored in bf16 and each layer computes in float32
     (`upcast_linears`). This x86 CPU has no native bf16: a bf16 matmul runs at 116 GFLOP/s
     against 491 for float32, while a float32 copy (17 GB) does not fit in 15 GB of RAM. On a
     Mac, the config comment says to use `device: mps`, `dtype: bfloat16`.
  3. Batches are sorted by text length, so less compute goes to padding.
  4. New tests: length-sorted batching maps results back correctly; dtype validation;
     thinking is disabled in the rendered prompt; `upcast_linears` equals a float32 model with
     the same weights; and on a **tiny random Qwen3.5-style hybrid model** (linear-attention
     plus full-attention layers), left padding does not change any score, and upcast equals
     float32. On the real Qwen3.5-4B, batched scores matched one-at-a-time scores to
     6.7e-7 on 4 complaints.
- `configs/llm.yaml`: model, revision, `weights_dtype: bfloat16`, `tokens_per_second: 32`
  (measured). `max_items` was raised to 10,000, because the cap is cumulative over the cache and
  the Qwen2.5-0.5B run already used 4,992.

### Scoring run

- 4,992 complaints scored (8 duplicate texts), **0 invalid**, $0, CPU only. The wall-clock was
  about 12.6 h of scoring, in seven chunks.
- The first attempt ran as a detached process and died when the container was reclaimed
  while the session was idle (after 72 items, which stayed cached). The rest ran as tracked
  background chunks of under 2 h each, resuming from the cache. Per-chunk logs are in the
  session; the ledger has one line per scored item.

### E6 with Qwen3.5-4B (`docs/results/e06_cfpb_*`, 1,000 replicates, same seed and pool)

The tabular and embedding results are byte-identical to the previous run. LLM rows:

| | Qwen2.5-0.5B (before) | Qwen3.5-4B (now) |
|---|---|---|
| ρ with disputing | −0.083 | **0.041** |
| mean P(detractor) vs true rate 21.3% | 73.8% | 81.8% (naive-model bias +60.5 points) |
| effective n at n = 1,000: PPI / PPI++ / cross-fit | 880 / 1,006 / 852 | 883 / 1,002 / 852 |
| PPI++ mean λ̂ | −0.33 → −0.25 | 0.10 → 0.09 |
| coverage cells in A1 band (of 20 valid) | 19 | **20** |

All valid cells: **58 of 60** in the A1 band. The two misses (tabular classical 0.940 and
tabular PPI 0.938, both at n = 1,000) are the finite-pool over-coverage predicted above.

**The stronger model does not help on this task.** Its correlation with disputing is
0.041, which is effectively zero; PPI++ correctly gives it almost no weight (λ̂ ≈ 0.1), so
the interval matches classical. Plain PPI loses 12% efficiency. My reading is that this is a
*task mismatch*, not a weak model. The zero-shot prompt asks whether the writer is a
detractor, passive or promoter, and almost every complaint is a detractor by that
standard. Whether the consumer later *disputes the company's response* depends on that
response, which the labeler does not see by design (`LABELER_FEATURES`). The supervised
embedding labeler, trained on dispute labels, reaches ρ = 0.18. A dispute-specific prompt
might do better; that is a prompt change, so it is listed as open question 2.

### Release download

- `labelers/cache.py`:
  - `pack` writes a deterministic `.tar.gz` (sorted members, zeroed times and owners, no
    gzip filename). Packing twice gives the same SHA-256.
  - `unpack` verifies the SHA-256 and rejects unsafe paths. It skips identical files and
    refuses to overwrite a differing one.
  - `fetch` downloads the archive named in `configs/labels.yaml` once.
- Makefile targets: `pack-labels` and `fetch-labels`. The README has the reproduce steps
  (`make cfpb`, `make fetch-labels`, `make reproduce`).
- **Archive `silent97-labels.tar.gz`:** 54,973,418 bytes, SHA-256
  `c35fa9b3816fe9ed272108e7bc8ff5f9282aab6fc8bdb16aef9be0f516e8b2ec`. It holds all 9,987
  cache files: both LLMs' scores, the ledger and the two embedding matrices. Unpacking it
  into an empty directory reproduces `cache/` exactly (`diff -r`).
- **Not yet published.** I cannot create releases from this environment. The
  owner creates release `labels-v1` and uploads the file; then `make fetch-labels` works.

### Open questions

1. **Release upload** (above). After it, I will verify `make fetch-labels` against the live URL.
2. **LLM prompt for dispute.** Keep the NPS-category prompt (consistent with the simulator
   framing; result reported as is), or add a dispute-specific prompt version for E6? The
   second would be another rescoring run of about 12.6 hours.
