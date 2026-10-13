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

### Follow-up (2026-10-12): release published; how predictable is disputing at all?

- **Release `labels-v1`** is published. Its asset digest equals the pinned SHA-256. Because the
  repository is private, the unauthenticated download in `make fetch-labels` returns 404; it
  works once the repository is public (or with token support, if wanted).
- **Ceiling check on the outcome.** This is a diagnostic only, on the labeler-training split
  (32,801 complaints; the evaluation pool was not touched). LightGBM on metadata, evaluated
  on a held-out half of that split. ρ with disputing:

  | features | ρ |
  |---|---|
  | product, issue, state (current tabular labeler) | 0.063 |
  | + sub-product, sub-issue | 0.078 |
  | company response alone | 0.120 |
  | product, issue, state, company response, timely, channel, company | 0.134 |
  | all of the above + public response, tags | 0.156 |

  Even with every pre-dispute field in the export, ρ stays below 0.2. The company's response
  is the strongest single signal (24% of complaints closed "with explanation" were disputed,
  against 12% of those closed with relief). Disputing is close to unpredictable from the
  public data, so **no labeler can make E6 show a large efficiency gain**; theory caps it near
  1/(1 − 0.2²) ≈ 1.04×. A dispute-specific LLM prompt is therefore not worth another 12.6 h
  scoring run. The CFPB benchmark's value is validity under a real, weak model, and showing
  the naive-LLM trap.

---

## Session 3 (2026-10-12): making the results useful (reviewer: "do 1, 2 and 3")

Reviewer decisions:
1. Reframe the CFPB benchmark (validity under weak models, plus the model-only trap).
2. Build a planning tool that estimates ρ from past surveys and turns it into survey sizes.
3. Add a real-text benchmark with a real satisfaction rating. Academic-use data is allowed
   (the owner has university credentials), and Amazon Reviews 2023 is approved.

### 1. CFPB reframed (no new compute)

- `docs/PLAN.md` E6 now describes two benchmarks:
  - **(a) CFPB:** validity under weak models plus the model-only trap. Disputing is nearly
    unpredictable from public data: ρ < 0.2 even from every metadata field.
  - **(b) Amazon reviews:** the gains on real text.
- New figure `docs/results/e06_cfpb_trap.png` (`viz.figures.trap_figure`): at n = 200, the
  average model-only interval against the truth, beside classical and PPI++.
  - The LLM-only estimate is 81.8% against a true 21.3%, with 0% coverage.
  - Even the supervised embedding and tabular labelers, whose model-only estimates land
    within about 1 point of the truth, get 0% coverage, because those intervals ignore
    model error.
  - Classical and PPI++ cover 89–91%.
- Rerunning E6 on CFPB reproduced the summary CSV byte for byte. It added the planning check
  below to `e06_cfpb_summary.md`.

### 2. Survey planning tool (`src/silent97/eval/planning.py`, tests first, 8 tests)

- `estimate_rho(y, p, rng=...)`: ρ = corr(actual NPS value, predicted NPS value), with a
  percentile bootstrap interval. It is the same quantity as the harness's `model_correlation`.
  A test checks the interval's coverage across 40 resampled datasets (at least 34 of 40 at
  nominal 95%).
- `gain(ρ, n, N)` = 1 / (1 − ρ²/(1 + n/N)), the curve E2 validated (A3).
  `required_n(margin, sd, ρ, N)` gives the smallest n whose interval half-width is at most
  the margin; it is exact, by bisection on the half-width formula. `planning_table` and
  `report` produce planning tables at the low, point and high values of ρ.
- CLI: `python -m silent97.eval.planning table`, and `estimate --data past.parquet --big-n N`
  for a team's own past responses with model scores. `make planning` writes
  `docs/results/planning_table.md` (NPS value sd 80, which fits NPS ≈ +20; N = 1 M; 90%
  intervals). For a ±5-point margin, a survey needs:

  | ρ | classical n | PPI++ n | saved |
  |---|---|---|---|
  | 0.5 | 693 | 520 | 25% |
  | 0.6 | 693 | 444 | 36% |
  | 0.7 | 693 | 354 | 49% |
  | 0.8 | 693 | 250 | 64% |
  | 0.9 | 693 | 132 | 81% |

- **Caveats in the docstring:** predictions must be out of sample for the past respondents;
  past respondents are self-selected, so plan with the low end of ρ's interval; gains below
  about 1.05 are noise-level at small n.
- **Predicted vs realized check** (`planning.compare_with_results`), added to both E6
  summaries. On CFPB it matches closely. At n = 1,000, for example: embedding predicted
  1.026, realized 1.027; tabular 1.004 vs 1.005; LLM 1.001 vs 1.002. At n = 50 the realized
  gain runs about 1–2 points above the prediction; that is the small-sample optimism of the
  in-sample λ̂ seen in E2. ρ estimated from one sample of 300 "past" responses lands near the
  pool value, with intervals of about ±0.12: embedding 0.160 [0.042, 0.277] against a pool
  value of 0.179.

### 3. Amazon gift-card reviews (`src/silent97/data/amazon.py`, `experiments/e06_amazon.py`)

- **Data:** McAuley Lab, Amazon Reviews 2023, category Gift_Cards. One file, 50.2 MB,
  SHA-256 `b7ceead0…544e` (pinned). It has 152,410 reviews; 1,421 exact duplicates of (user,
  product, timestamp) are dropped, leaving **150,989**. Gift cards were chosen as the
  stored-value, consumer-finance-adjacent category.
- **Outcome:** score = 2 × stars, so 5 stars is a promoter, 4 passive and 1–3 detractor.
  **True NPS +72.75** on the full data. Unlike CFPB, the full two-dimensional NPS
  estimators run on real text and a real satisfaction answer.
- **Labelers:**
  - tabular: metadata only (verified purchase, images, any helpful vote, year, length band);
  - embedding: MiniLM plus logistic regression on title and text;
  - LLM: Qwen3.5-4B with a review-specific prompt.
  The rating never reaches a labeler (`assert_labeler_columns`).
- **LLM prompts per source:** `SYSTEM_PROMPTS` maps each source to its prompt, and the review
  prompt does not mention complaints. The cache key uses the item's own prompt version. The
  CFPB keys are unchanged; 5,000 of 5,000 are still cache hits after the change. A scorer
  built for one prompt cannot score items from another source (guard plus tests).
- **Design** as in the CFPB benchmark: a 20% training split, then a pool of **10,000**
  reviews from the rest, with the truth being the pool's NPS. The larger pool keeps the
  finite-pool over-coverage at n = 1,000 near 91.7%, inside the band. Seed 20261015; 1,000
  replicates; n from 50 to 1,000.
- **LLM scoring:**
  - The dry run reported 10,000 items, of which 9,489 are unique texts; projected 12.9 h.
  - Measured about 3 s per item for the shortest reviews and 4.6 s on a typical sample.
  - It runs as `experiments/e06_amazon.py --score-only` in tracked chunks of under 2 h.
  - The item cap is now 20,000 cumulative (9,984 used by the two CFPB runs).
- Status: scoring is in progress; E6 (reviews) runs when it finishes.

`make test`: 1,345+ fast tests pass; `make lint` is clean.

### E6 (reviews) results (`docs/results/e06_amazon_*`, 1,000 replicates, seed 20261015)

**LLM scoring:** 9,489 unique reviews scored by Qwen3.5-4B in about 9.2 h of CPU (five chunks of
under 2 h). **0 invalid**, $0. The pool's true NPS is **+73.20** (10,000 reviews).

**Labeler quality (ρ on the pool):**
- embedding (trained on 30,198 labeled reviews): **0.809**
- LLM (zero-shot, no labels): **0.794**
- tabular (metadata only): 0.477

**Gains** (PPI++, effective n / n from widths; MSE ratio, the A3 measure, in brackets):

| labeler | n = 50 | n = 200 | n = 1,000 |
|---|---|---|---|
| LLM (zero-shot) | 2.52 [MSE 2.28] | 2.57 | 2.30 [MSE 2.85] |
| embedding | 2.53 [MSE 1.76] | 2.72 | 2.43 [MSE 2.81] |
| tabular | 1.29 [MSE 1.22] | 1.29 | 1.26 [MSE 1.26] |

This is the headline for real text. A zero-shot local LLM, with no training labels, makes
each survey response worth **about 2.3–2.6 classical responses** on real customer text with
real satisfaction ratings. For a ±5-point NPS interval, that means roughly 40% of the
responses. The width-based gain at n = 1,000 understates the MSE gain because of the
finite-pool effect (next paragraph).

**Planning check** (`e06_amazon_summary.md`): from n = 200 up, predicted and realized gains
agree within about 2%:
- LLM: 2.619 vs 2.565 at n = 200; 2.313 vs 2.303 at n = 1,000.
- tabular: 1.286 vs 1.290 at n = 200; 1.257 vs 1.258 at n = 1,000.

ρ estimated from 300 "past" responses: LLM 0.803 [0.745, 0.848] and embedding 0.795
[0.700, 0.873]. Both intervals contain the pool value. At n ≤ 100, PPI++ realizes less than
predicted (LLM 2.52 vs 2.69 at n = 50); see the next point.

**Coverage:** 57 of 60 valid cells are in the A1 band. All 15 naive-model cells for the LLM and
tabular labelers cover 0%; the LLM-only NPS is **51.95 against a truth of 73.20** (bias
−21.25). The embedding labeler's model-only estimate covers 100% (bias −0.22, width 1.8).
That model was trained on 30,198 labeled reviews from the same population, which is not
the small-survey setting. The three misses:

1. **PPI++ at n = 50, embedding labeler: 0.843 [0.819, 0.864] (under-coverage).**
   - *Investigation (one pass):* PPI++ estimates λ from the same n responses it uses for
     the rectifier, and that creates an O(1/n) bias. The bias roughly halves as n doubles
     (embedding: +2.23, +1.02, +0.48, +0.14 points at n = 50, 100, 200, 500), and the
     error is negatively correlated with λ̂ (corr −0.13 to −0.54 across labelers).
   - At n = 50 the reported SE is also 13% too small (embedding; reported se / empirical
     sd = 0.867, against 0.95–1.03 for classical and PPI).
   - The data are very skewed: 84% promoters, so about 5 detractors in a sample of 50. The
     plug-in λ̂ is noisy (0.95 ± 0.36).
   - The same pattern, within the band, holds at n = 50 for the LLM (0.869) and tabular
     (0.860) labelers.
   - Plain PPI (λ = 1) has no such bias and covers 0.888–0.906 at n = 50. Because the
     labelers are calibrated (λ* ≈ 1), it is also more efficient there: MSE gain 2.75 vs
     1.76 (embedding) and 2.89 vs 2.28 (LLM).
   - From n ≥ 500 the two agree.
   - Our `ppi_pp` matches `ppi_py` at 1e-6, so this is a property of the published
     plug-in estimator in small, skewed samples, not an implementation error.
   - **Open question 3.**
2. **PPI at n = 1,000, embedding labeler: 0.937 [0.920, 0.950] (over-coverage).** This is
   the finite-pool effect, and it is stronger for good models. The interval counts the
   unlabeled mean as noise (S_f²/N), but relative to the pool's own NPS it is not. For
   ρ ≈ 0.8, n = 1,000 and N = 9,000, the formula overstates the variance by about 30%,
   which predicts coverage of about 0.939. Observed: 0.935–0.937 for PPI and 0.926–0.936
   for PPI++.
3. **Cross-fit PPI at n = 50, tabular labeler: 0.855 [0.832, 0.875].** This is the cross-fit
   pattern already seen on CFPB. With 10 folds of 5 responses, refitting adds variance
   that the plug-in leaves out. Cross-fit also gains nothing here (effective n 43 at
   n = 50), because its own logistic model sees only metadata.

### Release

`make pack-labels` gives an archive of 113,669,500 bytes, SHA-256
`dba99b274dea495a2083ac1e91deaeacaad203c4b7aaef95a178ed2a3573cc56`. It contains the CFPB
labels (both LLMs), the review labels and both embedding caches. `configs/labels.yaml` now
points at release **`labels-v2`**, which needs uploading (labels-v1 is superseded).

### Open questions

3. **PPI++ at small n on skewed data.** Options:
   - (a) Document that at n < ~200 with a calibrated labeler, plain PPI (λ = 1) is
     preferred. This needs no code change, and the data support it.
   - (b) Add a cross-fitted-λ PPI++ variant: estimate λ on the other folds, which removes
     the same-sample bias. This would be a new, separately tested estimator, and it
     needs your approval under CLAUDE.md rules 3 and 6.
   - (c) Both.
4. Upload `labels-v2` (send-out below).

---

## Session 3, continued (2026-10-13): small-n PPI++ fix (reviewer chose option c); labels-v2 published

### `labels-v2`

Published by the owner. The asset digest equals the pinned SHA-256 (`dba99b27…cc56`).
Unauthenticated downloads still return 404 while the repository is private.

### (b) New estimator `ppi_pp_cf`: PPI++ with a cross-fitted λ (`estimators/ppi.py`)

- **Method.**
  - The labeled rows are split into K folds (`rng` or given `folds`).
  - λ_k is the same plug-in NPS-contrast λ as `ppi_pp`, computed from the labeled rows
    *outside* fold k only. The unlabeled set enters only through N.
  - θ̂ = mean_i(Z_i − λ_{k(i)} f_i) + λ̄ · mean_j f̃_j, with λ̄ = Σ_k (n_k/n) λ_k.
  - The variance treats the λ_k as fixed and uses full-sample moments; with equal λ_k it
    is exactly the PPI variance. This is the cross-fitting idea of DML (Chernozhukov et
    al. 2018, *Econometrics Journal* 21(1)), applied to the tuning parameter.
  - **Exactly unbiased:** λ_k is independent of fold k's rows and of the unlabeled set.
    The derivation is in the module docstring.
- **Tests first** (`tests/unit/test_ppi_pp_cf.py`, 9 tests):
  - reduces to PPI(λ) when the fold λs are equal;
  - the point formula holds with unequal λs;
  - λ_k ignores fold k's answers and the unlabeled values;
  - the fold split is reproducible;
  - permuting rows together with their folds changes nothing;
  - constant predictions reduce to classical;
  - input validation;
  - a Monte Carlo probe on a skewed population at n = 30: plug-in PPI++ bias +2.52
    (MC se 0.17); `ppi_pp_cf` −0.06.
- A harness test checks that adding `ppi_pp_cf` (its own random stream) leaves every other
  estimator's results byte-identical. This was also confirmed on E1 and both E6 reruns.
- **One existing test changed:** `tests/unit/test_ppi.py::test_registry` asserts the exact
  set of registered estimators, and now lists `ppi_pp_cf`. The check is as strict as
  before; it is flagged here per CLAUDE.md §1.3.
- `ppi_pp_cf` is in `VALID_UNDER_MCAR`, so the slow suite checks it. **`make test-slow`:
  18/18 pass**, including `ppi_pp_cf` at n = 50, 200 and 1,000.

### Evidence (E1, E6 CFPB and E6 reviews rerun with `ppi_pp_cf`; other rows unchanged)

| | `ppi` | `ppi_pp` | `ppi_pp_cf` |
|---|---|---|---|
| Reviews n = 50, bias (emb / LLM / tab) | −0.18 / −0.04 / −0.19 | **+2.23 / +1.75 / +1.15** | −0.30 / −0.06 / −0.10 |
| Reviews n = 50, coverage | 0.895 / 0.906 / 0.888 | **0.843** / 0.869 / 0.860 | 0.872 / 0.884 / 0.865 |
| Reviews n = 50, MSE gain vs classical | 2.75 / 2.89 / 1.32 | 1.76 / 2.28 / 1.22 | 2.54 / 2.77 / 1.24 |
| CFPB LLM (λ* ≈ 0.1), MSE gain n = 50 / 1,000 | **0.91 / 0.81** | 0.98 / 1.00 | 0.97 / 1.00 |
| E1 simulator n = 50, bias / coverage | +0.18 / 0.900 | +0.44 / 0.888 | +0.15 / 0.890 |
| Cells in A1 band (all three experiments) | 34 / 35 | 34 / 35 | 34 / 35 |

(The A1 misses: reviews `ppi` and `ppi_pp_cf` at n = 1,000, embedding labeler, 0.937, which
is the finite-pool over-coverage explained above; CFPB tabular `ppi` at n = 1,000, 0.938,
same cause; `ppi_pp` reviews n = 50, 0.843.)

**Conclusion (a + b, as documented in `docs/PLAN.md`):** `ppi_pp_cf` is the recommended
default for small surveys:
- It is unbiased.
- Its coverage is in the band.
- It is close to plain PPI's efficiency when the labeler is calibrated.
- It is safe when the labeler is not; plain PPI loses up to 19% against the survey alone
  on CFPB.

Plain PPI is marginally better at n ≤ 100 *if* the labeler is known to be calibrated.
Plug-in `ppi_pp` is fine from about n = 500.

**Remaining limitation (reported, not fixed):** at n ≤ 50 on very skewed data, `ppi_pp_cf`'s
SE is somewhat optimistic, because the fold λs' variability is ignored. On the skewed probe
population, reported se / empirical sd was 0.849 at n = 30 and 0.916 at n = 50; for classical
and plain PPI it was 0.95–0.98. At n ≥ 100 all are 0.96–0.98. A variance term for the
λ_k variability, or a bootstrap interval, would be a new interval type (CLAUDE.md rule 6)
and is not attempted here.

`make lint` is clean, `make test` gives 1,363 passed, and `make test-slow` gives 18/18.
