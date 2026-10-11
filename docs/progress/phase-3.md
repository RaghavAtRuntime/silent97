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
