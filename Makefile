.PHONY: setup test test-slow lint format coverage reproduce demo simulate e01 e02 e03 e04 e05 e06 e06a e07 cfpb amazon planning fetch-labels pack-labels

UV ?= uv

setup:
	$(UV) sync --all-extras
	$(UV) run pre-commit install

test:
	$(UV) run pytest -m "not slow"

test-slow:
	$(UV) run pytest -m slow || [ $$? -eq 5 ]  # exit 5 = no slow tests collected yet

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy src

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

coverage:
	$(UV) run pytest -m "not slow" --cov=silent97 --cov-report=term-missing

simulate:
	$(UV) run python -m silent97.data.simulate --config configs/simulate.yaml

e01:
	$(UV) run python experiments/e01_coverage.py --config configs/e01_coverage.yaml

e03:
	$(UV) run python experiments/e03_active.py --config configs/e03_active.yaml

e04:
	$(UV) run python experiments/e04_nonresponse.py --config configs/e04_nonresponse.yaml

e05:
	$(UV) run python experiments/e05_segments.py --config configs/e05_segments.yaml

e07:
	$(UV) run python experiments/e07_sensitivity.py --config configs/e07_sensitivity.yaml

e02:
	$(UV) run python experiments/e02_model_quality.py --config configs/e02_model_quality.yaml

# CFPB data (needs network access to files.consumerfinance.gov).
cfpb:
	$(UV) run python -m silent97.data.cfpb

# Amazon Reviews 2023, Gift Cards (SHA-256 pinned; research use).
amazon:
	$(UV) run python -m silent97.data.amazon

# The cached labels behind the published results: download the pinned release asset
# (configs/labels.yaml), check its SHA-256 and unpack it into cache/.
fetch-labels:
	$(UV) run python -m silent97.labelers.cache fetch --config configs/labels.yaml

# Maintainers: write the release archive from cache/ (then upload it to the release
# and pin its URL and SHA-256 in configs/labels.yaml).
LABELS_ARCHIVE ?= data/release/silent97-labels.tar.gz
pack-labels:
	$(UV) run python -m silent97.labelers.cache pack --out $(LABELS_ARCHIVE)

# E6 replays cached labels (needs `make cfpb` and `make fetch-labels` first). --run-llm
# includes the local LLM; with its labels cached nothing is re-scored, and the item cap
# in configs/llm.yaml stops any unexpected scoring.
e06:
	$(UV) run python experiments/e06_cfpb.py --config configs/e06_cfpb.yaml --run-llm

# E6 on Amazon gift-card reviews: real text with a real satisfaction rating.
e06a:
	$(UV) run python experiments/e06_amazon.py --config configs/e06_amazon.yaml --run-llm

# Responses needed by model quality (rho) and target margin.
planning:
	$(UV) run python -m silent97.eval.planning table --out docs/results/planning_table.md

# Every experiment result and figure. Experiments are added as their phases land.
# E6 needs the data (`make cfpb amazon`) and the cached labels (`make fetch-labels`).
reproduce: simulate e01 e02 e03 e04 e05 e06 e06a e07 planning

# The demo is built in Phase 5.
demo:
	@echo "demo/ is built in Phase 5; nothing to serve yet."
