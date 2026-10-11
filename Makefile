.PHONY: setup test test-slow lint format coverage reproduce demo simulate e01 e02 e06 cfpb fetch-labels pack-labels

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

e02:
	$(UV) run python experiments/e02_model_quality.py --config configs/e02_model_quality.yaml

# CFPB data (needs network access to files.consumerfinance.gov).
cfpb:
	$(UV) run python -m silent97.data.cfpb

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

# Every experiment result and figure. Experiments are added as their phases land.
# E6 needs the CFPB data (`make cfpb`) and its cached labels (`make fetch-labels`).
reproduce: simulate e01 e02 e06

# The demo is built in Phase 5.
demo:
	@echo "demo/ is built in Phase 5; nothing to serve yet."
