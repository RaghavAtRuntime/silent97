.PHONY: setup test test-slow lint format coverage reproduce demo simulate e01

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

# Every experiment result and figure. Experiments are added as their phases land.
reproduce: simulate e01

# The demo is built in Phase 5.
demo:
	@echo "demo/ is built in Phase 5; nothing to serve yet."
