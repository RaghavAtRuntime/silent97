.PHONY: setup test test-slow lint format coverage reproduce demo simulate

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

# Phase 1: the only artifact is the simulated population. Experiments are added
# to this target as they land (Phase 2 onward).
reproduce: simulate

# The demo is built in Phase 5.
demo:
	@echo "demo/ is built in Phase 5; nothing to serve yet."
