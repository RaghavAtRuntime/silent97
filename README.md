# silent97

Estimate customer satisfaction (NPS) for a whole customer base from a small survey plus
model predictions on everyone else, with confidence intervals that are valid and
verified by simulation.

**Status:** Phase 1 (foundations and simulator). See `docs/PLAN.md` for the
specification and `docs/progress/` for phase notes.

## Quickstart

```bash
make setup                 # uv sync --all-extras && pre-commit install
make test                  # fast tests
make lint                  # ruff + mypy
uv run python -m silent97.data.simulate --config configs/simulate.yaml
```

The last command writes a synthetic population of 200,000 customers to
`data/processed/sim/` as Parquet, with a JSON manifest of every parameter.
All data in this repository is synthetic.
