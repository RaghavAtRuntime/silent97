# silent97

Estimate customer satisfaction (NPS) for a whole customer base from a small survey plus
model predictions on everyone else, with confidence intervals that are valid and
verified by simulation.

**Status:** Phase 3 (labelers and the CFPB benchmark). See `docs/PLAN.md` for the
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

## Reproducing the results

```bash
make cfpb                  # public CFPB complaints (CC0 mirror, SHA-256 pinned, cross-checked)
make amazon                # Amazon Reviews 2023, Gift Cards (research use, SHA-256 pinned)
make fetch-labels          # cached LLM scores and embeddings from the labels-v1 release
make reproduce             # simulator, E1, E2, both E6 benchmarks and the planning table
```

The simulator data is synthetic. The real-text benchmarks use public CFPB complaints
and the research-use Amazon Reviews 2023 dataset; the LLM labeler runs locally and nothing is sent to an external API.
