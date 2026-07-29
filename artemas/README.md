# ARTEMAS

The project asks: **What would happen if I used an LLM transformer to model the stock market?**

Approaching stock market prediction from an LLM point of view, we use a next-week-prediction transformer trained on Vanguard's 10,000+ stocks with Monte Carlo to simulate 0.5-, 1-, 5-, and 10-year horizons of stock prices and movement. 


## Market definition

The modeled market is an equal-weight USD basket constructed from eligible equities with Yahoo Finance price histories.

- The holdings selector determines membership.
- Yahoo Finance supplies the price histories used for returns, PCA, model fitting, evaluation, and visuals.
- Yahoo metadata is preferred for listing metadata.
- When Yahoo metadata is sparse, selector country, exchange, currency, sector, or industry may be used as a disclosed descriptive fallback.
- Source-fund weights, market values, share quantities, and source ordering do not determine the modeled market.
- Every retained stock receives equal aggregation weight.

The result is a constructed global-equity basket, not Vanguard or any specific indexes.


## Model

```yaml
model:
  context_length: 104
  model_dimension: 128
  layers: 3
  heads: 4
  feedforward_dimension: 384
  mixture_components: 4
  covariance_type: full
```

With 13 factors, the production architecture has approximately 570 thousand parameters and preserves a 104-week context, a multivariate Student-t mixture output, full covariance, and recursive generation.


## Monte Carlo horizons

```text
1, 4, 13, 26, 52, 260, and 520 weeks
```

All paths come from recursive sampling. Long-horizon outputs include sanity and drift diagnostics and must be interpreted as model-implied explorations.


## Validation structure

```text
through 2024-12-31    development training and security eligibility
2025-01-01–09-30      checkpoint selection
2025-10-01–12-31      dispersion calibration
from 2026-01-01       frozen outcomes for evaluation
```

Frozen evaluation is joined by calendar date, so excluding a low-coverage realized week does not shift later forecast steps.


## Run

Smoke validation:

```bash
python -m src.main --mode smoke --force
```

Full run:

```bash
python -m src.main --mode real --force
```
