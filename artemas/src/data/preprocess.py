from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import download_fx_rates, normalize_currency
from src.utils.config import load_project
from src.utils.files import ensure_parent, read_frame, write_frame, write_json
from src.utils.runtime import seed_everything

LOGGER = logging.getLogger(__name__)


def _weekly_prices(price_frame: pd.DataFrame, rule: str) -> pd.Series:
    """Return one observed price per calendar week without collapsing missing weeks."""
    price_column = "adj_close" if "adj_close" in price_frame.columns else "close"
    if price_column not in price_frame.columns:
        raise ValueError("Price file has neither adjusted close nor close")
    prices = pd.to_numeric(price_frame[price_column], errors="coerce")
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    # Crucially, do not drop empty resampled weeks. A return is valid only when both
    # adjacent weekly endpoints are observed.
    return prices.resample(rule).last()


def _adjacent_log_return(weekly_prices: pd.Series) -> pd.Series:
    current = pd.to_numeric(weekly_prices, errors="coerce")
    previous = current.shift(1)
    valid = current.notna() & previous.notna() & (current > 0) & (previous > 0)
    result = pd.Series(np.nan, index=current.index, dtype=np.float64)
    result.loc[valid] = np.log(current.loc[valid] / previous.loc[valid])
    return result.replace([np.inf, -np.inf], np.nan)


def _load_fx_weekly(
    currency: str,
    config: dict,
    paths: dict,
    cache: dict[str, pd.Series],
) -> pd.Series | None:
    if currency in cache:
        return cache[currency]
    canonical, unit_scale = normalize_currency(currency)
    rule = str(config["data"]["week_rule"])
    if canonical == "USD":
        cache[currency] = pd.Series(dtype=np.float64)
        return cache[currency]

    manifest_path = Path(paths["data"]["fx_manifest"])
    if not manifest_path.exists():
        return None
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    match = manifest[
        (manifest["source_currency"].astype(str) == str(currency))
        & (manifest["status"].astype(str) == "ok")
    ]
    if match.empty:
        # Currency case sometimes changes between metadata sources.
        match = manifest[
            (manifest["source_currency"].astype(str).str.upper() == str(currency).upper())
            & (manifest["status"].astype(str) == "ok")
        ]
    if match.empty:
        return None
    source = Path(match.iloc[-1]["file"])
    if not source.exists():
        return None
    frame = read_frame(source)
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None)
    series = pd.to_numeric(frame["usd_per_currency_unit"], errors="coerce").sort_index()
    weekly = series.resample(rule).last()
    limit = int(config["data"].get("fx_max_forward_fill_weeks", 1))
    if limit > 0:
        weekly = weekly.ffill(limit=limit)
    cache[currency] = weekly
    return weekly


def _usd_weekly_return(
    frame: pd.DataFrame,
    currency: str,
    config: dict,
    paths: dict,
    fx_cache: dict[str, pd.Series],
) -> pd.Series:
    weekly_local = _weekly_prices(frame, str(config["data"]["week_rule"]))
    canonical, unit_scale = normalize_currency(currency)
    if canonical == "USD":
        weekly_usd = weekly_local * float(unit_scale)
    else:
        weekly_fx = _load_fx_weekly(currency, config, paths, fx_cache)
        if weekly_fx is None or weekly_fx.empty:
            raise ValueError(f"No USD FX series is available for listing currency {currency!r}")
        aligned_fx = weekly_fx.reindex(weekly_local.index)
        weekly_usd = weekly_local * aligned_fx
    return _adjacent_log_return(weekly_usd)


def preprocess_market(config: dict, paths: dict, force: bool = False) -> None:
    output = Path(paths["data"]["weekly_returns"])
    # Remove legacy selector-weight artifacts so they cannot be mistaken for the
    # active Yahoo Finance equal-weight market definition.
    for legacy_name in ("market_weights.parquet", "weekly_observed_weight.parquet"):
        legacy = output.parent / legacy_name
        if legacy.exists():
            legacy.unlink()
    required_outputs = [
        output,
        Path(paths["data"]["availability_mask"]),
        Path(paths["data"]["weekly_observed_universe"]),
        Path(paths["data"]["stock_metadata"]),
        Path(paths["data"]["market_aggregation_weights"]),
    ]
    if all(item.exists() for item in required_outputs) and not force:
        LOGGER.info("Using existing processed panel: %s", output)
        return

    universe_path = Path(paths["data"]["prepared_universe"])
    manifest_path = Path(paths["data"]["download_manifest"])
    if not universe_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Prepare the universe and download prices before preprocessing")

    universe = read_frame(universe_path)
    manifest = read_frame(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()
    if bool(config["data"].get("require_common_currency", True)):
        fx_manifest = Path(paths["data"]["fx_manifest"])
        if not fx_manifest.exists():
            LOGGER.info("FX manifest is missing; downloading required USD conversion series")
            download_fx_rates(config, paths, force=False)

    minimum_weeks = int(config["data"]["minimum_weeks"])
    minimum_training_weeks = int(config["data"]["minimum_training_weeks"])
    training_end = pd.Timestamp(config["data"]["train_end"])
    # Use a fixed pre-validation panel. Security inclusion is determined only from
    # information available by train_end, so 2025 validation availability cannot decide
    # which securities enter development or final fitting.
    eligibility_end = pd.Timestamp(config["data"]["train_end"])
    minimum_coverage = float(config["data"]["minimum_lifetime_coverage"])

    universe_index = universe.set_index("security_id", drop=False)
    series: dict[str, pd.Series] = {}
    quality: list[dict] = []
    exclusions: list[dict] = []
    fx_cache: dict[str, pd.Series] = {}
    for row in manifest.itertuples(index=False):
        security_id = str(row.security_id)
        if security_id not in universe_index.index:
            exclusions.append({"security_id": security_id, "reason": "not_in_prepared_universe"})
            continue
        meta = universe_index.loc[security_id]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        currency = str(meta.get("currency", "") or "").strip()
        if bool(config["data"].get("require_common_currency", True)) and not currency:
            exclusions.append({"security_id": security_id, "reason": "missing_currency"})
            continue
        try:
            frame = read_frame(row.file)
            frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None)
            frame = frame[~frame.index.isna()].sort_index()
            frame = frame[~frame.index.duplicated(keep="last")]
            weekly = _usd_weekly_return(frame, currency or "USD", config, paths, fx_cache)
            observed = weekly.dropna()
            # Security eligibility is frozen at train_end. Later 2025/2026 observations
            # may be used as validation/evaluation targets, but cannot decide membership.
            eligible_weekly = weekly.loc[:eligibility_end]
            eligible_observed = eligible_weekly.dropna()
            if len(eligible_observed) < minimum_weeks:
                exclusions.append({"security_id": security_id, "reason": "insufficient_pre_cutoff_weeks"})
                continue
            training_observed = eligible_observed.loc[:training_end]
            if int(training_observed.size) < minimum_training_weeks:
                exclusions.append({"security_id": security_id, "reason": "insufficient_training_weeks"})
                continue
            eligible_lifetime = eligible_weekly.loc[
                eligible_observed.index.min() : eligible_observed.index.max()
            ]
            coverage = float(eligible_lifetime.notna().mean())
            if coverage < minimum_coverage:
                exclusions.append({"security_id": security_id, "reason": "low_pre_cutoff_coverage"})
                continue
            series[security_id] = weekly
            quality.append(
                {
                    "security_id": security_id,
                    "valid_weeks": int(observed.size),
                    "eligible_weeks_through_cutoff": int(eligible_observed.size),
                    "training_weeks": int(training_observed.size),
                    "lifetime_coverage": coverage,
                    "eligibility_end": eligibility_end,
                    "first_week": observed.index.min(),
                    "last_week": observed.index.max(),
                    "currency": currency,
                    "canonical_currency": normalize_currency(currency)[0],
                }
            )
        except Exception as exc:
            exclusions.append(
                {"security_id": security_id, "reason": "processing_error", "detail": repr(exc)}
            )
            LOGGER.debug("Skipping %s during preprocessing: %s", security_id, exc)

    if not series:
        raise RuntimeError("No securities passed preprocessing")

    returns = pd.DataFrame(series).sort_index()
    mask = returns.notna()
    quality_frame = pd.DataFrame(quality)
    metadata = universe.merge(quality_frame, on="security_id", how="inner", suffixes=("", "_quality"))
    ordered = [column for column in returns.columns if column in set(metadata["security_id"])]
    returns = returns[ordered]
    mask = mask[ordered]
    metadata = metadata.set_index("security_id").loc[ordered].reset_index()

    # The selector only determines membership. Every retained Yahoo equity receives
    # the same aggregation weight for the observed market basket and simulations.
    weights = metadata[["security_id", "country", "sector", "currency"]].copy()
    if weights.empty:
        raise RuntimeError("No preprocessed Yahoo equities remain for market aggregation")
    retained_count = len(weights) / max(len(universe), 1)
    minimum_count = float(config["data"].get("minimum_processed_count_fraction", 0.0))
    if retained_count < minimum_count:
        raise RuntimeError(
            f"Preprocessing retained {retained_count:.1%} of prepared Yahoo equities; "
            f"configured minimum is {minimum_count:.1%}. Inspect "
            f"{paths['data']['preprocessing_report']} and the FX manifest before continuing."
        )
    weights["aggregation_weight"] = 1.0 / len(weights)
    weights["aggregation_method"] = str(config["market"]["aggregation_method"])
    weights["price_provider"] = str(config["market"]["price_provider"])
    valid_ids = set(weights["security_id"])
    ordered = [security_id for security_id in ordered if security_id in valid_ids]
    returns = returns[ordered]
    mask = mask[ordered]
    metadata = metadata.set_index("security_id").loc[ordered].reset_index()
    weights = weights.set_index("security_id").loc[ordered].reset_index()

    weight_vector = weights.set_index("security_id").loc[ordered, "aggregation_weight"].to_numpy(dtype=float)
    observed_universe = mask.to_numpy(dtype=float) @ weight_vector
    observed_count = mask.mean(axis=1).to_numpy(dtype=float)
    weekly_quality = pd.DataFrame(
        {
            "observed_universe_fraction": observed_universe,
            "observed_count_fraction": observed_count,
        },
        index=returns.index,
    )

    write_frame(returns, ensure_parent(paths["data"]["weekly_returns"]), index=True)
    write_frame(mask.astype("uint8"), ensure_parent(paths["data"]["availability_mask"]), index=True)
    write_frame(
        weekly_quality,
        ensure_parent(paths["data"]["weekly_observed_universe"]),
        index=True,
    )
    write_frame(metadata, ensure_parent(paths["data"]["stock_metadata"]), index=False)
    write_frame(weights, ensure_parent(paths["data"]["market_aggregation_weights"]), index=False)
    report = pd.DataFrame(exclusions)
    report.to_csv(ensure_parent(paths["data"]["preprocessing_report"]), index=False)
    summary_path = Path(paths["data"]["preprocessing_report"]).with_suffix(".json")
    write_json(
        summary_path,
        {
            "input_manifest_rows": int(len(manifest)),
            "processed_stocks": int(returns.shape[1]),
            "weeks": int(returns.shape[0]),
            "start_week": str(returns.index.min().date()),
            "end_week": str(returns.index.max().date()),
            "base_currency": str(config["data"].get("base_currency", "USD")),
            "eligibility_end": str(eligibility_end.date()),
            "eligibility_uses_post_train_end_data": False,
            "eligibility_uses_post_cutoff_data": False,
            "prepared_equities": int(len(universe)),
            "processed_count_fraction": float(retained_count),
            "market_aggregation_method": str(config["market"]["aggregation_method"]),
            "market_price_provider": str(config["market"]["price_provider"]),
            "median_weekly_observed_universe_fraction": float(np.nanmedian(observed_universe)),
            "minimum_weekly_observed_universe_fraction": float(np.nanmin(observed_universe)),
            "exclusion_counts": report.get("reason", pd.Series(dtype=str)).value_counts().to_dict(),
        },
    )
    LOGGER.info(
        "Processed %d Yahoo equities across %d weeks in %s; median observed universe fraction %.1f%%",
        returns.shape[1],
        returns.shape[0],
        config["data"].get("base_currency", "USD"),
        100 * float(np.nanmedian(observed_universe)),
    )


def create_smoke_dataset(config: dict, paths: dict, force: bool = True) -> None:
    output = Path(paths["data"]["weekly_returns"])
    if output.exists() and not force:
        return

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    smoke = config["smoke"]
    n_stocks = int(smoke["stocks"])
    n_weeks = int(smoke["weeks"])
    n_factors = int(smoke["latent_factors"])

    dates = pd.date_range(end="2026-07-17", periods=n_weeks, freq="W-FRI")
    factors = np.zeros((n_weeks, n_factors), dtype=np.float64)
    shocks = rng.standard_t(df=7, size=(n_weeks, n_factors)) * 0.012
    transition = np.diag(np.linspace(0.15, 0.55, n_factors))
    for index in range(1, n_weeks):
        factors[index] = transition @ factors[index - 1] + shocks[index]

    loadings = rng.normal(0.0, 0.55, size=(n_stocks, n_factors))
    residual = rng.standard_t(df=7, size=(n_weeks, n_stocks)) * rng.uniform(
        0.008, 0.022, size=n_stocks
    )
    returns = factors @ loadings.T + residual
    missing = rng.random(size=returns.shape) < 0.015
    returns[missing] = np.nan

    security_ids = [f"SMOKE_{index:04d}" for index in range(n_stocks)]
    return_frame = pd.DataFrame(returns, index=dates, columns=security_ids)
    mask = return_frame.notna().astype("uint8")
    countries = np.array(["United States", "Japan", "United Kingdom", "India"])
    sectors = np.array(["Technology", "Financials", "Industrials", "Health Care"])
    market_caps = np.exp(rng.normal(22.0, 1.1, size=n_stocks))
    weights = np.full(n_stocks, 1.0 / n_stocks, dtype=np.float64)
    metadata = pd.DataFrame(
        {
            "security_id": security_ids,
            "company_name": [f"Synthetic Company {i}" for i in range(n_stocks)],
            "yahoo_ticker": security_ids,
            "country": countries[np.arange(n_stocks) % len(countries)],
            "currency": "USD",
            "sector": sectors[np.arange(n_stocks) % len(sectors)],
            "industry": "Synthetic",
            "market_cap": market_caps,
            "selection_source_weight": np.nan,
            "snapshot_date": pd.Timestamp("2025-12-31"),
            "price_provider": "yfinance",
            "model_aggregation_method": "equal_weight",
        }
    )
    weight_frame = metadata[["security_id", "country", "sector", "currency"]].copy()
    weight_frame["aggregation_weight"] = weights
    weight_frame["aggregation_method"] = "equal_weight"
    weight_frame["price_provider"] = "yfinance"
    weekly_quality = pd.DataFrame(
        {
            "observed_universe_fraction": mask.to_numpy(dtype=float) @ weights,
            "observed_count_fraction": mask.mean(axis=1),
        },
        index=dates,
    )

    write_frame(return_frame, ensure_parent(paths["data"]["weekly_returns"]), index=True)
    write_frame(mask, ensure_parent(paths["data"]["availability_mask"]), index=True)
    write_frame(weekly_quality, ensure_parent(paths["data"]["weekly_observed_universe"]), index=True)
    write_frame(metadata, ensure_parent(paths["data"]["stock_metadata"]), index=False)
    write_frame(weight_frame, ensure_parent(paths["data"]["market_aggregation_weights"]), index=False)
    pd.DataFrame(columns=["security_id", "reason", "detail"]).to_csv(
        ensure_parent(paths["data"]["preprocessing_report"]), index=False
    )
    LOGGER.info("Created synthetic smoke panel: %s", return_frame.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cached prices into adjacent weekly USD returns.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    if args.smoke:
        create_smoke_dataset(config, paths, force=args.force)
    else:
        preprocess_market(config, paths, force=args.force)


if __name__ == "__main__":
    main()
