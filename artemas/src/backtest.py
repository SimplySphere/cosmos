from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.download import download_fx_rates, download_prices
from src.data.preprocess import preprocess_market
from src.data.universe import prepare_universe
from src.evaluation.benchmarks import run_benchmarks
from src.evaluation.evaluate import evaluate_project
from src.factors.pca import fit_pca
from src.factors.selection import select_factor_rank
from src.model.train import train_model
from src.simulation.engine import simulate_ordinary
from src.utils.config import load_project, validate_config
from src.utils.files import ensure_directories, write_json


def _origin_paths(paths: dict, origin: pd.Timestamp) -> dict:
    """Create isolated derived artifacts for one historical forecast origin."""
    updated = deepcopy(paths)
    root = Path(paths["project_root"])
    base = Path(paths["artifacts"].get("backtests_dir", root / "artifacts" / "backtests")) / origin.strftime("%Y%m%d")
    data = base / "data"
    factor = base / "factors"
    model = base / "models"
    simulations = base / "simulations"
    benchmarks = base / "benchmarks"
    evaluation = base / "evaluation"

    # Raw price/FX caches may be shared. Everything whose contents depend on the
    # point-in-time universe or cutoff is isolated by origin.
    updated["data"].update(
        {
            "prepared_universe": data / "prepared_universe.parquet",
            # Raw stock/FX files and their acquisition manifests are intentionally
            # shared across origins.  They contain observations, not model state;
            # preprocessing enforces each origin's temporal cutoff.  Sharing them
            # avoids redownloading the same Yahoo history for every backtest origin.
            "download_failures": data / "download_failures.csv",
            "weekly_returns": data / "weekly_returns.parquet",
            "availability_mask": data / "availability_mask.parquet",
            "preprocessing_report": data / "preprocessing_report.csv",
            "stock_metadata": data / "stock_metadata.parquet",
            "weekly_observed_universe": data / "weekly_observed_universe.parquet",
            "market_aggregation_weights": data / "market_aggregation_weights.parquet",
        }
    )
    updated["artifacts"].update(
        {
            "selected_rank": factor / "selected_rank.json",
            "factor_cv_results": factor / "factor_cv_results.csv",
            "factor_cv_folds": factor / "factor_cv_folds.csv",
            "development_pca": factor / "development_pca.npz",
            "development_factors": factor / "development_factor_scores.parquet",
            "development_factor_quality": factor / "development_factor_quality.parquet",
            "final_pca": factor / "final_pca.npz",
            "final_factors": factor / "final_factor_scores.parquet",
            "final_factor_quality": factor / "final_factor_quality.parquet",
            "development_checkpoint": model / "development_checkpoint.pt",
            "final_checkpoint": model / "final_checkpoint.pt",
            "training_history": model / "training_history.csv",
            "simulations_dir": simulations,
            "simulation_sanity": simulations / "sanity_summary.json",
            "benchmarks_dir": benchmarks,
            "evaluation_dir": evaluation,
            "rolling_evaluation": evaluation / "rolling_one_step_evaluation.csv",
            "rolling_summary": evaluation / "rolling_summary.json",
            "comparison_uncertainty": evaluation / "paired_metric_differences.csv",
            "leakage_audit": evaluation / "leakage_audit.json",
            "leakage_audit_table": evaluation / "leakage_audit.csv",
        }
    )
    return updated


def _snapshot_entry(config: dict, paths: dict, origin: pd.Timestamp) -> tuple[Path, Path]:
    snapshots = config["universe"].get("historical_snapshots", {})
    key = str(origin.date())
    entry = snapshots.get(key)
    if not isinstance(entry, dict):
        raise FileNotFoundError(
            f"No point-in-time universe is configured for backtest origin {key}. "
            "Add universe.historical_snapshots.<origin>.universe_source and ticker_map. "
            "The backtest refuses to reuse a later universe because that would reintroduce survivorship/membership leakage."
        )
    root = Path(paths["project_root"])
    universe_source = Path(str(entry.get("universe_source", "")))
    ticker_map = Path(str(entry.get("ticker_map", "")))
    if not universe_source.is_absolute():
        universe_source = root / universe_source
    if not ticker_map.is_absolute():
        ticker_map = root / ticker_map
    if not universe_source.exists() or not ticker_map.exists():
        raise FileNotFoundError(
            f"Point-in-time files for {key} are missing: {universe_source}, {ticker_map}"
        )
    return universe_source.resolve(), ticker_map.resolve()


def _configure_origin(config: dict, origin: pd.Timestamp) -> dict:
    local = deepcopy(config)
    validation_end = origin - pd.Timedelta(days=1)
    train_end = origin - pd.DateOffset(years=1) - pd.Timedelta(days=1)
    # Reserve roughly the final quarter of the validation year for scale calibration.
    model_selection_end = validation_end - pd.DateOffset(months=3)
    if model_selection_end <= train_end:
        model_selection_end = train_end + (validation_end - train_end) * 0.75

    local["data"]["test_start"] = str(origin.date())
    local["data"]["validation_end"] = str(validation_end.date())
    local["data"]["train_end"] = str(train_end.date())
    local["data"]["model_selection_end"] = str(pd.Timestamp(model_selection_end).date())
    local["benchmarks"]["horizon_weeks"] = int(local["evaluation"]["backtest_horizon_weeks"])
    horizon = int(local["benchmarks"]["horizon_weeks"])
    local["simulation"]["horizons_weeks"] = [horizon]
    local["simulation"]["paths_by_horizon"] = {str(horizon): int(local["benchmarks"]["paths"])}
    local["universe"]["membership_mode"] = "point_in_time"
    local["universe"]["membership_cutoff"] = "train_end"
    local["universe"]["auto_build"] = False
    validate_config(local)
    return local


def run_backtests(
    config: dict,
    paths: dict,
    force: bool = False,
    skip_download: bool = False,
) -> None:
    origins = [pd.Timestamp(value) for value in config["evaluation"].get("backtest_origins", [])]
    if not origins:
        raise ValueError("evaluation.backtest_origins is empty")

    rows: list[dict] = []
    status_rows: list[dict] = []
    for origin in origins:
        local = _configure_origin(config, origin)
        local_paths = _origin_paths(paths, origin)
        try:
            universe_source, ticker_map = _snapshot_entry(config, paths, origin)
            local_paths["data"]["universe_source"] = universe_source
            local_paths["data"]["ticker_map"] = ticker_map
            ensure_directories(local_paths)
            prepare_universe(local, local_paths, force=force)
            if not skip_download:
                # Match the top-level --force contract: rebuild origin-dependent
                # derived outputs, but never discard resumable Yahoo acquisition.
                download_prices(local, local_paths, force=False)
                if bool(local["data"].get("require_common_currency", True)):
                    download_fx_rates(local, local_paths, force=False)
            preprocess_market(local, local_paths, force=force)

            select_factor_rank(local, local_paths, force=force)
            fit_pca(local, local_paths, "development", force=force)
            train_model(local, local_paths, "development", force=force)
            fit_pca(local, local_paths, "final", force=force)
            train_model(local, local_paths, "final", force=force)
            horizon = int(local["benchmarks"]["horizon_weeks"])
            simulate_ordinary(local, local_paths, horizon, int(local["benchmarks"]["paths"]), force=force)
            run_benchmarks(local, local_paths, force=force)
            evaluate_project(local, local_paths, force=force)

            frozen = json.loads((Path(local_paths["artifacts"]["evaluation_dir"]) / "summary.json").read_text())
            for method, metrics in frozen.items():
                rows.append({"origin": str(origin.date()), "method": method, **metrics})
            status_rows.append({"origin": str(origin.date()), "status": "completed", "reason": ""})
        except FileNotFoundError as exc:
            status_rows.append({"origin": str(origin.date()), "status": "missing_point_in_time_universe", "reason": str(exc)})
            continue

    output_dir = Path(paths["artifacts"].get("backtests_dir", Path(paths["project_root"]) / "artifacts" / "backtests"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(output_dir / "backtest_status.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "multi_origin_results.csv", index=False)
    summary = frame.groupby("method").mean(numeric_only=True).to_dict("index") if not frame.empty else {}
    write_json(output_dir / "multi_origin_summary.json", summary)
    if not frame.empty:
        write_json(
            output_dir / "multi_origin_metadata.json",
            {
                "completed_origins": sorted(frame["origin"].unique().tolist()),
                "point_in_time_membership_required": True,
                "note": "Each completed origin rebuilt universe eligibility, PCA, model, simulation, benchmarks, and evaluation using that origin's configured point-in-time files.",
            },
        )
    missing = [row for row in status_rows if row["status"] != "completed"]
    if missing and bool(config["evaluation"].get("require_point_in_time_backtests", True)):
        raise RuntimeError(
            "One or more historical origins lack required point-in-time universe files. "
            f"See {output_dir / 'backtest_status.csv'}; no retrospective substitute was used."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run point-in-time historical forecast-origin backtests.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    run_backtests(config, paths, force=args.force, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
