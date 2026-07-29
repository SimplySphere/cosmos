from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing YAML file: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {source}")
    return data


def _resolve_tree(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_tree(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tree(item, root) for item in value]
    if isinstance(value, str):
        return (root / value).resolve()
    return value


def nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing configuration key: {dotted_key}")
        current = current[key]
    return current


def _require_positive(mapping: dict[str, Any], key: str, *, allow_zero: bool = False) -> None:
    value = nested_get(mapping, key)
    threshold = 0 if allow_zero else 1
    if not isinstance(value, (int, float)) or value < threshold:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Configuration key {key} must be {comparator}; got {value!r}")


def _require_fraction(mapping: dict[str, Any], key: str, *, positive: bool = False) -> None:
    value = float(nested_get(mapping, key))
    lower_ok = value > 0 if positive else value >= 0
    if not lower_ok or value > 1:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{key} must be in {interval}; got {value}")


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "data",
        "universe",
        "market",
        "factors",
        "model",
        "training",
        "simulation",
        "scenarios",
        "benchmarks",
        "evaluation",
        "visualization",
        "smoke",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    import pandas as pd

    train_end = str(nested_get(config, "data.train_end"))
    model_selection_end = str(nested_get(config, "data.model_selection_end"))
    validation_end = str(nested_get(config, "data.validation_end"))
    test_start = str(nested_get(config, "data.test_start"))
    if not (
        pd.Timestamp(train_end)
        < pd.Timestamp(model_selection_end)
        < pd.Timestamp(validation_end)
        < pd.Timestamp(test_start)
    ):
        raise ValueError(
            "Dates must satisfy train_end < model_selection_end < validation_end < test_start"
        )

    positive_keys = (
        "data.minimum_weeks",
        "data.minimum_training_weeks",
        "data.download_batch_size",
        "data.download_retries",
        "data.download_consecutive_failure_limit",
        "data.download_checkpoint_batches",
        "data.fx_download_batch_size",
        "data.fx_download_retries",
        "data.evaluation_complete_lag_days",
        "factors.minimum_rank",
        "factors.maximum_rank",
        "factors.stock_folds",
        "factors.time_folds",
        "factors.minimum_time_fold_train_weeks",
        "factors.minimum_fold_train_observations",
        "factors.minimum_fold_stocks",
        "factors.randomized_svd_iterations",
        "factors.imputation_iterations",
        "factors.forecast_lags",
        "factors.forecast_candidate_stride",
        "model.context_length",
        "model.model_dimension",
        "model.layers",
        "model.heads",
        "model.feedforward_dimension",
        "model.mixture_components",
        "training.batch_size",
        "training.maximum_epochs",
        "training.early_stopping_patience",
        "simulation.inference_batch_size",
        "benchmarks.horizon_weeks",
        "benchmarks.paths",
        "benchmarks.linear_lags",
        "evaluation.rolling_samples",
        "evaluation.paired_bootstrap_repetitions",
        "evaluation.paired_bootstrap_block_weeks",
        "visualization.default_horizon_weeks",
        "visualization.reference_notional_value",
        "visualization.animation_paths",
        "visualization.generation_animation_weeks",
    )
    for key in positive_keys:
        _require_positive(config, key)

    if int(nested_get(config, "factors.stock_folds")) < 2:
        raise ValueError("factors.stock_folds must be at least 2")
    if int(nested_get(config, "factors.time_folds")) < 2:
        raise ValueError("factors.time_folds must be at least 2")
    if int(nested_get(config, "factors.minimum_rank")) > int(
        nested_get(config, "factors.maximum_rank")
    ):
        raise ValueError("factors.minimum_rank cannot exceed factors.maximum_rank")

    model_dimension = int(nested_get(config, "model.model_dimension"))
    heads = int(nested_get(config, "model.heads"))
    if model_dimension % heads != 0:
        raise ValueError("model.model_dimension must be divisible by model.heads")
    if str(nested_get(config, "model.covariance_type")) not in {"full", "diagonal"}:
        raise ValueError("model.covariance_type must be full or diagonal")
    minimum_scale = float(nested_get(config, "model.minimum_scale"))
    maximum_scale = float(nested_get(config, "model.maximum_scale"))
    if not 0 < minimum_scale < maximum_scale:
        raise ValueError("model scales must satisfy 0 < minimum_scale < maximum_scale")
    minimum_df = float(nested_get(config, "model.minimum_degrees_of_freedom"))
    maximum_df = float(nested_get(config, "model.maximum_degrees_of_freedom"))
    if not 2 < minimum_df < maximum_df:
        raise ValueError("model degrees of freedom must satisfy 2 < minimum < maximum")
    if float(nested_get(config, "model.maximum_off_diagonal")) < 0:
        raise ValueError("model.maximum_off_diagonal cannot be negative")

    dropout = float(nested_get(config, "model.dropout"))
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    warmup = float(nested_get(config, "training.warmup_fraction"))
    if not 0 <= warmup < 1:
        raise ValueError("training.warmup_fraction must be in [0, 1)")
    grid = nested_get(config, "training.calibration_scale_grid")
    if not isinstance(grid, list) or not grid or any(float(value) <= 0 for value in grid):
        raise ValueError("training.calibration_scale_grid must be a non-empty positive list")

    for key in (
        "data.minimum_lifetime_coverage",
        "data.minimum_factor_week_observed_fraction",
        "data.minimum_evaluation_observed_fraction",
        "data.minimum_currency_count_fraction",
        "data.minimum_processed_count_fraction",
        "factors.minimum_fold_train_fraction",
        "universe.minimum_verified_count_fraction",
        "scenarios.outlier_quantile",
        "scenarios.winsor_lower_quantile",
        "scenarios.winsor_upper_quantile",
        "benchmarks.linear_max_spectral_radius",
    ):
        _require_fraction(config, key, positive=key == "data.minimum_lifetime_coverage")
    if float(nested_get(config, "scenarios.winsor_lower_quantile")) >= float(
        nested_get(config, "scenarios.winsor_upper_quantile")
    ):
        raise ValueError("scenario winsor lower quantile must be below upper quantile")


    aggregation_method = str(nested_get(config, "market.aggregation_method"))
    if aggregation_method != "equal_weight":
        raise ValueError(
            "market.aggregation_method must be 'equal_weight'. Source-fund weights are selector metadata only and may not enter modeling or evaluation."
        )
    if str(nested_get(config, "market.price_provider")) != "yfinance":
        raise ValueError("market.price_provider must be 'yfinance'")

    provider = str(nested_get(config, "universe.provider"))
    if provider != "vanguard_vt":
        raise ValueError("universe.provider must be 'vanguard_vt'")
    membership_mode = str(nested_get(config, "universe.membership_mode"))
    allowed_membership_modes = {"point_in_time", "retrospective_disclosed"}
    if membership_mode not in allowed_membership_modes:
        raise ValueError(
            "universe.membership_mode must be one of "
            f"{sorted(allowed_membership_modes)}; got {membership_mode!r}"
        )
    membership_cutoff = str(nested_get(config, "universe.membership_cutoff"))
    if membership_cutoff not in {"train_end", "validation_end"}:
        raise ValueError("universe.membership_cutoff must be train_end or validation_end")
    historical_snapshots = nested_get(config, "universe.historical_snapshots")
    if not isinstance(historical_snapshots, dict):
        raise ValueError("universe.historical_snapshots must be a mapping")
    profile_urls = nested_get(config, "universe.vanguard_profile_urls")
    if not isinstance(profile_urls, list) or not profile_urls:
        raise ValueError("universe.vanguard_profile_urls must be a non-empty list")
    resolver_seed_path = config.get("universe", {}).get("resolver_seed_path", "")
    if resolver_seed_path is not None and not isinstance(resolver_seed_path, str):
        raise ValueError("universe.resolver_seed_path must be a string path")
    selector_seed_path = config.get("universe", {}).get("selector_seed_path", "")
    if selector_seed_path is not None and not isinstance(selector_seed_path, str):
        raise ValueError("universe.selector_seed_path must be a string path")
    if not isinstance(config.get("universe", {}).get("prefer_selector_seed", True), bool):
        raise ValueError("universe.prefer_selector_seed must be true or false")
    if not isinstance(config.get("universe", {}).get("yahoo_search_enabled", True), bool):
        raise ValueError("universe.yahoo_search_enabled must be true or false")
    for key in (
        "universe.http_timeout_seconds",
        "universe.browser_timeout_seconds",
        "universe.yahoo_verification_lookback_days",
        "universe.yahoo_search_results",
        "universe.yahoo_batch_size",
        "universe.yahoo_batch_retries",
        "universe.yahoo_request_timeout_seconds",
        "universe.yahoo_search_chunk_size",
        "universe.yahoo_search_empty_healthcheck",
        "universe.yahoo_search_health_retries",
    ):
        _require_positive(config, key)
    for key in (
        "universe.yahoo_retry_backoff_seconds",
        "universe.yahoo_search_pause_seconds",
        "data.download_pause_seconds",
        "data.download_retry_backoff_seconds",
        "data.fx_download_pause_seconds",
        "data.fx_download_retry_backoff_seconds",
    ):
        if float(nested_get(config, key)) < 0:
            raise ValueError(f"{key} cannot be negative")
    for key in (
        "universe.yahoo_search_review_name_similarity",
        "universe.yahoo_search_verified_name_similarity",
        "universe.yahoo_cached_mapping_min_name_similarity",
    ):
        _require_fraction(config, key)
    canaries = nested_get(config, "universe.yahoo_canary_symbols")
    if not isinstance(canaries, list) or not canaries:
        raise ValueError("universe.yahoo_canary_symbols must be a non-empty list")

    if float(nested_get(config, "factors.residual_degrees_of_freedom")) <= 2:
        raise ValueError("factors.residual_degrees_of_freedom must be greater than 2")
    if str(nested_get(config, "factors.time_fold_mode")) != "expanding":
        raise ValueError("factors.time_fold_mode must be expanding in the leakage-safe pipeline")
    if str(nested_get(config, "factors.selection_objective")) not in {
        "forecast_mse",
        "reconstruction_mse",
    }:
        raise ValueError("factors.selection_objective must be forecast_mse or reconstruction_mse")


    seeds = nested_get(config, "evaluation.robustness_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(int(seed) < 0 for seed in seeds):
        raise ValueError("evaluation.robustness_seeds must contain at least two non-negative integers")
    origins = nested_get(config, "evaluation.backtest_origins")
    if not isinstance(origins, list) or not origins:
        raise ValueError("evaluation.backtest_origins must be a non-empty date list")
    for origin in origins:
        pd.Timestamp(origin)
    _require_positive(config, "evaluation.backtest_horizon_weeks")

    horizons = nested_get(config, "simulation.horizons_weeks")
    if not isinstance(horizons, list) or not horizons or any(int(value) <= 0 for value in horizons):
        raise ValueError("simulation.horizons_weeks must be a non-empty positive list")
    if float(nested_get(config, "simulation.factor_clip_std")) <= 0:
        raise ValueError("simulation.factor_clip_std must be positive")
    interpretation = str(nested_get(config, "simulation.long_horizon_interpretation"))
    if interpretation != "exploratory_recursive_scenario":
        raise ValueError(
            "simulation.long_horizon_interpretation must be exploratory_recursive_scenario"
        )


def load_project(
    config_path: str | Path = "configs/config.yaml",
    paths_path: str | Path = "configs/paths.yaml",
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_yaml(config_path)
    validate_config(config)
    paths_file = Path(paths_path).resolve()
    raw_paths = _load_yaml(paths_file)
    root_value = Path(raw_paths.get("project_root", ".."))
    project_root = root_value if root_value.is_absolute() else (paths_file.parent / root_value)
    project_root = project_root.resolve()
    raw_paths = {key: value for key, value in raw_paths.items() if key != "project_root"}
    paths = _resolve_tree(raw_paths, project_root)
    paths["project_root"] = project_root
    return config, paths


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    smoke = updated["smoke"]
    updated["factors"]["maximum_rank"] = smoke["factor_maximum_rank"]
    updated["factors"]["stock_folds"] = smoke["stock_folds"]
    updated["factors"]["time_folds"] = smoke["time_folds"]
    updated["factors"]["minimum_time_fold_train_weeks"] = 120
    updated["factors"]["forecast_candidate_stride"] = 1
    updated["factors"]["imputation_iterations"] = 2

    for key in (
        "context_length",
        "model_dimension",
        "layers",
        "heads",
        "feedforward_dimension",
        "mixture_components",
    ):
        updated["model"][key] = smoke[key]
    updated["training"]["batch_size"] = smoke["batch_size"]
    updated["training"]["maximum_epochs"] = smoke["epochs"]
    updated["training"]["early_stopping_patience"] = smoke["epochs"]
    updated["training"]["calibration_scale_grid"] = [0.8, 1.0, 1.2]

    updated["simulation"]["horizons_weeks"] = [smoke["horizon_weeks"]]
    updated["simulation"]["paths_by_horizon"] = {
        str(smoke["horizon_weeks"]): smoke["paths"]
    }
    updated["simulation"]["scenario_clusters"] = min(4, smoke["paths"])

    updated["benchmarks"]["horizon_weeks"] = smoke["horizon_weeks"]
    updated["benchmarks"]["paths"] = smoke["paths"]
    updated["benchmarks"]["linear_lags"] = 2
    updated["evaluation"]["rolling_samples"] = smoke["paths"]
    updated["evaluation"]["paired_bootstrap_repetitions"] = 100
    updated["evaluation"]["paired_bootstrap_block_weeks"] = 2
    updated["visualization"]["default_horizon_weeks"] = smoke["horizon_weeks"]
    updated["visualization"]["animation_paths"] = min(10, smoke["paths"])
    updated["visualization"]["generation_animation_weeks"] = min(12, smoke["horizon_weeks"])
    validate_config(updated)
    return updated


def apply_smoke_paths(paths: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(paths)
    root = Path(updated["project_root"])
    smoke_data = root / "data" / "processed" / "smoke"
    smoke_artifacts = root / "artifacts" / "smoke"

    for key, filename in {
        "weekly_returns": "weekly_returns.parquet",
        "availability_mask": "availability_mask.parquet",
        "weekly_observed_universe": "weekly_observed_universe.parquet",
        "preprocessing_report": "preprocessing_report.csv",
        "stock_metadata": "stock_metadata.parquet",
        "market_aggregation_weights": "market_aggregation_weights.parquet",
    }.items():
        updated["data"][key] = smoke_data / filename

    factor_dir = smoke_artifacts / "factors"
    model_dir = smoke_artifacts / "models"
    evaluation_dir = smoke_artifacts / "evaluation"
    updated["artifacts"].update(
        {
            "selected_rank": factor_dir / "selected_rank.json",
            "factor_cv_results": factor_dir / "factor_cv_results.csv",
            "factor_cv_folds": factor_dir / "factor_cv_folds.csv",
            "development_pca": factor_dir / "development_pca.npz",
            "development_factors": factor_dir / "development_factor_scores.parquet",
            "development_factor_quality": factor_dir / "development_factor_quality.parquet",
            "final_pca": factor_dir / "final_pca.npz",
            "final_factors": factor_dir / "final_factor_scores.parquet",
            "final_factor_quality": factor_dir / "final_factor_quality.parquet",
            "development_checkpoint": model_dir / "development_checkpoint.pt",
            "final_checkpoint": model_dir / "final_checkpoint.pt",
            "training_history": model_dir / "training_history.csv",
            "simulations_dir": smoke_artifacts / "simulations",
            "simulation_sanity": smoke_artifacts / "simulations" / "sanity_summary.json",
            "benchmarks_dir": smoke_artifacts / "benchmarks",
            "evaluation_dir": evaluation_dir,
            "rolling_evaluation": evaluation_dir / "rolling_one_step_evaluation.csv",
            "rolling_summary": evaluation_dir / "rolling_summary.json",
            "comparison_uncertainty": evaluation_dir / "paired_metric_differences.csv",
            "leakage_audit": evaluation_dir / "leakage_audit.json",
            "leakage_audit_table": evaluation_dir / "leakage_audit.csv",
            "visuals_dir": smoke_artifacts / "visuals",
            "visuals_manifest": smoke_artifacts / "visuals" / "visual_manifest.csv",
            "visuals_readme": smoke_artifacts / "visuals" / "README_VISUALS.md",
            "logs_dir": smoke_artifacts / "logs",
            "run_manifest": smoke_artifacts / "run_manifest.json",
            "audit_report": smoke_artifacts / "audit_report.json",
            "pipeline_schema": smoke_artifacts / "pipeline_schema.json",
            "config_snapshot_dir": smoke_artifacts / "config_snapshot",
        }
    )
    return updated
