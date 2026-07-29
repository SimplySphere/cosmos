from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.benchmarks import _factor_history, fit_stable_var, sample_stable_var_one_step
from src.model.train import load_model_bundle
from src.simulation.engine import (
    _factor_paths_to_market,
    _model_parameters,
    _projection_statistics,
    _sample_factor_step,
)
from src.utils.config import load_project
from src.utils.files import ensure_parent, load_npz, read_frame, write_json
from src.utils.runtime import select_device

LOGGER = logging.getLogger(__name__)


def _evaluation_cutoff(config: dict) -> pd.Timestamp:
    configured = config["data"].get("download_end")
    as_of = (
        pd.Timestamp(configured)
        if configured
        else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    )
    return as_of - pd.Timedelta(days=int(config["data"].get("evaluation_complete_lag_days", 3)))


def _actual_global_returns(config: dict, paths: dict) -> pd.DataFrame:
    """Observed equal-weight Yahoo Finance basket returns in USD.

    The universe selector contributes security membership and may supply disclosed
    descriptive-metadata fallbacks. No source-fund weights enter this realized series.
    """
    returns = read_frame(paths["data"]["weekly_returns"]).sort_index()
    weights = read_frame(paths["data"]["market_aggregation_weights"])
    weight_map = weights.set_index("security_id")["aggregation_weight"]
    common = [column for column in returns.columns if column in weight_map.index]
    values = returns[common]
    if not common:
        raise ValueError("No aggregation-weight securities overlap the Yahoo Finance return panel")
    base_weights = np.array(
        weight_map.loc[common].to_numpy(dtype=np.float64), dtype=np.float64, copy=True
    )
    observed = np.array(values.notna().to_numpy(), dtype=bool, copy=True)
    numeric = np.array(
        values.fillna(0.0).to_numpy(dtype=np.float64), dtype=np.float64, copy=True
    )
    weighted = numeric * base_weights[None, :]
    denominators = observed @ base_weights
    global_returns = np.full(len(values), np.nan, dtype=np.float64)
    valid = denominators > 1e-12
    global_returns[valid] = weighted.sum(axis=1)[valid] / denominators[valid]
    frame = pd.DataFrame(
        {
            "global_return": global_returns,
            "observed_universe_fraction": denominators,
            "observed_count_fraction": observed.mean(axis=1),
            "market_series_name": str(config["market"]["name"]),
            "price_provider": str(config["market"]["price_provider"]),
            "aggregation_method": str(config["market"]["aggregation_method"]),
        },
        index=values.index,
    )
    cutoff = _evaluation_cutoff(config)
    minimum_observed = float(config["data"].get("minimum_evaluation_observed_fraction", 0.9))
    frame["complete_date"] = frame.index <= cutoff
    frame["passes_observation_threshold"] = frame["observed_universe_fraction"] >= minimum_observed
    return frame


def _crps_samples(samples: np.ndarray, observation: float) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64))
    samples = samples[np.isfinite(samples)]
    n = len(samples)
    if n == 0 or not np.isfinite(observation):
        raise ValueError("CRPS requires finite samples and a finite observation")
    first = np.mean(np.abs(samples - observation))
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    pairwise = np.sum(coefficients * samples) / (n * n)
    return float(first - pairwise)


def _max_drawdown(path: np.ndarray) -> float:
    values = np.asarray(path, dtype=np.float64)
    running = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
    return float(np.min(values / np.maximum(running, 1e-12) - 1.0))


def _valid_actual_test(config: dict, actual_frame: pd.DataFrame) -> pd.DataFrame:
    result = actual_frame.loc[config["data"]["test_start"] :].copy()
    result = result[
        result["global_return"].notna()
        & result["complete_date"]
        & result["passes_observation_threshold"]
    ]
    return result


def _forecast_dates_from_payload(
    payload: dict[str, np.ndarray],
    horizon: int,
    config: dict,
) -> pd.DatetimeIndex:
    if "forecast_dates" in payload:
        dates = pd.DatetimeIndex(pd.to_datetime(payload["forecast_dates"].astype(str)))
        if len(dates) < horizon:
            raise ValueError("Simulation forecast_dates is shorter than the saved horizon")
        return dates[:horizon]
    if "forecast_origin" not in payload:
        raise ValueError("Simulation is missing forecast_origin and forecast_dates")
    origin = pd.Timestamp(str(np.asarray(payload["forecast_origin"]).reshape(-1)[0]))
    return pd.date_range(
        start=origin + pd.Timedelta(days=1),
        periods=horizon,
        freq=str(config["data"]["week_rule"]),
    )


def evaluate_distribution(
    name: str,
    simulation_path: Path,
    actual_frame: pd.DataFrame,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate a frozen-origin path distribution by calendar date, never by row position.

    Low-coverage realized weeks are excluded from scoring but remain in the realized
    cumulative path when their return is available. This prevents a skipped week from
    shifting every later forecast step backward in time.
    """
    payload = load_npz(simulation_path)
    simulated_index = np.array(payload["group_index_paths"][:, :, 0], dtype=np.float64, copy=True)
    simulated_returns = np.array(
        payload["group_weekly_returns"][:, :, 0], dtype=np.float64, copy=True
    )
    if simulated_index.ndim != 2 or simulated_index.shape[0] == 0:
        raise ValueError(f"Simulation {simulation_path} contains no paths")
    if simulated_returns.shape != simulated_index.shape:
        raise ValueError(f"Simulation {simulation_path} has inconsistent index/return shapes")

    full_horizon = simulated_index.shape[1]
    forecast_dates = _forecast_dates_from_payload(payload, full_horizon, config)
    calendar = actual_frame.reindex(forecast_dates).copy()

    # Only a contiguous observed prefix can define a realized cumulative index from the
    # frozen origin. Future/not-yet-complete dates end the prefix. Coverage failures do
    # not end it: their observed return is retained for compounding but not scored.
    available = (
        calendar["global_return"].notna()
        & calendar["complete_date"].fillna(False).astype(bool)
    ).to_numpy(dtype=bool)
    unavailable = np.flatnonzero(~available)
    observed_prefix = int(unavailable[0]) if unavailable.size else full_horizon
    if observed_prefix == 0:
        raise ValueError("No complete test-period return is available at the frozen forecast origin")

    calendar = calendar.iloc[:observed_prefix].copy()
    actual_weekly_all = calendar["global_return"].to_numpy(dtype=np.float64)
    actual_index_all = np.exp(np.cumsum(actual_weekly_all))
    score_mask = calendar["passes_observation_threshold"].fillna(False).to_numpy(dtype=bool)
    score_positions = np.flatnonzero(score_mask)
    if score_positions.size == 0:
        raise ValueError("No complete test-period weeks pass the observation threshold")

    scored_simulated_index = simulated_index[:, score_positions]
    scored_simulated_returns = simulated_returns[:, score_positions]
    actual_weekly = actual_weekly_all[score_positions]
    actual_index = actual_index_all[score_positions]
    scored_dates = calendar.index[score_positions]

    quantile_levels = np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])
    quantiles = np.quantile(scored_simulated_index, quantile_levels, axis=0)
    median = quantiles[3]
    weekly_rows: list[dict] = []
    crps_values: list[float] = []
    for column, position in enumerate(score_positions):
        samples = simulated_index[:, position]
        observation = actual_index_all[position]
        crps = _crps_samples(samples, observation)
        crps_values.append(crps)
        weekly_rows.append(
            {
                "model": name,
                "date": scored_dates[column],
                "forecast_step": int(position + 1),
                "actual_index": observation,
                "median_index": median[column],
                "q025": quantiles[0, column],
                "q10": quantiles[1, column],
                "q25": quantiles[2, column],
                "q75": quantiles[4, column],
                "q90": quantiles[5, column],
                "q975": quantiles[6, column],
                "actual_percentile": float(np.mean(samples <= observation)),
                "crps": crps,
                "observed_universe_fraction": float(
                    calendar.iloc[position]["observed_universe_fraction"]
                ),
            }
        )

    median_weekly = np.median(scored_simulated_returns, axis=0)
    last_position = int(score_positions[-1])
    summary = {
        "model": name,
        "weeks_evaluated": int(score_positions.size),
        "forecast_steps_observed": int(observed_prefix),
        "calendar_weeks_skipped_for_coverage": int(observed_prefix - score_positions.size),
        "evaluation_start": str(scored_dates.min().date()),
        "evaluation_end": str(scored_dates.max().date()),
        "last_evaluated_forecast_step": int(last_position + 1),
        "cumulative_index_mae": float(np.mean(np.abs(median - actual_index))),
        "weekly_return_mae": float(np.mean(np.abs(median_weekly - actual_weekly))),
        "weekly_direction_accuracy": float(
            np.mean(np.sign(median_weekly) == np.sign(actual_weekly))
        ),
        "coverage_50": float(
            np.mean((actual_index >= quantiles[2]) & (actual_index <= quantiles[4]))
        ),
        "coverage_80": float(
            np.mean((actual_index >= quantiles[1]) & (actual_index <= quantiles[5]))
        ),
        "coverage_95": float(
            np.mean((actual_index >= quantiles[0]) & (actual_index <= quantiles[6]))
        ),
        "mean_width_50": float(np.mean(quantiles[4] - quantiles[2])),
        "mean_width_80": float(np.mean(quantiles[5] - quantiles[1])),
        "mean_width_95": float(np.mean(quantiles[6] - quantiles[0])),
        "mean_crps": float(np.mean(crps_values)),
        "final_actual_percentile": float(
            np.mean(simulated_index[:, last_position] <= actual_index_all[last_position])
        ),
        "actual_max_drawdown": _max_drawdown(actual_index_all[: last_position + 1]),
        "median_simulated_max_drawdown": float(
            np.median(
                [
                    _max_drawdown(path[: last_position + 1])
                    for path in simulated_index
                ]
            )
        ),
        "minimum_observed_universe_fraction": float(
            calendar.iloc[score_positions]["observed_universe_fraction"].min()
        ),
        "calendar_alignment": "forecast_date_join",
    }
    return summary, pd.DataFrame(weekly_rows)

def _project_one_step_samples(
    factor_samples: np.ndarray,
    projection: dict,
    residual_df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    weekly, _ = _factor_paths_to_market(
        factor_samples[:, None, :], projection, residual_df, rng
    )
    return weekly[:, 0, 0].astype(np.float64)


def _rolling_method_row(
    method: str,
    date: pd.Timestamp,
    actual: float,
    samples: np.ndarray,
    observed_universe: float,
) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    quantiles = np.quantile(samples, [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])
    return {
        "model": method,
        "date": date,
        "actual_return": actual,
        "median_return": float(quantiles[3]),
        "q025": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q975": float(quantiles[6]),
        "actual_percentile": float(np.mean(samples <= actual)),
        "crps": _crps_samples(samples, actual),
        "absolute_error": float(abs(quantiles[3] - actual)),
        "direction_correct": bool(np.sign(quantiles[3]) == np.sign(actual)),
        "observed_universe_fraction": observed_universe,
    }


def rolling_one_step_evaluation(
    config: dict,
    paths: dict,
    actual_frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    import torch
    from sklearn.covariance import LedoitWolf

    device = select_device(str(config["project"]["device"]))
    model, checkpoint = load_model_bundle(paths["artifacts"]["final_checkpoint"], device)
    factors = read_frame(paths["artifacts"]["final_factors"]).sort_index()
    raw_values = np.array(factors.to_numpy(dtype=np.float64), copy=True)
    dates = pd.DatetimeIndex(factors.index)
    factor_mean = np.asarray(checkpoint["factor_mean"], dtype=np.float64)
    factor_scale = np.asarray(checkpoint["factor_scale"], dtype=np.float64)
    normalized_values = (raw_values - factor_mean) / factor_scale
    context_length = int(config["model"]["context_length"])
    n_samples = int(config["evaluation"]["rolling_samples"])
    base_seed = int(config["project"]["seed"]) + 60_000

    pca = load_npz(paths["artifacts"]["final_pca"])
    projection = _projection_statistics(paths, pca)
    residual_df = float(pca["residual_df"][0])
    history = _factor_history(config, paths)
    var_model = fit_stable_var(config, history)
    gaussian_mean = history.mean(axis=0)
    gaussian_covariance = LedoitWolf().fit(history).covariance_ + np.eye(history.shape[1]) * 1e-8
    gaussian_cholesky = np.linalg.cholesky(gaussian_covariance)

    actual = _valid_actual_test(config, actual_frame)
    rows: list[dict] = []
    for target_index in range(context_length, len(dates)):
        date = dates[target_index]
        if date not in actual.index:
            continue
        context = normalized_values[target_index - context_length : target_index]
        if not np.isfinite(context).all():
            continue
        actual_return = float(actual.loc[date, "global_return"])
        observed_universe = float(actual.loc[date, "observed_universe_fraction"])
        rng = np.random.default_rng(base_seed + target_index)

        parameters = _model_parameters(model, context[None, :, :], device, 1)
        repeated_parameters = {
            key: np.repeat(value, n_samples, axis=0) for key, value in parameters.items()
        }
        transformer_normalized, _ = _sample_factor_step(repeated_parameters, rng)
        transformer_factors = transformer_normalized * factor_scale + factor_mean
        transformer_samples = _project_one_step_samples(
            transformer_factors, projection, residual_df, rng
        )
        rows.append(
            _rolling_method_row(
                "transformer", date, actual_return, transformer_samples, observed_universe
            )
        )

        rows.append(
            _rolling_method_row(
                "zero_return",
                date,
                actual_return,
                np.zeros(n_samples, dtype=np.float64),
                observed_universe,
            )
        )

        bootstrap_indices = rng.integers(0, len(history), size=n_samples)
        bootstrap_samples = _project_one_step_samples(
            history[bootstrap_indices], projection, residual_df, rng
        )
        rows.append(
            _rolling_method_row(
                "block_bootstrap", date, actual_return, bootstrap_samples, observed_universe
            )
        )

        var_factors = sample_stable_var_one_step(
            var_model,
            raw_values[:target_index][np.isfinite(raw_values[:target_index]).all(axis=1)],
            n_samples,
            rng,
        )
        var_samples = _project_one_step_samples(var_factors, projection, residual_df, rng)
        rows.append(
            _rolling_method_row(
                "linear_factor", date, actual_return, var_samples, observed_universe
            )
        )

        gaussian_factors = gaussian_mean + rng.standard_normal(
            (n_samples, history.shape[1])
        ) @ gaussian_cholesky.T
        gaussian_samples = _project_one_step_samples(
            gaussian_factors, projection, residual_df, rng
        )
        rows.append(
            _rolling_method_row(
                "gaussian_factor", date, actual_return, gaussian_samples, observed_universe
            )
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Rolling evaluation produced no valid one-step forecast weeks")
    summaries: dict[str, dict] = {}
    for method, group in frame.groupby("model"):
        coverage_50 = np.mean(
            (group["actual_return"] >= group["q25"])
            & (group["actual_return"] <= group["q75"])
        )
        coverage_80 = np.mean(
            (group["actual_return"] >= group["q10"])
            & (group["actual_return"] <= group["q90"])
        )
        coverage_95 = np.mean(
            (group["actual_return"] >= group["q025"])
            & (group["actual_return"] <= group["q975"])
        )
        calibration_error = float(
            np.mean(np.abs(np.asarray([coverage_50, coverage_80, coverage_95]) - np.asarray([0.5, 0.8, 0.95])))
        )
        summaries[method] = {
            "weeks_evaluated": int(len(group)),
            "start": str(pd.to_datetime(group["date"]).min().date()),
            "end": str(pd.to_datetime(group["date"]).max().date()),
            "mean_crps": float(group["crps"].mean()),
            "weekly_return_mae": float(group["absolute_error"].mean()),
            "weekly_direction_accuracy": float(group["direction_correct"].mean()),
            "coverage_50": float(coverage_50),
            "coverage_80": float(coverage_80),
            "coverage_95": float(coverage_95),
            "calibration_error": calibration_error,
            "mean_interval_width_80": float((group["q90"] - group["q10"]).mean()),
            "mean_actual_percentile": float(group["actual_percentile"].mean()),
        }
    return summaries, frame


def _paired_block_bootstrap(
    rolling: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    transformer = rolling[rolling["model"] == "transformer"].set_index("date")
    repetitions = int(config["evaluation"]["paired_bootstrap_repetitions"])
    block = int(config["evaluation"]["paired_bootstrap_block_weeks"])
    rng = np.random.default_rng(int(config["project"]["seed"]) + 70_000)
    rows: list[dict] = []
    for method in sorted(set(rolling["model"]) - {"transformer"}):
        other = rolling[rolling["model"] == method].set_index("date")
        common = transformer.index.intersection(other.index)
        if len(common) < 2:
            continue
        for metric in ("crps", "absolute_error"):
            difference = (
                transformer.loc[common, metric].to_numpy(dtype=float)
                - other.loc[common, metric].to_numpy(dtype=float)
            )
            estimates: list[float] = []
            n = len(difference)
            for _ in range(repetitions):
                indices: list[int] = []
                while len(indices) < n:
                    start = int(rng.integers(0, n))
                    indices.extend([(start + offset) % n for offset in range(block)])
                estimates.append(float(np.mean(difference[np.asarray(indices[:n])])) )
            lower, upper = np.quantile(estimates, [0.025, 0.975])
            rows.append(
                {
                    "comparison": f"transformer_minus_{method}",
                    "metric": metric,
                    "mean_difference": float(np.mean(difference)),
                    "ci_025": float(lower),
                    "ci_975": float(upper),
                    "weeks": n,
                    "negative_favors_transformer": True,
                }
            )
    return pd.DataFrame(rows)


def evaluate_project(config: dict, paths: dict, force: bool = False) -> None:
    evaluation_dir = Path(paths["artifacts"]["evaluation_dir"])
    summary_path = evaluation_dir / "summary.json"
    weekly_path = evaluation_dir / "weekly_frozen_evaluation.csv"
    rolling_path = Path(paths["artifacts"]["rolling_evaluation"])
    rolling_summary_path = Path(paths["artifacts"]["rolling_summary"])
    if (
        summary_path.exists()
        and weekly_path.exists()
        and rolling_path.exists()
        and rolling_summary_path.exists()
        and not force
    ):
        LOGGER.info("Using existing frozen and rolling evaluation")
        return

    horizon = int(config["benchmarks"]["horizon_weeks"])
    simulation_dir = Path(paths["artifacts"]["simulations_dir"])
    benchmark_dir = Path(paths["artifacts"]["benchmarks_dir"])
    candidates = {
        "transformer": simulation_dir / f"ordinary_h{horizon:03d}.npz",
        "zero_return": benchmark_dir / f"zero_return_h{horizon:03d}.npz",
        "block_bootstrap": benchmark_dir / f"block_bootstrap_h{horizon:03d}.npz",
        "linear_factor": benchmark_dir / f"linear_factor_h{horizon:03d}.npz",
        "gaussian_factor": benchmark_dir / f"gaussian_factor_h{horizon:03d}.npz",
    }
    actual_frame = _actual_global_returns(config, paths)
    observed_path = evaluation_dir / "observed_yahoo_basket_returns.csv"
    actual_frame.to_csv(ensure_parent(observed_path), index_label="date")
    legacy_path = evaluation_dir / "actual_global_returns.csv"
    if legacy_path.exists():
        legacy_path.unlink()
    summaries: dict[str, dict] = {}
    weekly_frames: list[pd.DataFrame] = []
    for name, path in candidates.items():
        if not path.exists():
            LOGGER.warning("Skipping missing evaluation input: %s", path)
            continue
        summary, weekly = evaluate_distribution(name, path, actual_frame, config)
        summaries[name] = summary
        weekly_frames.append(weekly)
    if not summaries:
        raise RuntimeError("No simulation or benchmark outputs were available for evaluation")

    write_json(summary_path, summaries)
    pd.concat(weekly_frames, ignore_index=True).to_csv(ensure_parent(weekly_path), index=False)

    rolling_summaries, rolling = rolling_one_step_evaluation(config, paths, actual_frame)
    rolling.to_csv(ensure_parent(rolling_path), index=False)
    write_json(rolling_summary_path, rolling_summaries)
    paired = _paired_block_bootstrap(rolling, config)
    paired.to_csv(ensure_parent(paths["artifacts"]["comparison_uncertainty"]), index=False)
    LOGGER.info(
        "Evaluated %d methods on frozen paths and %d methods across %d rolling weeks",
        len(summaries),
        len(rolling_summaries),
        int(rolling["date"].nunique()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate complete frozen paths and rolling one-step forecasts."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    evaluate_project(config, paths, force=args.force)


if __name__ == "__main__":
    main()
