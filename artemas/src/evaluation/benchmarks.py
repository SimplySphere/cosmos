from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

from src.simulation.engine import _factor_paths_to_market, _projection_statistics
from src.utils.config import load_project
from src.utils.files import load_npz, read_frame, save_npz

LOGGER = logging.getLogger(__name__)


def _automatic_block_length(history: np.ndarray) -> int:
    if len(history) < 12:
        return 2
    activity = np.sum(np.asarray(history, dtype=np.float64) ** 2, axis=1)
    activity -= activity.mean()
    denominator = float(np.dot(activity, activity))
    if denominator <= 1e-12:
        return 2
    maximum_lag = min(26, len(activity) // 4)
    threshold = 1.96 / np.sqrt(len(activity))
    small_run = 0
    for lag in range(1, maximum_lag + 1):
        autocorrelation = float(np.dot(activity[:-lag], activity[lag:]) / denominator)
        if abs(autocorrelation) < threshold:
            small_run += 1
            if small_run == 3:
                return max(2, lag - 2)
        else:
            small_run = 0
    return max(2, maximum_lag)


def _factor_history(config: dict, paths: dict) -> np.ndarray:
    factors = read_frame(paths["artifacts"]["final_factors"]).sort_index()
    history = np.array(
        factors.loc[: config["data"]["validation_end"]].to_numpy(dtype=np.float64),
        dtype=np.float64,
        copy=True,
        order="C",
    )
    return history[np.isfinite(history).all(axis=1)]


def _save_market_paths(
    output: Path,
    factor_paths: np.ndarray,
    config: dict,
    paths: dict,
    rng: np.random.Generator,
    **metadata: np.ndarray,
) -> None:
    pca = load_npz(paths["artifacts"]["final_pca"])
    projection = _projection_statistics(paths, pca)
    weekly_returns, index_paths = _factor_paths_to_market(
        factor_paths,
        projection,
        float(pca["residual_df"][0]),
        rng,
    )
    save_npz(
        output,
        factor_paths=factor_paths.astype(np.float32),
        group_weekly_returns=weekly_returns,
        group_index_paths=index_paths,
        group_names=np.asarray(projection["group_names"], dtype="U"),
        fit_end=np.asarray([str(config["data"]["validation_end"])], dtype="U"),
        forecast_origin=np.asarray([str(config["data"]["validation_end"])], dtype="U"),
        **metadata,
    )


def zero_return_benchmark(config: dict, paths: dict, force: bool = False) -> Path:
    horizon = int(config["benchmarks"]["horizon_weeks"])
    output = Path(paths["artifacts"]["benchmarks_dir"]) / f"zero_return_h{horizon:03d}.npz"
    if output.exists() and not force:
        return output
    pca = load_npz(paths["artifacts"]["final_pca"])
    projection = _projection_statistics(paths, pca)
    groups = len(projection["group_names"])
    save_npz(
        output,
        group_weekly_returns=np.zeros((1, horizon, groups), dtype=np.float32),
        group_index_paths=np.ones((1, horizon, groups), dtype=np.float32),
        group_names=np.asarray(projection["group_names"], dtype="U"),
        method=np.asarray(["zero_return"], dtype="U"),
        fit_end=np.asarray([str(config["data"]["validation_end"])], dtype="U"),
        forecast_origin=np.asarray([str(config["data"]["validation_end"])], dtype="U"),
    )
    return output


def stationary_bootstrap_benchmark(config: dict, paths: dict, force: bool = False) -> Path:
    horizon = int(config["benchmarks"]["horizon_weeks"])
    n_paths = int(config["benchmarks"]["paths"])
    output = Path(paths["artifacts"]["benchmarks_dir"]) / f"block_bootstrap_h{horizon:03d}.npz"
    if output.exists() and not force:
        return output

    seed = int(config["project"]["seed"]) + 20_000
    rng = np.random.default_rng(seed)
    history = _factor_history(config, paths).astype(np.float32)
    configured_block = config["benchmarks"].get("bootstrap_expected_block_weeks")
    expected_block = (
        float(configured_block)
        if configured_block is not None
        else float(_automatic_block_length(history))
    )
    LOGGER.info("Stationary bootstrap expected block length: %.1f weeks", expected_block)
    restart_probability = 1.0 / max(expected_block, 1.0)
    indices = rng.integers(0, len(history), size=n_paths)
    paths_out = np.empty((n_paths, horizon, history.shape[1]), dtype=np.float32)
    for step in range(horizon):
        paths_out[:, step] = history[indices]
        restart = rng.random(n_paths) < restart_probability
        indices = np.where(
            restart,
            rng.integers(0, len(history), size=n_paths),
            (indices + 1) % len(history),
        )
    _save_market_paths(
        output,
        paths_out,
        config,
        paths,
        rng,
        seed=np.asarray([seed], dtype=np.int64),
        expected_block_weeks=np.asarray([expected_block], dtype=np.float32),
        method=np.asarray(["stationary_block_bootstrap"], dtype="U"),
    )
    LOGGER.info("Created stationary block-bootstrap benchmark")
    return output


def _companion_spectral_radius(coefficients: np.ndarray, factor_count: int, lags: int) -> float:
    if lags == 1:
        return float(np.max(np.abs(np.linalg.eigvals(coefficients))))
    blocks_oldest_to_newest = [
        coefficients[:, index * factor_count : (index + 1) * factor_count]
        for index in range(lags)
    ]
    top = np.hstack(blocks_oldest_to_newest[::-1])
    companion = np.zeros((factor_count * lags, factor_count * lags), dtype=np.float64)
    companion[:factor_count] = top
    companion[factor_count:, :-factor_count] = np.eye(factor_count * (lags - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def fit_stable_var(config: dict, history: np.ndarray) -> dict[str, np.ndarray | float | int]:
    history = np.asarray(history, dtype=np.float64)
    lags = min(int(config["benchmarks"].get("linear_lags", 4)), len(history) - 2)
    if lags < 1:
        raise ValueError("Not enough factor history for the stable linear benchmark")
    x = np.stack(
        [history[index - lags : index].reshape(-1) for index in range(lags, len(history))]
    )
    y = history[lags:]
    ridge = Ridge(alpha=float(config["benchmarks"]["ridge_alpha"]), fit_intercept=True)
    ridge.fit(x, y)
    coefficients = np.asarray(ridge.coef_, dtype=np.float64)
    spectral_radius_before = _companion_spectral_radius(coefficients, history.shape[1], lags)
    maximum_radius = float(config["benchmarks"].get("linear_max_spectral_radius", 0.98))
    if spectral_radius_before > maximum_radius:
        coefficients *= maximum_radius / spectral_radius_before
    spectral_radius_after = _companion_spectral_radius(coefficients, history.shape[1], lags)
    predictions = x @ coefficients.T + np.asarray(ridge.intercept_, dtype=np.float64)
    residuals = y - predictions
    covariance = LedoitWolf().fit(residuals).covariance_
    covariance += np.eye(covariance.shape[0]) * 1e-8
    return {
        "coefficients": coefficients,
        "intercept": np.asarray(ridge.intercept_, dtype=np.float64),
        "cholesky": np.linalg.cholesky(covariance),
        "lags": lags,
        "spectral_radius_before": spectral_radius_before,
        "spectral_radius_after": spectral_radius_after,
    }


def sample_stable_var_one_step(
    model: dict[str, np.ndarray | float | int],
    context: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lags = int(model["lags"])
    context = np.asarray(context, dtype=np.float64)[-lags:]
    mean = context.reshape(1, -1) @ np.asarray(model["coefficients"]).T + np.asarray(
        model["intercept"]
    )
    shocks = rng.standard_normal((n_samples, mean.shape[1])) @ np.asarray(model["cholesky"]).T
    return mean + shocks


def linear_factor_benchmark(config: dict, paths: dict, force: bool = False) -> Path:
    horizon = int(config["benchmarks"]["horizon_weeks"])
    n_paths = int(config["benchmarks"]["paths"])
    output = Path(paths["artifacts"]["benchmarks_dir"]) / f"linear_factor_h{horizon:03d}.npz"
    if output.exists() and not force:
        return output

    seed = int(config["project"]["seed"]) + 30_000
    rng = np.random.default_rng(seed)
    history = _factor_history(config, paths)
    model = fit_stable_var(config, history)
    lags = int(model["lags"])
    contexts = np.repeat(history[-lags:][None, :, :], n_paths, axis=0)
    factor_paths = np.empty((n_paths, horizon, history.shape[1]), dtype=np.float32)
    coefficients = np.asarray(model["coefficients"])
    intercept = np.asarray(model["intercept"])
    cholesky = np.asarray(model["cholesky"])
    for step in range(horizon):
        prediction = contexts.reshape(n_paths, -1) @ coefficients.T + intercept
        shocks = rng.standard_normal(size=prediction.shape) @ cholesky.T
        sampled = prediction + shocks
        factor_paths[:, step] = sampled.astype(np.float32)
        contexts = np.concatenate([contexts[:, 1:], sampled[:, None, :]], axis=1)
    _save_market_paths(
        output,
        factor_paths,
        config,
        paths,
        rng,
        seed=np.asarray([seed], dtype=np.int64),
        lags=np.asarray([lags], dtype=np.int32),
        spectral_radius_before=np.asarray([model["spectral_radius_before"]], dtype=np.float32),
        spectral_radius_after=np.asarray([model["spectral_radius_after"]], dtype=np.float32),
        method=np.asarray(["stable_ridge_var"], dtype="U"),
    )
    LOGGER.info(
        "Created stable linear factor benchmark (spectral radius %.3f -> %.3f)",
        model["spectral_radius_before"],
        model["spectral_radius_after"],
    )
    return output


def gaussian_factor_benchmark(config: dict, paths: dict, force: bool = False) -> Path:
    horizon = int(config["benchmarks"]["horizon_weeks"])
    n_paths = int(config["benchmarks"]["paths"])
    output = Path(paths["artifacts"]["benchmarks_dir"]) / f"gaussian_factor_h{horizon:03d}.npz"
    if output.exists() and not force:
        return output
    seed = int(config["project"]["seed"]) + 40_000
    rng = np.random.default_rng(seed)
    history = _factor_history(config, paths)
    mean = history.mean(axis=0)
    covariance = LedoitWolf().fit(history).covariance_ + np.eye(history.shape[1]) * 1e-8
    cholesky = np.linalg.cholesky(covariance)
    shocks = rng.standard_normal((n_paths, horizon, history.shape[1]))
    factor_paths = mean[None, None, :] + np.einsum("phk,jk->phj", shocks, cholesky)
    _save_market_paths(
        output,
        factor_paths.astype(np.float32),
        config,
        paths,
        rng,
        seed=np.asarray([seed], dtype=np.int64),
        method=np.asarray(["unconditional_gaussian_factor"], dtype="U"),
    )
    LOGGER.info("Created unconditional Gaussian factor benchmark")
    return output


def run_benchmarks(config: dict, paths: dict, force: bool = False) -> None:
    zero_return_benchmark(config, paths, force=force)
    stationary_bootstrap_benchmark(config, paths, force=force)
    linear_factor_benchmark(config, paths, force=force)
    gaussian_factor_benchmark(config, paths, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the project forecasting benchmarks.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    run_benchmarks(config, paths, force=args.force)


if __name__ == "__main__":
    main()
