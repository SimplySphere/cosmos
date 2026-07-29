from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model.train import load_model_bundle
from src.utils.config import load_project
from src.utils.files import load_npz, read_frame, save_npz, write_json
from src.utils.runtime import select_device, seed_everything

LOGGER = logging.getLogger(__name__)


def _aggregate_matrix(paths: dict, pca: dict) -> tuple[np.ndarray, list[str]]:
    """Build equal-weight aggregate projections; selector-source weights are never loaded."""
    stock_ids = [str(value) for value in pca["stock_ids"]]
    index = {security_id: position for position, security_id in enumerate(stock_ids)}
    weights = read_frame(paths["data"]["market_aggregation_weights"])
    weights = weights[weights["security_id"].isin(index)].copy()
    weights["position"] = weights["security_id"].map(index)

    rows: list[np.ndarray] = []
    names: list[str] = []

    def add_group(name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        vector = np.zeros(len(stock_ids), dtype=np.float64)
        positions = np.array(frame["position"].to_numpy(dtype=int), dtype=int, copy=True)
        values = np.array(
            frame["aggregation_weight"].to_numpy(dtype=np.float64), dtype=np.float64, copy=True
        )
        if values.sum() <= 0:
            return
        values /= values.sum()
        vector[positions] = values
        rows.append(vector)
        names.append(name)

    add_group("Yahoo Finance Equal-Weight Global Basket", weights)
    country_weights = weights.groupby("country", dropna=False)["aggregation_weight"].sum().sort_values(ascending=False)
    for country in country_weights.index:
        add_group(f"Country: {country}", weights[weights["country"] == country])
    sector_weights = weights.groupby("sector", dropna=False)["aggregation_weight"].sum().sort_values(ascending=False)
    for sector in sector_weights.index:
        add_group(f"Sector: {sector}", weights[weights["sector"] == sector])
    return np.vstack(rows), names


def _projection_statistics(paths: dict, pca: dict) -> dict[str, np.ndarray | list[str]]:
    aggregate_weights, group_names = _aggregate_matrix(paths, pca)
    means = pca["means"].astype(np.float64)
    scales = pca["scales"].astype(np.float64)
    components = pca["components"].astype(np.float64)
    residual_scale = pca["residual_scale"].astype(np.float64)

    residual_projection = aggregate_weights * (scales * residual_scale)[None, :]
    residual_covariance = residual_projection @ residual_projection.T
    eigenvalues, eigenvectors = np.linalg.eigh(residual_covariance)
    keep = eigenvalues > max(float(eigenvalues.max(initial=0.0)) * 1e-10, 1e-14)
    residual_factor = (
        eigenvectors[:, keep] * np.sqrt(np.maximum(eigenvalues[keep], 0.0))[None, :]
        if np.any(keep)
        else np.zeros((len(group_names), 0), dtype=np.float64)
    )
    return {
        "group_names": group_names,
        "intercepts": aggregate_weights @ means,
        "factor_loadings": (aggregate_weights * scales[None, :]) @ components.T,
        "residual_factor": residual_factor,
    }


def _sample_components(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = np.array(probabilities, dtype=np.float64, copy=True)
    normalized = np.clip(normalized, 0.0, None)
    totals = normalized.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise FloatingPointError("Mixture probabilities are invalid")
    normalized /= totals
    cumulative = np.cumsum(normalized, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random((normalized.shape[0], 1))
    return np.sum(draws > cumulative, axis=1).clip(max=normalized.shape[1] - 1)


def _model_parameters(
    model: torch.nn.Module,
    contexts: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = torch.as_tensor(
                np.array(contexts[start : start + batch_size], dtype=np.float32, copy=True),
                dtype=torch.float32,
                device=device,
            )
            parameters = model.distribution_parameters(batch)
            for key, value in parameters.items():
                collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _selected_parameters(
    parameters: dict[str, np.ndarray],
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(len(components))
    loc = parameters["loc"][rows, components]
    degrees = parameters["degrees_of_freedom"][rows, components]
    covariance = (
        parameters["cholesky"][rows, components]
        if "cholesky" in parameters
        else parameters["scale"][rows, components]
    )
    return loc, covariance, degrees


def _sample_selected_student_t(
    loc: np.ndarray,
    covariance: np.ndarray,
    degrees: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    normal = rng.standard_normal(size=loc.shape)
    radial = np.sqrt(rng.chisquare(degrees) / degrees)[:, None]
    if covariance.ndim == 3:
        transformed = np.einsum("bij,bj->bi", covariance, normal)
    else:
        transformed = covariance * normal
    return loc + transformed / np.maximum(radial, 1e-8)


def _sample_factor_step(
    parameters: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one factor vector from the fitted Student-t mixture."""
    probabilities = parameters["probabilities"]
    components = _sample_components(probabilities, rng)
    loc, covariance, degrees = _selected_parameters(parameters, components)
    factors = _sample_selected_student_t(loc, covariance, degrees, rng)
    return factors.astype(np.float32), components

def _factor_paths_to_market(
    factor_paths: np.ndarray,
    projection: dict[str, np.ndarray | list[str]],
    residual_df: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    intercepts = np.asarray(projection["intercepts"], dtype=np.float64)
    loadings = np.asarray(projection["factor_loadings"], dtype=np.float64)
    residual_factor = np.asarray(projection["residual_factor"], dtype=np.float64)
    weekly_returns = np.einsum("phk,gk->phg", factor_paths, loadings) + intercepts
    if residual_factor.shape[1] > 0:
        variance_correction = np.sqrt((residual_df - 2.0) / residual_df)
        latent = rng.standard_t(
            residual_df,
            size=(factor_paths.shape[0], factor_paths.shape[1], residual_factor.shape[1]),
        ) * variance_correction
        weekly_returns += np.einsum("phr,gr->phg", latent, residual_factor)
    cumulative = np.cumsum(weekly_returns, axis=1)
    cumulative = np.clip(cumulative, -80.0, 80.0)
    index_paths = np.exp(cumulative)
    return weekly_returns.astype(np.float32), index_paths.astype(np.float32)


def _initial_context(
    config: dict,
    paths: dict,
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = read_frame(paths["artifacts"]["final_factors"]).sort_index()
    history = scores.loc[: config["data"]["validation_end"]]
    context_length = int(config["model"]["context_length"])
    history = history[np.isfinite(history.to_numpy()).all(axis=1)]
    if len(history) < context_length:
        raise ValueError("Final finite factor history is shorter than the configured context")
    raw = np.array(
        history.iloc[-context_length:].to_numpy(dtype=np.float32),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    factor_mean = np.asarray(checkpoint["factor_mean"], dtype=np.float32)
    factor_scale = np.asarray(checkpoint["factor_scale"], dtype=np.float32)
    normalized = (raw - factor_mean) / factor_scale
    return normalized, factor_mean, factor_scale, pd.Timestamp(history.index.max())


def _sanity_payload(
    horizon: int,
    weekly_returns: np.ndarray,
    index_paths: np.ndarray,
    clip_count: int,
    total_factor_draws: int,
    config: dict,
) -> dict:
    global_weekly = np.asarray(weekly_returns[:, :, 0], dtype=np.float64)
    global_index = np.asarray(index_paths[:, :, 0], dtype=np.float64)
    terminal = global_index[:, -1] - 1.0
    annualized = np.expm1(np.log(np.maximum(global_index[:, -1], 1e-12)) * 52.0 / horizon)
    max_abs_week = float(np.nanmax(np.abs(global_weekly)))
    median_annualized = float(np.nanmedian(annualized))
    warnings: list[str] = []
    if not np.isfinite(global_index).all() or np.any(global_index <= 0):
        warnings.append("nonfinite_or_nonpositive_index_path")
    if max_abs_week > float(config["simulation"]["sanity_max_abs_weekly_global_return"]):
        warnings.append("weekly_global_return_exceeds_configured_sanity_bound")
    if horizon >= 52 and abs(median_annualized) > float(
        config["simulation"]["sanity_max_median_annualized_return"]
    ):
        warnings.append("median_annualized_return_exceeds_configured_sanity_bound")
    terminal_quantile_values = np.nanquantile(terminal, [0.01, 0.05, 0.5, 0.95, 0.99])
    annualized_quantile_values = np.nanquantile(annualized, [0.01, 0.05, 0.5, 0.95, 0.99])
    if horizon >= 260 and float(terminal_quantile_values[0]) > 0:
        warnings.append("long_horizon_lower_tail_entirely_positive_check_drift")
    return {
        "horizon_weeks": int(horizon),
        "path_count": int(len(global_index)),
        "minimum_weekly_global_return": float(np.nanmin(global_weekly)),
        "maximum_weekly_global_return": float(np.nanmax(global_weekly)),
        "maximum_absolute_weekly_global_return": max_abs_week,
        "terminal_return_quantiles": {
            str(level): float(value)
            for level, value in zip(
                [0.01, 0.05, 0.5, 0.95, 0.99],
                terminal_quantile_values,
            )
        },
        "annualized_return_quantiles": {
            str(level): float(value)
            for level, value in zip(
                [0.01, 0.05, 0.5, 0.95, 0.99], annualized_quantile_values
            )
        },
        "probability_terminal_loss": float(np.mean(terminal < 0.0)),
        "median_annualized_return": median_annualized,
        "interpretation": str(
            config["simulation"].get(
                "long_horizon_interpretation", "exploratory_recursive_scenario"
            )
        ),
        "factor_clip_count": int(clip_count),
        "factor_clip_fraction": float(clip_count / max(total_factor_draws, 1)),
        "warnings": warnings,
    }


def _update_sanity_summary(paths: dict, payload: dict) -> None:
    output = Path(paths["artifacts"]["simulation_sanity"])
    current: dict = {}
    if output.exists():
        import json

        with output.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                current = loaded
    current[str(payload["horizon_weeks"])] = payload
    write_json(output, current)


def simulate_ordinary(
    config: dict,
    paths: dict,
    horizon: int,
    n_paths: int,
    force: bool = False,
) -> Path:
    output = Path(paths["artifacts"]["simulations_dir"]) / f"ordinary_h{horizon:03d}.npz"
    if output.exists() and not force:
        LOGGER.info("Using existing ordinary simulation: %s", output)
        return output

    actual_seed = int(config["project"]["seed"]) + horizon
    seed_everything(actual_seed)
    rng = np.random.default_rng(actual_seed)
    device = select_device(str(config["project"]["device"]))
    model, checkpoint = load_model_bundle(paths["artifacts"]["final_checkpoint"], device)
    base_context, factor_mean, factor_scale, context_end = _initial_context(config, paths, checkpoint)
    contexts = np.repeat(base_context[None, :, :], n_paths, axis=0)
    normalized_paths = np.empty((n_paths, horizon, base_context.shape[1]), dtype=np.float32)
    batch_size = int(config["simulation"]["inference_batch_size"])
    clip_bound = float(config["simulation"].get("factor_clip_std", 0.0))
    clip_count = 0

    for step in range(horizon):
        parameters = _model_parameters(model, contexts, device, batch_size)
        sampled, _ = _sample_factor_step(parameters, rng)
        if clip_bound > 0:
            clipped = np.clip(sampled, -clip_bound, clip_bound)
            clip_count += int(np.count_nonzero(clipped != sampled))
            sampled = clipped
        normalized_paths[:, step] = sampled
        contexts = np.concatenate([contexts[:, 1:], sampled[:, None, :]], axis=1)
        if (step + 1) % 13 == 0 or step + 1 == horizon:
            LOGGER.info("Ordinary simulation week %d/%d", step + 1, horizon)

    factor_paths = normalized_paths * factor_scale[None, None, :] + factor_mean[None, None, :]
    pca = load_npz(paths["artifacts"]["final_pca"])
    projection = _projection_statistics(paths, pca)
    weekly_returns, index_paths = _factor_paths_to_market(
        factor_paths,
        projection,
        float(pca["residual_df"][0]),
        rng,
    )
    forecast_dates = pd.date_range(
        start=context_end + pd.Timedelta(days=1),
        periods=horizon,
        freq=str(config["data"]["week_rule"]),
    )
    save_npz(
        output,
        factor_paths=factor_paths.astype(np.float32),
        group_weekly_returns=weekly_returns,
        group_index_paths=index_paths,
        group_names=np.asarray(projection["group_names"], dtype="U"),
        horizon=np.asarray([horizon], dtype=np.int32),
        seed=np.asarray([actual_seed], dtype=np.int64),
        calibration_scale=np.asarray([checkpoint.get("calibration_scale", 1.0)], dtype=np.float32),
        factor_clip_count=np.asarray([clip_count], dtype=np.int64),
        forecast_origin=np.asarray([str(context_end.date())], dtype="U"),
        forecast_dates=np.asarray([str(date.date()) for date in forecast_dates], dtype="U"),
        model_fit_end=np.asarray([str(checkpoint.get("fit_end", config["data"]["validation_end"]))], dtype="U"),
        interpretation=np.asarray(
            [str(config["simulation"].get("long_horizon_interpretation", "exploratory_recursive_scenario"))],
            dtype="U",
        ),
    )
    sanity = _sanity_payload(
        horizon,
        weekly_returns,
        index_paths,
        clip_count,
        normalized_paths.size,
        config,
    )
    write_json(output.with_name(f"sanity_h{horizon:03d}.json"), sanity)
    _update_sanity_summary(paths, sanity)
    if sanity["warnings"]:
        LOGGER.warning("Simulation sanity warnings for %d weeks: %s", horizon, sanity["warnings"])
    LOGGER.info("Saved %d ordinary paths for %d weeks", n_paths, horizon)
    return output


def run_simulations(config: dict, paths: dict, force: bool = False) -> None:
    """Generate ordinary recursive Monte Carlo paths at every configured horizon.

    Long horizons are intentionally retained as exploratory recursive scenarios. They
    are not presented as horizon-validated forecasts merely because the code can roll
    the one-step distribution forward for many years.
    """
    horizons = sorted(set(int(value) for value in config["simulation"]["horizons_weeks"]))
    paths_by_horizon = config["simulation"]["paths_by_horizon"]
    one_year_paths = int(paths_by_horizon.get("52", 5000))
    for horizon in horizons:
        if str(horizon) in paths_by_horizon:
            n_paths = int(paths_by_horizon[str(horizon)])
        elif horizon <= 52:
            n_paths = one_year_paths
        else:
            n_paths = max(500, one_year_paths // 4)
        simulate_ordinary(config, paths, horizon, n_paths, force=force)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recursive ordinary Monte Carlo futures.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--paths-count", type=int)
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    if args.horizon is not None:
        count = args.paths_count or int(
            config["simulation"]["paths_by_horizon"].get(str(args.horizon), 5000)
        )
        simulate_ordinary(config, paths, args.horizon, count, force=args.force)
    else:
        run_simulations(config, paths, force=args.force)


if __name__ == "__main__":
    main()
