from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from src.factors.matrix import iterative_svd_impute, masked_factor_scores, standardize_from_rows
from src.utils.config import load_project
from src.utils.files import load_npz, read_frame, read_json, save_npz, write_frame

LOGGER = logging.getLogger(__name__)


def _paths_for_phase(paths: dict, phase: str) -> tuple[Path, Path, Path]:
    if phase == "development":
        return (
            Path(paths["artifacts"]["development_pca"]),
            Path(paths["artifacts"]["development_factors"]),
            Path(paths["artifacts"]["development_factor_quality"]),
        )
    if phase == "final":
        return (
            Path(paths["artifacts"]["final_pca"]),
            Path(paths["artifacts"]["final_factors"]),
            Path(paths["artifacts"]["final_factor_quality"]),
        )
    raise ValueError("phase must be 'development' or 'final'")


def fit_pca(config: dict, paths: dict, phase: str, force: bool = False) -> None:
    model_path, scores_path, quality_path = _paths_for_phase(paths, phase)
    if model_path.exists() and scores_path.exists() and quality_path.exists() and not force:
        LOGGER.info("Using existing %s PCA artifacts", phase)
        return

    selected = read_json(paths["artifacts"]["selected_rank"])
    rank = int(selected["selected_rank"])
    returns = read_frame(paths["data"]["weekly_returns"]).sort_index()
    fit_end = (
        config["data"]["train_end"] if phase == "development" else config["data"]["validation_end"]
    )
    fit_rows = np.flatnonzero(returns.index <= pd.Timestamp(fit_end))
    if len(fit_rows) == 0:
        raise ValueError(f"No data available through {fit_end}")

    values = np.array(returns.to_numpy(dtype=np.float64), copy=True, order="C")
    standardized, observed, means, scales = standardize_from_rows(values, fit_rows)
    fit_standardized = standardized[fit_rows]
    fit_observed = observed[fit_rows]
    effective_rank = min(rank, min(fit_standardized.shape) - 1)
    if effective_rank < 1:
        raise ValueError("The processed panel is too small to fit PCA")

    imputed_fit = iterative_svd_impute(
        fit_standardized,
        fit_observed,
        effective_rank,
        int(config["factors"].get("imputation_iterations", 4)),
        int(config["project"]["seed"]),
        int(config["factors"]["randomized_svd_iterations"]),
    )
    _, singular_values, components = randomized_svd(
        imputed_fit,
        n_components=effective_rank,
        n_iter=int(config["factors"]["randomized_svd_iterations"]),
        random_state=int(config["project"]["seed"]),
    )

    factor_scores, observed_fraction = masked_factor_scores(
        standardized,
        observed,
        components,
        ridge=float(config["factors"].get("score_ridge", 1e-4)),
        minimum_observed_fraction=float(
            config["data"].get("minimum_factor_week_observed_fraction", 0.5)
        ),
    )
    fit_scores = factor_scores[fit_rows]
    valid_fit_rows = np.isfinite(fit_scores).all(axis=1)
    if valid_fit_rows.sum() < max(20, effective_rank + 2):
        raise RuntimeError("Too few complete masked factor scores were produced for PCA fitting")

    reconstructed_fit = fit_scores @ components
    residuals = fit_standardized - reconstructed_fit
    residuals[~fit_observed] = np.nan
    residual_scale = np.nanstd(residuals, axis=0, ddof=1)
    residual_scale = np.where(
        np.isfinite(residual_scale) & (residual_scale > 1e-6), residual_scale, 1e-3
    )
    total_variance = float(np.nansum(np.nanvar(fit_standardized, axis=0, ddof=1)))
    explained = (singular_values**2) / max((len(fit_rows) - 1) * total_variance, 1e-12)

    save_npz(
        model_path,
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        components=components.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
        residual_scale=residual_scale.astype(np.float32),
        residual_df=np.array(
            [config["factors"]["residual_degrees_of_freedom"]], dtype=np.float32
        ),
        stock_ids=np.asarray(returns.columns.astype(str), dtype="U"),
        fit_end=np.asarray([fit_end], dtype="U"),
        imputation_method=np.asarray(["iterative_svd_masked_scores"], dtype="U"),
    )
    score_frame = pd.DataFrame(
        factor_scores,
        index=returns.index,
        columns=[f"factor_{index + 1:03d}" for index in range(effective_rank)],
    )
    weekly_quality = read_frame(paths["data"]["weekly_observed_universe"]).reindex(returns.index)
    quality_frame = pd.DataFrame(
        {
            "observed_stock_fraction": observed_fraction,
            "factor_scores_valid": np.isfinite(factor_scores).all(axis=1),
        },
        index=returns.index,
    ).join(weekly_quality, how="left")
    write_frame(score_frame, scores_path, index=True)
    write_frame(quality_frame, quality_path, index=True)
    LOGGER.info(
        "Fitted %s PCA with rank %d; %.1f%% cumulative explained variance; %d/%d weeks valid",
        phase,
        effective_rank,
        100 * float(np.sum(explained)),
        int(quality_frame["factor_scores_valid"].sum()),
        len(quality_frame),
    )


def load_pca(path: str | Path) -> dict[str, np.ndarray]:
    return load_npz(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit masked PCA factors for development or final training.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--phase", choices=["development", "final", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    phases = ["development", "final"] if args.phase == "both" else [args.phase]
    for phase in phases:
        fit_pca(config, paths, phase, force=args.force)


if __name__ == "__main__":
    main()
