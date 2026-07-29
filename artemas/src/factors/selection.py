from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd

from src.factors.matrix import iterative_svd_impute, masked_factor_scores, standardize_from_rows
from src.utils.config import load_project
from src.utils.files import ensure_parent, read_frame, write_json

LOGGER = logging.getLogger(__name__)


def _expanding_time_splits(
    n_time: int,
    fold_count: int,
    minimum_train: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    minimum_train = min(max(int(minimum_train), 8), n_time - 2)
    remaining = n_time - minimum_train
    fold_count = min(int(fold_count), remaining)
    if fold_count < 2:
        raise ValueError("Expanding factor selection requires at least two held-out time blocks")
    boundaries = np.linspace(minimum_train, n_time, fold_count + 1, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(fold_count):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        if end <= start:
            continue
        splits.append((np.arange(start, dtype=int), np.arange(start, end, dtype=int)))
    if len(splits) < 2:
        raise ValueError("Unable to create at least two expanding time folds")
    return splits


def _mean_and_se(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(finite))
    standard_error = (
        float(np.std(finite, ddof=1) / np.sqrt(count)) if count > 1 else 0.0
    )
    return mean, standard_error, count


def _aggregate_metric(
    frame: pd.DataFrame,
    value_column: str,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate long-form fold results without requiring every fold to test every rank."""
    means: list[float] = []
    standard_errors: list[float] = []
    counts: list[int] = []
    for rank in ranks:
        if frame.empty or value_column not in frame:
            values = np.asarray([], dtype=np.float64)
        else:
            values = pd.to_numeric(
                frame.loc[frame["rank"] == int(rank), value_column], errors="coerce"
            ).to_numpy(dtype=np.float64)
        mean, standard_error, count = _mean_and_se(values)
        means.append(mean)
        standard_errors.append(standard_error)
        counts.append(count)
    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(standard_errors, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def _eligible_stocks_for_time_fold(
    values: np.ndarray,
    train_time: np.ndarray,
    factor_cfg: dict,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Select stocks that can be standardized and evaluated inside one time fold.

    The real panel contains securities that listed after the beginning of the history.
    Such columns can have hundreds of observations by the global training cutoff but
    zero observations inside the earliest expanding fold. They must not make the fold
    fail, and they also must not be standardized from one or two accidental values.
    """
    # Eligibility is based only on the expanding training block. Held-out observations
    # are never consulted to decide whether a stock participates in a fold.
    train_counts = np.isfinite(values[train_time]).sum(axis=0)
    minimum_train_observations = max(
        2,
        int(factor_cfg.get("minimum_fold_train_observations", 26)),
        int(
            np.ceil(
                float(factor_cfg.get("minimum_fold_train_fraction", 0.25))
                * len(train_time)
            )
        ),
    )
    eligible = np.flatnonzero(train_counts >= minimum_train_observations)
    diagnostics: dict[str, int | float] = {
        "minimum_train_observations": int(minimum_train_observations),
        "eligible_stocks": int(len(eligible)),
        "excluded_stocks": int(values.shape[1] - len(eligible)),
        "eligible_stock_fraction": float(len(eligible) / max(values.shape[1], 1)),
    }
    return eligible, diagnostics


def _forecast_error(
    train_scores: np.ndarray,
    held_scores: np.ndarray,
    lags: int,
    alpha: float,
) -> float:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    held_scores = np.asarray(held_scores, dtype=np.float64)
    if len(train_scores) <= lags or held_scores.size == 0:
        return float("nan")
    x = np.stack(
        [train_scores[index - lags : index].reshape(-1) for index in range(lags, len(train_scores))]
    )
    y = train_scores[lags:]
    valid_train = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if valid_train.sum() < max(20, x.shape[1] // 2):
        return float("nan")
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(x[valid_train], y[valid_train])

    history = [row.copy() for row in train_scores]
    errors: list[np.ndarray] = []
    variance = np.nanvar(train_scores, axis=0, ddof=1)
    variance = np.where(np.isfinite(variance) & (variance > 1e-10), variance, 1.0)
    for actual in held_scores:
        context = np.asarray(history[-lags:], dtype=np.float64)
        if len(context) < lags or not np.isfinite(context).all() or not np.isfinite(actual).all():
            history.append(actual.copy())
            continue
        prediction = model.predict(context.reshape(1, -1))[0]
        errors.append((prediction - actual) ** 2 / variance)
        history.append(actual.copy())
    if not errors:
        return float("nan")
    return float(np.mean(np.vstack(errors)))


def _fold_output_frame(
    reconstruction_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
) -> pd.DataFrame:
    reconstruction = reconstruction_frame.copy()
    forecast = forecast_frame.copy()
    reconstruction["metric_scope"] = "held_stock_reconstruction"
    forecast["metric_scope"] = "full_cross_section_forecast"
    all_columns = sorted(set(reconstruction.columns) | set(forecast.columns))
    for frame in (reconstruction, forecast):
        for column in all_columns:
            if column not in frame:
                frame[column] = np.nan
    return pd.concat(
        [reconstruction[all_columns], forecast[all_columns]],
        ignore_index=True,
        sort=False,
    )


def select_factor_rank(config: dict, paths: dict, force: bool = False) -> dict:
    output = Path(paths["artifacts"]["selected_rank"])
    results_path = Path(paths["artifacts"]["factor_cv_results"])
    fold_path = Path(paths["artifacts"]["factor_cv_folds"])
    if output.exists() and results_path.exists() and fold_path.exists() and not force:
        LOGGER.info("Using existing factor-rank selection: %s", output)
        return json.loads(output.read_text(encoding="utf-8"))

    returns = read_frame(paths["data"]["weekly_returns"]).sort_index()
    train = returns.loc[: config["data"]["train_end"]]
    if train.empty:
        raise ValueError("No observations fall inside the configured training period")
    values = np.array(train.to_numpy(dtype=np.float64), copy=True, order="C")
    n_time, n_stocks = values.shape
    factor_cfg = config["factors"]
    requested_min = int(factor_cfg["minimum_rank"])
    requested_max = min(int(factor_cfg["maximum_rank"]), n_time - 1, n_stocks - 1)
    stride = max(1, int(factor_cfg.get("forecast_candidate_stride", 1)))
    ranks = list(range(requested_min, requested_max + 1, stride))
    if requested_max not in ranks:
        ranks.append(requested_max)
    ranks = np.asarray(sorted(set(ranks)), dtype=int)
    if len(ranks) < 2:
        raise ValueError("The panel is too small for the configured factor-rank range")

    time_splits = _expanding_time_splits(
        n_time,
        int(factor_cfg["time_folds"]),
        int(factor_cfg.get("minimum_time_fold_train_weeks", max(52, n_time // 3))),
    )
    requested_stock_folds = int(factor_cfg["stock_folds"])
    minimum_fold_stocks = max(
        requested_stock_folds,
        requested_min + 2,
        int(factor_cfg.get("minimum_fold_stocks", 20)),
    )
    max_rank = int(ranks.max())
    imputation_iterations = int(factor_cfg.get("imputation_iterations", 3))
    svd_iterations = int(factor_cfg["randomized_svd_iterations"])
    seed = int(config["project"]["seed"])

    reconstruction_rows: list[dict] = []
    forecast_rows: list[dict] = []
    time_fold_diagnostics: list[dict] = []
    for time_index, (train_time, held_time) in enumerate(time_splits):
        eligible_stocks, diagnostics = _eligible_stocks_for_time_fold(
            values,
            train_time,
            factor_cfg,
        )
        diagnostics.update(
            {
                "time_fold": time_index + 1,
                "train_weeks": len(train_time),
                "held_weeks": len(held_time),
            }
        )
        time_fold_diagnostics.append(diagnostics)
        if len(eligible_stocks) < minimum_fold_stocks:
            LOGGER.warning(
                "Skipping factor time fold %d/%d: only %d/%d stocks have at least %d "
                "training observations",
                time_index + 1,
                len(time_splits),
                len(eligible_stocks),
                n_stocks,
                diagnostics["minimum_train_observations"],
            )
            continue

        fold_values = values[:, eligible_stocks]
        standardized, observed, _, _ = standardize_from_rows(fold_values, train_time)
        fold_max_rank = min(max_rank, len(train_time) - 1, len(eligible_stocks) - 1)
        fold_ranks = ranks[ranks <= fold_max_rank]
        if len(fold_ranks) == 0:
            LOGGER.warning(
                "Skipping factor time fold %d/%d because no configured rank fits %d eligible stocks",
                time_index + 1,
                len(time_splits),
                len(eligible_stocks),
            )
            continue

        LOGGER.info(
            "Factor time fold %d/%d uses %d/%d stocks (minimum %d training observations); "
            "testing ranks through %d",
            time_index + 1,
            len(time_splits),
            len(eligible_stocks),
            n_stocks,
            diagnostics["minimum_train_observations"],
            int(fold_ranks.max()),
        )

        # Forecast-oriented rank assessment on the fold-eligible cross-section.
        train_matrix = standardized[train_time]
        train_observed = observed[train_time]
        imputed = iterative_svd_impute(
            train_matrix,
            train_observed,
            fold_max_rank,
            imputation_iterations,
            seed + time_index * 10_000,
            svd_iterations,
        )
        _, _, full_components = randomized_svd(
            imputed,
            n_components=fold_max_rank,
            n_iter=svd_iterations,
            random_state=seed + 50_000 + time_index,
        )
        for rank in fold_ranks:
            components = full_components[:rank]
            train_scores = imputed @ components.T
            held_scores, _ = masked_factor_scores(
                standardized[held_time],
                observed[held_time],
                components,
                ridge=float(factor_cfg.get("score_ridge", 1e-4)),
                minimum_observed_fraction=float(
                    config["data"].get("minimum_factor_week_observed_fraction", 0.5)
                ),
            )
            forecast_rows.append(
                {
                    "time_fold": time_index + 1,
                    "rank": int(rank),
                    "forecast_mse": _forecast_error(
                        train_scores,
                        held_scores,
                        int(factor_cfg.get("forecast_lags", 4)),
                        float(factor_cfg.get("forecast_ridge_alpha", 1.0)),
                    ),
                    "train_weeks": len(train_time),
                    "held_weeks": len(held_time),
                    "eligible_stocks": len(eligible_stocks),
                    "excluded_stocks": n_stocks - len(eligible_stocks),
                    "minimum_train_observations": diagnostics[
                        "minimum_train_observations"
                    ],
                }
            )

        # Twice K-fold bi-cross-validation with a stock partition built from the
        # stocks that actually existed in this expanding time fold.
        stock_fold_count = min(requested_stock_folds, len(eligible_stocks))
        stock_splitter = KFold(
            n_splits=stock_fold_count,
            shuffle=True,
            random_state=seed + time_index,
        )
        local_stock_groups = [
            test for _, test in stock_splitter.split(np.arange(len(eligible_stocks)))
        ]
        for stock_index, held_stocks in enumerate(local_stock_groups):
            train_stocks = np.setdiff1d(
                np.arange(len(eligible_stocks)), held_stocks, assume_unique=True
            )
            a_raw = standardized[np.ix_(train_time, train_stocks)]
            a_observed = observed[np.ix_(train_time, train_stocks)]
            effective_max = min(fold_max_rank, min(a_raw.shape) - 1)
            if effective_max < requested_min:
                continue
            a = iterative_svd_impute(
                a_raw,
                a_observed,
                effective_max,
                imputation_iterations,
                seed + time_index * 100 + stock_index,
                svd_iterations,
            )
            b = np.where(
                observed[np.ix_(train_time, held_stocks)],
                standardized[np.ix_(train_time, held_stocks)],
                0.0,
            )
            c = np.where(
                observed[np.ix_(held_time, train_stocks)],
                standardized[np.ix_(held_time, train_stocks)],
                0.0,
            )
            d = standardized[np.ix_(held_time, held_stocks)]
            d_observed = observed[np.ix_(held_time, held_stocks)]
            u, singular_values, vt = randomized_svd(
                a,
                n_components=effective_max,
                n_iter=svd_iterations,
                random_state=seed + 100_000 + time_index * 100 + stock_index,
            )
            v = vt.T
            left = c @ v
            right = u.T @ b
            prediction = np.zeros_like(d)
            rank_set = set(int(value) for value in fold_ranks if value <= effective_max)
            for component in range(effective_max):
                singular = max(float(singular_values[component]), 1e-10)
                prediction += np.outer(left[:, component] / singular, right[component])
                rank_value = component + 1
                if rank_value in rank_set and d_observed.any():
                    residual = d[d_observed] - prediction[d_observed]
                    reconstruction_rows.append(
                        {
                            "time_fold": time_index + 1,
                            "stock_fold": stock_index + 1,
                            "rank": rank_value,
                            "reconstruction_mse": float(np.mean(residual**2)),
                            "train_weeks": len(train_time),
                            "held_weeks": len(held_time),
                            "held_stocks": len(held_stocks),
                            "eligible_stocks": len(eligible_stocks),
                            "excluded_stocks": n_stocks - len(eligible_stocks),
                            "minimum_train_observations": diagnostics[
                                "minimum_train_observations"
                            ],
                        }
                    )
            LOGGER.info(
                "Completed factor fold time=%d/%d stock=%d/%d",
                time_index + 1,
                len(time_splits),
                stock_index + 1,
                len(local_stock_groups),
            )

    reconstruction_frame = pd.DataFrame(reconstruction_rows)
    forecast_frame = pd.DataFrame(forecast_rows)
    if reconstruction_frame.empty or forecast_frame.empty:
        raise RuntimeError(
            "Factor selection produced no valid cross-validation results. Inspect the "
            "fold eligibility columns and lower factors.minimum_fold_train_observations "
            "only if the historical panel truly lacks sufficient data."
        )
    fold_frame = _fold_output_frame(reconstruction_frame, forecast_frame)
    fold_frame.to_csv(ensure_parent(fold_path), index=False)

    rec_mean, rec_se, rec_count = _aggregate_metric(
        reconstruction_frame, "reconstruction_mse", ranks
    )
    fore_mean, fore_se, fore_count = _aggregate_metric(
        forecast_frame, "forecast_mse", ranks
    )

    valid_rec = np.isfinite(rec_mean) & (rec_count > 0)
    if not np.any(valid_rec):
        raise RuntimeError("Every factor rank had an invalid reconstruction score")
    best_rec_position = np.flatnonzero(valid_rec)[np.argmin(rec_mean[valid_rec])]
    rec_threshold = rec_mean[best_rec_position]
    if bool(factor_cfg["one_standard_error_rule"]):
        rec_threshold += rec_se[best_rec_position]
    rec_eligible = valid_rec & (rec_mean <= rec_threshold)

    objective = str(factor_cfg.get("selection_objective", "forecast_mse"))
    valid_forecast = np.isfinite(fore_mean) & (fore_count > 0)
    if objective == "forecast_mse" and np.any(rec_eligible & valid_forecast):
        candidate_positions = np.flatnonzero(rec_eligible & valid_forecast)
        best_position = candidate_positions[np.argmin(fore_mean[candidate_positions])]
        selection_threshold = fore_mean[best_position]
        if bool(factor_cfg["one_standard_error_rule"]):
            selection_threshold += fore_se[best_position]
        eligible_positions = candidate_positions[
            fore_mean[candidate_positions] <= selection_threshold
        ]
        selected_position = int(eligible_positions[0])
        metric_name = "forecast_mse"
    else:
        candidate_positions = np.flatnonzero(rec_eligible)
        best_position = best_rec_position
        selection_threshold = rec_threshold
        selected_position = int(candidate_positions[0])
        metric_name = "reconstruction_mse"

    results = pd.DataFrame(
        {
            "rank": ranks,
            "mean_reconstruction_mse": rec_mean,
            "reconstruction_standard_error": rec_se,
            "reconstruction_fold_count": rec_count,
            "within_reconstruction_one_standard_error": rec_eligible,
            "mean_forecast_mse": fore_mean,
            "forecast_standard_error": fore_se,
            "forecast_fold_count": fore_count,
            "selected": ranks == int(ranks[selected_position]),
        }
    )
    results.to_csv(ensure_parent(results_path), index=False)

    finite_forecast_positions = np.flatnonzero(valid_forecast)
    minimum_forecast_position = (
        int(finite_forecast_positions[np.argmin(fore_mean[finite_forecast_positions])])
        if len(finite_forecast_positions)
        else None
    )
    completed_diagnostics = [
        item for item in time_fold_diagnostics if int(item["eligible_stocks"]) >= minimum_fold_stocks
    ]
    summary = {
        "selected_rank": int(ranks[selected_position]),
        "selection_objective": metric_name,
        "minimum_reconstruction_rank": int(ranks[best_rec_position]),
        "minimum_reconstruction_mse": float(rec_mean[best_rec_position]),
        "minimum_forecast_rank": (
            int(ranks[minimum_forecast_position])
            if minimum_forecast_position is not None
            else None
        ),
        "minimum_forecast_mse": (
            float(fore_mean[minimum_forecast_position])
            if minimum_forecast_position is not None
            else None
        ),
        "selection_threshold": float(selection_threshold),
        "time_folds_requested": len(time_splits),
        "time_folds_completed": len(completed_diagnostics),
        "stock_folds_requested": requested_stock_folds,
        "training_weeks": int(n_time),
        "stocks": int(n_stocks),
        "training_start": str(train.index.min().date()),
        "training_end": str(train.index.max().date()),
        "time_fold_mode": "expanding",
        "candidate_ranks": ranks.tolist(),
        "minimum_fold_train_observations": int(
            factor_cfg.get("minimum_fold_train_observations", 26)
        ),
        "minimum_fold_train_fraction": float(
            factor_cfg.get("minimum_fold_train_fraction", 0.25)
        ),
        "minimum_fold_stocks": minimum_fold_stocks,
        "minimum_eligible_stocks_across_completed_folds": (
            min(int(item["eligible_stocks"]) for item in completed_diagnostics)
            if completed_diagnostics
            else 0
        ),
        "maximum_eligible_stocks_across_completed_folds": (
            max(int(item["eligible_stocks"]) for item in completed_diagnostics)
            if completed_diagnostics
            else 0
        ),
    }
    write_json(output, summary)
    LOGGER.info(
        "Selected %d PCA factors using %s after %d/%d expanding-time folds and "
        "fold-specific stock eligibility",
        summary["selected_rank"],
        metric_name,
        summary["time_folds_completed"],
        summary["time_folds_requested"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select PCA rank with expanding-time forecast CV and held-stock bi-cross-validation."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    select_factor_rank(config, paths, force=args.force)


if __name__ == "__main__":
    main()
