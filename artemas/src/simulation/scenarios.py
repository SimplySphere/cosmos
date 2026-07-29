from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from src.utils.config import load_project
from src.utils.files import ensure_parent, load_npz, save_npz

LOGGER = logging.getLogger(__name__)


def _max_drawdown(index_paths: np.ndarray) -> np.ndarray:
    initial = np.ones((index_paths.shape[0], 1), dtype=index_paths.dtype)
    running_max = np.maximum.accumulate(
        np.concatenate([initial, index_paths], axis=1), axis=1
    )[:, 1:]
    drawdowns = index_paths / np.maximum(running_max, 1e-12) - 1.0
    return drawdowns.min(axis=1)


def _recovery_fraction(index_paths: np.ndarray) -> np.ndarray:
    terminal = index_paths[:, -1]
    trough = index_paths.min(axis=1)
    loss = np.maximum(1.0 - trough, 1e-8)
    return np.clip((terminal - trough) / loss, 0.0, 5.0)


def _safe_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def cluster_scenarios(config: dict, paths: dict, force: bool = False) -> None:
    horizon = int(config["visualization"]["default_horizon_weeks"])
    source = Path(paths["artifacts"]["simulations_dir"]) / f"ordinary_h{horizon:03d}.npz"
    summary_path = Path(paths["artifacts"]["simulations_dir"]) / f"scenario_summary_h{horizon:03d}.csv"
    labels_path = Path(paths["artifacts"]["simulations_dir"]) / f"scenario_labels_h{horizon:03d}.npz"
    if summary_path.exists() and labels_path.exists() and not force:
        LOGGER.info("Using existing scenario clusters")
        return

    payload = load_npz(source)
    index_paths = np.asarray(payload["group_index_paths"], dtype=np.float64)
    weekly_returns = np.asarray(payload["group_weekly_returns"], dtype=np.float64)
    group_names = [str(value) for value in payload["group_names"]]
    global_paths = index_paths[:, :, 0]
    global_returns = weekly_returns[:, :, 0]
    terminal = global_paths[:, -1] - 1.0
    volatility = global_returns.std(axis=1) * np.sqrt(52.0)
    drawdown = _max_drawdown(global_paths)
    worst_week = global_returns.min(axis=1)
    best_week = global_returns.max(axis=1)
    recovery = _recovery_fraction(global_paths)

    scenario_cfg = config.get("scenarios", {})
    country_indices = [index for index, name in enumerate(group_names) if name.startswith("Country: ")][
        : int(scenario_cfg.get("top_country_groups", 6))
    ]
    sector_indices = [index for index, name in enumerate(group_names) if name.startswith("Sector: ")][
        : int(scenario_cfg.get("top_sector_groups", 6))
    ]
    selected_group_indices = country_indices + sector_indices
    selected_terminal = (
        index_paths[:, -1, selected_group_indices] - 1.0
        if selected_group_indices
        else np.empty((len(terminal), 0), dtype=np.float64)
    )
    features = np.column_stack(
        [terminal, volatility, drawdown, worst_week, best_week, recovery, selected_terminal]
    )

    lower_q = float(scenario_cfg.get("winsor_lower_quantile", 0.005))
    upper_q = float(scenario_cfg.get("winsor_upper_quantile", 0.995))
    lower = np.quantile(features, lower_q, axis=0)
    upper = np.quantile(features, upper_q, axis=0)
    winsorized = np.clip(features, lower, upper)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(winsorized)
    distances = np.sqrt(np.sum(scaled * scaled, axis=1))
    outlier_threshold = float(np.quantile(distances, float(scenario_cfg.get("outlier_quantile", 0.995))))
    inlier_mask = distances <= outlier_threshold
    inlier_indices = np.flatnonzero(inlier_mask)
    outlier_indices = np.flatnonzero(~inlier_mask)

    n_clusters = min(int(config["simulation"]["scenario_clusters"]), len(inlier_indices))
    model = KMeans(
        n_clusters=n_clusters,
        n_init=30,
        random_state=int(config["project"]["seed"]),
    )
    inlier_labels = model.fit_predict(scaled[inlier_indices])
    labels = np.full(len(features), -1, dtype=np.int16)
    labels[inlier_indices] = inlier_labels.astype(np.int16)

    rows: list[dict] = []
    representatives: list[int] = []
    for cluster in range(n_clusters):
        members = inlier_indices[inlier_labels == cluster]
        center = model.cluster_centers_[cluster]
        representative = int(
            members[np.argmin(np.sum((scaled[members] - center) ** 2, axis=1))]
        )
        representatives.append(representative)
        row = {
            "scenario_id": cluster,
            "scenario_type": "cluster",
            "probability": len(members) / len(labels),
            "path_count": len(members),
            "median_terminal_return": float(np.median(terminal[members])),
            "q10_terminal_return": float(np.quantile(terminal[members], 0.10)),
            "q90_terminal_return": float(np.quantile(terminal[members], 0.90)),
            "median_annualized_volatility": float(np.median(volatility[members])),
            "median_max_drawdown": float(np.median(drawdown[members])),
            "median_worst_week": float(np.median(worst_week[members])),
            "median_recovery_fraction": float(np.median(recovery[members])),
            "representative_path": representative,
            "label": "Interpret from quantified global, country, and sector behavior",
        }
        for group_index in selected_group_indices:
            row[f"median_{_safe_column(group_names[group_index])}_return"] = float(
                np.median(index_paths[members, -1, group_index] - 1.0)
            )
        rows.append(row)

    if len(outlier_indices):
        representative = int(outlier_indices[np.argmax(distances[outlier_indices])])
        rows.append(
            {
                "scenario_id": -1,
                "scenario_type": "robust_outlier_set",
                "probability": len(outlier_indices) / len(labels),
                "path_count": len(outlier_indices),
                "median_terminal_return": float(np.median(terminal[outlier_indices])),
                "q10_terminal_return": float(np.quantile(terminal[outlier_indices], 0.10)),
                "q90_terminal_return": float(np.quantile(terminal[outlier_indices], 0.90)),
                "median_annualized_volatility": float(np.median(volatility[outlier_indices])),
                "median_max_drawdown": float(np.median(drawdown[outlier_indices])),
                "median_worst_week": float(np.median(worst_week[outlier_indices])),
                "median_recovery_fraction": float(np.median(recovery[outlier_indices])),
                "representative_path": representative,
                "label": "Paths beyond the robust feature-distance threshold; inspect as model-risk tails",
            }
        )

    pd.DataFrame(rows).sort_values("probability", ascending=False).to_csv(
        ensure_parent(summary_path), index=False
    )
    save_npz(
        labels_path,
        labels=labels,
        representative_paths=np.asarray(representatives, dtype=np.int32),
        selected_group_names=np.asarray(
            [group_names[index] for index in selected_group_indices], dtype="U"
        ),
        outlier_threshold=np.asarray([outlier_threshold], dtype=np.float32),
    )
    LOGGER.info(
        "Created %d robust scenario clusters and isolated %d outlier paths",
        n_clusters,
        len(outlier_indices),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Monte Carlo paths into robust scenario branches.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    cluster_scenarios(config, paths, force=args.force)


if __name__ == "__main__":
    main()
