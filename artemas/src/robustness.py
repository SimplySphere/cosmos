from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import evaluate_project
from src.model.train import train_model
from src.simulation.engine import simulate_ordinary
from src.utils.config import load_project
from src.utils.files import ensure_directories, write_json


def _seed_paths(paths: dict, seed: int) -> dict:
    updated = deepcopy(paths)
    base = Path(paths["artifacts"]["robustness_dir"]) / f"seed_{seed}"
    model_dir = base / "models"
    simulation_dir = base / "simulations"
    evaluation_dir = base / "evaluation"
    updated["artifacts"].update(
        {
            "development_checkpoint": model_dir / "development_checkpoint.pt",
            "final_checkpoint": model_dir / "final_checkpoint.pt",
            "training_history": model_dir / "training_history.csv",
            "simulations_dir": simulation_dir,
            "simulation_sanity": simulation_dir / "sanity_summary.json",
            "evaluation_dir": evaluation_dir,
            "rolling_evaluation": evaluation_dir / "rolling_one_step_evaluation.csv",
            "rolling_summary": evaluation_dir / "rolling_summary.json",
            "comparison_uncertainty": evaluation_dir / "paired_metric_differences.csv",
        }
    )
    return updated


def run_robustness(config: dict, paths: dict, force: bool = False) -> None:
    seeds = [int(value) for value in config["evaluation"].get("robustness_seeds", [])]
    if len(seeds) < 2:
        raise ValueError("evaluation.robustness_seeds must contain at least two seeds")
    horizon = int(config["benchmarks"]["horizon_weeks"])
    path_count = int(config["benchmarks"]["paths"])
    rows: list[dict] = []
    for seed in seeds:
        local_config = deepcopy(config)
        local_config["project"]["seed"] = seed
        local_paths = _seed_paths(paths, seed)
        ensure_directories(local_paths)
        train_model(local_config, local_paths, "development", force=force)
        train_model(local_config, local_paths, "final", force=force)
        simulate_ordinary(local_config, local_paths, horizon, path_count, force=force)
        evaluate_project(local_config, local_paths, force=force)
        frozen = json.loads((Path(local_paths["artifacts"]["evaluation_dir"]) / "summary.json").read_text())
        rolling = json.loads(Path(local_paths["artifacts"]["rolling_summary"]).read_text())
        rows.append({"seed": seed, "evaluation": "frozen", **frozen["transformer"]})
        rows.append({"seed": seed, "evaluation": "rolling", **rolling["transformer"]})
    output_dir = Path(paths["artifacts"]["robustness_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "seed_results.csv", index=False)
    numeric = frame.select_dtypes(include="number").columns.difference(["seed"])
    summary = {}
    for evaluation, group in frame.groupby("evaluation"):
        summary[evaluation] = {
            column: {"mean": float(group[column].mean()), "std": float(group[column].std(ddof=1))}
            for column in numeric if column in group and group[column].notna().any()
        }
    write_json(output_dir / "seed_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat transformer training and evaluation across configured seeds.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    run_robustness(config, paths, force=args.force)


if __name__ == "__main__":
    main()
