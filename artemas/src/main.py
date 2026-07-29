from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.audit import PIPELINE_SCHEMA_VERSION, run_audit
from src.data.download import download_fx_rates, download_prices
from src.data.preprocess import create_smoke_dataset, preprocess_market
from src.data.universe import prepare_universe
from src.evaluation.benchmarks import run_benchmarks
from src.evaluation.evaluate import evaluate_project
from src.factors.pca import fit_pca
from src.factors.selection import select_factor_rank
from src.model.train import train_model
from src.simulation.engine import run_simulations
from src.simulation.scenarios import cluster_scenarios
from src.utils.config import apply_smoke_overrides, apply_smoke_paths, load_project
from src.utils.files import bootstrap_workspace, load_npz, write_json
from src.utils.provenance import (
    finish_run_manifest,
    snapshot_configuration,
    start_run_manifest,
    update_run_manifest,
)
from src.utils.runtime import configure_logging, seed_everything
from src.utils.selfcheck import run_self_checks
from src.visualization.figures import create_visuals

LOGGER = logging.getLogger(__name__)
STAGES = ["data", "factors", "train", "simulate", "benchmarks", "evaluate", "visuals"]


def _selected_stages(start: str, end: str) -> list[str]:
    start_index = STAGES.index(start)
    end_index = STAGES.index(end)
    if end_index < start_index:
        raise ValueError("--to-stage must not come before --from-stage")
    return STAGES[start_index : end_index + 1]


def run_pipeline(
    config: dict,
    paths: dict,
    mode: str,
    stages: list[str],
    force: bool,
    skip_download: bool,
    manifest: dict | None = None,
) -> None:
    smoke = mode == "smoke"
    effective_force = force or smoke
    seed_everything(
        int(config["project"]["seed"]),
        strict=bool(config["project"].get("strict_reproducibility", True)),
    )

    for stage in stages:
        LOGGER.info("Starting stage: %s", stage)
        if manifest is not None:
            manifest["stage_status"][stage] = "running"
            update_run_manifest(paths, manifest)
        try:
            if stage == "data":
                if smoke:
                    create_smoke_dataset(config, paths, force=True)
                else:
                    # --force rebuilds derived outputs but intentionally preserves
                    # expensive, resumable network caches.  Missing raw/reference data
                    # are still acquired automatically.
                    prepare_universe(config, paths, force=effective_force)
                    if not skip_download:
                        download_prices(config, paths, force=False)
                        if bool(config["data"].get("require_common_currency", True)):
                            download_fx_rates(config, paths, force=False)
                    preprocess_market(config, paths, force=effective_force)

            elif stage == "factors":
                select_factor_rank(config, paths, force=effective_force)
                fit_pca(config, paths, "development", force=effective_force)

            elif stage == "train":
                train_model(config, paths, "development", force=effective_force)
                fit_pca(config, paths, "final", force=effective_force)
                train_model(config, paths, "final", force=effective_force)

            elif stage == "simulate":
                run_simulations(config, paths, force=effective_force)
                cluster_scenarios(config, paths, force=effective_force)

            elif stage == "benchmarks":
                run_benchmarks(config, paths, force=effective_force)

            elif stage == "evaluate":
                evaluate_project(config, paths, force=effective_force)

            elif stage == "visuals":
                create_visuals(config, paths, force=effective_force)
        except Exception:
            if manifest is not None:
                manifest["stage_status"][stage] = "failed"
                update_run_manifest(paths, manifest)
            raise
        if manifest is not None:
            manifest["stage_status"][stage] = "completed"
            update_run_manifest(paths, manifest)
        LOGGER.info("Completed stage: %s", stage)


def _validate_smoke_outputs(config: dict, paths: dict, stages: list[str]) -> None:
    selected = set(stages)

    def require(path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.exists():
            raise RuntimeError(f"Smoke validation expected an output that was not created: {resolved}")
        return resolved

    if "data" in selected:
        require(paths["data"]["weekly_returns"])
        aggregation_path = require(paths["data"]["market_aggregation_weights"])
        require(paths["data"]["weekly_observed_universe"])
        from src.utils.files import read_frame
        aggregation = read_frame(aggregation_path)
        if "index_weight" in aggregation.columns:
            raise RuntimeError("Smoke aggregation leaked selector-source index_weight")
        if set(aggregation.get("aggregation_method", [])) != {"equal_weight"}:
            raise RuntimeError("Smoke aggregation method is not equal_weight")
        values = aggregation["aggregation_weight"].to_numpy(dtype=float)
        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise RuntimeError("Smoke aggregation weights do not sum to one")
        if not np.allclose(values, np.full_like(values, 1.0 / len(values)), atol=1e-12):
            raise RuntimeError("Smoke aggregation weights are not equal")
    if "factors" in selected:
        require(paths["artifacts"]["selected_rank"])
        require(paths["artifacts"]["factor_cv_results"])
        require(paths["artifacts"]["factor_cv_folds"])
        require(paths["artifacts"]["development_pca"])
        require(paths["artifacts"]["development_factors"])
        require(paths["artifacts"]["development_factor_quality"])
    if "train" in selected:
        require(paths["artifacts"]["development_checkpoint"])
        require(paths["artifacts"]["final_checkpoint"])
        require(paths["artifacts"]["final_pca"])
        require(paths["artifacts"]["final_factors"])
    if "simulate" in selected:
        horizon = int(config["smoke"]["horizon_weeks"])
        expected_paths = int(config["smoke"]["paths"])
        ordinary = load_npz(
            require(Path(paths["artifacts"]["simulations_dir"]) / f"ordinary_h{horizon:03d}.npz")
        )
        index_paths = ordinary["group_index_paths"]
        if index_paths.shape[0] != expected_paths or index_paths.shape[1] != horizon:
            raise RuntimeError(
                f"Smoke simulation has shape {index_paths.shape}; expected "
                f"[{expected_paths}, {horizon}, groups]"
            )
        if not np.isfinite(index_paths).all() or np.any(index_paths <= 0):
            raise RuntimeError("Smoke simulation contains invalid market-index values")
        require(paths["artifacts"]["simulation_sanity"])
    if "benchmarks" in selected:
        horizon = int(config["smoke"]["horizon_weeks"])
        for name in ("zero_return", "block_bootstrap", "linear_factor", "gaussian_factor"):
            require(Path(paths["artifacts"]["benchmarks_dir"]) / f"{name}_h{horizon:03d}.npz")
    if "evaluate" in selected:
        summary_path = require(Path(paths["artifacts"]["evaluation_dir"]) / "summary.json")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        expected = {
            "transformer",
            "zero_return",
            "block_bootstrap",
            "linear_factor",
            "gaussian_factor",
        }
        if set(summary) != expected:
            raise RuntimeError(
                f"Smoke evaluation methods were {sorted(summary)}; expected {sorted(expected)}"
            )
        require(paths["artifacts"]["rolling_evaluation"])
        require(paths["artifacts"]["rolling_summary"])
        require(paths["artifacts"]["comparison_uncertainty"])
    if "visuals" in selected:
        visual_dir = Path(paths["artifacts"]["visuals_dir"])
        for name in (
            "transformer_generation_process.html",
            "ten_path_rollout_animation.html",
            "global_probability_river.html",
            "global_probability_river.png",
            "benchmark_path_cloud_comparison.html",
            "benchmark_path_cloud_comparison.png",
            "rolling_prediction_vs_actual.html",
            "rolling_prediction_vs_actual.png",
            "notional_100k_forecast_vs_observed.html",
            "notional_100k_forecast_vs_observed.png",
            "factor_selection.png",
            "factor_forecast_selection.png",
            "factor_explained_variance.png",
            "training_history.png",
            "benchmark_scorecard.html",
            "benchmark_scorecard.png",
            "benchmark_summary.csv",
            "uncertainty_by_horizon.html",
            "uncertainty_by_horizon.png",
            "uncertainty_by_horizon.csv",
            "temporal_leakage_timeline.png",
            "visual_manifest.csv",
            "README_VISUALS.md",
        ):
            require(visual_dir / name)

    run_self_checks(config)
    LOGGER.info("Smoke output validation passed")



def _guard_pipeline_schema(paths: dict, mode: str, force: bool) -> None:
    if mode == "smoke" or force:
        return
    schema_path = Path(paths["artifacts"]["pipeline_schema"])
    if schema_path.exists():
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        version = int(payload.get("pipeline_schema_version", -1))
        if version != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Existing artifacts use pipeline schema {version}, but this code requires "
                f"schema {PIPELINE_SCHEMA_VERSION}. Rerun with --force. Use --skip-download "
                "to preserve cached stock prices while rebuilding derived artifacts."
            )
        return
    generated = [
        Path(paths["data"]["weekly_returns"]),
        Path(paths["artifacts"]["final_checkpoint"]),
        Path(paths["artifacts"]["evaluation_dir"]) / "summary.json",
    ]
    if any(path.exists() for path in generated):
        raise RuntimeError(
            "Existing artifacts predate the scientifically revised pipeline and cannot be "
            "safely reused. Run `python -m src.main --mode real --force --skip-download` "
            "after downloading any missing FX series. This preserves cached stock files but "
            "rebuilds preprocessing, factors, models, simulations, and evaluation."
        )


def _write_pipeline_schema(paths: dict) -> None:
    write_json(
        paths["artifacts"]["pipeline_schema"],
        {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "description": "USD Yahoo Finance equity pipeline with selector membership with disclosed descriptive-metadata fallbacks, equal-weight market aggregation, cutoff audits, calibrated forecasts, and demonstrative visuals",
        },
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ARTEMAS pipeline or its synthetic smoke check."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--mode", choices=["real", "smoke"], default="real")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild derived data, factors, models, simulations, evaluation, and visuals "
            "while preserving resumable Vanguard/Yahoo raw caches."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already cached stock and FX files during the data stage.",
    )
    args = parser.parse_args()

    config, paths = load_project(args.config, args.paths)
    if args.mode == "smoke":
        config = apply_smoke_overrides(config)
        paths = apply_smoke_paths(paths)
    created_workspace_roots = bootstrap_workspace(paths)
    _guard_pipeline_schema(paths, args.mode, args.force)
    log_name = f"pipeline_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(
        str(config["project"]["log_level"]),
        Path(paths["artifacts"]["logs_dir"]) / log_name,
    )
    if created_workspace_roots:
        LOGGER.info(
            "Bootstrapped missing generated workspace roots: %s",
            ", ".join(str(path) for path in created_workspace_roots),
        )
    stages = _selected_stages(args.from_stage, args.to_stage)
    snapshot_configuration(config, paths, args.config, args.paths)
    manifest = start_run_manifest(config, paths, args.mode, stages, sys.argv)
    LOGGER.info("Mode: %s | stages: %s", args.mode, ", ".join(stages))
    try:
        run_pipeline(
            config,
            paths,
            args.mode,
            stages,
            args.force,
            args.skip_download,
            manifest,
        )
        if args.mode == "smoke":
            _validate_smoke_outputs(config, paths, stages)
        _write_pipeline_schema(paths)
        if args.mode == "real" and "evaluate" in stages:
            run_audit(config, paths)
        finish_run_manifest(paths, manifest, "completed")
    except Exception as exc:
        finish_run_manifest(paths, manifest, "failed", repr(exc))
        raise


if __name__ == "__main__":
    main()
