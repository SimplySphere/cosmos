from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_project
from src.utils.files import load_npz, read_frame, write_json

PIPELINE_SCHEMA_VERSION = 10


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def _date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(root.resolve()))
    except Exception:
        return str(resolved)


def _temporal_leakage_audit(config: dict, paths: dict) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(paths["project_root"])
    train_end = pd.Timestamp(config["data"]["train_end"])
    model_selection_end = pd.Timestamp(config["data"]["model_selection_end"])
    validation_end = pd.Timestamp(config["data"]["validation_end"])
    membership_cutoff = pd.Timestamp(config["data"][str(config["universe"].get("membership_cutoff", "train_end"))])
    test_start = pd.Timestamp(config["data"]["test_start"])
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        artifact: str | Path,
        field: str,
        value: Any,
        allowed_max: pd.Timestamp | None,
        interpretation: str,
        required: bool = True,
    ) -> None:
        parsed = _date(value)
        passed = parsed is not None and (allowed_max is None or parsed <= allowed_max)
        if parsed is None and not required:
            passed = True
        rows.append(
            {
                "category": category,
                "artifact": _relative(artifact, root),
                "field": field,
                "value": None if parsed is None else str(parsed.date()),
                "allowed_max": None if allowed_max is None else str(allowed_max.date()),
                "passed": bool(passed),
                "interpretation": interpretation,
            }
        )

    selected_path = Path(paths["artifacts"]["selected_rank"])
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        add(
            "factor_selection",
            selected_path,
            "training_end",
            selected.get("training_end"),
            train_end,
            "Rank selection must use returns no later than the development-training cutoff.",
        )

    for phase, key, allowed in (
        ("development", "development_pca", train_end),
        ("final", "final_pca", validation_end),
    ):
        pca_path = Path(paths["artifacts"][key])
        if pca_path.exists():
            payload = load_npz(pca_path)
            add(
                f"{phase}_pca",
                pca_path,
                "fit_end",
                payload.get("fit_end"),
                allowed,
                f"{phase.title()} PCA means, scales, components, and residuals must be fitted by {allowed.date()}.",
            )

    for phase, key, allowed in (
        ("development", "development_checkpoint", train_end),
        ("final", "final_checkpoint", validation_end),
    ):
        checkpoint_path = Path(paths["artifacts"][key])
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            add(
                f"{phase}_model",
                checkpoint_path,
                "fit_end",
                checkpoint.get("fit_end"),
                allowed,
                f"{phase.title()} factor normalization and training targets must stop by {allowed.date()}.",
            )
            add(
                f"{phase}_model",
                checkpoint_path,
                "train_target_end",
                checkpoint.get("train_target_end"),
                allowed,
                f"The latest target used to optimize the {phase} model must not exceed its cutoff.",
            )
            if phase == "development":
                add(
                    "development_validation",
                    checkpoint_path,
                    "validation_target_end",
                    checkpoint.get("validation_target_end"),
                    model_selection_end,
                    "Checkpoint selection must stop by model_selection_end; later 2025 weeks are reserved for calibration.",
                )
                add(
                    "development_calibration",
                    checkpoint_path,
                    "calibration_target_end",
                    checkpoint.get("calibration_target_end"),
                    validation_end,
                    "Dispersion calibration may use the later 2025 holdout but not 2026.",
                )
                calibration_start = _date(checkpoint.get("calibration_target_start"))
                rows.append(
                    {
                        "category": "development_calibration",
                        "artifact": _relative(checkpoint_path, root),
                        "field": "calibration_target_start",
                        "value": None if calibration_start is None else str(calibration_start.date()),
                        "allowed_max": f">{model_selection_end.date()}",
                        "passed": bool(calibration_start is not None and calibration_start > model_selection_end),
                        "interpretation": "Calibration observations must be strictly later than the model-selection block.",
                    }
                )

    preprocessing_json = Path(paths["data"]["preprocessing_report"]).with_suffix(".json")
    if preprocessing_json.exists():
        payload = json.loads(preprocessing_json.read_text(encoding="utf-8"))
        add(
            "security_eligibility",
            preprocessing_json,
            "eligibility_end",
            payload.get("eligibility_end"),
            train_end,
            "Security inclusion and coverage filters must be frozen by train_end, before the 2025 validation year.",
        )
        rows.append(
            {
                "category": "security_eligibility",
                "artifact": _relative(preprocessing_json, root),
                "field": "eligibility_uses_post_train_end_data",
                "value": bool(payload.get("eligibility_uses_post_train_end_data", True)),
                "allowed_max": False,
                "passed": payload.get("eligibility_uses_post_train_end_data") is False,
                "interpretation": "2025/2026 returns may be retained as targets but must not determine fitted-panel eligibility.",
            }
        )

    simulation_dir = Path(paths["artifacts"]["simulations_dir"])
    if simulation_dir.exists():
        for simulation_path in sorted(simulation_dir.glob("ordinary_h*.npz")):
            payload = load_npz(simulation_path)
            add(
                "frozen_simulation",
                simulation_path,
                "forecast_origin",
                payload.get("forecast_origin"),
                validation_end,
                "Every frozen Monte Carlo rollout must begin from a context ending no later than 2025-12-31.",
            )
            add(
                "frozen_simulation",
                simulation_path,
                "model_fit_end",
                payload.get("model_fit_end"),
                validation_end,
                "The checkpoint used for frozen simulation must be fitted no later than 2025-12-31.",
            )

    benchmark_dir = Path(paths["artifacts"]["benchmarks_dir"])
    if benchmark_dir.exists():
        for benchmark_path in sorted(benchmark_dir.glob("*_h*.npz")):
            payload = load_npz(benchmark_path)
            add(
                "benchmark",
                benchmark_path,
                "fit_end",
                payload.get("fit_end"),
                validation_end,
                "Benchmark parameters and historical samples must stop by the same 2025 cutoff.",
            )

    actual_path = Path(paths["artifacts"]["evaluation_dir"]) / "observed_yahoo_basket_returns.csv"
    if actual_path.exists():
        actual = pd.read_csv(actual_path, parse_dates=["date"])
        tested = actual[
            (actual["date"] >= test_start)
            & actual.get("complete_date", True).astype(bool)
            & actual.get("passes_observation_threshold", True).astype(bool)
        ]
        first = tested["date"].min() if not tested.empty else None
        rows.append(
            {
                "category": "evaluation",
                "artifact": _relative(actual_path, root),
                "field": "first_test_observation",
                "value": None if pd.isna(first) else str(pd.Timestamp(first).date()),
                "allowed_max": f">={test_start.date()}",
                "passed": first is not None and pd.Timestamp(first) >= test_start,
                "interpretation": "Observed 2026 returns are used only as evaluation targets.",
            }
        )

    universe_source = Path(paths["data"]["universe_source"])
    snapshot = None
    if universe_source.exists():
        universe = pd.read_csv(universe_source)
        snapshot = pd.to_datetime(universe.get("snapshot_date"), errors="coerce").max()
        rows.append(
            {
                "category": "membership_selection",
                "artifact": _relative(universe_source, root),
                "field": "snapshot_date",
                "value": None if pd.isna(snapshot) else str(pd.Timestamp(snapshot).date()),
                "allowed_max": str(membership_cutoff.date()),
                "passed": bool(pd.notna(snapshot) and pd.Timestamp(snapshot) <= membership_cutoff),
                "interpretation": (
                    "A later selector snapshot leaks future membership. Selector-source weights are discarded before modeling and evaluation."
                ),
            }
        )

    frame = pd.DataFrame(rows)
    price_categories = {
        "factor_selection",
        "development_pca",
        "final_pca",
        "development_model",
        "development_validation",
        "development_calibration",
        "final_model",
        "security_eligibility",
        "frozen_simulation",
        "benchmark",
    }
    price_rows = frame[frame["category"].isin(price_categories)] if not frame.empty else frame
    price_leakage = bool((~price_rows["passed"].astype(bool)).any()) if not price_rows.empty else True
    membership_rows = frame[frame["category"] == "membership_selection"] if not frame.empty else frame
    membership_leakage = bool((~membership_rows["passed"].astype(bool)).any()) if not membership_rows.empty else True
    membership_mode = str(config["universe"].get("membership_mode", "point_in_time"))
    claim_scope = (
        "point_in_time_validation"
        if not membership_leakage
        else "retrospective_exploration"
    )
    summary = {
        "train_end": str(train_end.date()),
        "model_selection_end": str(model_selection_end.date()),
        "validation_end": str(validation_end.date()),
        "membership_cutoff": str(membership_cutoff.date()),
        "test_start": str(test_start.date()),
        "membership_mode": membership_mode,
        "claim_scope": claim_scope,
        "prospective_membership_valid": not membership_leakage,
        "post_2025_return_or_parameter_leakage_detected": price_leakage,
        "future_membership_leakage_detected": membership_leakage,
        "future_membership_or_weight_leakage_detected": membership_leakage,
        "selector_source_weights_used_in_modeling": False,
        "rolling_evaluation_uses_prior_realized_2026_context": True,
        "rolling_context_interpretation": (
            "This is valid one-step rolling evaluation: each week may use earlier 2026 observations, "
            "but the PCA and transformer weights remain frozen at the 2025 cutoff."
        ),
        "frozen_origin_interpretation": (
            "Frozen-origin simulations must use only the final 104-week context ending by 2025-12-31."
        ),
        "overall": (
            "failed"
            if price_leakage
            else (
                "retrospective_membership_disclosed"
                if membership_leakage and membership_mode == "retrospective_disclosed"
                else (
                    "membership_leakage_only"
                    if membership_leakage
                    else "passed"
                )
            )
        ),
    }
    return summary, frame


def build_audit(config: dict, paths: dict) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    report: dict[str, Any] = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "checks": {},
        "warnings": warnings,
        "failures": failures,
    }

    leakage_summary, leakage_rows = _temporal_leakage_audit(config, paths)
    report["temporal_leakage_audit"] = leakage_summary
    leakage_json = Path(paths["artifacts"].get("leakage_audit", Path(paths["artifacts"]["evaluation_dir"]) / "leakage_audit.json"))
    leakage_csv = Path(paths["artifacts"].get("leakage_audit_table", Path(paths["artifacts"]["evaluation_dir"]) / "leakage_audit.csv"))
    write_json(leakage_json, leakage_summary)
    leakage_csv.parent.mkdir(parents=True, exist_ok=True)
    leakage_rows.to_csv(leakage_csv, index=False)
    if leakage_summary["post_2025_return_or_parameter_leakage_detected"]:
        failures.append("post_2025_return_or_parameter_leakage_detected")
    if leakage_summary["future_membership_leakage_detected"]:
        if str(config["universe"].get("membership_mode")) == "point_in_time":
            failures.append("universe_membership_snapshot_after_point_in_time_cutoff")
        else:
            warnings.append("universe_membership_snapshot_after_configured_cutoff")

    universe_source = Path(paths["data"]["universe_source"])
    prepared = Path(paths["data"]["prepared_universe"])
    if universe_source.exists():
        universe = pd.read_csv(universe_source)
        snapshots = pd.to_datetime(universe.get("snapshot_date"), errors="coerce")
        snapshot = snapshots.max()
        report["universe"] = {
            "source_rows": int(len(universe)),
            "snapshot_date": None if pd.isna(snapshot) else str(snapshot.date()),
            "membership_mode": config["universe"].get("membership_mode"),
            "claim_scope": leakage_summary.get("claim_scope"),
            "prospective_membership_valid": leakage_summary.get("prospective_membership_valid"),
        }
    universe_report_path = Path(paths["data"].get("universe_report", ""))
    if universe_report_path.exists():
        try:
            resolver_report = json.loads(universe_report_path.read_text(encoding="utf-8"))
            report.setdefault("universe", {})["verified_yahoo_count_fraction"] = resolver_report.get(
                "verified_count_fraction"
            )
            if "resolver_seed" in resolver_report:
                report["universe"]["resolver_seed"] = resolver_report.get("resolver_seed")
        except Exception as exc:
            warnings.append(f"universe_report_read_failed:{type(exc).__name__}")

    if prepared.exists():
        try:
            prepared_frame = read_frame(prepared)
        except Exception as exc:
            warnings.append(f"prepared_universe_read_failed:{type(exc).__name__}")
        else:
            report.setdefault("universe", {})["prepared_rows"] = int(len(prepared_frame))
            if "currency" in prepared_frame:
                known = prepared_frame["currency"].fillna("").astype(str).str.strip().ne("")
                report["universe"]["known_currency_count_fraction"] = float(known.mean())
            if "currency_source" in prepared_frame:
                report["universe"]["currency_source_counts"] = {
                    str(key): int(value)
                    for key, value in prepared_frame["currency_source"].value_counts(dropna=False).items()
                }
            for field in ("country", "sector", "industry", "exchange"):
                source_column = f"{field}_source"
                if source_column in prepared_frame:
                    report["universe"][f"{field}_source_counts"] = {
                        str(key): int(value)
                        for key, value in prepared_frame[source_column].value_counts(dropna=False).items()
                    }
            selector_weight_columns = sorted(
                column
                for column in ("index_weight", "selection_source_weight", "source_weight", "position_market_value", "shares")
                if column in prepared_frame.columns
            )
            report["universe"]["selector_weight_columns_present_in_prepared_data"] = selector_weight_columns
            if selector_weight_columns:
                failures.append("selector_weight_leaked_into_prepared_universe")
            report["universe"]["market_aggregation_method"] = config["market"]["aggregation_method"]

    aggregation_path = Path(paths["data"]["market_aggregation_weights"])
    if aggregation_path.exists():
        try:
            aggregation = read_frame(aggregation_path)
        except Exception as exc:
            warnings.append(f"market_aggregation_read_failed:{type(exc).__name__}")
        else:
            values = pd.to_numeric(aggregation.get("aggregation_weight"), errors="coerce")
            methods = set(aggregation.get("aggregation_method", pd.Series(dtype=str)).astype(str))
            report["market_aggregation"] = {
                "method": config["market"]["aggregation_method"],
                "rows": int(len(aggregation)),
                "weight_sum": float(values.sum()),
                "minimum_weight": float(values.min()),
                "maximum_weight": float(values.max()),
                "selector_weight_columns_present": sorted(
                    column
                    for column in ("index_weight", "selection_source_weight", "source_weight", "position_market_value", "shares")
                    if column in aggregation.columns
                ),
                "methods_in_artifact": sorted(methods),
            }
            if any(
                column in aggregation.columns
                for column in ("index_weight", "selection_source_weight", "source_weight", "position_market_value", "shares")
            ):
                failures.append("selector_weight_leaked_into_market_aggregation")
            if methods != {"equal_weight"}:
                failures.append("market_aggregation_not_equal_weight")

    returns_path = Path(paths["data"]["weekly_returns"])
    if returns_path.exists():
        try:
            returns = read_frame(returns_path)
        except Exception as exc:
            warnings.append(f"weekly_returns_read_failed:{type(exc).__name__}")
        else:
            report["data"] = {
                "weeks": int(returns.shape[0]),
                "stocks": int(returns.shape[1]),
                "start": str(pd.DatetimeIndex(returns.index).min().date()),
                "end": str(pd.DatetimeIndex(returns.index).max().date()),
                "missing_fraction": float(returns.isna().mean().mean()),
                "base_currency": config["data"].get("base_currency"),
            }

    selected_rank = Path(paths["artifacts"]["selected_rank"])
    if selected_rank.exists():
        report["factors"] = json.loads(selected_rank.read_text(encoding="utf-8"))

    checkpoint_path = Path(paths["artifacts"]["final_checkpoint"])
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        shape = checkpoint.get("shape", {})
        report["model"] = {
            "parameter_count": int(checkpoint.get("parameter_count", -1)),
            "rank": int(shape.get("factor_count", checkpoint.get("factor_count", -1))),
            "calibration_scale": float(checkpoint.get("calibration_scale", 1.0)),
            "phase": checkpoint.get("phase", "final"),
            "fit_end": checkpoint.get("fit_end"),
            "train_target_end": checkpoint.get("train_target_end"),
            "device_used": checkpoint.get("device_used"),
        }

    sanity_path = Path(paths["artifacts"]["simulation_sanity"])
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        report["simulation_sanity"] = sanity
        for horizon, values in sanity.items():
            for warning in values.get("warnings", []):
                warnings.append(f"simulation_h{horizon}:{warning}")


    frozen_path = Path(paths["artifacts"]["evaluation_dir"]) / "summary.json"
    rolling_path = Path(paths["artifacts"]["rolling_summary"])
    if frozen_path.exists():
        frozen_evaluation = json.loads(frozen_path.read_text(encoding="utf-8"))
        report["frozen_evaluation"] = frozen_evaluation
        transformer_summary = frozen_evaluation.get("transformer", {})
        if transformer_summary.get("calendar_alignment") != "forecast_date_join":
            failures.append("frozen_evaluation_not_calendar_aligned")
    if rolling_path.exists():
        report["rolling_evaluation"] = json.loads(rolling_path.read_text(encoding="utf-8"))
    comparison = Path(paths["artifacts"]["comparison_uncertainty"])
    if comparison.exists():
        report["paired_comparisons"] = pd.read_csv(comparison).to_dict("records")

    required = [
        paths["data"]["weekly_returns"],
        paths["artifacts"]["selected_rank"],
        paths["artifacts"]["final_checkpoint"],
        paths["artifacts"]["simulation_sanity"],
        paths["artifacts"]["rolling_summary"],
    ]
    for item in required:
        if not _exists(item):
            failures.append(
                f"missing_required_artifact:{_relative(item, Path(paths['project_root']))}"
            )
    report["status"] = "failed" if failures else ("warning" if warnings else "passed")
    return report


def run_audit(config: dict, paths: dict) -> dict[str, Any]:
    report = build_audit(config, paths)
    write_json(paths["artifacts"]["audit_report"], report)
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit scientific, temporal, and artifact integrity of a completed run."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    report = run_audit(config, paths)
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
