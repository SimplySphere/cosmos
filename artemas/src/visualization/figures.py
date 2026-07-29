from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation.evaluate import _actual_global_returns, _valid_actual_test
from src.utils.config import load_project
from src.utils.files import load_npz

LOGGER = logging.getLogger(__name__)

ACTUAL_GOLD = "#D4AF37"
MEDIAN_BLUE = "#1F77B4"
ERROR_RED = "#D62728"
INTERVAL_BLUE = "#7DB7E8"
NEUTRAL_GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
SUCCESS_GREEN = "#2CA02C"
DARK = "#111827"
METHOD_COLORS = {
    "transformer": MEDIAN_BLUE,
    "block_bootstrap": "#2CA02C",
    "gaussian_factor": "#FF7F0E",
    "linear_factor": "#9467BD",
    "zero_return": "#6B7280",
}
METHOD_LABELS = {
    "transformer": "Transformer + Monte Carlo",
    "block_bootstrap": "Stationary block bootstrap",
    "gaussian_factor": "Gaussian factor model",
    "linear_factor": "Stable linear factor model",
    "zero_return": "Zero-return baseline",
}


def _read_json(path: str | Path, default: dict | None = None) -> dict:
    source = Path(path)
    if not source.exists():
        return {} if default is None else default
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default
    return payload if isinstance(payload, dict) else ({} if default is None else default)


def _format_date_axis(axis: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        end = start + pd.Timedelta(days=7)
    axis.set_xlim(start, end)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.margins(x=0)


def _forecast_dates(config: dict, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(config["data"]["test_start"]),
        periods=int(horizon),
        freq=str(config["data"]["week_rule"]),
    )


def _origin_and_dates(config: dict, horizon: int) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    dates = _forecast_dates(config, horizon)
    origin = dates[0] - pd.Timedelta(days=7)
    return origin, dates


def _actual_complete(config: dict, paths: dict) -> pd.DataFrame:
    return _valid_actual_test(config, _actual_global_returns(config, paths))


def _actual_index(config: dict, paths: dict, dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Observed cumulative basket path on the forecast calendar.

    Coverage-threshold failures remain in the cumulative path when an observed return
    exists; otherwise later dates would shift backward and no longer match forecast steps.
    """
    frame = _actual_global_returns(config, paths).reindex(dates)
    available = (
        frame["global_return"].notna()
        & frame["complete_date"].fillna(False).astype(bool)
    ).to_numpy(dtype=bool)
    unavailable = np.flatnonzero(~available)
    prefix = int(unavailable[0]) if unavailable.size else len(frame)
    if prefix == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    frame = frame.iloc[:prefix]
    values = np.exp(np.cumsum(frame["global_return"].to_numpy(dtype=float)))
    return pd.DatetimeIndex(frame.index), values


def _prepend_origin(paths_array: np.ndarray, origin_value: float = 1.0) -> np.ndarray:
    values = np.asarray(paths_array, dtype=float)
    return np.concatenate(
        [np.full((values.shape[0], 1), origin_value, dtype=float), values], axis=1
    )


def _path_segments(dates: pd.DatetimeIndex, paths_array: np.ndarray) -> np.ndarray:
    x = mdates.date2num(pd.DatetimeIndex(dates).to_pydatetime())
    x_grid = np.broadcast_to(x[None, :], paths_array.shape)
    return np.stack([x_grid, paths_array], axis=2)


def _add_all_paths_static(
    axis: plt.Axes,
    dates: pd.DatetimeIndex,
    paths_array: np.ndarray,
    color: str,
    *,
    alpha: float | None = None,
    linewidth: float = 0.35,
) -> None:
    count = max(1, len(paths_array))
    if alpha is None:
        alpha = min(0.08, max(0.006, 15.0 / count))
    collection = LineCollection(
        _path_segments(dates, paths_array),
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        rasterized=True,
        zorder=1,
    )
    axis.add_collection(collection)
    axis.autoscale_view()


def _plotly_path_cloud_trace(
    dates: pd.DatetimeIndex,
    paths_array: np.ndarray,
    color: str,
    name: str,
    *,
    opacity: float = 0.10,
    showlegend: bool = True,
) -> go.Scattergl:
    values = np.asarray(paths_array, dtype=np.float32)
    n_paths, n_steps = values.shape
    x_matrix = np.empty((n_paths, n_steps + 1), dtype="datetime64[ns]")
    x_matrix[:, :n_steps] = np.asarray(dates, dtype="datetime64[ns]")[None, :]
    x_matrix[:, n_steps] = np.datetime64("NaT")
    y_matrix = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    y_matrix[:, :n_steps] = values
    y_matrix[:, n_steps] = np.nan
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    return go.Scattergl(
        x=x_matrix.ravel(),
        y=y_matrix.ravel(),
        mode="lines",
        line={"width": 0.55, "color": f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity})"},
        name=name,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def _quantiles(paths_array: np.ndarray) -> np.ndarray:
    return np.quantile(
        np.asarray(paths_array, dtype=float),
        [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975],
        axis=0,
    )


def _simulation_payload(config: dict, paths: dict) -> tuple[int, dict[str, np.ndarray]] | None:
    horizon = int(config["visualization"]["default_horizon_weeks"])
    source = Path(paths["artifacts"]["simulations_dir"]) / f"ordinary_h{horizon:03d}.npz"
    if not source.exists():
        return None
    return horizon, load_npz(source)


def _benchmark_sources(config: dict, paths: dict) -> dict[str, Path]:
    horizon = int(config["benchmarks"]["horizon_weeks"])
    simulation_dir = Path(paths["artifacts"]["simulations_dir"])
    benchmark_dir = Path(paths["artifacts"]["benchmarks_dir"])
    return {
        "transformer": simulation_dir / f"ordinary_h{horizon:03d}.npz",
        "block_bootstrap": benchmark_dir / f"block_bootstrap_h{horizon:03d}.npz",
        "gaussian_factor": benchmark_dir / f"gaussian_factor_h{horizon:03d}.npz",
        "linear_factor": benchmark_dir / f"linear_factor_h{horizon:03d}.npz",
        "zero_return": benchmark_dir / f"zero_return_h{horizon:03d}.npz",
    }


def _frozen_summary(paths: dict) -> dict:
    return _read_json(Path(paths["artifacts"]["evaluation_dir"]) / "summary.json")


def _rolling_summary(paths: dict) -> dict:
    return _read_json(paths["artifacts"]["rolling_summary"])


def _shade_future(axis: plt.Axes, actual_end: pd.Timestamp | None, forecast_end: pd.Timestamp) -> None:
    if actual_end is None or pd.Timestamp(actual_end) >= pd.Timestamp(forecast_end):
        return
    axis.axvspan(actual_end, forecast_end, color=LIGHT_GRAY, alpha=0.35, zorder=0)
    axis.axvline(actual_end, color=NEUTRAL_GRAY, linestyle="--", linewidth=1.2, zorder=4)
    axis.text(
        actual_end,
        0.985,
        f"Latest complete weekly actual: {pd.Timestamp(actual_end).date()}",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=NEUTRAL_GRAY,
    )


def _save_factor_selection(paths: dict, output_dir: Path) -> list[str]:
    source = Path(paths["artifacts"]["factor_cv_results"])
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    if "selected" in frame:
        selected_rows = frame[frame["selected"].astype(str).str.lower().isin({"true", "1", "yes"})]
    else:
        selected_rows = pd.DataFrame()
    if selected_rows.empty:
        selected_rank = int(_read_json(paths["artifacts"]["selected_rank"])["selected_rank"])
    else:
        selected_rank = int(selected_rows.iloc[0]["rank"])

    outputs: list[str] = []
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.plot(frame["rank"], frame["mean_reconstruction_mse"], marker="o", markersize=3, color=MEDIAN_BLUE)
    error = pd.to_numeric(
        frame.get("reconstruction_standard_error", frame.get("standard_error", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    axis.fill_between(
        frame["rank"],
        frame["mean_reconstruction_mse"] - error,
        frame["mean_reconstruction_mse"] + error,
        color=INTERVAL_BLUE,
        alpha=0.22,
        label="±1 standard error",
    )
    axis.axvline(selected_rank, color=ERROR_RED, linestyle="--", label=f"Selected rank: {selected_rank}")
    best_rank = int(frame.loc[frame["mean_reconstruction_mse"].idxmin(), "rank"])
    axis.scatter(
        [best_rank],
        [frame["mean_reconstruction_mse"].min()],
        color=ACTUAL_GOLD,
        edgecolor="black",
        zorder=5,
        label=f"Minimum reconstruction rank: {best_rank}",
    )
    axis.set_xlabel("PCA factor count")
    axis.set_ylabel("Held-out reconstruction MSE")
    axis.set_title("Leakage-Safe Expanding-Fold PCA Reconstruction")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    name = "factor_selection.png"
    figure.savefig(output_dir / name, dpi=220)
    plt.close(figure)
    outputs.append(name)

    if "mean_forecast_mse" in frame:
        figure, axis = plt.subplots(figsize=(10, 5.8))
        axis.plot(frame["rank"], frame["mean_forecast_mse"], marker="o", markersize=3, color=MEDIAN_BLUE)
        forecast_error = pd.to_numeric(frame.get("forecast_standard_error", 0.0), errors="coerce").fillna(0.0)
        axis.fill_between(
            frame["rank"],
            frame["mean_forecast_mse"] - forecast_error,
            frame["mean_forecast_mse"] + forecast_error,
            color=INTERVAL_BLUE,
            alpha=0.18,
            label="±1 standard error",
        )
        axis.axvline(selected_rank, color=ERROR_RED, linestyle="--", label=f"Selected rank: {selected_rank}")
        best_rank = int(frame.loc[frame["mean_forecast_mse"].idxmin(), "rank"])
        axis.scatter(
            [best_rank],
            [frame["mean_forecast_mse"].min()],
            color=ACTUAL_GOLD,
            edgecolor="black",
            zorder=5,
            label=f"Minimum forecast rank: {best_rank}",
        )
        axis.set_xlabel("PCA factor count")
        axis.set_ylabel("One-step factor forecast MSE")
        axis.set_title("Forecast-Oriented PCA Factor-Rank Selection")
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        name = "factor_forecast_selection.png"
        figure.savefig(output_dir / name, dpi=220)
        plt.close(figure)
        outputs.append(name)
    return outputs


def _save_factor_variance(paths: dict, output_dir: Path) -> list[str]:
    source = Path(paths["artifacts"]["final_pca"])
    if not source.exists():
        return []
    payload = load_npz(source)
    explained = np.asarray(payload.get("explained_variance_ratio", []), dtype=float)
    if explained.size == 0:
        return []
    factor_number = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(factor_number, explained * 100, color=MEDIAN_BLUE, alpha=0.75, label="Individual factor")
    axis.plot(factor_number, cumulative * 100, color=ERROR_RED, marker="o", label="Cumulative")
    axis.set_xlabel("PCA factor")
    axis.set_ylabel("Explained standardized variance (%)")
    axis.set_title("Variance Preserved by the Selected Market Factors")
    axis.set_xticks(factor_number)
    axis.legend(loc="best")
    figure.tight_layout()
    name = "factor_explained_variance.png"
    figure.savefig(output_dir / name, dpi=220)
    plt.close(figure)
    return [name]


def _save_training_history(paths: dict, output_dir: Path) -> list[str]:
    source = Path(paths["artifacts"]["training_history"])
    if not source.exists():
        return []
    frame = pd.read_csv(source)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    phase_colors = {"development": MEDIAN_BLUE, "final": SUCCESS_GREEN}
    for phase, group in frame.groupby("phase"):
        color = phase_colors.get(str(phase), NEUTRAL_GRAY)
        axis.plot(group["epoch"], group["training_loss"], color=color, linewidth=2.2, label=f"{phase} training")
        if "validation_loss" in group and group["validation_loss"].notna().any():
            axis.plot(group["epoch"], group["validation_loss"], color=ACTUAL_GOLD, linewidth=2.2, label=f"{phase} validation")
        if "selected_checkpoint" in group:
            selected = group[group["selected_checkpoint"].astype(str).str.lower().isin({"true", "1", "yes"})]
            if not selected.empty:
                y = selected["validation_loss"].fillna(selected["training_loss"])
                axis.scatter(selected["epoch"], y, marker="*", s=180, color=ERROR_RED, edgecolor="black", zorder=5, label=f"{phase} selected")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Negative log-likelihood (lower is better)")
    axis.set_title("Transformer Training, Validation, and Early Stopping")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    name = "training_history.png"
    figure.savefig(output_dir / name, dpi=220)
    plt.close(figure)
    return [name]


def _save_temporal_leakage_timeline(config: dict, paths: dict, output_dir: Path) -> list[str]:
    train_start = pd.Timestamp(config["data"]["start_date"])
    train_end = pd.Timestamp(config["data"]["train_end"])
    validation_end = pd.Timestamp(config["data"]["validation_end"])
    test_start = pd.Timestamp(config["data"]["test_start"])
    actual = _actual_complete(config, paths)
    actual_end = None if actual.empty else pd.Timestamp(actual.index.max())
    universe_path = Path(paths["data"]["universe_source"])
    snapshot = None
    if universe_path.exists():
        universe = pd.read_csv(universe_path)
        snapshot = pd.to_datetime(universe.get("snapshot_date"), errors="coerce").max()
    end = max([value for value in [actual_end, snapshot, validation_end + pd.DateOffset(years=1)] if pd.notna(value)])

    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.hlines(3, train_start, train_end, linewidth=16, color=MEDIAN_BLUE, label="Development fit: returns through 2024")
    model_selection_end = pd.Timestamp(config["data"]["model_selection_end"])
    axis.hlines(2, train_end + pd.Timedelta(days=1), model_selection_end, linewidth=16, color=ACTUAL_GOLD, label="2025 checkpoint selection")
    axis.hlines(1.65, model_selection_end + pd.Timedelta(days=1), validation_end, linewidth=10, color=NEUTRAL_GRAY, label="2025 calibration holdout")
    if actual_end is not None:
        axis.hlines(1, test_start, actual_end, linewidth=16, color=SUCCESS_GREEN, label="Observed 2026 evaluation only")
    axis.hlines(0, test_start, validation_end + pd.DateOffset(years=1), linewidth=6, color=ERROR_RED, alpha=0.45, label="Frozen 52-week forecast")
    axis.axvline(validation_end, color="black", linestyle="--", linewidth=1.5)
    axis.text(validation_end, 3.45, "Model parameters frozen\n2025-12-31", ha="right", va="top", fontsize=9)
    membership_mode = str(config["universe"].get("membership_mode", "point_in_time"))
    cutoff_key = str(config["universe"].get("membership_cutoff", "train_end"))
    membership_cutoff = pd.Timestamp(config["data"][cutoff_key])
    retrospective = bool(pd.notna(snapshot) and pd.Timestamp(snapshot) > membership_cutoff)
    if pd.notna(snapshot):
        axis.axvline(pd.Timestamp(snapshot), color=ERROR_RED, linestyle=":", linewidth=2)
        disclosure = (
            "RETROSPECTIVE MEMBERSHIP\nnot a prospective test"
            if retrospective
            else "point-in-time membership"
        )
        axis.text(
            pd.Timestamp(snapshot),
            3.45,
            f"Stock-list selector snapshot\n{pd.Timestamp(snapshot).date()}\n{disclosure}; weights ignored",
            ha="left",
            va="top",
            color=ERROR_RED if retrospective else SUCCESS_GREEN,
            fontsize=9,
        )
    axis.set_yticks([0, 1, 2, 3], ["Forecast", "Test outcomes", "2025 selection/calibration", "Development fit"])
    _format_date_axis(axis, train_start, end)
    title_scope = (
        "Retrospective Membership Disclosed"
        if retrospective or membership_mode == "retrospective_disclosed"
        else "Point-in-Time Membership"
    )
    axis.set_title(
        f"What Was Known When: {title_scope}, Separate Calibration, Frozen 2026 Returns"
    )
    axis.legend(loc="lower left", fontsize=8, ncol=2)
    figure.tight_layout()
    name = "temporal_leakage_timeline.png"
    figure.savefig(output_dir / name, dpi=220)
    plt.close(figure)
    return [name]


def _save_probability_river(config: dict, paths: dict, output_dir: Path) -> list[str]:
    item = _simulation_payload(config, paths)
    if item is None:
        return []
    horizon, payload = item
    simulated = np.asarray(payload["group_index_paths"][:, :, 0], dtype=float)
    origin, forecast_dates = _origin_and_dates(config, horizon)
    dates = pd.DatetimeIndex([origin, *forecast_dates])
    paths_with_origin = _prepend_origin(simulated)
    quantiles = _quantiles(paths_with_origin)
    actual_dates, actual_values = _actual_index(config, paths, forecast_dates)
    actual_dates_with_origin = pd.DatetimeIndex([origin, *actual_dates])
    actual_with_origin = np.concatenate([[1.0], actual_values])
    observed_steps = len(actual_values)
    median_observed = quantiles[3, 1 : observed_steps + 1]
    gap = actual_values - median_observed
    frozen = _frozen_summary(paths).get("transformer", {})
    final_gap = float(gap[-1]) if len(gap) else float("nan")
    summary_text = (
        f"All {len(simulated):,} saved transformer paths shown<br>"
        f"Observed-period cumulative MAE: {float(frozen.get('cumulative_index_mae', np.nan)):.3%}<br>"
        f"Final actual − median: {final_gap:+.2%}<br>"
        f"Final actual percentile: {float(frozen.get('final_actual_percentile', np.nan)):.1%}"
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.06,
        subplot_titles=("Every Monte Carlo future generated from the frozen end-of-2025 model", "Observed Yahoo basket minus transformer median"),
    )
    figure.add_trace(
        _plotly_path_cloud_trace(dates, paths_with_origin, MEDIAN_BLUE, f"All {len(simulated):,} predicted paths", opacity=0.055),
        row=1,
        col=1,
    )
    for lower, upper, label, dash in [
        (0, 6, "95% boundaries", "dot"),
        (1, 5, "80% boundaries", "dash"),
        (2, 4, "50% boundaries", "solid"),
    ]:
        figure.add_trace(go.Scatter(x=dates, y=quantiles[lower], mode="lines", line={"color": "rgba(31,119,180,0.42)", "width": 1, "dash": dash}, name=label, legendgroup=label, showlegend=True), row=1, col=1)
        figure.add_trace(go.Scatter(x=dates, y=quantiles[upper], mode="lines", line={"color": "rgba(31,119,180,0.42)", "width": 1, "dash": dash}, legendgroup=label, showlegend=False, hoverinfo="skip"), row=1, col=1)
    figure.add_trace(go.Scatter(x=dates, y=quantiles[3], mode="lines", name="Transformer median", line={"color": MEDIAN_BLUE, "width": 4}), row=1, col=1)
    if len(actual_values):
        connector_x: list[object] = []
        connector_y: list[float | None] = []
        for date, model_value, actual_value in zip(actual_dates, median_observed, actual_values):
            connector_x.extend([date, date, None])
            connector_y.extend([model_value, actual_value, None])
        figure.add_trace(go.Scatter(x=connector_x, y=connector_y, mode="lines", name="Median-to-actual gap", line={"color": "rgba(214,39,40,0.35)", "width": 1}, hoverinfo="skip"), row=1, col=1)
        figure.add_trace(go.Scatter(x=actual_dates_with_origin, y=actual_with_origin, mode="lines+markers", name="Observed Yahoo Finance equal-weight basket", line={"color": ACTUAL_GOLD, "width": 4}, marker={"size": 5}), row=1, col=1)
        colors = [ACTUAL_GOLD if value >= 0 else ERROR_RED for value in gap]
        figure.add_trace(go.Bar(x=actual_dates, y=gap, name="Actual − median", marker_color=colors, hovertemplate="%{x|%Y-%m-%d}<br>Actual − median: %{y:.2%}<extra></extra>"), row=2, col=1)
        figure.add_vline(x=actual_dates[-1], line_dash="dash", line_color=NEUTRAL_GRAY, row=1, col=1)
        figure.add_vrect(x0=actual_dates[-1], x1=forecast_dates[-1], fillcolor="rgba(107,114,128,0.08)", line_width=0, annotation_text="Forecast-only after latest complete week", annotation_position="top left", row=1, col=1)
    figure.add_hline(y=0, line_color=NEUTRAL_GRAY, line_width=1, row=2, col=1)
    figure.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98, text=summary_text, align="left", showarrow=False, bgcolor="rgba(255,255,255,0.92)", bordercolor=NEUTRAL_GRAY, borderwidth=1)
    figure.update_yaxes(title_text="Cumulative equal-weight Yahoo basket index<br>(forecast origin = 1.0)", row=1, col=1)
    figure.update_yaxes(title_text="Gap", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(title_text="Week", row=2, col=1, range=[dates[0], dates[-1]])
    figure.update_layout(title="Transformer + Monte Carlo Forecast Versus Observed Yahoo Finance Prices", template="plotly_white", height=820, legend={"orientation": "h", "y": -0.12})
    html_name = "global_probability_river.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.08})
    top, lower = axes
    _add_all_paths_static(top, dates, paths_with_origin, MEDIAN_BLUE)
    top.plot(dates, quantiles[3], color=MEDIAN_BLUE, linewidth=3.2, label="Transformer median", zorder=5)
    for q_index, style, label in [(0, ":", "95% boundaries"), (1, "--", "80% boundaries"), (2, "-", "50% boundaries")]:
        top.plot(dates, quantiles[q_index], color=MEDIAN_BLUE, linewidth=0.9, linestyle=style, alpha=0.55, label=label)
        top.plot(dates, quantiles[6 - q_index], color=MEDIAN_BLUE, linewidth=0.9, linestyle=style, alpha=0.55)
    if len(actual_values):
        for date, model_value, actual_value in zip(actual_dates, median_observed, actual_values):
            top.plot([date, date], [model_value, actual_value], color=ERROR_RED, alpha=0.24, linewidth=0.7, zorder=2)
        top.plot(actual_dates_with_origin, actual_with_origin, color=ACTUAL_GOLD, linewidth=3.4, marker="o", markersize=3.5, label="Observed Yahoo Finance equal-weight basket", zorder=6)
        _shade_future(top, actual_dates[-1], forecast_dates[-1])
        lower.bar(actual_dates, gap, width=5.5, color=[ACTUAL_GOLD if value >= 0 else ERROR_RED for value in gap], alpha=0.85)
    lower.axhline(0, color=NEUTRAL_GRAY, linewidth=1)
    lower.yaxis.set_major_formatter(PercentFormatter(1.0))
    lower.set_ylabel("Actual − median")
    top.set_ylabel("Cumulative equal-weight Yahoo basket index")
    top.set_title("All Transformer Monte Carlo Paths Versus Observed Yahoo Finance Prices\nThin blue lines are predicted futures; gold is the realized equal-weight basket from Yahoo prices")
    top.text(0.99, 0.97, summary_text.replace("<br>", "\n"), transform=top.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": NEUTRAL_GRAY, "alpha": 0.94})
    top.legend(loc="upper left", fontsize=8, ncol=2)
    _format_date_axis(lower, dates[0], dates[-1])
    lower.set_xlabel("Week")
    static.tight_layout()
    png_name = "global_probability_river.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return [html_name, png_name]


def _save_benchmark_path_clouds(config: dict, paths: dict, output_dir: Path) -> list[str]:
    sources = {name: path for name, path in _benchmark_sources(config, paths).items() if path.exists()}
    if not sources:
        return []
    horizon = int(config["benchmarks"]["horizon_weeks"])
    origin, forecast_dates = _origin_and_dates(config, horizon)
    dates = pd.DatetimeIndex([origin, *forecast_dates])
    actual_dates, actual_values = _actual_index(config, paths, forecast_dates)
    actual_dates_with_origin = pd.DatetimeIndex([origin, *actual_dates])
    actual_with_origin = np.concatenate([[1.0], actual_values])
    summaries = _frozen_summary(paths)

    methods = [name for name in METHOD_LABELS if name in sources]
    figure = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=[METHOD_LABELS[name] for name in methods] + [""] * (6 - len(methods)), vertical_spacing=0.08, horizontal_spacing=0.07)
    all_arrays: dict[str, np.ndarray] = {}
    y_values: list[np.ndarray] = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 2)
        row += 1
        col += 1
        payload = load_npz(sources[method])
        array = _prepend_origin(np.asarray(payload["group_index_paths"][:, :, 0], dtype=float))
        all_arrays[method] = array
        y_values.append(array)
        color = METHOD_COLORS[method]
        figure.add_trace(_plotly_path_cloud_trace(dates, array, color, f"{METHOD_LABELS[method]} paths", opacity=0.045, showlegend=index == 0), row=row, col=col)
        median = np.median(array, axis=0)
        figure.add_trace(go.Scatter(x=dates, y=median, mode="lines", name="Method median", line={"color": color, "width": 3}, showlegend=index == 0), row=row, col=col)
        if len(actual_values):
            figure.add_trace(go.Scatter(x=actual_dates_with_origin, y=actual_with_origin, mode="lines", name="Observed Yahoo equal-weight basket", line={"color": ACTUAL_GOLD, "width": 3}, showlegend=index == 0), row=row, col=col)
        metric = summaries.get(method, {})
        final_gap = actual_values[-1] - median[len(actual_values)] if len(actual_values) else np.nan
        text = f"CRPS {float(metric.get('mean_crps', np.nan)):.4f}<br>Path MAE {float(metric.get('cumulative_index_mae', np.nan)):.3%}<br>Final gap {final_gap:+.2%}"
        figure.add_annotation(xref=f"x{index + 1 if index else ''} domain", yref=f"y{index + 1 if index else ''} domain", x=0.98, y=0.98, text=text, showarrow=False, align="right", bgcolor="rgba(255,255,255,0.88)", font={"size": 10})
    figure.update_layout(title="Same Observed Yahoo Finance Basket, Five Forecast Distributions", template="plotly_white", height=1120, legend={"orientation": "h", "y": -0.05})
    figure.update_xaxes(range=[dates[0], dates[-1]])
    figure.update_yaxes(title_text="Cumulative equal-weight Yahoo basket index")
    html_name = "benchmark_path_cloud_comparison.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for axis in axes_flat:
        axis.set_visible(False)
    finite_values = np.concatenate([array[np.isfinite(array)] for array in y_values])
    ymin, ymax = np.quantile(finite_values, [0.002, 0.998])
    padding = 0.06 * max(ymax - ymin, 0.1)
    for index, method in enumerate(methods):
        axis = axes_flat[index]
        axis.set_visible(True)
        array = all_arrays[method]
        color = METHOD_COLORS[method]
        _add_all_paths_static(axis, dates, array, color)
        median = np.median(array, axis=0)
        axis.plot(dates, median, color=color, linewidth=2.7, label="Forecast median", zorder=5)
        if len(actual_values):
            axis.plot(actual_dates_with_origin, actual_with_origin, color=ACTUAL_GOLD, linewidth=3, label="Observed Yahoo equal-weight basket", zorder=6)
            _shade_future(axis, actual_dates[-1], forecast_dates[-1])
        metric = summaries.get(method, {})
        final_gap = actual_values[-1] - median[len(actual_values)] if len(actual_values) else np.nan
        axis.text(0.98, 0.96, f"CRPS: {float(metric.get('mean_crps', np.nan)):.4f}\nPath MAE: {float(metric.get('cumulative_index_mae', np.nan)):.2%}\nFinal actual − median: {final_gap:+.2%}", transform=axis.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.9})
        axis.set_title(METHOD_LABELS[method])
        axis.set_ylim(ymin - padding, ymax + padding)
        _format_date_axis(axis, dates[0], dates[-1])
        if index % 2 == 0:
            axis.set_ylabel("Cumulative equal-weight Yahoo basket index")
        if index >= 4:
            axis.set_xlabel("Week")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    static.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    static.suptitle("All Predicted Paths Compared With the Same Observed Yahoo Finance Basket\nGold is the realized equal-weight basket; source-fund weights are never used", fontsize=16)
    static.tight_layout(rect=[0, 0.04, 1, 0.95])
    png_name = "benchmark_path_cloud_comparison.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return [html_name, png_name]


def _quantile_spanning_indices(paths_array: np.ndarray, count: int) -> np.ndarray:
    count = max(1, min(int(count), len(paths_array)))
    ordering = np.argsort(paths_array[:, -1])
    positions = np.linspace(0, len(ordering) - 1, count).round().astype(int)
    return ordering[positions]


def _historical_context(config: dict, paths: dict, points: int = 104) -> tuple[list[str], list[float]]:
    frame = _actual_global_returns(config, paths)
    history = frame.loc[: config["data"]["validation_end"], "global_return"].dropna().tail(points)
    if history.empty:
        return [], []
    index = np.exp(np.cumsum(history.to_numpy(dtype=float)))
    index = index / index[-1]
    return [str(pd.Timestamp(value).date()) for value in history.index], [float(value) for value in index]


def _save_generation_animation(config: dict, paths: dict, output_dir: Path) -> list[str]:
    item = _simulation_payload(config, paths)
    if item is None:
        return []
    horizon, payload = item
    paths_array = np.asarray(payload["group_index_paths"][:, :, 0], dtype=float)
    path_count = int(config.get("visualization", {}).get("animation_paths", 10))
    weeks = min(int(config.get("visualization", {}).get("generation_animation_weeks", 12)), horizon)
    selected = paths_array[_quantile_spanning_indices(paths_array, path_count), :weeks]
    history_dates, history_values = _historical_context(config, paths, points=int(config["model"]["context_length"]))
    selected_json = json.dumps(selected.tolist())
    history_json = json.dumps(history_values)
    history_date_json = json.dumps(history_dates)
    rank = _read_json(paths["artifacts"]["selected_rank"]).get("selected_rank", "K")
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Transformer Generation Process</title>
<style>
body{{margin:0;background:#07111f;color:#f8fafc;font-family:Inter,system-ui,sans-serif;overflow:hidden}}
#wrap{{height:100vh;display:grid;grid-template-rows:23% 60% 17%;padding:22px;box-sizing:border-box}}
#stages{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;align-items:center}}
.stage{{position:relative;border:2px solid #334155;border-radius:18px;padding:18px;text-align:center;background:#0f172a;transition:.35s;box-shadow:0 0 0 rgba(31,119,180,0)}}
.stage.active{{border-color:#60a5fa;box-shadow:0 0 30px rgba(96,165,250,.45);transform:translateY(-4px)}}
.stage h2{{margin:0 0 6px;font-size:19px}} .stage p{{margin:0;color:#94a3b8;font-size:13px}}
.arrow{{position:absolute;right:-23px;top:45%;font-size:25px;color:#64748b;z-index:3}}
#chart{{width:100%;height:100%;border:1px solid #253348;border-radius:18px;background:#081522}}
#footer{{display:flex;align-items:center;justify-content:space-between;gap:20px}}
#caption{{font-size:24px;font-weight:600;color:#f8fafc}} #detail{{color:#94a3b8;font-size:14px;margin-top:5px}}
button,select{{background:#1e293b;color:white;border:1px solid #475569;border-radius:9px;padding:9px 14px;font-size:14px}}
.badge{{color:#93c5fd;border:1px solid #1d4ed8;background:#172554;padding:7px 10px;border-radius:999px;font-size:13px}}
</style></head>
<body><div id='wrap'>
<div id='stages'>
<div class='stage' id='s0'><h2>1. Read the context</h2><p>104 observed weeks × {rank} PCA factors</p><span class='arrow'>→</span></div>
<div class='stage' id='s1'><h2>2. Predict a distribution</h2><p>Causal transformer → Student-t mixture</p><span class='arrow'>→</span></div>
<div class='stage' id='s2'><h2>3. Sample next week</h2><p>One draw for every Monte Carlo future</p><span class='arrow'>→</span></div>
<div class='stage' id='s3'><h2>4. Append and repeat</h2><p>The sampled week becomes the next input</p></div>
</div>
<canvas id='chart'></canvas>
<div id='footer'><div><div id='caption'></div><div id='detail'></div></div><div><span class='badge' id='counter'></span> <button id='play'>Pause</button> <button id='reset'>Reset</button> <select id='speed'><option value='900'>Slow</option><option value='550' selected>Normal</option><option value='280'>Fast</option></select></div></div>
</div>
<script>
const paths={selected_json}; const history={history_json}; const historyDates={history_date_json};
const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'); let week=0,phase=0,playing=true,timer;
const captions=[
['The transformer reads the same historical context for every future.','No 2026 outcome is visible to the frozen-origin model.'],
['It outputs probabilities, locations, scales, covariance, and tail thickness.','This is a distribution—not one single market guess.'],
['Monte Carlo draws one possible next week for each future.','Ten saved paths are shown here for clarity; the real run generated thousands.'],
['Each sampled week is appended, then the transformer predicts again.','Repeated one-step generation creates the branching forecast tree.']];
function resize(){{const r=canvas.getBoundingClientRect();canvas.width=r.width*devicePixelRatio;canvas.height=r.height*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw();}}
function bounds(){{let vals=[...history];paths.forEach(p=>vals.push(...p.slice(0,Math.max(1,week))));let lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo||.1)*.12;return[lo-pad,hi+pad];}}
function draw(){{
 const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);ctx.fillStyle='#081522';ctx.fillRect(0,0,w,h);
 const margin={{l:65,r:35,t:35,b:45}},plotW=w-margin.l-margin.r,plotH=h-margin.t-margin.b; const [lo,hi]=bounds();
 ctx.strokeStyle='#243244';ctx.lineWidth=1;ctx.fillStyle='#94a3b8';ctx.font='12px system-ui';
 for(let i=0;i<=4;i++){{let y=margin.t+plotH*i/4;ctx.beginPath();ctx.moveTo(margin.l,y);ctx.lineTo(w-margin.r,y);ctx.stroke();let v=hi-(hi-lo)*i/4;ctx.fillText(v.toFixed(2),12,y+4);}}
 const total=history.length+paths[0].length; const x=i=>margin.l+plotW*i/(total-1), y=v=>margin.t+plotH*(hi-v)/(hi-lo);
 ctx.strokeStyle='#64748b';ctx.lineWidth=2;ctx.beginPath();history.forEach((v,i)=>{{i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v));}});ctx.stroke();
 const colors=['#60a5fa','#93c5fd','#38bdf8','#818cf8','#22d3ee','#a5b4fc','#0ea5e9','#67e8f9','#3b82f6','#7dd3fc'];
 paths.forEach((p,j)=>{{ctx.strokeStyle=colors[j%colors.length];ctx.globalAlpha=.82;ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(x(history.length-1),y(1));for(let k=0;k<week;k++)ctx.lineTo(x(history.length+k),y(p[k]));ctx.stroke();}});ctx.globalAlpha=1;
 ctx.strokeStyle='#d4af37';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(x(history.length-1),margin.t);ctx.lineTo(x(history.length-1),h-margin.b);ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#d4af37';ctx.fillText('Forecast origin',x(history.length-1)-42,margin.t+15);ctx.fillStyle='#94a3b8';ctx.fillText('Observed Yahoo-price factor context',margin.l, h-15);ctx.fillText('Recursively generated Yahoo-basket future',x(history.length+2),h-15);
 }}
function update(){{document.querySelectorAll('.stage').forEach((e,i)=>e.classList.toggle('active',i===phase));document.getElementById('caption').textContent=captions[phase][0];document.getElementById('detail').textContent=captions[phase][1];document.getElementById('counter').textContent=`Generated week ${{week}} / ${{paths[0].length}}`;draw();}}
function tick(){{if(!playing)return;phase++;if(phase>3){{phase=0;week++;if(week>paths[0].length)week=0;}}update();schedule();}}
function schedule(){{clearTimeout(timer);timer=setTimeout(tick,+document.getElementById('speed').value);}}
document.getElementById('play').onclick=()=>{{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';if(playing)schedule();}};
document.getElementById('reset').onclick=()=>{{week=0;phase=0;update();schedule();}};document.getElementById('speed').onchange=schedule;window.onresize=resize;resize();update();schedule();
</script></body></html>"""
    name = "transformer_generation_process.html"
    (output_dir / name).write_text(html, encoding="utf-8")
    return [name]


def _save_ten_path_animation(config: dict, paths: dict, output_dir: Path) -> list[str]:
    item = _simulation_payload(config, paths)
    if item is None:
        return []
    horizon, payload = item
    all_paths = np.asarray(payload["group_index_paths"][:, :, 0], dtype=float)
    count = int(config.get("visualization", {}).get("animation_paths", 10))
    selected = all_paths[_quantile_spanning_indices(all_paths, count)]
    origin, forecast_dates = _origin_and_dates(config, horizon)
    dates = pd.DatetimeIndex([origin, *forecast_dates])
    selected = _prepend_origin(selected)
    median = np.concatenate([[1.0], np.median(all_paths, axis=0)])
    actual_dates, actual_values = _actual_index(config, paths, forecast_dates)
    actual_dates = pd.DatetimeIndex([origin, *actual_dates])
    actual_values = np.concatenate([[1.0], actual_values])
    colors = [f"hsl({205 + index * 4},75%,{38 + index * 3}%)" for index in range(len(selected))]

    traces: list[go.Scatter] = []
    for index, path in enumerate(selected):
        traces.append(go.Scatter(x=[dates[0]], y=[path[0]], mode="lines", line={"color": colors[index], "width": 1.7}, name=f"Sampled path {index + 1}", showlegend=False))
    traces.append(go.Scatter(x=[dates[0]], y=[median[0]], mode="lines", line={"color": MEDIAN_BLUE, "width": 4}, name="Median of all Monte Carlo paths"))
    traces.append(go.Scatter(x=[actual_dates[0]], y=[actual_values[0]], mode="lines+markers", line={"color": ACTUAL_GOLD, "width": 4}, marker={"size": 6}, name="Observed Yahoo equal-weight basket"))
    frames: list[go.Frame] = []
    for step in range(1, horizon + 1):
        frame_data: list[go.Scatter] = []
        for index, path in enumerate(selected):
            frame_data.append(go.Scatter(x=dates[: step + 1], y=path[: step + 1]))
        frame_data.append(go.Scatter(x=dates[: step + 1], y=median[: step + 1]))
        actual_step = min(step, len(actual_values) - 1)
        frame_data.append(go.Scatter(x=actual_dates[: actual_step + 1], y=actual_values[: actual_step + 1]))
        frames.append(go.Frame(name=str(step), data=frame_data, layout=go.Layout(title_text=f"Ten Sampled Transformer Futures Growing One Week at a Time — Week {step}")))
    figure = go.Figure(data=traces, frames=frames)
    figure.update_layout(
        title="Ten Sampled Transformer Futures Growing One Week at a Time",
        template="plotly_white",
        xaxis={"title": "Week", "range": [dates[0], dates[-1]]},
        yaxis={"title": "Cumulative equal-weight Yahoo basket index"},
        height=700,
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.02, "y": 1.12, "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 260, "redraw": False}, "transition": {"duration": 120}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
        ]}],
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Generated week: "},
            "steps": [
                {
                    "label": str(step),
                    "method": "animate",
                    "args": [[str(step)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                }
                for step in range(1, horizon + 1)
            ],
        }],
        legend={"orientation": "h", "y": -0.18},
    )
    name = "ten_path_rollout_animation.html"
    figure.write_html(output_dir / name, include_plotlyjs="cdn", auto_play=False)
    return [name]


def _save_rolling_prediction_difference(paths: dict, output_dir: Path) -> list[str]:
    source = Path(paths["artifacts"]["rolling_evaluation"])
    if not source.exists():
        return []
    frame = pd.read_csv(source, parse_dates=["date"])
    frame = frame[frame["model"] == "transformer"].sort_values("date")
    if frame.empty:
        return []
    error = frame["actual_return"] - frame["median_return"]
    correct = frame["direction_correct"].astype(str).str.lower().isin({"true", "1", "yes"})
    summary = _rolling_summary(paths).get("transformer", {})

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.08, subplot_titles=("What the frozen transformer predicted before each week, after seeing earlier weeks", "Signed prediction error: observed return minus predicted median"))
    figure.add_trace(go.Bar(x=frame["date"], y=frame["actual_return"], name="Observed weekly Yahoo-basket return", marker_color=ACTUAL_GOLD, opacity=0.75, hovertemplate="%{x|%Y-%m-%d}<br>Observed: %{y:.2%}<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["date"], y=frame["median_return"], mode="lines+markers", name="Transformer median prediction", line={"color": MEDIAN_BLUE, "width": 3}, marker={"size": 7}, error_y={"type": "data", "symmetric": False, "array": frame["q90"] - frame["median_return"], "arrayminus": frame["median_return"] - frame["q10"], "color": "rgba(31,119,180,0.45)", "thickness": 1.2, "width": 3}, hovertemplate="%{x|%Y-%m-%d}<br>Predicted median: %{y:.2%}<extra></extra>"), row=1, col=1)
    connector_x: list[object] = []
    connector_y: list[float | None] = []
    for date, predicted, actual in zip(frame["date"], frame["median_return"], frame["actual_return"]):
        connector_x.extend([date, date, None])
        connector_y.extend([predicted, actual, None])
    figure.add_trace(go.Scatter(x=connector_x, y=connector_y, mode="lines", name="Prediction gap", line={"color": "rgba(107,114,128,0.48)", "width": 1}, hoverinfo="skip"), row=1, col=1)
    error_colors = [SUCCESS_GREEN if is_correct else ERROR_RED for is_correct in correct]
    figure.add_trace(go.Bar(x=frame["date"], y=error, name="Actual − predicted", marker_color=error_colors, hovertemplate="%{x|%Y-%m-%d}<br>Error: %{y:+.2%}<extra></extra>"), row=2, col=1)
    figure.add_hline(y=0, line_color=NEUTRAL_GRAY, row=2, col=1)
    figure.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.93)", bordercolor=NEUTRAL_GRAY, text=f"Weekly MAE: {float(summary.get('weekly_return_mae', np.nan)):.2%}<br>Direction accuracy: {float(summary.get('weekly_direction_accuracy', np.nan)):.1%}<br>80% interval coverage: {float(summary.get('coverage_80', np.nan)):.1%}<br>CRPS: {float(summary.get('mean_crps', np.nan)):.5f}")
    figure.update_yaxes(tickformat=".1%", title_text="Weekly return", row=1, col=1)
    figure.update_yaxes(tickformat="+.1%", title_text="Error", row=2, col=1)
    figure.update_xaxes(title_text="Prediction week", row=2, col=1)
    figure.update_layout(title="Rolling One-Week Transformer Predictions Versus What Happened", template="plotly_white", height=820, legend={"orientation": "h", "y": -0.12})
    html_name = "rolling_prediction_vs_actual.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08})
    top, lower = axes
    top.bar(frame["date"], frame["actual_return"], width=5.5, color=ACTUAL_GOLD, alpha=0.72, label="Observed weekly return")
    top.errorbar(frame["date"], frame["median_return"], yerr=[frame["median_return"] - frame["q10"], frame["q90"] - frame["median_return"]], fmt="o-", color=MEDIAN_BLUE, linewidth=2.4, markersize=4.5, elinewidth=1, capsize=2, label="Transformer median + 80% interval")
    for date, predicted, actual in zip(frame["date"], frame["median_return"], frame["actual_return"]):
        top.plot([date, date], [predicted, actual], color=NEUTRAL_GRAY, alpha=0.38, linewidth=0.8)
    top.axhline(0, color=NEUTRAL_GRAY, linewidth=1)
    top.yaxis.set_major_formatter(PercentFormatter(1.0))
    top.set_ylabel("Weekly return")
    top.set_title("Rolling One-Week Predictions Versus Observed Yahoo Finance Basket\nEach prediction may use earlier 2026 prices; PCA and transformer parameters remain frozen at 2025-12-31")
    top.legend(loc="upper left", fontsize=8)
    top.text(0.99, 0.97, f"Weekly MAE: {float(summary.get('weekly_return_mae', np.nan)):.2%}\nDirection accuracy: {float(summary.get('weekly_direction_accuracy', np.nan)):.1%}\n80% coverage: {float(summary.get('coverage_80', np.nan)):.1%}\nCRPS: {float(summary.get('mean_crps', np.nan)):.5f}", transform=top.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": NEUTRAL_GRAY, "alpha": 0.93})
    lower.bar(frame["date"], error, width=5.5, color=error_colors, alpha=0.85)
    lower.axhline(0, color=NEUTRAL_GRAY, linewidth=1)
    lower.yaxis.set_major_formatter(PercentFormatter(1.0))
    lower.set_ylabel("Actual − predicted")
    lower.set_xlabel("Prediction week")
    _format_date_axis(lower, frame["date"].min(), frame["date"].max())
    static.tight_layout()
    png_name = "rolling_prediction_vs_actual.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return [html_name, png_name]


def _save_notional_value(config: dict, paths: dict, output_dir: Path) -> list[str]:
    item = _simulation_payload(config, paths)
    if item is None:
        return []
    horizon, payload = item
    simulated = np.asarray(payload["group_index_paths"][:, :, 0], dtype=float)
    origin, forecast_dates = _origin_and_dates(config, horizon)
    dates = pd.DatetimeIndex([origin, *forecast_dates])
    quantiles = _quantiles(_prepend_origin(simulated))
    notional = float(config.get("visualization", {}).get("reference_notional_value", 100000.0))
    actual_dates, actual_values = _actual_index(config, paths, forecast_dates)
    actual_dates_with_origin = pd.DatetimeIndex([origin, *actual_dates])
    actual_notional = notional * np.concatenate([[1.0], actual_values])
    median_notional = notional * quantiles[3]
    observed_steps = len(actual_values)
    latest_median = median_notional[observed_steps]
    latest_actual = actual_notional[-1] if len(actual_notional) else np.nan
    difference = latest_actual - latest_median
    lower80 = notional * quantiles[1, observed_steps]
    upper80 = notional * quantiles[5, observed_steps]
    percentile = float(_frozen_summary(paths).get("transformer", {}).get("final_actual_percentile", np.nan))

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dates, y=notional * quantiles[5], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=dates, y=notional * quantiles[1], mode="lines", fill="tonexty", fillcolor="rgba(125,183,232,0.22)", line={"width": 0}, name="Central 80% forecast range"))
    figure.add_trace(go.Scatter(x=dates, y=notional * quantiles[4], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=dates, y=notional * quantiles[2], mode="lines", fill="tonexty", fillcolor="rgba(125,183,232,0.38)", line={"width": 0}, name="Central 50% forecast range"))
    figure.add_trace(go.Scatter(x=dates, y=median_notional, mode="lines", name="Transformer median", line={"color": MEDIAN_BLUE, "width": 4}))
    if len(actual_values):
        figure.add_trace(go.Scatter(x=actual_dates_with_origin, y=actual_notional, mode="lines+markers", name="Observed value of equal-weight Yahoo basket", line={"color": ACTUAL_GOLD, "width": 4}, marker={"size": 5}))
        figure.add_vline(x=actual_dates[-1], line_dash="dash", line_color=NEUTRAL_GRAY)
        figure.add_vrect(x0=actual_dates[-1], x1=forecast_dates[-1], fillcolor="rgba(107,114,128,0.08)", line_width=0, annotation_text="Forecast-only", annotation_position="top left")
    figure.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.94)", bordercolor=NEUTRAL_GRAY, text=f"Latest complete week: {actual_dates[-1].date() if len(actual_dates) else 'n/a'}<br>Observed index × ${notional:,.0f}: ${latest_actual:,.0f}<br>Transformer median: ${latest_median:,.0f}<br>Difference: ${difference:+,.0f}<br>80% range: ${lower80:,.0f}–${upper80:,.0f}<br>Observed percentile: {percentile:.1%}")
    figure.update_layout(title=f"Notional ${notional:,.0f} Scale: Forecast Versus Observed Yahoo Finance Basket", template="plotly_white", height=690, xaxis_title="Week", yaxis_title="Notional value (equal-weight Yahoo basket index × starting amount)", legend={"orientation": "h", "y": -0.18})
    figure.add_annotation(xref="paper", yref="paper", x=0, y=-0.24, showarrow=False, align="left", text="This is not a traded portfolio. It multiplies both the predicted and observed equal-weight Yahoo Finance basket indices by the same starting amount so their gap is easy to read.", font={"size": 11, "color": NEUTRAL_GRAY})
    html_name = "notional_100k_forecast_vs_observed.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axis = plt.subplots(figsize=(13, 6.8))
    axis.fill_between(dates, notional * quantiles[1], notional * quantiles[5], color=INTERVAL_BLUE, alpha=0.20, label="Central 80% forecast range")
    axis.fill_between(dates, notional * quantiles[2], notional * quantiles[4], color=INTERVAL_BLUE, alpha=0.38, label="Central 50% forecast range")
    axis.plot(dates, median_notional, color=MEDIAN_BLUE, linewidth=3.2, label="Transformer median")
    if len(actual_values):
        axis.plot(actual_dates_with_origin, actual_notional, color=ACTUAL_GOLD, linewidth=3.4, marker="o", markersize=3.5, label="Observed value of equal-weight Yahoo basket")
        _shade_future(axis, actual_dates[-1], forecast_dates[-1])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.set_ylabel("Notional value (equal-weight Yahoo basket index × starting amount)")
    axis.set_title(f"Forecast Versus Observed Yahoo Finance Basket on a ${notional:,.0f} Scale\nBoth lines use the same equal-weight basket; the gold line is observed prices, not a model output")
    axis.text(0.99, 0.97, f"Observed: ${latest_actual:,.0f}\nTransformer median: ${latest_median:,.0f}\nObserved − median: ${difference:+,.0f}\n80% range: ${lower80:,.0f}–${upper80:,.0f}\nObserved percentile: {percentile:.1%}", transform=axis.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": NEUTRAL_GRAY, "alpha": 0.94})
    axis.legend(loc="upper left", fontsize=8)
    _format_date_axis(axis, dates[0], dates[-1])
    axis.set_xlabel("Week")
    static.tight_layout()
    png_name = "notional_100k_forecast_vs_observed.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return [html_name, png_name]


def _metric_format(metric: str, value: float) -> str:
    if "accuracy" in metric or "coverage" in metric or "error_80" in metric:
        return f"{value:.1%}"
    if "mae" in metric and value < 1:
        return f"{value:.3%}"
    return f"{value:.5f}"


def _save_benchmark_scorecard(paths: dict, output_dir: Path) -> list[str]:
    frozen = _frozen_summary(paths)
    rolling = _rolling_summary(paths)
    methods = [method for method in METHOD_LABELS if method in frozen and method in rolling]
    if not methods:
        return []
    metrics = [
        ("Rolling CRPS ↓", lambda method: float(rolling[method]["mean_crps"]), False),
        ("Rolling weekly MAE ↓", lambda method: float(rolling[method]["weekly_return_mae"]), False),
        ("Rolling calibration error ↓", lambda method: float(rolling[method]["calibration_error"]), False),
        ("Rolling direction accuracy ↑", lambda method: float(rolling[method]["weekly_direction_accuracy"]), True),
        ("Frozen CRPS ↓", lambda method: float(frozen[method]["mean_crps"]), False),
        ("Frozen path MAE ↓", lambda method: float(frozen[method]["cumulative_index_mae"]), False),
        ("Frozen 80% coverage error ↓", lambda method: abs(float(frozen[method]["coverage_80"]) - 0.8), False),
    ]
    values = np.array([[getter(method) for method in methods] for _, getter, _ in metrics], dtype=float)
    ranks = np.empty_like(values)
    for row, (_, _, higher_better) in enumerate(metrics):
        ranks[row] = pd.Series(values[row]).rank(
            method="min", ascending=not higher_better
        ).to_numpy(dtype=float)
    text = np.array([[_metric_format(metrics[row][0], values[row, col]) for col in range(len(methods))] for row in range(len(metrics))], dtype=object)
    table_rows = []
    for row, (metric, _, _) in enumerate(metrics):
        for col, method in enumerate(methods):
            table_rows.append({"metric": metric, "method": method, "value": values[row, col], "rank": int(ranks[row, col])})
    pd.DataFrame(table_rows).to_csv(output_dir / "benchmark_summary.csv", index=False)

    figure = go.Figure(go.Heatmap(z=ranks, x=[METHOD_LABELS[method] for method in methods], y=[metric[0] for metric in metrics], text=text, texttemplate="%{text}<br><b>rank %{z:.0f}</b>", hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>", colorscale=[[0, "#1a9850"], [0.5, "#ffffbf"], [1, "#d73027"]], reversescale=False, zmin=1, zmax=len(methods), colorbar={"title": "Rank", "tickvals": list(range(1, len(methods) + 1))}))
    figure.update_layout(title="All Forecasting Methods in One Scorecard — Values and Per-Metric Ranks", template="plotly_white", height=650, xaxis_title="Method", yaxis_title="Metric")
    figure.add_annotation(xref="paper", yref="paper", x=0, y=-0.18, showarrow=False, align="left", text="Green is best for that metric, red is worst. Ranks are not combined into one arbitrary overall score.", font={"size": 11, "color": NEUTRAL_GRAY})
    html_name = "benchmark_scorecard.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axis = plt.subplots(figsize=(13, 7.2))
    image = axis.imshow(ranks, cmap="RdYlGn_r", vmin=1, vmax=len(methods), aspect="auto")
    axis.set_xticks(np.arange(len(methods)), [METHOD_LABELS[method] for method in methods], rotation=22, ha="right")
    axis.set_yticks(np.arange(len(metrics)), [metric[0] for metric in metrics])
    for row in range(len(metrics)):
        for col in range(len(methods)):
            axis.text(col, row, f"{text[row, col]}\nrank {int(ranks[row, col])}", ha="center", va="center", fontsize=8, color="black")
    colorbar = static.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Per-metric rank (1 = best)")
    axis.set_title("Benchmark Scorecard: Actual Values and Per-Metric Ranks\nNo single overall score is imposed")
    static.tight_layout()
    png_name = "benchmark_scorecard.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return ["benchmark_summary.csv", html_name, png_name]


def _save_uncertainty_ladder(config: dict, paths: dict, output_dir: Path) -> list[str]:
    simulation_dir = Path(paths["artifacts"]["simulations_dir"])
    records: list[dict] = []
    for source in sorted(simulation_dir.glob("ordinary_h*.npz")):
        payload = load_npz(source)
        horizon = int(np.asarray(payload.get("horizon", [int(source.stem.split("h")[-1])])).ravel()[0])
        terminal = np.asarray(payload["group_index_paths"][:, -1, 0], dtype=float)
        q = np.quantile(terminal, [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])
        records.append({"horizon": horizon, "paths": len(terminal), "q025": q[0], "q10": q[1], "q25": q[2], "median": q[3], "q75": q[4], "q90": q[5], "q975": q[6], "probability_loss": float(np.mean(terminal < 1.0))})
    if not records:
        return []
    frame = pd.DataFrame(records).sort_values("horizon")
    frame.to_csv(output_dir / "uncertainty_by_horizon.csv", index=False)
    notional = float(config.get("visualization", {}).get("reference_notional_value", 100000.0))
    labels = [f"{int(value)} week{'s' if value != 1 else ''}" if value < 52 else ("1 year" if value == 52 else f"{value / 52:.0f} years") for value in frame["horizon"]]

    figure = go.Figure()
    for index, row in frame.reset_index(drop=True).iterrows():
        y = labels[index]
        figure.add_trace(go.Scatter(x=[notional * row["q025"], notional * row["q975"]], y=[y, y], mode="lines", line={"color": "rgba(31,119,180,0.35)", "width": 4}, showlegend=index == 0, name="95% range", hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=[notional * row["q10"], notional * row["q90"]], y=[y, y], mode="lines", line={"color": INTERVAL_BLUE, "width": 10}, showlegend=index == 0, name="80% range", hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=[notional * row["q25"], notional * row["q75"]], y=[y, y], mode="lines", line={"color": MEDIAN_BLUE, "width": 18}, showlegend=index == 0, name="50% range", hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=[notional * row["median"]], y=[y], mode="markers+text", marker={"color": DARK, "size": 10}, text=[f"  median ${notional * row['median']:,.0f}<br>P(loss) {row['probability_loss']:.1%}"], textposition="middle right", showlegend=index == 0, name="Median and loss probability", hovertemplate=f"{y}<br>Median: $%{{x:,.0f}}<br>P(end below start): {row['probability_loss']:.1%}<extra></extra>"))
    figure.add_vline(x=notional, line_dash="dash", line_color=ACTUAL_GOLD, annotation_text=f"Starting value ${notional:,.0f}")
    figure.update_xaxes(type="log", title="Possible ending notional value (log scale)", tickprefix="$", separatethousands=True)
    figure.update_yaxes(title="Forecast horizon", categoryorder="array", categoryarray=labels[::-1])
    figure.update_layout(title="Uncertainty Ladder: Longer Forecasts Produce a Wider Range of Possible Outcomes", template="plotly_white", height=720, legend={"orientation": "h", "y": -0.12})
    figure.add_annotation(xref="paper", yref="paper", x=0, y=-0.19, showarrow=False, align="left", text="Each row uses every saved recursive Monte Carlo path. Multi-year rows are exploratory scenario distributions, not horizon-validated ten-year forecasts; the log axis preserves readability.", font={"size": 11, "color": NEUTRAL_GRAY})
    html_name = "uncertainty_by_horizon.html"
    figure.write_html(output_dir / html_name, include_plotlyjs="cdn")

    static, axis = plt.subplots(figsize=(13, 7.2))
    y_positions = np.arange(len(frame))
    for y, (_, row) in zip(y_positions, frame.iterrows()):
        axis.plot([notional * row["q025"], notional * row["q975"]], [y, y], color=MEDIAN_BLUE, alpha=0.28, linewidth=3)
        axis.plot([notional * row["q10"], notional * row["q90"]], [y, y], color=INTERVAL_BLUE, linewidth=8, solid_capstyle="round")
        axis.plot([notional * row["q25"], notional * row["q75"]], [y, y], color=MEDIAN_BLUE, linewidth=14, solid_capstyle="round")
        axis.scatter([notional * row["median"]], [y], color=DARK, s=45, zorder=5)
        axis.text(notional * row["median"], y + 0.18, f"median ${notional * row['median']:,.0f} · P(loss) {row['probability_loss']:.1%}", fontsize=8, ha="center")
    axis.axvline(notional, color=ACTUAL_GOLD, linestyle="--", linewidth=2, label=f"Starting value ${notional:,.0f}")
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.set_yticks(y_positions, labels)
    axis.set_ylim(-0.7, max(0.7, len(frame) - 0.3))
    axis.set_xlabel("Possible ending notional value (log scale)")
    axis.set_ylabel("Forecast horizon")
    axis.set_title("Exploratory Recursive Uncertainty Ladder\nMulti-year paths show model-implied scenarios, not validated long-horizon accuracy")
    axis.legend(loc="lower right")
    static.tight_layout()
    png_name = "uncertainty_by_horizon.png"
    static.savefig(output_dir / png_name, dpi=220)
    plt.close(static)
    return ["uncertainty_by_horizon.csv", html_name, png_name]


def _write_visual_guide(paths: dict, output_dir: Path, outputs: list[str]) -> None:
    descriptions = {
        "global_probability_river.html": "Interactive hero visual showing every transformer Monte Carlo path, observed Yahoo basket, median, interval boundaries, and observed-minus-median gap.",
        "global_probability_river.png": "Static hero visual showing every transformer path and the direct prediction gap.",
        "benchmark_path_cloud_comparison.html": "Interactive small multiples showing every path for all five forecasting methods against the same observed Yahoo basket.",
        "benchmark_path_cloud_comparison.png": "Static all-method path-cloud comparison on shared axes.",
        "transformer_generation_process.html": "3Blue1Brown-inspired animation of read → predict distribution → sample → append → repeat.",
        "ten_path_rollout_animation.html": "Animated rollout of ten quantile-spanning sampled futures, the full-path median, and observed Yahoo basket.",
        "rolling_prediction_vs_actual.html": "Interactive week-by-week prediction-versus-observation comparison with signed errors.",
        "rolling_prediction_vs_actual.png": "Static week-by-week prediction-versus-observation comparison.",
        "notional_100k_forecast_vs_observed.html": "Interactive dollar-scale translation of forecast versus observed Yahoo-basket outcome; explicitly not a trading strategy.",
        "notional_100k_forecast_vs_observed.png": "Static dollar-scale translation with exact observed-minus-median Yahoo-basket difference.",
        "factor_selection.png": "Leakage-safe held-stock PCA reconstruction criterion.",
        "factor_forecast_selection.png": "Forecast-oriented PCA rank-selection criterion.",
        "factor_explained_variance.png": "Variance preserved by the selected PCA factors.",
        "training_history.png": "Transformer training, validation, overfitting, and selected checkpoint.",
        "benchmark_scorecard.html": "Interactive all-metric benchmark scorecard with values and per-metric ranks.",
        "benchmark_scorecard.png": "Static all-metric benchmark scorecard.",
        "benchmark_summary.csv": "Machine-readable benchmark scorecard values and ranks.",
        "uncertainty_by_horizon.html": "Interactive recursive-scenario uncertainty ladder from one week through ten years; multi-year horizons are explicitly exploratory.",
        "uncertainty_by_horizon.png": "Static recursive-scenario uncertainty ladder with explicit long-horizon caveat.",
        "uncertainty_by_horizon.csv": "Machine-readable horizon quantiles and loss probabilities.",
        "temporal_leakage_timeline.png": "Data-cutoff and remaining universe-membership leakage timeline.",
    }
    rows = []
    for name in outputs:
        path = output_dir / name
        if path.exists():
            rows.append({"file": name, "bytes": path.stat().st_size, "description": descriptions.get(name, "Generated project visual or companion table.")})
    manifest_path = Path(paths["artifacts"].get("visuals_manifest", output_dir / "visual_manifest.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    guide_path = Path(paths["artifacts"].get("visuals_readme", output_dir / "README_VISUALS.md"))
    lines = [
        "# Visual Guide",
        "",
        "## Core meaning",
        "",
        "The stock-list selector chooses securities and may supply disclosed country, exchange, sector, industry, or currency fallbacks when Yahoo metadata is sparse. All realized returns still come from Yahoo Finance price histories, converted to USD. The reported market series is equal-weight; no source-fund portfolio weights or market values enter training, simulation, evaluation, or visuals.",
        "",
        "- **Gold:** realized equal-weight basket calculated from observed Yahoo Finance prices.",
        "- **Blue:** transformer forecasts and Monte Carlo paths.",
        "- **Red:** prediction error or miss—not a selected portfolio.",
        "- **Other method colors:** benchmark-specific forecast distributions.",
        "",
        "The transformer creates the blue recursive scenario distribution. The gold line is calculated afterward from observed Yahoo Finance adjusted prices, converted to USD and averaged equally across eligible securities. The stock-list selector contributes membership plus disclosed descriptive-metadata fallbacks; its weights and market values are never used.",
        "",
        "Horizons through 520 weeks are intentionally retained for exploration. Multi-year paths are model-implied recursive scenarios with accumulating uncertainty, not claims of validated ten-year forecasting accuracy.",
        "",
        "The dollar visual multiplies the observed and predicted equal-weight Yahoo basket indices by the same starting amount. It is a comparison scale, not a trading strategy.",
        "",
        "## Recommended presentation order",
        "",
        "1. `transformer_generation_process.html`",
        "2. `ten_path_rollout_animation.html`",
        "3. `global_probability_river.html`",
        "4. `rolling_prediction_vs_actual.html`",
        "5. `benchmark_path_cloud_comparison.html`",
        "6. `benchmark_scorecard.html`",
        "7. `uncertainty_by_horizon.html`",
        "8. PCA and training figures",
        "",
        "## Files",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['file']}` — {row['description']}")
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_visuals(config: dict, paths: dict, force: bool = False) -> None:
    output_dir = Path(paths["artifacts"]["visuals_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for item in output_dir.iterdir():
            if item.is_file() and item.name != ".gitkeep":
                item.unlink()

    generators: list[tuple[Callable[..., list[str]], bool]] = [
        (_save_generation_animation, True),
        (_save_ten_path_animation, True),
        (_save_probability_river, True),
        (_save_benchmark_path_clouds, True),
        (_save_rolling_prediction_difference, False),
        (_save_notional_value, True),
        (_save_factor_selection, False),
        (_save_factor_variance, False),
        (_save_training_history, False),
        (_save_benchmark_scorecard, False),
        (_save_uncertainty_ladder, True),
        (_save_temporal_leakage_timeline, True),
    ]
    outputs: list[str] = []
    for generator, needs_config in generators:
        try:
            if needs_config:
                outputs.extend(generator(config, paths, output_dir))
            else:
                outputs.extend(generator(paths, output_dir))
        except Exception as exc:
            LOGGER.warning("Skipping visual %s because: %s", generator.__name__, exc)
    outputs = list(dict.fromkeys(outputs))
    _write_visual_guide(paths, output_dir, outputs)
    LOGGER.info("Created %d high-priority visual outputs in %s", len(outputs), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create focused, demonstrative transformer and Monte Carlo visuals.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, paths = load_project(args.config, args.paths)
    create_visuals(config, paths, force=args.force)


if __name__ == "__main__":
    main()
