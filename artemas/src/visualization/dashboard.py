from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import load_project


@st.cache_data
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"] if "evaluation" in path.name else None)


def main() -> None:
    st.set_page_config(page_title="ARTEMAS", layout="wide")
    config, paths = load_project()
    evaluation_dir = Path(paths["artifacts"]["evaluation_dir"])
    simulation_dir = Path(paths["artifacts"]["simulations_dir"])
    frozen_summary = evaluation_dir / "summary.json"
    frozen_weekly = evaluation_dir / "weekly_frozen_evaluation.csv"
    rolling_summary = Path(paths["artifacts"]["rolling_summary"])
    rolling_weekly = Path(paths["artifacts"]["rolling_evaluation"])

    st.title("ARTEMAS")
    st.caption("A point-in-time probabilistic experiment on an equal-weight USD basket built from Yahoo Finance prices.")
    st.info(
        "The configured stock-list snapshot must be known by the pre-validation membership cutoff. "
        "Horizons through ten years are retained as exploratory recursive Monte Carlo scenarios, not "
        "as claims of validated ten-year predictive accuracy. This is not financial advice."
    )
    if not frozen_summary.exists() or not frozen_weekly.exists():
        st.warning("Run the evaluation stage before opening the dashboard.")
        return

    frozen = _load_json(frozen_summary)
    weekly = pd.read_csv(frozen_weekly, parse_dates=["date"])
    methods = list(frozen)
    selected = st.sidebar.selectbox(
        "Method", methods, index=methods.index("transformer") if "transformer" in methods else 0
    )
    selected_weekly = weekly[weekly["model"] == selected]
    metrics = frozen[selected]

    columns = st.columns(5)
    columns[0].metric("Frozen cumulative MAE", f"{metrics['cumulative_index_mae']:.4f}")
    columns[1].metric("Frozen weekly MAE", f"{metrics['weekly_return_mae']:.4f}")
    columns[2].metric("Direction", f"{metrics['weekly_direction_accuracy']:.1%}")
    columns[3].metric("80% coverage", f"{metrics['coverage_80']:.1%}")
    columns[4].metric("CRPS", f"{metrics['mean_crps']:.4f}")

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=selected_weekly["date"], y=selected_weekly["q975"], mode="lines", line={"width": 0}, showlegend=False))
    figure.add_trace(go.Scatter(x=selected_weekly["date"], y=selected_weekly["q025"], mode="lines", line={"width": 0}, fill="tonexty", name="95% interval"))
    figure.add_trace(go.Scatter(x=selected_weekly["date"], y=selected_weekly["median_index"], mode="lines", name="Median"))
    figure.add_trace(go.Scatter(x=selected_weekly["date"], y=selected_weekly["actual_index"], mode="lines", name="Observed Yahoo equal-weight basket"))
    figure.update_layout(template="plotly_white", title=f"Frozen-Origin Calendar-Aligned Comparison: {selected}")
    st.plotly_chart(figure, use_container_width=True)

    if rolling_summary.exists() and rolling_weekly.exists():
        st.subheader("Rolling one-step evaluation")
        rolling_metrics = _load_json(rolling_summary).get(selected)
        if rolling_metrics:
            columns = st.columns(5)
            columns[0].metric("Rolling CRPS", f"{rolling_metrics['mean_crps']:.4f}")
            columns[1].metric("Rolling MAE", f"{rolling_metrics['weekly_return_mae']:.4f}")
            columns[2].metric("Direction", f"{rolling_metrics['weekly_direction_accuracy']:.1%}")
            columns[3].metric("80% coverage", f"{rolling_metrics['coverage_80']:.1%}")
            columns[4].metric("Calibration error", f"{rolling_metrics['calibration_error']:.4f}")
        rolling = pd.read_csv(rolling_weekly, parse_dates=["date"])
        selected_rolling = rolling[rolling["model"] == selected]
        rolling_figure = go.Figure()
        rolling_figure.add_trace(go.Scatter(x=selected_rolling["date"], y=selected_rolling["q90"], mode="lines", line={"width": 0}, showlegend=False))
        rolling_figure.add_trace(go.Scatter(x=selected_rolling["date"], y=selected_rolling["q10"], mode="lines", line={"width": 0}, fill="tonexty", name="80% interval"))
        rolling_figure.add_trace(go.Scatter(x=selected_rolling["date"], y=selected_rolling["median_return"], mode="lines", name="Median weekly return"))
        rolling_figure.add_trace(go.Scatter(x=selected_rolling["date"], y=selected_rolling["actual_return"], mode="lines", name="Observed Yahoo-basket weekly return"))
        rolling_figure.update_layout(template="plotly_white")
        st.plotly_chart(rolling_figure, use_container_width=True)

    st.subheader("Observed Yahoo-basket outcome percentile")
    percentile = go.Figure(go.Scatter(x=selected_weekly["date"], y=selected_weekly["actual_percentile"], mode="lines"))
    percentile.update_yaxes(range=[0, 1])
    percentile.update_layout(template="plotly_white")
    st.plotly_chart(percentile, use_container_width=True)

    scenario_path = simulation_dir / f"scenario_summary_h{int(config['visualization']['default_horizon_weeks']):03d}.csv"
    if scenario_path.exists():
        st.subheader("Generated scenario clusters")
        st.dataframe(pd.read_csv(scenario_path), use_container_width=True)

    sanity_path = Path(paths["artifacts"]["simulation_sanity"])
    if sanity_path.exists():
        st.subheader("Simulation sanity audit")
        st.json(_load_json(sanity_path))


if __name__ == "__main__":
    main()
