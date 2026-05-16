"""
Concept: DashboardView
Purpose: Present predictions visually, calling `PredictionEngine` in-process.
State: Cached PredictionEngine via @st.cache_resource; selected state/year/county.
Actions: main() — orchestrates rendering.
Operational principle: Selecting IA/2024 renders a choropleth; clicking Story County
    opens a detail panel with prediction + drivers.

Deployment: This is the only inference path live on Hugging Face Spaces. The root
`app.py` imports `main` from here.

Gotchas applied:
- #10: `@st.cache_resource` for engine, `@st.cache_data` for predictions.
- #21: filtered county geojson is cached to data/processed/counties_geo.json.
- #36: SHAP explainer is reused — built once inside PredictionEngine.__init__.
- #39: cache resets only on Space restart, which is fine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paths import BEST_MODEL_PATH, COUNTIES_GEOJSON, FEATURES_PARQUET

US_COUNTIES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)


def _import_streamlit():  # pragma: no cover - import shim for environments without streamlit
    import streamlit as st

    return st


def main() -> None:
    """Entry point used by the root `app.py` and the `make dashboard` target."""
    st = _import_streamlit()
    import pandas as pd
    import plotly.express as px

    from src.inference.prediction_engine import PredictionEngine

    @st.cache_resource(show_spinner="Loading model…")
    def load_engine() -> PredictionEngine:
        return PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_predict(state: str, county_fips: str, year: int) -> dict[str, Any]:
        engine = load_engine()
        return engine.predict(state, county_fips, year).model_dump()

    @st.cache_data(show_spinner="Loading map…")
    def load_geo(states: tuple[str, ...]) -> dict:
        if COUNTIES_GEOJSON.exists():
            return json.loads(COUNTIES_GEOJSON.read_text(encoding="utf-8"))
        import requests

        full = requests.get(US_COUNTIES_GEOJSON_URL, timeout=60).json()
        state_fips_for = {"IA": "19", "IL": "17", "NE": "31"}
        keep_prefixes = {state_fips_for[s] for s in states if s in state_fips_for}
        filtered = {
            "type": "FeatureCollection",
            "features": [
                feat
                for feat in full["features"]
                if str(feat["id"])[:2] in keep_prefixes
            ],
        }
        COUNTIES_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
        COUNTIES_GEOJSON.write_text(json.dumps(filtered), encoding="utf-8")
        return filtered

    @st.cache_data(show_spinner=False)
    def features_df() -> pd.DataFrame:
        import pandas as pd

        df = pd.read_parquet(FEATURES_PARQUET)
        df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
        df["year"] = df["year"].astype(int)
        return df

    # ------------------------------------------------------------------
    # Page setup
    # ------------------------------------------------------------------
    st.set_page_config(page_title="CropIQ — Corn Yield Forecasting", page_icon="🌽", layout="wide")
    st.title("🌽 CropIQ — County-Level Corn Yield Forecasting")
    st.caption(
        "Iowa, Illinois, Nebraska · 2010–2024 · USDA NASS yields + Open-Meteo weather + "
        "USDA SSURGO soil · XGBoost with SHAP-grounded explanations."
    )

    engine = load_engine()
    metadata = engine.get_metadata()
    features = features_df()

    # ------------------------------------------------------------------
    # Sidebar selectors
    # ------------------------------------------------------------------
    states_available = sorted(features["state"].unique())
    with st.sidebar:
        st.header("Selectors")
        state = st.selectbox("State", states_available, index=0)
        years_for_state = sorted(features.loc[features["state"] == state, "year"].unique())
        default_year_idx = years_for_state.index(2024) if 2024 in years_for_state else len(years_for_state) - 1
        year = st.selectbox("Year", years_for_state, index=default_year_idx)
        st.markdown("---")
        st.markdown(f"**Model:** {metadata.model_name}")
        st.markdown(f"**Trained:** {metadata.training_date}")
        st.markdown(f"**Residual σ (val):** {metadata.residual_std:.2f} bu/acre")
        st.markdown(f"**Features:** {metadata.feature_count}")

    # ------------------------------------------------------------------
    # Build predictions for the chosen (state, year)
    # ------------------------------------------------------------------
    state_year_features = features.loc[
        (features["state"] == state) & (features["year"] == year)
    ].copy()
    if state_year_features.empty:
        st.warning(f"No feature rows for {state} {year}.")
        return

    preds = []
    for cf in state_year_features["county_fips"]:
        try:
            r = cached_predict(state, cf, int(year))
            preds.append(
                {
                    "county_fips": cf,
                    "predicted": r["predicted_yield_bu_per_acre"],
                    "ci_low": r["confidence_interval_80pct"][0],
                    "ci_high": r["confidence_interval_80pct"][1],
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Failed to predict for {cf}: {exc}")
    if not preds:
        st.error("No predictions produced.")
        return
    pred_df = pd.DataFrame(preds)
    pred_df = pred_df.merge(
        state_year_features.loc[:, ["county_fips", "yield_bu_per_acre"]],
        on="county_fips",
        how="left",
    ).rename(columns={"yield_bu_per_acre": "actual"})

    # ------------------------------------------------------------------
    # Choropleth map
    # ------------------------------------------------------------------
    col_map, col_detail = st.columns([3, 2])
    with col_map:
        st.subheader(f"Predicted yield — {state} {year}")
        geo = load_geo(tuple(states_available))
        fig = px.choropleth(
            pred_df,
            geojson=geo,
            locations="county_fips",
            color="predicted",
            color_continuous_scale="YlGn",
            range_color=(pred_df["predicted"].min(), pred_df["predicted"].max()),
            scope="usa",
            labels={"predicted": "bu/acre"},
            hover_data={"county_fips": True, "predicted": ":.1f", "actual": ":.1f"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # County detail
    # ------------------------------------------------------------------
    with col_detail:
        st.subheader("County detail")
        county_options = sorted(pred_df["county_fips"].tolist())
        default_idx = county_options.index("19169") if state == "IA" and "19169" in county_options else 0
        county_fips = st.selectbox("County FIPS", county_options, index=default_idx)
        result = engine.predict(state, county_fips, int(year))

        st.metric(
            "Predicted yield",
            f"{result.predicted_yield_bu_per_acre:.1f} bu/acre",
            help=(
                f"80% CI: [{result.confidence_interval_80pct[0]:.1f}, "
                f"{result.confidence_interval_80pct[1]:.1f}] bu/acre"
            ),
        )
        actual_row = pred_df.loc[pred_df["county_fips"] == county_fips, "actual"]
        if not actual_row.empty and pd.notna(actual_row.iloc[0]):
            st.metric("Actual (NASS)", f"{actual_row.iloc[0]:.1f} bu/acre")

        st.markdown("**Top SHAP drivers**")
        driver_df = pd.DataFrame(
            [
                {
                    "feature": d.feature,
                    "contribution": d.contribution,
                    "direction": d.direction,
                }
                for d in result.top_drivers
            ]
        )
        if not driver_df.empty:
            bar = px.bar(
                driver_df,
                x="contribution",
                y="feature",
                orientation="h",
                color="direction",
                color_discrete_map={"positive": "#2c7a3a", "negative": "#a83232"},
                labels={"contribution": "Δ bu/acre", "feature": ""},
            )
            bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=240)
            st.plotly_chart(bar, use_container_width=True)

    # ------------------------------------------------------------------
    # Predicted-vs-actual table
    # ------------------------------------------------------------------
    st.subheader(f"Predicted vs actual ({state} {year})")
    display_df = pred_df.copy()
    display_df["error_bu"] = display_df["predicted"] - display_df["actual"]
    st.dataframe(
        display_df.sort_values("predicted", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=320,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
