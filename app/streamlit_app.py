"""Interactive explorer.   Run:  streamlit run app/streamlit_app.py
Requires `python -m src.train` to have been run once (creates models/ and data/processed/).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.predict import estimate, load_model
from src.visualize import stock_pivot, style_axes, _fmt_count

st.set_page_config(page_title="India's Diaspora, 2000-2026", page_icon="🌏", layout="wide")
cfg = load_config()
COLORS = cfg["plot"]["country_colors"]


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_csv(resolve(cfg, "panel_wide"))


@st.cache_data
def load_metrics() -> dict:
    return json.loads(resolve(cfg, "metrics").read_text())


@st.cache_resource
def get_model():
    return load_model(cfg)


try:
    panel, metrics, model = load_panel(), load_metrics(), get_model()
except FileNotFoundError:
    st.error("Artifacts not found. Run `python -m src.train` from the project root first.")
    st.stop()

st.title("Where India Goes: diaspora growth across 7 destinations")
st.caption("UN DESA migrant stock (quinquennial), national visa data, World Bank/ILOSTAT macro series.")

tab_explore, tab_models, tab_estimate = st.tabs(["Explore", "Model results", "Estimator"])

# ---------------------------------------------------------------- Explore
with tab_explore:
    piv = stock_pivot(panel)
    chosen = st.multiselect("Destinations", list(piv.columns), default=list(piv.columns))
    log_scale = st.toggle("Log scale", value=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c in chosen:
        s = piv[c].dropna()
        ax.plot(s.index, s.values, marker="o", lw=2.2, color=COLORS[c], label=c)
    if log_scale:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_count))
    if chosen:
        ax.legend(frameon=False)
    style_axes(ax, "Indian-born population by destination", "UN DESA stock estimates", "Residents", "Year")
    st.pyplot(fig)

    if {"UAE"} & set(chosen):
        uae = piv["UAE"].dropna()
        st.info(f"UAE peaked at {uae.max():,.0f} in {uae.idxmax()} and was {uae.iloc[-1]:,.0f} in "
                f"{uae.index[-1]}: the only contraction in the panel.")
    with st.expander("Raw panel (wide format)"):
        st.dataframe(panel)

# ---------------------------------------------------------------- Models
with tab_models:
    st.subheader("Leave-one-out CV leaderboard")
    st.dataframe(pd.DataFrame(metrics["leaderboard"]).set_index("model").style.format("{:.3f}"),
)
    st.markdown(f"Served model: **`{metrics['best_model']}`**, trained on **{metrics['n_train_rows']}** rows. "
                "At this sample size the gap between train and held-out R2 matters more than the headline score. "
                "LightGBM is statistically indistinguishable from the linear baseline here; Random Forest is clearly worse.")
    fig_path = resolve(cfg, "figures_dir") / "06_residuals_linear.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Largest misses: Germany (4 observations only) and the UK's endpoints")

# ---------------------------------------------------------------- Estimator
with tab_estimate:
    st.subheader("Estimate Indian-born population")
    st.warning("Pooled-trend estimate from 35 observations. Not a forecast: no calibrated uncertainty, "
               "and values outside 2000-2024 are extrapolation.")
    c1, c2, c3 = st.columns(3)
    country = c1.selectbox("Destination", sorted(COLORS))
    year = c2.slider("Year", 2000, 2035, 2024)
    unemp_default = float(panel.dropna(subset=["Unemployment_India"]).iloc[-1]["Unemployment_India"])
    unemp = c3.number_input("India unemployment rate (%)", 0.0, 20.0, round(unemp_default, 2), 0.1)
    est = estimate(model, country, year, unemp)
    st.metric(f"{country}, {year}", f"{est:,.0f}")
    if year > 2024 or year < 2000:
        st.caption("Outside the observed range: treat as illustrative only.")

    years = np.arange(2000, 2036)
    curve = [estimate(model, country, int(y), unemp) for y in years]
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.plot(years, curve, color=COLORS[country], lw=2.2, label="model")
    obs = stock_pivot(panel)[country].dropna()
    ax2.scatter(obs.index, obs.values, color="#222222", zorder=3, label="observed (UN DESA)")
    ax2.axvspan(2024.5, 2035.5, color="#888888", alpha=0.08)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_count)); ax2.legend(frameon=False)
    style_axes(ax2, f"{country}: model vs observed", "Shaded = extrapolation beyond data", "Residents", "Year")
    st.pyplot(fig2)
