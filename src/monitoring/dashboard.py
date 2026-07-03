"""Streamlit dashboard over the trade journal (read-only).

Run locally with the dashboard extras installed:

    pip install -r requirements-dashboard.txt
    streamlit run src/monitoring/dashboard.py

This module is a thin shell: every query and figure comes from
``src.monitoring.queries`` (unit-tested headlessly); streamlit is imported
only here. Excluded from coverage for that reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from sqlalchemy import Engine

from config.settings import get_settings
from src.monitoring import queries

st.set_page_config(page_title="PaperPilot", layout="wide")


@st.cache_resource
def _engine(db_url: str) -> Engine:
    return queries.read_only_engine(db_url)


@st.cache_data(ttl=30)
def _equity(db_url: str) -> pd.DataFrame:
    return queries.load_equity_curve(_engine(db_url))


@st.cache_data(ttl=30)
def _orders(db_url: str) -> pd.DataFrame:
    return queries.load_recent_orders(_engine(db_url))


@st.cache_data(ttl=30)
def _signals(db_url: str) -> pd.DataFrame:
    return queries.load_recent_signals(_engine(db_url))


@st.cache_data(ttl=30)
def _halt(db_url: str) -> queries.HaltStatus:
    return queries.load_halt_status(db_url)


settings = get_settings()
db_url = settings.db_url

st.title("PaperPilot")
st.caption(f"Journal: `{db_url}` (read-only). Educational project; not financial advice.")

equity_df = _equity(db_url)
halt = _halt(db_url)

col_equity, col_drawdown, col_halt = st.columns(3)
latest_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 0.0
col_equity.metric("Equity", f"${latest_equity:,.2f}")
col_drawdown.metric("Drawdown from peak", f"{queries.compute_drawdown(equity_df):.2%}")
col_halt.metric("Halt state", "HALTED" if halt.halted else "OK", help=halt.detail)

if halt.halted:
    st.error(
        f"Trading halted: {halt.detail}. Clear with `python scripts/run_live.py "
        f"--reset-halt` after review."
    )

st.plotly_chart(queries.build_equity_figure(equity_df), use_container_width=True)

col_orders, col_signals = st.columns(2)
with col_orders:
    st.subheader("Recent orders")
    st.dataframe(_orders(db_url), use_container_width=True, hide_index=True)
with col_signals:
    st.subheader("Recent signals")
    st.dataframe(_signals(db_url), use_container_width=True, hide_index=True)

if st.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()
