"""Read-only journal queries and figure builders for the dashboard.

Everything here is a pure function over a SQLAlchemy engine or a DataFrame, so
the dashboard's logic is unit-testable headlessly. Only the thin Streamlit
shell (``dashboard.py``) imports streamlit; plotly is imported inside the one
figure builder so this module works without the dashboard extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from src.journal.store import Journal


def read_only_engine(db_url: str) -> Engine:
    """An engine that cannot write, so the dashboard can never lock the journal."""
    url = make_url(db_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, ":memory:"):
        return create_engine(
            f"sqlite:///file:{url.database}?mode=ro&uri=true",
            connect_args={"uri": True},
        )
    return create_engine(db_url)


def load_equity_curve(engine: Engine, limit: int = 2000) -> pd.DataFrame:
    """Equity snapshots as a DataFrame (ts, equity, cash, halted, halt_reason)."""
    query = (
        "SELECT ts, equity, cash, halted, halt_reason FROM equity_snapshots "
        f"ORDER BY id DESC LIMIT {int(limit)}"
    )
    df = pd.read_sql(query, engine)
    return df.iloc[::-1].reset_index(drop=True)  # oldest-first for plotting


def load_recent_orders(engine: Engine, limit: int = 50) -> pd.DataFrame:
    query = (
        "SELECT ts, symbol, side, qty, status, filled_qty, filled_avg_price, reason "
        f"FROM orders ORDER BY id DESC LIMIT {int(limit)}"
    )
    return pd.read_sql(query, engine)


def load_recent_signals(engine: Engine, limit: int = 50) -> pd.DataFrame:
    query = (
        "SELECT ts, symbol, strategy, action, confidence, reason "
        f"FROM signals ORDER BY id DESC LIMIT {int(limit)}"
    )
    return pd.read_sql(query, engine)


@dataclass(frozen=True)
class HaltStatus:
    halted: bool
    detail: str


def load_halt_status(db_url: str) -> HaltStatus:
    """Current halt state from the journal's latest-transition view."""
    states = Journal(db_url).latest_halt_states()
    active = {name: row for name, row in states.items() if row.active}
    if not active:
        return HaltStatus(False, "not halted")
    detail = "; ".join(f"{name}: {row.reason or 'tripped'}" for name, row in active.items())
    return HaltStatus(True, detail)


def compute_drawdown(equity_df: pd.DataFrame) -> float:
    """Current drawdown fraction from the running equity peak (0.0 if empty)."""
    if equity_df.empty:
        return 0.0
    equity = equity_df["equity"].astype(float)
    peak = equity.cummax().iloc[-1]
    if peak <= 0:
        return 0.0
    return float((peak - equity.iloc[-1]) / peak)


def build_equity_figure(equity_df: pd.DataFrame) -> Any:
    """A plotly equity-curve figure (imported lazily; dashboard extra only)."""
    import plotly.graph_objects as go

    figure = go.Figure()
    if not equity_df.empty:
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(equity_df["ts"]),
                y=equity_df["equity"].astype(float),
                mode="lines",
                name="equity",
            )
        )
    figure.update_layout(
        title="Equity curve",
        xaxis_title=None,
        yaxis_title="Equity ($)",
        margin={"l": 40, "r": 20, "t": 40, "b": 30},
    )
    return figure
