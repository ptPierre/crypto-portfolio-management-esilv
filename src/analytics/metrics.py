from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def bars_per_year_from_timeframe(timeframe: str) -> int:
    # rough annualization
    tf = timeframe.strip().lower()
    if tf.endswith("h"):
        hours = int(tf[:-1])
        return int(365 * 24 / hours)
    if tf.endswith("d"):
        days = int(tf[:-1])
        return int(365 / days)
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        return int(365 * 24 * 60 / minutes)
    # fallback
    return 365 * 24


def compute_metrics(equity_series: pd.Series, timeframe: str) -> Dict[str, float]:
    if len(equity_series) < 3:
        return {"return_total": 0.0, "vol": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    equity_series = equity_series.astype(float)
    rets = equity_series.pct_change().dropna()

    total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0)

    bpy = bars_per_year_from_timeframe(timeframe)
    vol = float(rets.std() * np.sqrt(bpy))
    mean = float(rets.mean() * bpy)
    sharpe = float(mean / vol) if vol > 1e-12 else 0.0

    roll_max = equity_series.cummax()
    dd = (equity_series / roll_max) - 1.0
    mdd = float(dd.min())

    return {"return_total": total_return, "vol": vol, "max_drawdown": mdd, "sharpe": sharpe}
