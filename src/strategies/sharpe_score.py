from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def sharpe_scores(
    price_history: Dict[str, pd.DataFrame],
    i: int,
    lookback_bars: int,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute rolling Sharpe score for each asset at index i.

    Sharpe = mean(returns) / std(returns)
    Returns a dict {symbol: sharpe}
    """
    sharpes: Dict[str, float] = {}

    for sym, df in price_history.items():
        if i - lookback_bars < 1 or i >= len(df):
            continue

        closes = df.loc[i - lookback_bars : i, "close"].astype(float)
        rets = closes.pct_change().dropna()

        if len(rets) < 2:
            continue

        mu = rets.mean()
        sigma = rets.std()

        if sigma < eps:
            continue

        sharpes[sym] = float(mu / sigma)

    return sharpes


def relative_sharpe_scores(
    price_history: Dict[str, pd.DataFrame],
    i: int,
    lookback_bars: int,
) -> Dict[str, float]:
    """
    Compute relative Sharpe scores:
    score_i = sharpe_i - median(sharpe_all)
    """
    sharpes = sharpe_scores(price_history, i, lookback_bars)
    if not sharpes:
        return {}

    median_sharpe = float(np.median(list(sharpes.values())))
    return {sym: s - median_sharpe for sym, s in sharpes.items()}