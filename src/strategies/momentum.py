from __future__ import annotations

from typing import Dict, List
import pandas as pd


def momentum_scores(price_history: Dict[str, pd.DataFrame], i: int, lookback_bars: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for sym, df in price_history.items():
        if i - lookback_bars < 0 or i >= len(df):
            continue
        p0 = float(df.loc[i - lookback_bars, "close"])
        p1 = float(df.loc[i, "close"])
        if p0 > 0:
            scores[sym] = (p1 / p0) - 1.0
    return scores


def select_top_k(scores: Dict[str, float], k: int) -> List[str]:
    return [s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
