from __future__ import annotations

from typing import Dict, List, Tuple


def compute_rebalance_orders(
    prices: Dict[str, float],
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    equity: float,
    min_trade_usd: float = 10.0,
) -> List[Tuple[str, float]]:
    """
    Returns list of (symbol, delta_usd):
      delta_usd > 0 => buy delta_usd
      delta_usd < 0 => sell abs(delta_usd)
    """
    orders: List[Tuple[str, float]] = []

    for sym, w in target_weights.items():
        if sym == "USDC":
            continue
        px = float(prices.get(sym, 0.0))
        if px <= 0:
            continue

        target_val = equity * float(w)
        cur_val = float(current_values.get(sym, 0.0))
        delta = target_val - cur_val

        if abs(delta) >= min_trade_usd:
            orders.append((sym, delta))

    return orders
