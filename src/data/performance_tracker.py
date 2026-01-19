from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: str
    account_value: float
    total_unrealized_pnl: float
    num_positions: int


class PerformanceTracker:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.portfolio_csv = os.path.join(self.output_dir, "portfolio_history.csv")
        self.trades_csv = os.path.join(self.output_dir, "trades.csv")

        self._ensure_portfolio_csv()
        self._ensure_trades_csv()

    def _ensure_portfolio_csv(self) -> None:
        if not os.path.exists(self.portfolio_csv):
            with open(self.portfolio_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "account_value", "total_unrealized_pnl", "num_positions"])

    def _ensure_trades_csv(self) -> None:
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "symbol", "side", "qty", "price", "success", "reason"])

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def write_snapshot(self, account_value: float, positions: Dict[str, Dict[str, Any]]) -> PortfolioSnapshot:
        total_unrealized = 0.0
        for p in positions.values():
            total_unrealized += float(p.get("unrealized_pnl", 0.0))

        snap = PortfolioSnapshot(
            timestamp=self.now_iso(),
            account_value=float(account_value),
            total_unrealized_pnl=float(total_unrealized),
            num_positions=len(positions),
        )

        with open(self.portfolio_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([snap.timestamp, snap.account_value, snap.total_unrealized_pnl, snap.num_positions])

        return snap

    def log_trade(self, symbol: str, side: str, qty: float, price: float, success: bool, reason: str) -> None:
        with open(self.trades_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([self.now_iso(), symbol, side, float(qty), float(price), bool(success), reason])
