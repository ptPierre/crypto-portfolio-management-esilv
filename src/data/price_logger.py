from __future__ import annotations

import os
import csv


class PriceLogger:
    def __init__(self, path: str = "logs/prices_long.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "datetime", "symbol", "close"])

    def log_close(self, timestamp: int, dt_str: str, symbol: str, close: float) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([int(timestamp), dt_str, symbol, float(close)])
