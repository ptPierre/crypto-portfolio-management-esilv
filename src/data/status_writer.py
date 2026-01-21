from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class BotStatus:
    running: bool
    mode: str
    exchange: str
    timeframe: str
    start_date: str
    end_date: str
    step: int
    max_steps: int
    last_update_utc: str
    rebalance_period: str
    dca_rate: float


class StatusWriter:
    def __init__(self, path: str = "logs/bot_status.json"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def write(self, status: BotStatus) -> None:
        payload = asdict(status)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
