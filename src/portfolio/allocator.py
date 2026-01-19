from __future__ import annotations

import json
from typing import Dict, List


class TargetAllocator:
    def __init__(self, target_path: str = "config/target_allocation.json"):
        with open(target_path, "r", encoding="utf-8") as f:
            self.targets: Dict[str, float] = json.load(f)
        self._normalize()

    def _normalize(self) -> None:
        s = sum(float(v) for v in self.targets.values())
        if s <= 0:
            raise ValueError("Target weights sum <= 0")
        for k in list(self.targets.keys()):
            self.targets[k] = float(self.targets[k]) / s

    def symbols(self) -> List[str]:
        return list(self.targets.keys())

    def weights(self) -> Dict[str, float]:
        return dict(self.targets)
