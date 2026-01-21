from __future__ import annotations

import json
from typing import Dict, List


class DynamicAllocator:
    """
    Dynamic portfolio allocator.

    - Uses a base allocation as a reference (base_weight)
    - Deforms weights using a score (e.g. Sharpe)
    - Applies min / max constraints
    - Ensures USDC minimum reserve
    """

    def __init__(
        self,
        config_path: str = "config/target_allocation.json",
        score_scale: float = 1.0,
    ):
        """
        score_scale controls how aggressively scores impact weights.
        Typical values: 0.5 (conservative) to 1.5 (aggressive)
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Dict[str, float]] = json.load(f)

        self.score_scale = float(score_scale)
        self._validate_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        for sym, cfg in self.config.items():
            if "base_weight" not in cfg:
                raise ValueError(f"{sym}: missing base_weight")
            if "min" not in cfg or "max" not in cfg:
                raise ValueError(f"{sym}: missing min/max")
            if cfg["min"] > cfg["max"]:
                raise ValueError(f"{sym}: min > max")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def symbols(self) -> List[str]:
        return list(self.config.keys())

    def allocate(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute dynamic weights from scores.

        Parameters
        ----------
        scores : dict
            score per asset (e.g. Sharpe). Can be negative.

        Returns
        -------
        weights : dict
            normalized portfolio weights
        """

        # 1️⃣ Start from base weights
        raw_weights: Dict[str, float] = {}
        for sym, cfg in self.config.items():
            base = float(cfg["base_weight"])
            score = float(scores.get(sym, 0.0))

            # deformation function
            # score > 0  -> overweight
            # score <= 0 -> no reinforcement
            adj = max(0.0, 1.0 + self.score_scale * score)

            raw_weights[sym] = base * adj

        # 2️⃣ Apply min / max constraints
        constrained: Dict[str, float] = {}
        for sym, w in raw_weights.items():
            cfg = self.config[sym]
            constrained[sym] = min(max(w, cfg["min"]), cfg["max"])

        # 3️⃣ Ensure USDC minimum reserve
        if "USDC" in constrained:
            usdc_min = self.config["USDC"]["min"]
            constrained["USDC"] = max(constrained["USDC"], usdc_min)

        # 4️⃣ Renormalize to 1
        total = sum(constrained.values())
        if total <= 0:
            raise ValueError("Total allocation weight <= 0")

        weights = {sym: w / total for sym, w in constrained.items()}
        return weights