from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json, os

@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PaperPortfolio:
    def __init__(self, initial_cash: float, fee_rate: float, slippage_bps: float, state_path: str="logs/portfolio_state.json"):
        self.cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.positions: Dict[str, Position] = {}
        self.state_path = state_path
        os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.state_path):
            self.save(); return
        data = json.load(open(self.state_path, "r", encoding="utf-8"))
        self.cash = float(data.get("cash", self.cash))
        self.positions = {}
        for sym, p in data.get("positions", {}).items():
            self.positions[sym] = Position(sym, float(p["qty"]), float(p["entry_price"]))

    def save(self) -> None:
        data = {"cash": self.cash, "positions": {s: p.to_dict() for s,p in self.positions.items()}}
        json.dump(data, open(self.state_path, "w", encoding="utf-8"), indent=2)

    def equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            eq += pos.qty * float(prices.get(sym, 0.0))
        return float(eq)

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        pnl = 0.0
        for sym, pos in self.positions.items():
            px = float(prices.get(sym, 0.0))
            pnl += pos.qty * (px - pos.entry_price)
        return float(pnl)

    def positions_view(self, prices: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for sym, pos in self.positions.items():
            px = float(prices.get(sym, 0.0))
            out[sym] = {
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "price": px,
                "position_value": pos.qty * px,
                "unrealized_pnl": pos.qty * (px - pos.entry_price),
            }
        return out

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = (self.slippage_bps / 10_000.0) * price
        return price + slip if side == "buy" else price - slip

    def buy_value(self, symbol: str, usd_value: float, price: float) -> Dict[str, Any]:
        if usd_value <= 0:
            return {"success": False, "error": "usd_value<=0"}
        exec_price = self._apply_slippage(price, "buy")
        qty = usd_value / exec_price
        fee = usd_value * self.fee_rate
        total_cost = usd_value + fee
        if total_cost > self.cash:
            return {"success": False, "error": f"Not enough cash. Need {total_cost:.2f}, have {self.cash:.2f}"}

        self.cash -= total_cost
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, qty, exec_price)
        else:
            p = self.positions[symbol]
            new_qty = p.qty + qty
            p.entry_price = (p.entry_price * p.qty + exec_price * qty) / new_qty
            p.qty = new_qty

        self.save()
        return {"success": True, "qty": qty, "price": exec_price, "fee": fee}

    def sell_value(self, symbol: str, usd_value: float, price: float) -> Dict[str, Any]:
        """
        Sell enough qty to free approx usd_value (based on exec price).
        """
        if symbol not in self.positions:
            return {"success": False, "error": "No position"}

        usd_value = float(usd_value)
        if usd_value <= 0:
            return {"success": False, "error": "usd_value<=0"}

        p = self.positions[symbol]
        exec_price = self._apply_slippage(float(price), "sell")

        qty = usd_value / exec_price
        qty = min(qty, p.qty)
        if qty <= 0:
            return {"success": False, "error": "qty<=0"}

        gross = qty * exec_price
        fee = gross * self.fee_rate
        net = gross - fee

        realized = qty * (exec_price - p.entry_price)
        self.cash += net

        p.qty -= qty
        if p.qty <= 1e-12:
            del self.positions[symbol]

        self.save()
        return {"success": True, "qty": qty, "price": exec_price, "fee": fee, "realized_pnl": realized}


    def sell_qty(self, symbol: str, qty: float, price: float) -> Dict[str, Any]:
        if symbol not in self.positions:
            return {"success": False, "error": "No position"}
        p = self.positions[symbol]
        qty = min(float(qty), p.qty)
        if qty <= 0:
            return {"success": False, "error": "qty<=0"}
        exec_price = self._apply_slippage(price, "sell")
        gross = qty * exec_price
        fee = gross * self.fee_rate
        net = gross - fee

        realized = qty * (exec_price - p.entry_price)
        self.cash += net

        p.qty -= qty
        if p.qty <= 1e-12:
            del self.positions[symbol]

        self.save()
        return {"success": True, "qty": qty, "price": exec_price, "fee": fee, "realized_pnl": realized}
