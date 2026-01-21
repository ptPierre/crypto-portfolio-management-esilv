from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# NEW: for live mode
import ccxt

from data.historical_data import HistoricalDataLoader
from data.performance_tracker import PerformanceTracker
from data.price_logger import PriceLogger
from data.status_writer import BotStatus, StatusWriter
from portfolio.allocator import DynamicAllocator
from portfolio.rebalancer import compute_rebalance_orders
from strategies.sharpe_score import relative_sharpe_scores
from trading.paper_portfolio import PaperPortfolio


def _utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _tf_to_seconds(tf: str) -> int:
    """
    Rough mapping for common ccxt timeframes.
    """
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    # fallback
    return 60


class PortfolioBot:
    def __init__(self):
        load_dotenv("config/.env")

        self.exchange = os.getenv("EXCHANGE", "binance")
        self.timeframe = os.getenv("TIMEFRAME", "1h")
        self.start_date = os.getenv("START_DATE", "2024-01-01")
        self.end_date = os.getenv("END_DATE", "2024-06-30")

        # NEW: mode
        self.mode = os.getenv("MODE", "backtest").lower()  # "backtest" or "live"
        self.live_poll_seconds = float(os.getenv("LIVE_POLL_SECONDS", "10"))
        self.live_bootstrap_bars = int(os.getenv("LIVE_BOOTSTRAP_BARS", "200"))

        self.initial_cash = float(os.getenv("INITIAL_CASH", "10000"))
        self.fee_rate = float(os.getenv("FEE_RATE", "0.0005"))
        self.slippage_bps = float(os.getenv("SLIPPAGE_BPS", "5"))

        self.week_every = int(os.getenv("WEEK_EVERY_N_BARS", "168"))
        self.month_every = int(os.getenv("MONTH_EVERY_N_BARS", "720"))
        self.sleep_per_bar = float(os.getenv("SLEEP_PER_BAR", "0.1"))

        self.dca_rate = float(os.getenv("DCA_WEEKLY_RATE", "0.05"))
        self.rebalance_period = os.getenv("REBALANCE_PERIOD", "monthly")

        self.score_lookback = int(os.getenv("SCORE_LOOKBACK_BARS", "72"))

        self.usdc_reserve_target = float(os.getenv("USDC_RESERVE_TARGET", "0.05"))

        self.tracker = PerformanceTracker("logs")
        self.price_logger = PriceLogger("logs/prices_long.csv")
        self.status_writer = StatusWriter("logs/bot_status.json")

        self.allocator = DynamicAllocator(
        config_path="config/target_allocation.json",
        score_scale=1.0  # ajustable
        )

        self.universe_requested = [
            s for s in self.allocator.symbols() if s != "USDC"
        ]
        # Paper portfolio
        self.portfolio = PaperPortfolio(
            initial_cash=self.initial_cash,
            fee_rate=self.fee_rate,
            slippage_bps=self.slippage_bps,
            state_path="logs/portfolio_state.json",
        )

        # -----------------------------
        # Backtest: load full history
        # Live: create a rolling history from ccxt (bootstrap)
        # -----------------------------
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.universe: List[str] = []

        if self.mode == "live":
            self._init_live()
            # N is infinite in live; we still keep a max_steps for UI
            self.N = -1
        else:
            self._init_backtest()
            self.N = min(len(df) for df in self.price_history.values())
            if self.N < max(10, self.score_lookback + 5):
                raise ValueError("Not enough historical bars for the selected timeframe/date range.")
            

    def _compute_scores(self, i: int) -> Dict[str, float]:
        """
        Rolling relative Sharpe score used for dynamic allocation.
        """
        return relative_sharpe_scores(
            self.price_history,
            i,
            lookback_bars=self.score_lookback,
        )

    # ---------- INIT METHODS ----------
    def _init_backtest(self) -> None:
        self.data_loader = HistoricalDataLoader(self.exchange)
        self.price_history = self.data_loader.load_universe(
            self.universe_requested, self.timeframe, self.start_date, self.end_date
        )

        self.universe = sorted(self.price_history.keys())
        if not self.universe:
            raise ValueError("No symbols available from data source. Check tickers or exchange.")

    def _init_live(self) -> None:
        # Create ccxt exchange instance
        ex_class = getattr(ccxt, self.exchange, None)
        if ex_class is None:
            raise ValueError(f"Exchange '{self.exchange}' not found in ccxt.")

        self.ccxt_ex = ex_class({"enableRateLimit": True})
        # Some exchanges need this; safe to try
        try:
            self.ccxt_ex.load_markets()
        except Exception:
            pass

        # Filter to symbols that exist on exchange
        available = []
        for s in self.universe_requested:
            if hasattr(self.ccxt_ex, "markets") and self.ccxt_ex.markets:
                if s in self.ccxt_ex.markets:
                    available.append(s)
            else:
                # If markets not loaded, just attempt later
                available.append(s)

        self.universe = sorted(list(set(available)))
        if not self.universe:
            raise ValueError("No symbols available for live mode. Check tickers or exchange markets.")

        # Bootstrap recent OHLCV so momentum can work immediately
        bootstrap = max(self.live_bootstrap_bars, self.score_lookback + 10)

        for sym in self.universe:
            try:
                ohlcv = self.ccxt_ex.fetch_ohlcv(sym, timeframe=self.timeframe, limit=bootstrap)
                if not ohlcv or len(ohlcv) < 5:
                    continue
                df = pd.DataFrame(ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                # Normalize columns to match backtest format
                df["timestamp"] = (df["timestamp_ms"] / 1000).astype(int)
                df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).astype(str)
                df = df[["timestamp", "datetime", "close"]].copy()
                df.reset_index(drop=True, inplace=True)
                self.price_history[sym] = df
            except Exception:
                continue

        # Keep only symbols that successfully bootstrapped
        self.universe = sorted(self.price_history.keys())
        if not self.universe:
            raise ValueError("Live bootstrap failed for all symbols (fetch_ohlcv). Try fewer symbols or another exchange.")

        # For live loop: track last bar timestamp so we only act on new bars
        self._last_live_bar_ts: Optional[int] = None

    # ---------- HELPERS ----------
    def _prices_at(self, i: int) -> Dict[str, float]:
        prices = {}
        for sym, df in self.price_history.items():
            prices[sym] = float(df.loc[i, "close"])
        return prices

    def _log_prices_at(self, i: int) -> None:
        for sym, df in self.price_history.items():
            row = df.loc[i]
            self.price_logger.log_close(
                timestamp=int(row["timestamp"]),
                dt_str=str(row["datetime"]),
                symbol=sym,
                close=float(row["close"]),
            )

    def _current_values(self, prices: Dict[str, float]) -> Dict[str, float]:
        vals = {}
        for sym, pos in self.portfolio.positions.items():
            vals[sym] = pos.qty * float(prices.get(sym, 0.0))
        return vals

    def _cash_reserve_min(self, equity: float) -> float:
        return float(self.usdc_reserve_target) * float(equity)
    
    def _weights_ex_usdc(self, weights: Dict[str, float]) -> Dict[str, float]:
        w = {s: float(v) for s, v in weights.items() if s != "USDC"}
        ssum = sum(w.values())
        if ssum <= 0:
            return {}
        return {s: v / ssum for s, v in w.items()}

    # ---------- LIVE BAR FETCH ----------
    def _fetch_latest_live_bar(self) -> Optional[Tuple[int, Dict[str, float], Dict[str, Tuple[int, str, float]]]]:
        """
        Fetch latest OHLCV candle for each symbol.
        Returns:
          - bar_ts (seconds, int): chosen common bar timestamp (min across symbols to be safe)
          - prices dict: {symbol: close}
          - rows dict: {symbol: (timestamp, datetime_str, close)} for logging
        If the latest bar timestamp hasn't changed since last call, returns None.
        """
        closes: Dict[str, float] = {}
        rows: Dict[str, Tuple[int, str, float]] = {}
        bar_ts_candidates: List[int] = []

        for sym in self.universe:
            try:
                # limit=2 so we can handle exchanges returning current forming bar
                ohlcv = self.ccxt_ex.fetch_ohlcv(sym, timeframe=self.timeframe, limit=2)
                if not ohlcv:
                    continue
                last = ohlcv[-1]
                ts_ms = int(last[0])
                close = float(last[4])
                ts_s = ts_ms // 1000
                dt_str = datetime.fromtimestamp(ts_s, tz=timezone.utc).isoformat()
                closes[sym] = close
                rows[sym] = (ts_s, dt_str, close)
                bar_ts_candidates.append(ts_s)
            except Exception:
                continue

        if not closes or not bar_ts_candidates:
            return None

        # Choose a common bar timestamp to index "i" coherently
        bar_ts = min(bar_ts_candidates)

        # Only proceed on new bar
        if self._last_live_bar_ts is not None and bar_ts <= self._last_live_bar_ts:
            return None

        self._last_live_bar_ts = bar_ts
        return bar_ts, closes, rows

    def _append_live_bar_to_history(self, bar_ts: int, rows: Dict[str, Tuple[int, str, float]]) -> None:
        """
        Append latest bar to in-memory price_history DataFrames.
        Keep only a rolling window to avoid memory blow-up.
        """
        keep = max(self.live_bootstrap_bars, self.score_lookback + 50)

        for sym, (ts_s, dt_str, close) in rows.items():
            if sym not in self.price_history:
                self.price_history[sym] = pd.DataFrame(columns=["timestamp", "datetime", "close"])
            df = self.price_history[sym]
            # append
            df.loc[len(df)] = [int(ts_s), str(dt_str), float(close)]
            # keep rolling
            if len(df) > keep:
                self.price_history[sym] = df.iloc[-keep:].reset_index(drop=True)

        # Ensure universe consistent with available histories
        self.universe = sorted(self.price_history.keys())

    # ---------- MAIN LOOP ----------
    def run(self) -> None:
        if self.mode == "live":
            self._run_live()
        else:
            self._run_backtest()

    def _run_backtest(self) -> None:
        for i in range(self.N):
            prices = self._prices_at(i)
            self._log_prices_at(i)

            equity = self.portfolio.equity(prices)
            reserve_min = self._cash_reserve_min(equity)

            # ---------- SNAPSHOT ----------
            positions_view = self.portfolio.positions_view(prices)
            self.tracker.write_snapshot(
                account_value=equity,
                positions={sym: {"unrealized_pnl": v["unrealized_pnl"]} for sym, v in positions_view.items()},
            )

            self.status_writer.write(
                BotStatus(
                    running=True,
                    mode="PAPER_BACKTEST_ACCEL",
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    step=i,
                    max_steps=self.N,
                    last_update_utc=StatusWriter.now_iso(),
                    rebalance_period=self.rebalance_period,
                    dca_rate=self.dca_rate,
                )
            )

            # ============================================================
            # WEEKLY DCA (dynamic allocation)
            # ============================================================
            do_weekly = (i % max(1, self.week_every) == 0) and (i >= self.score_lookback)

            if do_weekly:
                invest_usd = self.dca_rate * equity
                invest_usd = min(invest_usd, max(0.0, self.portfolio.cash - reserve_min))

                if invest_usd > 0:
                    scores = self._compute_scores(i)
                    weights = self._weights_ex_usdc(self.allocator.allocate(scores))

                    for sym, wi in weights.items():
                        if sym == "USDC":
                            continue
                        if sym not in self.universe:
                            continue

                        usd = invest_usd * wi
                        px = float(prices.get(sym, 0.0))
                        if px <= 0 or usd <= 0:
                            continue

                        res = self.portfolio.buy_value(sym, usd, px)
                        self.tracker.log_trade(
                            sym,
                            "buy",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Weekly DCA (dynamic allocation)"
                            if res.get("success")
                            else res.get("error", ""),
                        )

            # ============================================================
            # PERIODIC REBALANCE (dynamic allocation)
            # ============================================================
            do_rebalance = False
            if self.rebalance_period == "monthly":
                do_rebalance = (i % max(1, self.month_every) == 0) and (i >= self.score_lookback)
            elif self.rebalance_period == "quarterly":
                do_rebalance = (i % max(1, 3 * self.month_every) == 0) and (i >= self.score_lookback)

            if do_rebalance:
                scores = self._compute_scores(i)
                target_weights = self._weights_ex_usdc(self.allocator.allocate(scores))

                current_vals = self._current_values(prices)
                orders = compute_rebalance_orders(
                    prices,
                    current_vals,
                    target_weights,
                    equity,
                    min_trade_usd=20.0,
                )

                for sym, delta in orders:
                    px = float(prices.get(sym, 0.0))
                    if px <= 0:
                        continue

                    if delta < 0:
                        res = self.portfolio.sell_value(sym, abs(delta), px)
                        self.tracker.log_trade(
                            sym,
                            "sell",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Rebalance sell"
                            if res.get("success")
                            else res.get("error", ""),
                        )
                    else:
                        available = max(0.0, self.portfolio.cash - reserve_min)
                        usd = min(delta, available)
                        if usd <= 0:
                            continue

                        res = self.portfolio.buy_value(sym, usd, px)
                        self.tracker.log_trade(
                            sym,
                            "buy",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Rebalance buy"
                            if res.get("success")
                            else res.get("error", ""),
                        )

            time.sleep(max(0.0, self.sleep_per_bar))

        # ---------- END ----------
        self.status_writer.write(
            BotStatus(
                running=False,
                mode="PAPER_BACKTEST_ACCEL",
                exchange=self.exchange,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
                step=self.N,
                max_steps=self.N,
                last_update_utc=StatusWriter.now_iso(),
                rebalance_period=self.rebalance_period,
                dca_rate=self.dca_rate,
            )
        )

    def _run_live(self) -> None:
        i = 0
        tf_seconds = _tf_to_seconds(self.timeframe)
        min_sleep = max(1.0, min(self.live_poll_seconds, float(tf_seconds)))

        while True:
            got = self._fetch_latest_live_bar()
            if got is None:
                time.sleep(min_sleep)
                continue

            bar_ts, prices, rows = got
            self._append_live_bar_to_history(bar_ts, rows)

            i = min(len(df) for df in self.price_history.values()) - 1
            if i < self.score_lookback:
                time.sleep(min_sleep)
                continue

            # ---------- LOG PRICES ----------
            for sym, (ts_s, dt_str, close) in rows.items():
                self.price_logger.log_close(
                    timestamp=int(ts_s),
                    dt_str=str(dt_str),
                    symbol=sym,
                    close=float(close),
                )

            equity = self.portfolio.equity(prices)
            reserve_min = self._cash_reserve_min(equity)

            positions_view = self.portfolio.positions_view(prices)
            self.tracker.write_snapshot(
                account_value=equity,
                positions={sym: {"unrealized_pnl": v["unrealized_pnl"]} for sym, v in positions_view.items()},
            )

            self.status_writer.write(
                BotStatus(
                    running=True,
                    mode="PAPER_LIVE",
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    step=i,
                    max_steps=-1,
                    last_update_utc=StatusWriter.now_iso(),
                    rebalance_period=self.rebalance_period,
                    dca_rate=self.dca_rate,
                )
            )

            # ============================================================
            # WEEKLY DCA (live)
            # ============================================================
            do_weekly = (i % max(1, self.week_every) == 0)

            if do_weekly:
                invest_usd = self.dca_rate * equity
                invest_usd = min(invest_usd, max(0.0, self.portfolio.cash - reserve_min))

                if invest_usd > 0:
                    scores = self._compute_scores(i)
                    weights = self._weights_ex_usdc(self.allocator.allocate(scores))

                    for sym, wi in weights.items():
                        if sym == "USDC":
                            continue
                        if sym not in self.universe:
                            continue

                        usd = invest_usd * wi
                        px = float(prices.get(sym, 0.0))
                        if px <= 0 or usd <= 0:
                            continue

                        res = self.portfolio.buy_value(sym, usd, px)
                        self.tracker.log_trade(
                            sym,
                            "buy",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Weekly DCA (live, dynamic allocation)"
                            if res.get("success")
                            else res.get("error", ""),
                        )

            # ============================================================
            # PERIODIC REBALANCE (live)
            # ============================================================
            do_rebalance = False
            if self.rebalance_period == "monthly":
                do_rebalance = (i % max(1, self.month_every) == 0)
            elif self.rebalance_period == "quarterly":
                do_rebalance = (i % max(1, 3 * self.month_every) == 0)

            if do_rebalance:
                scores = self._compute_scores(i)
                target_weights = self._weights_ex_usdc(self.allocator.allocate(scores))

                current_vals = self._current_values(prices)
                orders = compute_rebalance_orders(
                    prices,
                    current_vals,
                    target_weights,
                    equity,
                    min_trade_usd=20.0,
                )

                for sym, delta in orders:
                    px = float(prices.get(sym, 0.0))
                    if px <= 0:
                        continue

                    if delta < 0:
                        res = self.portfolio.sell_value(sym, abs(delta), px)
                        self.tracker.log_trade(
                            sym,
                            "sell",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Rebalance sell (live)"
                            if res.get("success")
                            else res.get("error", ""),
                        )
                    else:
                        available = max(0.0, self.portfolio.cash - reserve_min)
                        usd = min(delta, available)
                        if usd <= 0:
                            continue

                        res = self.portfolio.buy_value(sym, usd, px)
                        self.tracker.log_trade(
                            sym,
                            "buy",
                            res.get("qty", 0.0),
                            res.get("price", px),
                            bool(res.get("success", False)),
                            "Rebalance buy (live)"
                            if res.get("success")
                            else res.get("error", ""),
                        )

            time.sleep(min_sleep)


def main():
    bot = PortfolioBot()
    bot.run()


if __name__ == "__main__":
    main()