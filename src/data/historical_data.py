from __future__ import annotations

from typing import Dict, List
import ccxt
import pandas as pd


class HistoricalDataLoader:
    """
    Fetch historical OHLCV for symbols using ccxt.
    Robust: if a symbol is unavailable, it is skipped (no crash).
    """

    def __init__(self, exchange_name: str = "binance"):
        ex_cls = getattr(ccxt, exchange_name, None)
        if ex_cls is None:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        self.exchange = ex_cls({"enableRateLimit": True})

    def _market_symbol(self, symbol: str) -> str:
        # For most cryptos on Binance, it's SYMBOL/USDT
        return f"{symbol}/USDT"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        mkt = self._market_symbol(symbol)
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        all_rows = []
        while since < end_ts:
            rows = self.exchange.fetch_ohlcv(mkt, timeframe=timeframe, since=since, limit=limit)
            if not rows:
                break
            all_rows.extend(rows)
            since = rows[-1][0] + 1

            # safety: avoid infinite loop
            if len(all_rows) > 5_000_000:
                break

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df[(df["timestamp"] >= int(pd.Timestamp(start_date).timestamp() * 1000)) &
                (df["timestamp"] <= int(pd.Timestamp(end_date).timestamp() * 1000))]
        df.reset_index(drop=True, inplace=True)
        return df

    def load_universe(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            if s == "USDC":
                continue
            try:
                df = self.fetch_ohlcv(s, timeframe, start_date, end_date)
                if df.empty or len(df) < 5:
                    # skip unavailable / too short
                    continue
                data[s] = df
            except Exception:
                # symbol not found or exchange error -> skip silently
                continue
        return data
