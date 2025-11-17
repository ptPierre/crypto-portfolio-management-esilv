"""
Data Fetcher Module
Fetches market data from Hyperliquid for specified tokens.
"""

from typing import Dict, List, Optional
from hyperliquid.info import Info


class DataFetcher:
    """Fetches market data from Hyperliquid."""

    def __init__(self, base_url: str):
        """
        Initialize data fetcher.

        Args:
            base_url: Hyperliquid API URL
        """
        self.base_url = base_url
        self.info = Info(base_url=base_url, skip_ws=True)

    def get_token_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch current market data for given symbols.

        Args:
            symbols: List of token symbols (e.g., ["BTC", "ETH"])

        Returns:
            Dictionary mapping symbol to market data
        """
        token_data = {}

        try:
            # Get all mid prices
            all_mids = self.info.all_mids()

            # Get meta and contexts for funding rates
            meta = self.info.meta()
            contexts = self.info.meta_and_asset_ctxs()
            universe = meta.get('universe', [])
            asset_ctxs = contexts[1]

            # Build index mapping
            symbol_to_idx = {}
            for idx, asset in enumerate(universe):
                if isinstance(asset, str):
                    symbol_to_idx[asset] = idx
                elif isinstance(asset, dict):
                    symbol_to_idx[asset.get('name')] = idx

            # Fetch data for each symbol
            for symbol in symbols:
                try:
                    # Get price
                    price = float(all_mids.get(symbol, 0))
                    if price == 0:
                        continue

                    # Get funding rate if available
                    funding_rate = None
                    idx = symbol_to_idx.get(symbol)
                    if idx is not None and idx < len(asset_ctxs):
                        ctx = asset_ctxs[idx]
                        funding_8h = float(ctx.get('funding', 0))
                        funding_rate = funding_8h / 8.0

                    token_data[symbol] = {
                        'symbol': symbol,
                        'price': price,
                        'funding_rate': funding_rate,
                        'timestamp': None  # Add timestamp if needed
                    }

                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    continue

            return token_data

        except Exception as e:
            print(f"Error in get_token_data: {e}")
            return {}

    def get_user_positions(self, wallet_address: str) -> Dict:
        """
        Get current user positions.

        Args:
            wallet_address: User's wallet address

        Returns:
            Dictionary of current positions
        """
        try:
            user_state = self.info.user_state(wallet_address)

            positions = {}
            for asset_position in user_state.get('assetPositions', []):
                position = asset_position['position']
                symbol = position['coin']
                size = float(position['szi'])

                if size != 0:
                    positions[symbol] = {
                        'size': size,
                        'entry_price': float(position['entryPx']),
                        'unrealized_pnl': float(position['unrealizedPnl']),
                        'position_value': float(position['positionValue'])
                    }

            return positions

        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}