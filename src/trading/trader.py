"""
Trader Module
Executes trades on Hyperliquid.
"""

from typing import Optional, Dict
from eth_account import Account
from hyperliquid.exchange import Exchange


class Trader:
    """Executes trades on Hyperliquid."""

    def __init__(self, wallet_address: str, private_key: str, base_url: str, dry_run: bool = True):
        """
        Initialize trader.

        Args:
            wallet_address: Wallet address
            private_key: Private key for signing
            base_url: Hyperliquid API URL
            dry_run: If True, simulate trades without executing
        """
        self.wallet_address = wallet_address
        self.base_url = base_url
        self.dry_run = dry_run

        if not dry_run:
            # Create wallet object
            wallet = Account.from_key(private_key)
            self.exchange = Exchange(wallet=wallet, base_url=base_url)
        else:
            self.exchange = None
            print("TRADER: Running in DRY RUN mode - no real trades will be executed")

    def execute_trade(self, symbol: str, side: str, size: float, price: float) -> bool:
        """
        Execute a trade.

        Args:
            symbol: Token symbol
            side: 'buy' or 'sell'
            size: Size in tokens
            price: Limit price

        Returns:
            True if successful
        """
        if self.dry_run:
            print(f"[DRY RUN] {side.upper()} {size} {symbol} @ ${price:.2f}")
            return True

        try:
            is_buy = (side.lower() == 'buy')

            result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=False
            )

            if result.get('status') == 'ok':
                print(f"Trade executed: {side.upper()} {size} {symbol} @ ${price:.2f}")
                return True
            else:
                print(f"Trade failed: {result}")
                return False

        except Exception as e:
            print(f"Error executing trade: {e}")
            return False

    def close_position(self, symbol: str, size: float, price: float) -> bool:
        """
        Close a position.

        Args:
            symbol: Token symbol
            size: Size to close (positive value)
            price: Limit price

        Returns:
            True if successful
        """
        if self.dry_run:
            print(f"[DRY RUN] CLOSE {size} {symbol} @ ${price:.2f}")
            return True

        try:
            result = self.exchange.order(
                name=symbol,
                is_buy=True,  # Adjust based on position direction
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True
            )

            if result.get('status') == 'ok':
                print(f"Position closed: {size} {symbol} @ ${price:.2f}")
                return True
            else:
                print(f"Close failed: {result}")
                return False

        except Exception as e:
            print(f"Error closing position: {e}")
            return False