import os
import sys
import time
from typing import Dict, Optional
from dotenv import load_dotenv
from hyperliquid.utils import constants

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.csv_loader import TokenLoader
from data.data_fetcher import DataFetcher
from trading.trader import Trader
from algorithms.simple_momentum import SimpleMomentumAlgorithm




class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self):
        """Initialize trading bot components."""

        # Load environment variables
        load_dotenv('../config/.env')

        # Configuration
        self.use_testnet = os.getenv('USE_TESTNET', 'True').lower() == 'true'
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '60'))
        self.dry_run = os.getenv('DRY_RUN', 'True').lower() == 'true'
        self.wallet_address = os.getenv('HYPERLIQUID_WALLET', '')
        self.private_key = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')

        # Set API URL
        self.base_url = constants.TESTNET_API_URL if self.use_testnet else constants.MAINNET_API_URL

        print(f"\n{'='*70}")
        print(f"TRADING BOT - {'TESTNET' if self.use_testnet else 'MAINNET'}")
        print(f"{'='*70}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        print(f"Wallet: {self.wallet_address[:8]}...{self.wallet_address[-6:]}")
        print(f"Check interval: {self.check_interval}s")
        print(f"{'='*70}\n")

        # Initialize components
        self.token_loader = TokenLoader('../config/tokens.csv')
        self.data_fetcher = DataFetcher(self.base_url)
        self.trader = Trader(self.wallet_address, self.private_key, self.base_url, self.dry_run)

        # Initialize algorithm (can be swapped out)
        self.algorithm = SimpleMomentumAlgorithm(funding_threshold=0.0001)

        # Load tokens
        self.tokens = self.token_loader.load_tokens()
        self.symbols = self.token_loader.get_enabled_symbols()

        if not self.symbols:
            raise ValueError("No tokens loaded from CSV")

    def run_iteration(self):
        """Execute one iteration of the trading loop."""

        print(f"\n{'='*70}")
        print(f"Iteration - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # 1. Fetch market data for all tokens
        print(f"\nFetching data for {len(self.symbols)} tokens...")
        market_data = self.data_fetcher.get_token_data(self.symbols)
        print(f"Received data for {len(market_data)} tokens")

        # 2. Get current positions
        current_positions = self.data_fetcher.get_user_positions(self.wallet_address)

        # 3. Run algorithm for each token
        print(f"\nAnalyzing tokens:")
        for symbol in self.symbols:
            if symbol not in market_data:
                print(f"  {symbol}: No data available")
                continue

            token_data = market_data[symbol]
            current_position = current_positions.get(symbol)

            # Run algorithm
            signal = self.algorithm.analyze(symbol, token_data, current_position)

            # Display signal
            action = signal['action']
            if action != 'hold':
                print(f"  {symbol}: {action.upper()} - {signal['reason']}")

                # Execute trade based on signal
                self.execute_signal(symbol, signal, token_data, current_position)
            else:
                # Show price info even when holding
                price = token_data.get('price', 0)
                funding = token_data.get('funding_rate', 0)
                funding_str = f"{funding*100:.4f}%" if funding else "N/A"
                print(f"  {symbol}: HOLD - Price: ${price:,.2f}, Funding: {funding_str}")

        print(f"\n{'='*70}")

    def execute_signal(self, symbol: str, signal: Dict, token_data: Dict, current_position: Optional[Dict]):
        """
        Execute trading signal.

        Args:
            symbol: Token symbol
            signal: Algorithm signal
            token_data: Current market data
            current_position: Current position info
        """
        action = signal['action']
        price = token_data['price']

        if action == 'buy':
            # Calculate size based on position_size_usd from CSV
            position_size_usd = self.token_loader.get_position_size(symbol)
            size = position_size_usd / price
            size = round(size, 4)

            self.trader.execute_trade(symbol, 'buy', size, price)

        elif action == 'sell':
            size = signal.get('size', 0)
            if size > 0:
                self.trader.execute_trade(symbol, 'sell', size, price)

        elif action == 'close':
            size = signal.get('size', 0)
            if size > 0:
                self.trader.close_position(symbol, size, price)

    def run(self):
        """Run the trading bot continuously."""

        print(f"\nBot started. Press Ctrl+C to stop.\n")

        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n{'#'*70}")
                print(f"# Iteration {iteration}")
                print(f"{'#'*70}")

                self.run_iteration()

                print(f"\nNext iteration in {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print("Bot stopped by user")
            print(f"{'='*70}")
            print(f"Total iterations: {iteration}")

            # Show final positions
            print(f"\nFinal Positions:")
            positions = self.data_fetcher.get_user_positions(self.wallet_address)
            if positions:
                for symbol, pos in positions.items():
                    print(f"  {symbol}: {pos['size']} @ ${pos['entry_price']:,.2f} | PnL: ${pos['unrealized_pnl']:+,.2f}")
            else:
                print("  No open positions")


def main():
    """Main entry point."""
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()