"""
Simple Momentum Algorithm
Educational example: buys when funding rate is positive and high.
"""

from typing import Dict, Optional
from .base_algorithm import BaseAlgorithm


class SimpleMomentumAlgorithm(BaseAlgorithm):
    """
    Simple test algorithm based on funding rates.

    Strategy:
    - BUY (long) when funding rate is positive and above threshold
    - SELL (close) when funding rate drops below threshold
    - This is a simplified educational example
    """

    def __init__(self, funding_threshold: float = 0.0001):
        """
        Initialize simple momentum algorithm.

        Args:
            funding_threshold: Minimum funding rate to enter position (e.g., 0.0001 = 0.01%)
        """
        super().__init__(name="SimpleMomentum")
        self.funding_threshold = funding_threshold
        print(f"Algorithm: {self.name} initialized")
        print(f"  Funding threshold: {funding_threshold * 100:.4f}% per hour")

    def analyze(self, symbol: str, market_data: Dict, current_position: Optional[Dict]) -> Dict:
        """
        Analyze token and return trading signal.

        Logic:
        1. If no position and funding rate > threshold: BUY
        2. If position exists and funding rate < threshold: CLOSE
        3. Otherwise: HOLD

        Args:
            symbol: Token symbol (e.g., "BTC")
            market_data: Dict with 'price', 'funding_rate', etc.
            current_position: Current position info or None

        Returns:
            Trading signal dictionary
        """
        price = market_data.get('price', 0)
        funding_rate = market_data.get('funding_rate', 0)

        # Default: hold
        signal = {
            'action': 'hold',
            'size': 0,
            'reason': 'No action'
        }

        # Check if we have a position
        has_position = current_position is not None and current_position.get('size', 0) != 0

        # Decision logic
        if not has_position:
            # No position - check if we should enter
            if funding_rate and funding_rate > self.funding_threshold:
                signal = {
                    'action': 'buy',
                    'size': None,  # Will be set by position manager
                    'reason': f'Positive funding: {funding_rate*100:.4f}% > {self.funding_threshold*100:.4f}%'
                }
        else:
            # Have position - check if we should exit
            if not funding_rate or funding_rate < self.funding_threshold * 0.5:
                signal = {
                    'action': 'close',
                    'size': abs(current_position['size']),
                    'reason': f'Funding dropped: {(funding_rate or 0)*100:.4f}% < threshold'
                }

        return signal