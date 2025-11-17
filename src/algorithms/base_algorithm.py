"""
Base Algorithm Module
Abstract base class for all trading algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseAlgorithm(ABC):
    """
    Abstract base class for trading algorithms.

    All custom algorithms should inherit from this class and implement
    the required methods.
    """

    def __init__(self, name: str = "BaseAlgorithm"):
        """
        Initialize algorithm.

        Args:
            name: Algorithm name for logging
        """
        self.name = name
        self.positions = {}  # Track algorithm's view of positions

    @abstractmethod
    def analyze(self, symbol: str, market_data: Dict, current_position: Optional[Dict]) -> Dict:
        """
        Analyze market data and return trading signal.

        Args:
            symbol: Token symbol
            market_data: Current market data for the token
            current_position: Current position info (None if no position)

        Returns:
            Dictionary with keys:
                - 'action': 'buy', 'sell', 'hold', or 'close'
                - 'size': Position size (if action is buy/sell)
                - 'reason': Text explanation of decision
        """
        pass

    def get_name(self) -> str:
        """Get algorithm name."""
        return self.name

    def reset(self):
        """Reset algorithm state."""
        self.positions = {}