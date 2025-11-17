"""
CSV Loader Module
Loads token configuration from CSV file.
"""

import pandas as pd
from typing import List, Dict


class TokenLoader:
    """Loads and manages token list from CSV."""

    def __init__(self, csv_path: str = "config/tokens.csv"):
        """
        Initialize token loader.

        Args:
            csv_path: Path to tokens CSV file
        """
        self.csv_path = csv_path
        self.tokens = []

    def load_tokens(self) -> List[Dict]:
        """
        Load tokens from CSV file.

        Returns:
            List of token dictionaries
        """
        try:
            df = pd.read_csv(self.csv_path)

            # Filter only enabled tokens
            df_enabled = df[df['enabled'].astype(str).str.lower() == 'true']

            # Convert to list of dicts
            self.tokens = df_enabled.to_dict('records')

            print(f"Loaded {len(self.tokens)} enabled tokens from {self.csv_path}")
            return self.tokens

        except FileNotFoundError:
            print(f"Error: Token file not found at {self.csv_path}")
            return []
        except Exception as e:
            print(f"Error loading tokens: {e}")
            return []

    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled token symbols."""
        return [token['symbol'] for token in self.tokens]

    def get_position_size(self, symbol: str) -> float:
        """Get position size for a specific token."""
        for token in self.tokens:
            if token['symbol'] == symbol:
                return float(token['position_size_usd'])
        return 100.0  # Default