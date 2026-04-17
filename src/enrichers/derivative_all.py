"""Derivative (growth rate) feature enricher."""

import pandas as pd

from src.enrichers.base import BaseEnricher


class DerivativeAllEnricher(BaseEnricher):
    """Add derivative (growth rate) features between consecutive time windows."""

    def __init__(self, target_col: str = "CN.VIEWS604800") -> None:
        self.target_col = target_col

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add growth rate derivatives between consecutive CN.VIEWS columns.

        For each pair of consecutive time windows, computes:
        derivative = (views[t] - views[t-1]) / views[t-1]

        Excludes the target variable (CN.VIEWS604800) from derivative calculation.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added derivative features.
        """
        enriched_df = df.copy()

        # Find all CN.VIEWS columns and sort by time window
        views_cols = [c for c in df.columns if c != self.target_col]

        for i in range(1, len(views_cols)):
            prev_col = views_cols[i - 1]
            curr_col = views_cols[i]

            prev_values = df[prev_col]
            curr_values = df[curr_col]

            # Only compute derivative where previous value is significant (> 100)
            # This avoids extreme values from division by very small numbers
            valid_mask = prev_values > 100

            derivative = pd.Series(0.0, index=df.index)

            # Calculate growth rate only for valid rows
            if valid_mask.sum() > 0:
                derivative[valid_mask] = (
                    curr_values[valid_mask] - prev_values[valid_mask]
                ) / prev_values[valid_mask]

            enriched_df[f"derivative_{curr_col}"] = derivative

        return enriched_df
