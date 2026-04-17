"""Derivative (growth rate) feature enricher."""

import numpy as np
import pandas as pd

from src.enrichers.base import BaseEnricher


class DerivativeEnricher(BaseEnricher):
    """Add derivative (growth rate) features between consecutive time windows."""

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
        views_cols = sorted(
            [c for c in df.columns if c.startswith("CN.VIEWS")],
            key=lambda x: int(x.replace("CN.VIEWS", "")),
        )

        if len(views_cols) < 2:
            return enriched_df

        # Exclude target variable (CN.VIEWS604800) from derivative calculation
        # Only compute derivatives between non-target columns
        views_cols_for_deriv = [c for c in views_cols if c != "CN.VIEWS604800"]

        # Add derivatives between consecutive windows
        for i in range(1, len(views_cols_for_deriv)):
            prev_col = views_cols_for_deriv[i - 1]
            curr_col = views_cols_for_deriv[i]

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

                # Clip extreme values to [-10, 100] to avoid model destabilization
                derivative = derivative.clip(-10, 100)

            # Create new column name
            interval_seconds = int(curr_col.replace("CN.VIEWS", ""))
            prev_seconds = int(prev_col.replace("CN.VIEWS", ""))

            deriv_col_name = f"DERIV_{prev_seconds}_to_{interval_seconds}"
            enriched_df[deriv_col_name] = derivative

            # Final safety: replace any remaining infinities and NaNs
            enriched_df[deriv_col_name] = enriched_df[deriv_col_name].replace(
                [np.inf, -np.inf], 0
            )
            enriched_df[deriv_col_name] = enriched_df[deriv_col_name].fillna(0)

        return enriched_df
