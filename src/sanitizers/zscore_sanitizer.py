"""Z-score based outlier removal sanitizer."""

import numpy as np
import pandas as pd
from scipy import stats

from src.sanitizers.base import BaseSanitizer


class ZScoreSanitizer(BaseSanitizer):
    """Remove outliers using Z-score method.

    Uses standard deviations from mean:
    - Z-score = (value - mean) / std
    - Threshold: |Z| > threshold (default 3.0 = 99.7% confidence)

    Removes rows where ANY feature has |Z-score| exceeding threshold.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        """Initialize Z-score sanitizer.

        Args:
            threshold: Z-score threshold for outlier detection (default 3.0).
                      Common values: 2.0 (95%), 2.5 (98%), 3.0 (99.7%)
        """
        self.threshold = threshold
        self.outlier_rows = set()
        self.removed_count = 0
        self.total_rows = 0

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from all numeric columns using Z-score method.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with outlier rows removed.
        """
        self.total_rows = len(df)
        sanitized_df = df.copy()

        # Process all numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            try:
                z_scores = np.abs(stats.zscore(df[col]))
            except ValueError:
                # Skip columns with zero variance
                continue

            # Detect outliers
            is_outlier = z_scores > self.threshold
            outlier_indices = df.index[is_outlier]
            self.outlier_rows.update(outlier_indices)

        # Remove rows with outliers
        sanitized_df = df[~df.index.isin(self.outlier_rows)].copy()
        self.removed_count = len(df) - len(sanitized_df)

        return sanitized_df

    def get_report(self) -> dict:
        """Get sanitization report.

        Returns:
            Dictionary with outlier removal statistics.
        """
        return {
            "method": "Z-score",
            "threshold": self.threshold,
            "total_rows": self.total_rows,
            "removed_rows": self.removed_count,
            "removed_percentage": self.removed_count / self.total_rows * 100
            if self.total_rows > 0
            else 0,
            "remaining_rows": self.total_rows - self.removed_count,
            "outlier_indices": sorted(list(self.outlier_rows)),
        }
