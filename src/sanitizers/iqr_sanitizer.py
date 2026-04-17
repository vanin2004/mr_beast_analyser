"""IQR (Interquartile Range) based outlier removal sanitizer."""

import pandas as pd

from src.sanitizers.base import BaseSanitizer


class IQRSanitizer(BaseSanitizer):
    """Remove outliers using IQR (Interquartile Range) method.

    Uses the 3×IQR threshold:
    - Lower bound: Q1 - 3×IQR
    - Upper bound: Q3 + 3×IQR

    Removes rows where ANY feature value exceeds these bounds.
    """

    def __init__(self, iqr_multiplier: float = 3.0) -> None:
        """Initialize IQR sanitizer.

        Args:
            iqr_multiplier: Multiplier for IQR threshold (default 3.0 for 99.7% confidence).
                           Use 1.5 for less aggressive outlier removal.
        """
        self.iqr_multiplier = iqr_multiplier
        self.outlier_rows = set()
        self.removed_count = 0
        self.total_rows = 0

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from all numeric columns using IQR method.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with outlier rows removed.
        """
        self.total_rows = len(df)
        sanitized_df = df.copy()

        # Process all numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR

            # Detect outliers
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
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
            "method": "IQR",
            "iqr_multiplier": self.iqr_multiplier,
            "total_rows": self.total_rows,
            "removed_rows": self.removed_count,
            "removed_percentage": self.removed_count / self.total_rows * 100
            if self.total_rows > 0
            else 0,
            "remaining_rows": self.total_rows - self.removed_count,
            "outlier_indices": sorted(list(self.outlier_rows)),
        }
