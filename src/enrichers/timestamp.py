"""Timestamp feature enricher."""

import pandas as pd

from src.enrichers.base import BaseEnricher


class TimestampEnricher(BaseEnricher):
    """Extract temporal features from timestamp column."""

    TIMESTAMP_FEATURES = ["ts_dayofweek", "ts_month", "ts_day", "ts_hour"]

    @staticmethod
    def extract_timestamp_features(ts_series: pd.Series) -> pd.DataFrame:
        """Extract features from timestamp: day of week, month, day, hour.

        Args:
            ts_series: Timestamp series to extract features from.

        Returns:
            DataFrame with columns: ts_dayofweek, ts_month, ts_day, ts_hour.
        """
        ts = pd.to_datetime(ts_series, errors="coerce")
        features = pd.DataFrame(
            {
                "ts_dayofweek": ts.dt.dayofweek,
                "ts_month": ts.dt.month,
                "ts_day": ts.dt.day,
                "ts_hour": ts.dt.hour,
            }
        )
        return features

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enrich DataFrame with timestamp features.

        Args:
            df: Input DataFrame.
            **kwargs: Should contain 'timestamp_col' (defaults to "timestamp").

        Returns:
            DataFrame with added timestamp features.
        """
        timestamp_col = kwargs.get("timestamp_col", "timestamp")
        return self.enrich_dataframe(df, timestamp_col)

    def enrich_dataframe(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp", **kwargs
    ) -> pd.DataFrame:
        """Enrich DataFrame with timestamp features.

        Args:
            df: Input DataFrame.
            timestamp_col: Column name with timestamps. Defaults to "timestamp".
            **kwargs: Additional arguments (ignored).

        Returns:
            DataFrame with added timestamp features.
        """
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found in DataFrame"
            )

        enriched_df = df.copy()
        ts_features = self.extract_timestamp_features(df[timestamp_col])
        enriched_df = pd.concat([enriched_df, ts_features], axis=1)
        return enriched_df
