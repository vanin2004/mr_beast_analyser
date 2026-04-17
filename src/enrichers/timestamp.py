"""Timestamp feature enricher."""

import pandas as pd

from src.enrichers.base import BaseEnricher


class TimestampEnricher(BaseEnricher):
    """Extract temporal features from timestamp column."""

    TIMESTAMP_FEATURES = ["ts_dayofweek", "ts_month", "ts_day", "ts_hour"]

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        """Initialize TimestampEnricher.

        Args:
            timestamp_col: Column name containing timestamps. Defaults to "timestamp".
        """
        self.timestamp_col = timestamp_col

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

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with timestamp features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with added timestamp features.
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_col}' not found in DataFrame"
            )

        enriched_df = df.copy()
        ts_features = self.extract_timestamp_features(df[self.timestamp_col])
        enriched_df = pd.concat([enriched_df, ts_features], axis=1)
        return enriched_df
