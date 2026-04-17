import pandas as pd

from src.enrichers.base import BaseEnricher


class AddTimestampEnricher(BaseEnricher):
    TIMESTAMP_FEATURES = ["raw_timestamp"]

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        self.timestamp_col = timestamp_col

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

        df["raw_timestamp"] = df[self.timestamp_col]
        return df
