"""Base class for data sanitizers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseSanitizer(ABC):
    """Abstract base class for data sanitizers.

    Sanitizers are responsible for cleaning and validating data,
    including outlier removal and data quality checks.
    """

    @abstractmethod
    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sanitization to DataFrame.

        Args:
            df: Input DataFrame to sanitize.

        Returns:
            Sanitized DataFrame (potentially with fewer rows/columns).
        """
        raise NotImplementedError

    def get_report(self) -> dict:
        """Get sanitization report.

        Returns:
            Dictionary with sanitization statistics.
        """
        return {}
