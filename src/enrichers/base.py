"""Base class for data enrichers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseEnricher(ABC):
    """Abstract base class for data enrichers."""

    @abstractmethod
    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply enrichment to DataFrame.

        Args:
            df: Input DataFrame.
            **kwargs: Additional arguments specific to enricher.

        Returns:
            Enriched DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
    def enrich_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Public interface for enrichment (alias for enrich).

        Args:
            df: Input DataFrame.
            **kwargs: Additional arguments specific to enricher.

        Returns:
            Enriched DataFrame.
        """
        raise NotImplementedError
