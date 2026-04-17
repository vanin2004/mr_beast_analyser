"""Base class for data enrichers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseEnricher(ABC):
    """Abstract base class for data enrichers."""

    @abstractmethod
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply enrichment to DataFrame.

        Args:
            df: Input DataFrame to enrich.

        Returns:
            Enriched DataFrame.
        """
        raise NotImplementedError
