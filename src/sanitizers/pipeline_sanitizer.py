"""Pipeline sanitizer for composing multiple sanitizers."""

import pandas as pd

from src.sanitizers.base import BaseSanitizer


class PipelineSanitizer(BaseSanitizer):
    """Apply multiple sanitizers in sequence.

    Each sanitizer is applied to the output of the previous one,
    allowing for complex data cleaning workflows.
    """

    def __init__(self, sanitizers: list = None) -> None:
        """Initialize pipeline sanitizer.

        Args:
            sanitizers: List of BaseSanitizer instances to apply in order.
        """
        self.sanitizers = sanitizers or []
        self.reports = []

    def add_sanitizer(self, sanitizer: BaseSanitizer) -> "PipelineSanitizer":
        """Add a sanitizer to the pipeline.

        Args:
            sanitizer: BaseSanitizer instance to add.

        Returns:
            Self for method chaining.
        """
        self.sanitizers.append(sanitizer)
        return self

    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all sanitizers in sequence.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame after all sanitization steps.
        """
        sanitized_df = df.copy()
        self.reports = []

        for sanitizer in self.sanitizers:
            sanitized_df = sanitizer.sanitize(sanitized_df)
            self.reports.append(sanitizer.get_report())

        return sanitized_df

    def get_report(self) -> dict:
        """Get combined sanitization report from all steps.

        Returns:
            Dictionary with all sanitization reports.
        """
        return {
            "pipeline_steps": len(self.sanitizers),
            "step_reports": self.reports,
            "final_summary": {
                "initial_rows": self.reports[0]["total_rows"] if self.reports else 0,
                "final_rows": self.reports[-1]["remaining_rows"]
                if self.reports
                else 0,
                "total_removed": sum(
                    r["removed_rows"] for r in self.reports if "removed_rows" in r
                ),
            },
        }
