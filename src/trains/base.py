from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.enrichers import TimestampEnricher


class BaseTrainer(ABC):
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        test_size: float = 0.2,
    ) -> None:
        self.timestamp_col = timestamp_col
        self.test_size = test_size
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.target_col: str = ""

    def get_feature_target_columns(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
    ) -> tuple[list[str], str]:
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Column '{self.timestamp_col}' is required")

        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            feature_cols = [
                c for c in df.columns if c not in {self.timestamp_col, target_col}
            ]
            return feature_cols, target_col

        inferred_target = df.columns[-1]
        feature_cols = [
            c for c in df.columns if c not in {self.timestamp_col, inferred_target}
        ]
        return feature_cols, inferred_target

    def split_by_time(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        work_df = df.copy()
        work_df[self.timestamp_col] = pd.to_datetime(
            work_df[self.timestamp_col], errors="coerce"
        )
        work_df = work_df.dropna(
            subset=[self.timestamp_col] + feature_cols + [target_col]
        )
        work_df = work_df.sort_values(self.timestamp_col).reset_index(drop=True)

        split_idx = int(len(work_df) * (1 - self.test_size))
        if split_idx <= 0 or split_idx >= len(work_df):
            raise ValueError("Not enough rows for time-based train/test split")

        train_part = work_df.iloc[:split_idx]
        test_part = work_df.iloc[split_idx:]

        x_train = train_part[feature_cols]
        y_train = train_part[target_col]
        x_test = test_part[feature_cols]
        y_test = test_part[target_col]

        return x_train, x_test, y_train, y_test

    @abstractmethod
    def build_model(self) -> Any:
        raise NotImplementedError

    def plot_diagnostics(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
    ) -> None:
        """Generate and save diagnostic plots for model evaluation."""
        assert self.model is not None, (
            "Model must be trained before plotting diagnostics"
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{self.__class__.__name__} - Training Diagnostics", fontsize=16)

        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
        axes[0, 0].plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
        )
        axes[0, 0].set_xlabel(f"Actual {self.target_col}")
        axes[0, 0].set_ylabel("Predicted")
        axes[0, 0].set_title("Predicted vs Actual (Test Set)")
        axes[0, 0].grid(True, alpha=0.3)

        residuals = y_test.values - y_pred
        axes[0, 1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residuals Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
        axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Predicted Values")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].set_title("Residuals vs Predicted")
        axes[1, 0].grid(True, alpha=0.3)

        train_score = self.model.score(x_train, y_train)
        test_score = self.model.score(x_test, y_test)
        metric_text = f"Train R²: {train_score:.4f}\nTest R²: {test_score:.4f}"
        if hasattr(self.model, "coef_"):
            coef_text = "\n".join(
                [
                    f"{col}: {coef:.4e}"
                    for col, coef in zip(self.feature_cols, self.model.coef_)
                ]
            )
            metric_text += f"\n\nCoefficients:\n{coef_text}"
        axes[1, 1].text(
            0.1,
            0.9,
            metric_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig("./diagnostics.png", dpi=100, bbox_inches="tight")
        print("✓ Diagnostics plot saved to ./diagnostics.png")
        plt.close()

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
        enrichers: list | None = None,
        verbose: bool = True,
    ) -> tuple[Any, dict[str, float]]:
        """Fit model with optional enrichers applied to data.

        Args:
            df: Input DataFrame.
            feature_cols: Feature column names. If None, auto-detected.
            target_col: Target column name. If None, auto-detected as last column (BEFORE enrichment).
            enrichers: List of enricher instances to apply to data. Defaults to None.
            verbose: Whether to print training report and diagnostics. Defaults to True.

        Returns:
            Tuple of (trained model, metrics dict).
        """
        enrichers = enrichers or []

        if target_col is None:
            target_col = df.columns[-1]

        work_df = df.copy()
        for enricher in enrichers:
            if isinstance(enricher, TimestampEnricher):
                work_df = enricher.enrich_dataframe(work_df, self.timestamp_col)
            else:
                work_df = enricher.enrich(work_df)

        if feature_cols is None:
            auto_features, _ = self.get_feature_target_columns(
                work_df, target_col=target_col
            )
            feature_cols = auto_features

        x_train, x_test, y_train, y_test = self.split_by_time(
            df=work_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        model = self.build_model()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

        self.model = model
        self.feature_cols = x_train.columns.tolist()
        self.target_col = target_col

        if verbose:
            enricher_names = [e.__class__.__name__ for e in enrichers]
            print("\n=== Train Report ===")
            print(f"Model: {self.__class__.__name__}")
            print(f"Enrichers: {enricher_names if enricher_names else 'None'}")
            print(f"Features ({len(self.feature_cols)}): {self.feature_cols}")
            print(f"Target: {target_col}")
            print(f"Train size: {len(x_train)}")
            print(f"Test size: {len(x_test)}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")
            print("\nGenerating diagnostic plots...")
            self.plot_diagnostics(x_train, x_test, y_train, y_test, y_pred)

        return model, metrics

    def predict(self, inference_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() before predict().")
        aligned = inference_df.reindex(columns=self.feature_cols)
        if aligned.isna().any(axis=None):
            missing = [c for c in self.feature_cols if c not in inference_df.columns]
            raise ValueError(f"Inference row is missing feature columns: {missing}")
        return self.model.predict(aligned)
