import argparse

import pandas as pd

from src.data_prep import VideoDataPreparer
from src.enrichers import TimestampEnricher
from src.enrichers.derivative_all import DerivativeAllEnricher
from src.sanitizers import IQRSanitizer, PipelineSanitizer, ZScoreSanitizer
from src.trains.ensemble_trainer import EnsembleTrainer
from src.trains.lightgbm_trainer import LightGBMTrainer
from src.trains.random_forest import RandomForestTrainer
from src.trains.ridge import RidgeTrainer
from src.trains.svr_trainer import SVRTrainer
from src.trains.xgboost_trainer import XGBoostTrainer

OUT_CSV_PATH = "./data/out.csv"
ROW_GLOB = "./data/row/*"
SAMPLE_INFERENCE_PATH = "./data/row/_AbFXuGDRTs_hourly.json"
REBUILD_DATASET = False


def main(model_name: str = "lightgbm", sanitizer_name: str | None = None) -> None:
    preparer = VideoDataPreparer()

    if REBUILD_DATASET:
        preparer.build_training_dataset(ROW_GLOB, out_csv_path=OUT_CSV_PATH)

    df = pd.read_csv(OUT_CSV_PATH, delimiter=";")
    print(f"\nLoaded dataset: {len(df)} rows × {len(df.columns)} columns")

    # Apply sanitizer if specified
    sanitizer = None
    if sanitizer_name:
        print("\n" + "=" * 60)
        print(f"APPLYING SANITIZER: {sanitizer_name}")
        print("=" * 60)

        sanitizer_name = sanitizer_name.lower()
        if sanitizer_name == "iqr":
            sanitizer = IQRSanitizer(iqr_multiplier=3.0)
        elif sanitizer_name == "zscore":
            sanitizer = ZScoreSanitizer(threshold=3.0)
        elif sanitizer_name == "pipeline":
            # Progressive filtering: less aggressive then stricter
            sanitizer = PipelineSanitizer(
                sanitizers=[
                    IQRSanitizer(iqr_multiplier=1.5),
                    ZScoreSanitizer(threshold=2.5),
                ]
            )
        else:
            raise ValueError(
                f"Unknown sanitizer: {sanitizer_name}. Choose from: iqr, zscore, pipeline"
            )

        df = sanitizer.sanitize(df)
        report = sanitizer.get_report()

        if isinstance(report, dict) and "pipeline_steps" in report:
            # Pipeline report
            print(f"\nPipeline steps: {report['pipeline_steps']}")
            for i, step_report in enumerate(report["step_reports"], 1):
                print(
                    f"  Step {i} ({step_report['method']}): "
                    f"Removed {step_report['removed_rows']} rows ({step_report['removed_percentage']:.2f}%)"
                )
            print(f"Total removed: {report['final_summary']['total_removed']} rows")
            print(f"Final dataset: {report['final_summary']['final_rows']} rows")
        else:
            # Single sanitizer report
            print(f"\nMethod: {report['method']}")
            print(
                f"Removed: {report['removed_rows']} rows ({report['removed_percentage']:.2f}%)"
            )
            print(f"Remaining: {report['remaining_rows']} rows")

    print("\n" + "=" * 60)
    enrichers_list = [
        TimestampEnricher(timestamp_col="timestamp"),
        DerivativeAllEnricher(),
    ]
    enrichers_list = [e for e in enrichers_list if e is not None]
    enricher_str = ", ".join([e.__class__.__name__ for e in enrichers_list]) or "None"
    print(f"TRAINING WITH ENRICHERS: {enricher_str}")
    print("=" * 60)

    # Select trainer based on model_name
    model_name = model_name.lower()

    if model_name == "ridge":
        trainer = RidgeTrainer(timestamp_col="timestamp", test_size=0.2, alpha=60.0)
    elif model_name == "random_forest":
        trainer = RandomForestTrainer(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=15,
            max_features=1.0,
            random_state=42,
        )
    elif model_name == "lightgbm":
        trainer = LightGBMTrainer(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_data_in_leaf=20,
            random_state=42,
        )
    elif model_name == "xgboost":
        trainer = XGBoostTrainer(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    elif model_name == "svr":
        trainer = SVRTrainer(
            C=1000.0,
            kernel="rbf",
            gamma="scale",
            epsilon=0.1,
        )
    elif model_name == "ensemble":
        trainer = EnsembleTrainer(
            lightgbm_weight=0.5,
            xgboost_weight=0.3,
            svr_weight=0.2,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: "
            "ridge, random_forest, lightgbm, xgboost, svr, ensemble"
        )

    target_col = df.columns[-1]
    _, metrics = trainer.fit(
        df,
        feature_cols=None,
        target_col=target_col,
        enrichers=enrichers_list,
        verbose=True,
    )

    print("\n=== Metrics dict ===")
    print(metrics)

    inference_row = preparer.build_inference_row(SAMPLE_INFERENCE_PATH)

    # Apply enrichers to inference row
    for enricher in enrichers_list:
        inference_row = enricher.enrich(inference_row)

    pred = trainer.predict(inference_row)

    print("\n=== Inference ===")
    print(f"File: {SAMPLE_INFERENCE_PATH}")
    print(f"Predicted {trainer.target_col}: {float(pred[0]):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regression model for video view prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["ridge", "random_forest", "lightgbm", "xgboost", "svr", "ensemble"],
        help="Model to train (default: lightgbm)",
    )
    parser.add_argument(
        "--sanitizer",
        type=str,
        default=None,
        choices=["iqr", "zscore", "pipeline"],
        help="Data sanitizer for outlier removal (default: None)",
    )
    args = parser.parse_args()

    main(model_name=args.model, sanitizer_name=args.sanitizer)
