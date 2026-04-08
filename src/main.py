import pandas as pd

from src.data_prep import VideoDataPreparer
from src.enrichers import TimestampEnricher
from src.trains.simple_train import SimpleLinearTrainer

OUT_CSV_PATH = "./data/out.csv"
ROW_GLOB = "./data/row/*"
SAMPLE_INFERENCE_PATH = "./data/row/_AbFXuGDRTs_hourly.json"
REBUILD_DATASET = False
USE_TIMESTAMP_FEATURES = True


def main() -> None:
    preparer = VideoDataPreparer()

    if REBUILD_DATASET:
        preparer.build_training_dataset(ROW_GLOB, out_csv_path=OUT_CSV_PATH)

    df = pd.read_csv(OUT_CSV_PATH, delimiter=";")

    print("=" * 60)
    enrichers_list = [TimestampEnricher()] if USE_TIMESTAMP_FEATURES else []
    enricher_str = ", ".join([e.__class__.__name__ for e in enrichers_list]) or "None"
    print(f"TRAINING WITH ENRICHERS: {enricher_str}")
    print("=" * 60)

    trainer = SimpleLinearTrainer(timestamp_col="timestamp", test_size=0.2)
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

    if enrichers_list:
        for enricher in enrichers_list:
            if isinstance(enricher, TimestampEnricher):
                inference_row = enricher.enrich_dataframe(inference_row, "timestamp")

    pred = trainer.predict(inference_row)

    print("\n=== Inference ===")
    print(f"File: {SAMPLE_INFERENCE_PATH}")
    print(f"Predicted {trainer.target_col}: {float(pred[0]):.2f}")


if __name__ == "__main__":
    main()
