"""Example usage of data sanitizers."""

import pandas as pd

from src.sanitizers import IQRSanitizer, PipelineSanitizer, ZScoreSanitizer


def main():
    """Demonstrate sanitizer usage patterns."""

    # Load data
    df = pd.read_csv("data/out.csv", index_col=0, sep=";")
    print(f"Original dataset: {len(df)} rows × {len(df.columns)} columns\n")

    # Example 1: IQR method alone
    print("=" * 70)
    print("EXAMPLE 1: IQR Sanitizer Only")
    print("=" * 70)
    iqr_sanitizer = IQRSanitizer(iqr_multiplier=3.0)
    df_iqr = iqr_sanitizer.sanitize(df)
    report_iqr = iqr_sanitizer.get_report()
    print(f"Removed: {report_iqr['removed_rows']} rows ({report_iqr['removed_percentage']:.2f}%)")
    print(f"Remaining: {report_iqr['remaining_rows']} rows\n")

    # Example 2: Z-score method alone
    print("=" * 70)
    print("EXAMPLE 2: Z-Score Sanitizer Only")
    print("=" * 70)
    zscore_sanitizer = ZScoreSanitizer(threshold=3.0)
    df_zscore = zscore_sanitizer.sanitize(df)
    report_zscore = zscore_sanitizer.get_report()
    print(f"Removed: {report_zscore['removed_rows']} rows ({report_zscore['removed_percentage']:.2f}%)")
    print(f"Remaining: {report_zscore['remaining_rows']} rows\n")

    # Example 3: Pipeline (less aggressive then more aggressive)
    print("=" * 70)
    print("EXAMPLE 3: Pipeline (IQR 1.5 then Z-score 2.5)")
    print("=" * 70)
    pipeline = PipelineSanitizer(
        sanitizers=[
            IQRSanitizer(iqr_multiplier=1.5),  # Less aggressive
            ZScoreSanitizer(threshold=2.5),  # Then more strict
        ]
    )
    df_pipeline = pipeline.sanitize(df)
    pipeline_report = pipeline.get_report()
    print(f"Step 1 (IQR 1.5): Removed {pipeline_report['step_reports'][0]['removed_rows']} rows")
    print(f"Step 2 (Z-score 2.5): Removed {pipeline_report['step_reports'][1]['removed_rows']} rows")
    print(f"Total removed: {pipeline_report['final_summary']['total_removed']} rows")
    print(f"Final dataset: {pipeline_report['final_summary']['final_rows']} rows\n")

    # Save cleaned dataset
    output_path = "data/out_cleaned_sanitizers.csv"
    df_iqr.to_csv(output_path, sep=";")
    print(f"✓ Saved example cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
