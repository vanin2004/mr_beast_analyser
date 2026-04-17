# Data Sanitizers

Object-oriented data cleaning and validation framework, following the same architectural pattern as `enrichers` and `trains`.

## Overview

Sanitizers are responsible for:
- **Outlier detection and removal** using statistical methods
- **Data quality validation**
- **Composable data cleaning pipelines**

## Architecture

```
BaseSanitizer (abstract)
├── IQRSanitizer          (Interquartile Range method)
├── ZScoreSanitizer       (Z-score method)
└── PipelineSanitizer     (Composite: chains multiple sanitizers)
```

## Available Sanitizers

### 1. IQRSanitizer
**Method:** Interquartile Range (IQR) with 3×IQR threshold

```python
from src.sanitizers import IQRSanitizer

sanitizer = IQRSanitizer(iqr_multiplier=3.0)
df_clean = sanitizer.sanitize(df)
report = sanitizer.get_report()
```

**Parameters:**
- `iqr_multiplier` (float): How many IQRs from Q1/Q3 define outliers
  - `3.0` (default): 99.7% confidence, removes extreme outliers
  - `1.5`: Less aggressive, removes moderate outliers
  - `2.0`: Balanced approach

**How it works:**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

Outlier if: value < (Q1 - multiplier×IQR) OR value > (Q3 + multiplier×IQR)
```

**Advantages:**
- ✅ Robust to extreme values
- ✅ Works with any distribution
- ✅ Simple and interpretable
- ✅ Industry standard

---

### 2. ZScoreSanitizer
**Method:** Z-score with configurable threshold

```python
from src.sanitizers import ZScoreSanitizer

sanitizer = ZScoreSanitizer(threshold=3.0)
df_clean = sanitizer.sanitize(df)
report = sanitizer.get_report()
```

**Parameters:**
- `threshold` (float): Z-score absolute value threshold
  - `2.0`: 95% confidence
  - `2.5`: 98% confidence  
  - `3.0` (default): 99.7% confidence

**How it works:**
```
Z-score = |value - mean| / std

Outlier if: |Z-score| > threshold
```

**Advantages:**
- ✅ Based on standard deviations
- ✅ Sensitive to distribution shape
- ✅ Commonly used in statistics
- ⚠️ Assumes roughly normal distribution

---

### 3. PipelineSanitizer
**Purpose:** Chain multiple sanitizers sequentially

```python
from src.sanitizers import PipelineSanitizer, IQRSanitizer, ZScoreSanitizer

pipeline = PipelineSanitizer(
    sanitizers=[
        IQRSanitizer(iqr_multiplier=1.5),      # Step 1: Less aggressive
        ZScoreSanitizer(threshold=2.5),        # Step 2: Stricter filter
    ]
)

df_clean = pipeline.sanitize(df)
report = pipeline.get_report()

# Access individual reports
print(report['step_reports'][0])  # IQR results
print(report['step_reports'][1])  # Z-score results
```

**Method chaining:**
```python
pipeline = (
    PipelineSanitizer()
    .add_sanitizer(IQRSanitizer(1.5))
    .add_sanitizer(ZScoreSanitizer(2.5))
)
```

---

## Reports and Statistics

Every sanitizer provides a detailed report:

```python
report = sanitizer.get_report()

# Contains:
{
    "method": "IQR",                    # Sanitizer method name
    "iqr_multiplier": 3.0,              # Method-specific params
    "total_rows": 257,                  # Input rows
    "removed_rows": 5,                  # Removed rows
    "removed_percentage": 1.95,         # Removal %
    "remaining_rows": 252,              # Output rows
    "outlier_indices": [12, 45, ...],   # Which rows were removed
}
```

---

## Usage Examples

### Example 1: Basic IQR Cleaning
```python
from src.sanitizers import IQRSanitizer

sanitizer = IQRSanitizer(iqr_multiplier=3.0)
df_clean = sanitizer.sanitize(df)

report = sanitizer.get_report()
print(f"Removed {report['removed_rows']} outliers ({report['removed_percentage']:.2f}%)")
```

### Example 2: Progressive Filtering
```python
from src.sanitizers import PipelineSanitizer, IQRSanitizer, ZScoreSanitizer

# First pass: remove obvious outliers
# Second pass: refine with stricter criteria
pipeline = PipelineSanitizer([
    IQRSanitizer(iqr_multiplier=3.0),      # Remove extreme outliers
    ZScoreSanitizer(threshold=2.0),        # Then strict filtering
])

df_clean = pipeline.sanitize(df)
```

### Example 3: Integration with Enrichers
```python
from src.enrichers import DerivativeAllEnricher
from src.sanitizers import IQRSanitizer

# 1. Load data
df = pd.read_csv("data/out.csv", sep=";", index_col=0)

# 2. Enrich features (add derivatives)
enricher = DerivativeAllEnricher()
df = enricher.enrich(df)

# 3. Sanitize (remove outliers)
sanitizer = IQRSanitizer(3.0)
df = sanitizer.sanitize(df)

# 4. Train model
```

---

## Comparison: IQR vs Z-Score

| Aspect | IQR | Z-Score |
|--------|-----|---------|
| **Distribution dependency** | None (distribution-free) | Assumes normality |
| **Robustness** | Very robust | Sensitive to extremes |
| **Common threshold** | 1.5×IQR (mild), 3×IQR (strict) | 2, 2.5, 3 std.dev |
| **Use case** | Unknown distributions | Well-behaved data |
| **Speed** | Very fast | Fast |
| **Interpretability** | Intuitive (quartiles) | Intuitive (std.dev) |

**Recommendation:** Start with IQR for exploratory analysis, use Z-score for well-understood datasets.

---

## Running the Example

```bash
cd /mnt/projects/my/mr_beast_analyser
python -m src.sanitizers.example
```

Output shows 4 cleaning approaches and a comparison table.

---

## Integration with Main Pipeline

Sanitizers can be integrated into the main training pipeline:

```python
# In src/main.py or notebook

df = pd.read_csv("data/out.csv", sep=";", index_col=0)

# Apply sanitization
sanitizer = IQRSanitizer(iqr_multiplier=3.0)
df = sanitizer.sanitize(df)

# Then enrich and train as usual
enricher = DerivativeAllEnricher()
df = enricher.enrich(df)

trainer = LightGBMTrainer()
trainer.fit(df, "CN.VIEWS604800")
```

---

## Notes

- ✅ All sanitizers handle numeric columns automatically
- ✅ Non-numeric columns are preserved
- ✅ Empty columns are skipped gracefully
- ✅ Sanitizers are **immutable** (return new DataFrames)
- ⚠️ Order matters in pipelines (apply less aggressive first)
- ⚠️ Always check `get_report()` to verify expected removal rates
