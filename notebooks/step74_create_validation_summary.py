import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT = os.path.join(
    BASE,
    "processed",
    "step74_model_validation_summary.csv"
)

results = {
    "Model": ["Random Forest"],
    "Factors": [8],
    "Training_samples": [991],
    "Trees": [700],
    "Independent_test_region": ["Meghalaya"],
    "Independent_test_samples": [74],
    "ROC_AUC": [0.835526],
    "PR_AUC": [0.753481],
    "Precision": [0.794118],
    "Recall": [0.750000],
    "F1": [0.771429],
    "Accuracy": [0.783784],
    "Balanced_accuracy": [0.782895],
    "TN": [31],
    "FP": [7],
    "FN": [9],
    "TP": [27],
    "NER_prediction_coverage_percent": [85.03]
}

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT,
    index=False,
    float_format="%.6f"
)

print("=" * 70)
print("STEP 74 — MODEL VALIDATION SUMMARY")
print("=" * 70)

print(df.to_string(index=False))

print("\nSaved:")
print(OUTPUT)

print("\nSTEP 74 COMPLETE")
