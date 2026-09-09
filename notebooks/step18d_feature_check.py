import pandas as pd

df = pd.read_csv("processed/training_data.csv")

feature_cols = [
    c for c in df.columns
    if c not in ["label", "lat", "lon", "date", "source"]
]

print("========================================")
print("FEATURE STATISTICS")
print("========================================")

print(df[feature_cols].describe())

print("\n========================================")
print("CORRELATION WITH LABEL")
print("========================================")

correlation = (
    df[feature_cols + ["label"]]
    .corr()["label"]
    .sort_values(ascending=False)
)

print(correlation)