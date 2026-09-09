import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

df = pd.read_csv("processed/training_data.csv")
feature_cols = [c for c in df.columns if c not in ["label", "lat", "lon", "date", "source"]]
X = df[feature_cols]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42)

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy")

print(f"LOOCV accuracy: {scores.mean():.2f}")
print(f"Correct predictions: {int(scores.sum())} / {len(scores)}")