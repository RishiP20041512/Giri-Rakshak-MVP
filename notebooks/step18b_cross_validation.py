import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("processed/training_data.csv")

feature_cols = [
    c for c in df.columns
    if c not in ["label", "lat", "lon", "date", "source"]
]

X = df[feature_cols]
y = df["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("F1 scores per fold:", scores)

print(
    f"Mean F1: {scores.mean():.2f} "
    f"(+/- {scores.std():.2f})"
)