import pandas as pd
import joblib


# Load training data
df = pd.read_csv(
    "processed/training_data.csv"
)


# Load trained Random Forest
model = joblib.load(
    "processed/random_forest_model.joblib"
)


# Features used by the model
features = [
    "elevation",
    "slope",
    "aspect",
    "ndvi",
    "rainfall_3day",
    "soil_moisture",
    "dist_to_road",
    "dist_to_river"
]


# Get feature importance
importance = model.feature_importances_


# Create table
result = pd.DataFrame({
    "feature": features,
    "importance": importance
})


# Sort highest to lowest
result = result.sort_values(
    "importance",
    ascending=False
)


# Print results
print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(
    result.to_string(index=False)
)


# Save results
result.to_csv(
    "processed/feature_importance.csv",
    index=False
)


print("\nSaved:")
print(
    "processed/feature_importance.csv"
)