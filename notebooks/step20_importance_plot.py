import pandas as pd
import matplotlib.pyplot as plt

# Load feature importance
df = pd.read_csv(
    "processed/feature_importance.csv"
)

# Sort for plotting
df = df.sort_values(
    "importance"
)

# Create chart
plt.figure(figsize=(9, 6))

plt.barh(
    df["feature"],
    df["importance"]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Environmental Feature")
plt.title("Factors Influencing Landslide Susceptibility")

plt.tight_layout()

plt.savefig(
    "processed/feature_importance.png",
    dpi=300
)

plt.show()

print("Saved: processed/feature_importance.png")