import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("raw_data/landslides/landslides.csv")
plt.scatter(df["lon"], df["lat"], c="red", marker="x", s=100)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"{len(df)} landslide points - Dibang Valley pilot area")
plt.gca().set_xlim(95.6, 96.1)
plt.gca().set_ylim(28.2, 28.9)
plt.savefig("processed/landslide_points_check.png")
print("Saved plot to processed/landslide_points_check.png")
