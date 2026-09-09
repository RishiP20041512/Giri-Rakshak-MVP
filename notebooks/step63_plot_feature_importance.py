"""
STEP 63 — FEATURE IMPORTANCE VISUALIZATION
===========================================

Purpose
-------
Create publication-ready visualizations of the finalized
Random Forest feature importance results from Step 62.

Input
-----
processed\step62_feature_importance_summary.csv

Outputs
-------
figures\step63_roc_auc_permutation_importance.png
figures\step63_pr_auc_permutation_importance.png
figures\step63_permutation_vs_impurity.png

Also creates:
processed\step63_feature_importance_plot_data.csv

Important
---------
These figures interpret the finalized model.

No:
- hyperparameter tuning
- feature selection
- model modification
- train/test split modification
is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "processed"
)

FIGURES_DIR = (
    PROJECT_ROOT / "figures"
)

INPUT_FILE = (
    PROCESSED_DIR /
    "step62_feature_importance_summary.csv"
)

OUTPUT_DATA = (
    PROCESSED_DIR /
    "step63_feature_importance_plot_data.csv"
)

ROC_FIGURE = (
    FIGURES_DIR /
    "step63_roc_auc_permutation_importance.png"
)

PR_FIGURE = (
    FIGURES_DIR /
    "step63_pr_auc_permutation_importance.png"
)

COMPARISON_FIGURE = (
    FIGURES_DIR /
    "step63_permutation_vs_impurity.png"
)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("STEP 63 — FEATURE IMPORTANCE VISUALIZATION")
print("=" * 70)


# ================================================================
# 3. CHECK INPUT
# ================================================================

print("\n[1] Checking input file...")
print("-" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required file not found:\n{INPUT_FILE}"
    )

print(
    "  Found:",
    INPUT_FILE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 4. CREATE FIGURES DIRECTORY
# ================================================================

print("\n[2] Preparing figures directory...")
print("-" * 70)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "  Figures directory:",
    FIGURES_DIR.relative_to(PROJECT_ROOT)
)


# ================================================================
# 5. LOAD DATA
# ================================================================

print("\n[3] Loading Step 62 results...")
print("-" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"  Rows loaded: {len(df)}"
)

print(
    f"  Columns loaded: {len(df.columns)}"
)


# ================================================================
# 6. VERIFY FEATURES
# ================================================================

print("\n[4] Verifying nine conditioning factors...")
print("-" * 70)

expected_features = [
    "distance_to_road_m",
    "elevation_m",
    "geomorph_origin",
    "slope_deg",
    "rainfall_3day",
    "Level_I",
    "soil_moisture",
    "lineament_density",
    "ndvi",
]

if "feature" not in df.columns:
    raise ValueError(
        "feature column not found."
    )

actual_features = set(
    df["feature"]
)

missing_features = [
    f
    for f in expected_features
    if f not in actual_features
]

if missing_features:
    raise ValueError(
        "Missing expected features:\n"
        + str(missing_features)
    )

if len(df) != 9:
    raise ValueError(
        f"Expected 9 feature rows, found {len(df)}."
    )

print(
    "  All 9 conditioning factors present."
)


# ================================================================
# 7. CREATE DISPLAY NAMES
# ================================================================

print("\n[5] Creating display labels...")
print("-" * 70)

display_names = {
    "distance_to_road_m":
        "Distance to road",

    "elevation_m":
        "Elevation",

    "geomorph_origin":
        "Geomorphology",

    "slope_deg":
        "Slope",

    "rainfall_3day":
        "Rainfall",

    "Level_I":
        "LUCC",

    "soil_moisture":
        "Soil moisture",

    "lineament_density":
        "Lineament density",

    "ndvi":
        "NDVI",
}

df["display_name"] = (
    df["feature"]
    .map(display_names)
)

if df["display_name"].isna().any():
    raise ValueError(
        "One or more features have no display name."
    )


# ================================================================
# 8. SORT BY ROC-AUC IMPORTANCE
# ================================================================

roc_df = (
    df
    .sort_values(
        "roc_auc_importance_mean",
        ascending=True
    )
    .copy()
)


# ================================================================
# 9. SORT BY PR-AUC IMPORTANCE
# ================================================================

pr_df = (
    df
    .sort_values(
        "pr_auc_importance_mean",
        ascending=True
    )
    .copy()
)


# ================================================================
# 10. SAVE PLOT DATA
# ================================================================

print("\n[6] Saving plot data...")
print("-" * 70)

plot_data = df[
    [
        "overall_rank",
        "feature",
        "display_name",
        "roc_auc_importance_mean",
        "roc_auc_importance_std",
        "roc_auc_rank",
        "pr_auc_importance_mean",
        "pr_auc_importance_std",
        "pr_auc_rank",
        "mean_rank",
        "impurity_importance",
        "impurity_rank",
    ]
].copy()

plot_data.to_csv(
    OUTPUT_DATA,
    index=False
)

print(
    "  Saved:",
    OUTPUT_DATA.relative_to(PROJECT_ROOT)
)


# ================================================================
# 11. FIGURE 1 — ROC-AUC PERMUTATION IMPORTANCE
# ================================================================

print("\n[7] Creating ROC-AUC permutation-importance figure...")
print("-" * 70)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

y_positions = np.arange(
    len(roc_df)
)

ax.barh(
    y_positions,
    roc_df[
        "roc_auc_importance_mean"
    ],
    xerr=roc_df[
        "roc_auc_importance_std"
    ],
    capsize=4
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    roc_df["display_name"]
)

ax.set_xlabel(
    "Decrease in ROC-AUC after permutation"
)

ax.set_ylabel(
    "Conditioning factor"
)

ax.set_title(
    "Permutation Importance of Landslide Susceptibility Factors\n"
    "Evaluated Using ROC-AUC"
)

ax.axvline(
    0,
    linewidth=1
)

ax.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

fig.savefig(
    ROC_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "  Saved:",
    ROC_FIGURE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 12. FIGURE 2 — PR-AUC PERMUTATION IMPORTANCE
# ================================================================

print("\n[8] Creating PR-AUC permutation-importance figure...")
print("-" * 70)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

y_positions = np.arange(
    len(pr_df)
)

ax.barh(
    y_positions,
    pr_df[
        "pr_auc_importance_mean"
    ],
    xerr=pr_df[
        "pr_auc_importance_std"
    ],
    capsize=4
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    pr_df["display_name"]
)

ax.set_xlabel(
    "Decrease in PR-AUC after permutation"
)

ax.set_ylabel(
    "Conditioning factor"
)

ax.set_title(
    "Permutation Importance of Landslide Susceptibility Factors\n"
    "Evaluated Using PR-AUC"
)

ax.axvline(
    0,
    linewidth=1
)

ax.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

fig.savefig(
    PR_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "  Saved:",
    PR_FIGURE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 13. FIGURE 3 — PERMUTATION VS IMPURITY
# ================================================================

print("\n[9] Creating permutation-vs-impurity comparison...")
print("-" * 70)

comparison_df = (
    df
    .sort_values(
        "overall_rank",
        ascending=False
    )
    .copy()
)

fig, ax = plt.subplots(
    figsize=(11, 7)
)

x_positions = np.arange(
    len(comparison_df)
)

width = 0.35

# Normalize impurity importance only for visualization.
# This does NOT change the original numerical results.

impurity_sum = (
    comparison_df[
        "impurity_importance"
    ].sum()
)

if impurity_sum > 0:

    impurity_normalized = (
        comparison_df[
            "impurity_importance"
        ]
        / impurity_sum
    )

else:

    impurity_normalized = np.zeros(
        len(comparison_df)
    )


# Normalize absolute ROC permutation importance
# only for visual comparison.

roc_abs_sum = (
    comparison_df[
        "roc_auc_importance_mean"
    ]
    .abs()
    .sum()
)

if roc_abs_sum > 0:

    roc_normalized = (
        comparison_df[
            "roc_auc_importance_mean"
        ]
        .abs()
        / roc_abs_sum
    )

else:

    roc_normalized = np.zeros(
        len(comparison_df)
    )


ax.bar(
    x_positions - width / 2,
    roc_normalized,
    width,
    label="Permutation importance (ROC-AUC)"
)

ax.bar(
    x_positions + width / 2,
    impurity_normalized,
    width,
    label="RF impurity importance"
)

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    comparison_df["display_name"],
    rotation=45,
    ha="right"
)

ax.set_ylabel(
    "Relative importance"
)

ax.set_xlabel(
    "Conditioning factor"
)

ax.set_title(
    "Comparison of Random Forest Feature Importance Methods"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

fig.savefig(
    COMPARISON_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "  Saved:",
    COMPARISON_FIGURE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 14. PRINT RANKING
# ================================================================

print("\n[10] Final ROC-AUC permutation ranking...")
print("-" * 70)

ranking = (
    df[
        [
            "roc_auc_rank",
            "display_name",
            "roc_auc_importance_mean",
            "roc_auc_importance_std",
        ]
    ]
    .sort_values(
        "roc_auc_rank"
    )
)

print(
    ranking.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ================================================================
# 15. VALIDATION
# ================================================================

print("\n[11] Final validation...")
print("-" * 70)

checks = {}

checks[
    "Nine features"
] = len(df) == 9

checks[
    "Plot data rows"
] = len(plot_data) == 9

checks[
    "ROC figure exists"
] = ROC_FIGURE.exists()

checks[
    "PR figure exists"
] = PR_FIGURE.exists()

checks[
    "Comparison figure exists"
] = COMPARISON_FIGURE.exists()

checks[
    "CSV output exists"
] = OUTPUT_DATA.exists()

checks[
    "Finite ROC values"
] = np.isfinite(
    df[
        "roc_auc_importance_mean"
    ]
).all()

checks[
    "Finite PR values"
] = np.isfinite(
    df[
        "pr_auc_importance_mean"
    ]
).all()


for name, result in checks.items():

    status = (
        "PASSED"
        if result
        else
        "FAILED"
    )

    print(
        f"  {name:<30} : {status}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "One or more Step 63 validation checks failed."
    )


# ================================================================
# 16. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("STEP 63 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFigures created:")

print(
    "  figures\\step63_roc_auc_permutation_importance.png"
)

print(
    "  figures\\step63_pr_auc_permutation_importance.png"
)

print(
    "  figures\\step63_permutation_vs_impurity.png"
)

print("\nData created:")

print(
    "  processed\\step63_feature_importance_plot_data.csv"
)

print("\nPrimary interpretation:")
print(
    "  Permutation importance"
)

print("\nFinal model remains unchanged.")

print("=" * 70)