from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

MODEL_FILE = PROCESSED / "ner_final_modeling_table_8factor.csv"
SPLIT_FILE = PROCESSED / "ner_spatial_split_05deg.csv"

OUTPUT = PROCESSED / "step67_8factor_feature_importance.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

CATEGORICAL_FEATURES = [
    "geomorph_origin",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "label"
GROUP = "spatial_group"
SPLIT = "split"

N_REPEATS = 30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("8-FACTOR PERMUTATION FEATURE IMPORTANCE")
print("=" * 70)

df = pd.read_csv(MODEL_FILE)
split_df = pd.read_csv(SPLIT_FILE)

print(f"\nModel rows: {len(df)}")
print(f"Split rows: {len(split_df)}")


# ============================================================
# MERGE SPATIAL SPLIT
# ============================================================

key_cols = [
    "lat",
    "lon",
    "label",
]

split_info = split_df[
    key_cols + [GROUP, SPLIT]
].copy()

df = df.merge(
    split_info,
    on=key_cols,
    how="left",
    validate="one_to_one",
)

if len(df) != 991:
    raise ValueError(
        f"Unexpected merged row count: {len(df)}"
    )


# ============================================================
# TRAIN / TEST
# ============================================================

train_df = df[
    df[SPLIT] == "train"
].copy()

test_df = df[
    df[SPLIT] == "test"
].copy()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")


# ============================================================
# DATA
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# ============================================================
# FINAL 8-FACTOR RF
# ============================================================

rf = RandomForestClassifier(
    n_estimators=700,
    max_features="sqrt",
    min_samples_leaf=5,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None,
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            rf,
        ),
    ]
)


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\nFitting model on training data...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# BASELINE TEST PERFORMANCE
# ============================================================

baseline_prob = model.predict_proba(
    X_test
)[:, 1]

baseline_roc = roc_auc_score(
    y_test,
    baseline_prob,
)

baseline_pr = average_precision_score(
    y_test,
    baseline_prob,
)

print("\nBaseline test performance:")
print(
    f"ROC-AUC: {baseline_roc:.4f}"
)
print(
    f"PR-AUC : {baseline_pr:.4f}"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    f"Repeats per factor: {N_REPEATS}"
)

results = []


# ============================================================
# ORIGINAL-PREDICTOR-LEVEL PERMUTATION
# ============================================================

rng = np.random.RandomState(
    RANDOM_STATE
)

for feature in FEATURES:

    print(
        f"\nProcessing: {feature}"
    )

    roc_scores = []
    pr_scores = []

    for repeat in range(
        N_REPEATS
    ):

        X_perm = X_test.copy()

        shuffled = X_perm[
            feature
        ].values.copy()

        rng.shuffle(
            shuffled
        )

        X_perm[
            feature
        ] = shuffled

        perm_prob = model.predict_proba(
            X_perm
        )[:, 1]

        perm_roc = roc_auc_score(
            y_test,
            perm_prob,
        )

        perm_pr = average_precision_score(
            y_test,
            perm_prob,
        )

        roc_scores.append(
            baseline_roc - perm_roc
        )

        pr_scores.append(
            baseline_pr - perm_pr
        )

    roc_scores = np.array(
        roc_scores
    )

    pr_scores = np.array(
        pr_scores
    )

    results.append(
        {
            "feature": feature,

            "roc_auc_importance_mean":
                roc_scores.mean(),

            "roc_auc_importance_std":
                roc_scores.std(
                    ddof=1
                ),

            "roc_auc_importance_min":
                roc_scores.min(),

            "roc_auc_importance_max":
                roc_scores.max(),

            "pr_auc_importance_mean":
                pr_scores.mean(),

            "pr_auc_importance_std":
                pr_scores.std(
                    ddof=1
                ),

            "pr_auc_importance_min":
                pr_scores.min(),

            "pr_auc_importance_max":
                pr_scores.max(),
        }
    )


# ============================================================
# RF IMPURITY IMPORTANCE
# ============================================================

print("\nCalculating Random Forest impurity importance...")

classifier = model.named_steps[
    "classifier"
]

pre = model.named_steps[
    "preprocessor"
]

feature_names = (
    pre.get_feature_names_out()
)

impurity_values = (
    classifier.feature_importances_
)

impurity_df = pd.DataFrame(
    {
        "encoded_feature":
            feature_names,
        "importance":
            impurity_values,
    }
)


# ============================================================
# AGGREGATE IMPURITY TO ORIGINAL PREDICTORS
# ============================================================

aggregated_impurity = []

for feature in FEATURES:

    if feature in NUMERIC_FEATURES:

        matches = [
            feature
        ]

    else:

        prefix = (
            f"categorical__{feature}_"
        )

        matches = [
            name
            for name in feature_names
            if name.startswith(prefix)
        ]

    mask = [
        name in matches
        for name in feature_names
    ]

    total_importance = (
        impurity_values[mask].sum()
    )

    aggregated_impurity.append(
        {
            "feature": feature,
            "impurity_importance":
                total_importance,
        }
    )


impurity_original = pd.DataFrame(
    aggregated_impurity
)


# ============================================================
# COMBINE
# ============================================================

importance_df = pd.DataFrame(
    results
)

importance_df = importance_df.merge(
    impurity_original,
    on="feature",
    how="left",
)


# ============================================================
# RANKINGS
# ============================================================

importance_df[
    "rank_roc_auc"
] = (
    importance_df[
        "roc_auc_importance_mean"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

importance_df[
    "rank_pr_auc"
] = (
    importance_df[
        "pr_auc_importance_mean"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

importance_df[
    "rank_impurity"
] = (
    importance_df[
        "impurity_importance"
    ]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)


# Sort primarily by ROC-AUC permutation importance.

importance_df = importance_df.sort_values(
    "roc_auc_importance_mean",
    ascending=False,
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

importance_df.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("FINAL 8-FACTOR FEATURE IMPORTANCE")
print("=" * 70)

display_cols = [
    "feature",
    "roc_auc_importance_mean",
    "roc_auc_importance_std",
    "pr_auc_importance_mean",
    "pr_auc_importance_std",
    "impurity_importance",
]

print(
    importance_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print("\n" + "=" * 70)
print("RANKING")
print("=" * 70)

for i, row in importance_df.iterrows():

    print(
        f"{i + 1}. "
        f"{row['feature']} | "
        f"ROC importance="
        f"{row['roc_auc_importance_mean']:.6f} | "
        f"PR importance="
        f"{row['pr_auc_importance_mean']:.6f}"
    )


print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"ROC-AUC: {baseline_roc:.4f}"
)

print(
    f"PR-AUC : {baseline_pr:.4f}"
)

print(
    f"\nSaved: {OUTPUT}"
)

print("\nDONE")