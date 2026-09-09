"""
STEP 64D-F
Validate reconstructed categorical rasters against the original
Bhuvan-derived training-point observations.

Inputs:
    processed/ner_training_points.geojson
    processed/ner_geomorphology.csv
    processed/ner_lulc.csv

    processed/predictors/geomorph_origin_250m.tif
    processed/predictors/lulc_level1_250m.tif

Outputs:
    processed/step64dF_categorical_validation_summary.csv
    processed/step64dF_geomorphology_validation.csv
    processed/step64dF_lulc_validation.csv
    processed/step64dF_categorical_point_comparison.csv

Important:
    - This is QA only.
    - No raster values are modified.
    - No missing values are fabricated.
    - Mizoram geomorphology remains NoData.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from sklearn.metrics import confusion_matrix


# ================================================================
# PATHS
# ================================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

POINTS = (
    PROJECT
    / "processed"
    / "ner_training_points.geojson"
)

GEOM_CSV = (
    PROJECT
    / "processed"
    / "ner_geomorphology.csv"
)

LULC_CSV = (
    PROJECT
    / "processed"
    / "ner_lulc.csv"
)

GEOM_RASTER = (
    PROJECT
    / "processed"
    / "predictors"
    / "geomorph_origin_250m.tif"
)

LULC_RASTER = (
    PROJECT
    / "processed"
    / "predictors"
    / "lulc_level1_250m.tif"
)

OUT_DIR = (
    PROJECT
    / "processed"
)

SUMMARY_OUT = (
    OUT_DIR
    / "step64dF_categorical_validation_summary.csv"
)

GEOM_OUT = (
    OUT_DIR
    / "step64dF_geomorphology_validation.csv"
)

LULC_OUT = (
    OUT_DIR
    / "step64dF_lulc_validation.csv"
)

POINT_OUT = (
    OUT_DIR
    / "step64dF_categorical_point_comparison.csv"
)


# ================================================================
# EXPECTED CODES
# ================================================================

GEOM_CODES = {
    1: "Denudational Origin",
    2: "Fluvial Origin",
    3: "Glacial Origin",
    4: "Lacustrine Origin",
    5: "Structural Origin",
    6: "Water Bodies",
}

LULC_CODES = {
    1: "Forest",
    2: "Wastelands",
    3: "Water Bodies",
    4: "Agriculture",
    5: "Others",
    6: "Built-up",
    7: "Grasslands / Grazing Lands",
}


# ================================================================
# HELPERS
# ================================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_column(df, candidates):
    """
    Return first matching column from candidates.
    """

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def read_raster_points(raster_path, points):
    """
    Extract nearest raster cell value at each point.
    """

    with rasterio.open(raster_path) as src:

        if points.crs != src.crs:
            points_projected = points.to_crs(
                src.crs
            )
        else:
            points_projected = points.copy()

        coords = [
            (geom.x, geom.y)
            for geom in points_projected.geometry
        ]

        values = []

        for value in src.sample(
            coords,
            indexes=1,
        ):
            values.append(
                int(value[0])
            )

    return np.array(
        values,
        dtype=np.int64,
    )


def make_confusion_table(
    expected,
    predicted,
    code_map,
    factor_name,
):
    """
    Create detailed expected/predicted confusion table.
    """

    codes = sorted(code_map.keys())

    rows = []

    cm = confusion_matrix(
        expected,
        predicted,
        labels=codes,
    )

    for i, expected_code in enumerate(codes):

        for j, predicted_code in enumerate(codes):

            count = int(cm[i, j])

            if count == 0:
                continue

            rows.append(
                {
                    "factor": factor_name,
                    "expected_code": expected_code,
                    "expected_class": code_map[
                        expected_code
                    ],
                    "predicted_code": predicted_code,
                    "predicted_class": code_map[
                        predicted_code
                    ],
                    "count": count,
                }
            )

    return pd.DataFrame(rows)


def calculate_validation(
    expected,
    predicted,
    code_map,
    factor_name,
):
    """
    Calculate coverage and agreement statistics.
    """

    expected = np.asarray(expected)
    predicted = np.asarray(predicted)

    total = len(expected)

    expected_valid = np.isin(
        expected,
        list(code_map.keys()),
    )

    predicted_valid = np.isin(
        predicted,
        list(code_map.keys()),
    )

    comparable = (
        expected_valid
        & predicted_valid
    )

    comparable_count = int(
        np.sum(comparable)
    )

    agreement_count = int(
        np.sum(
            expected[comparable]
            == predicted[comparable]
        )
    )

    if comparable_count > 0:

        agreement = (
            agreement_count
            / comparable_count
            * 100
        )

    else:
        agreement = np.nan

    return {
        "factor": factor_name,
        "total_points": total,
        "expected_valid": int(
            np.sum(expected_valid)
        ),
        "predicted_valid": int(
            np.sum(predicted_valid)
        ),
        "comparable_points": comparable_count,
        "agreement_points": agreement_count,
        "agreement_percent": agreement,
        "expected_missing": int(
            np.sum(~expected_valid)
        ),
        "raster_nodata": int(
            np.sum(~predicted_valid)
        ),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "STEP 64D-F — CATEGORICAL RASTER VALIDATION"
    )

    # ------------------------------------------------------------
    # Load training points
    # ------------------------------------------------------------

    print_header(
        "[1] Loading training points"
    )

    points = gpd.read_file(
        POINTS
    )

    print(
        "Training points:",
        len(points),
    )

    print(
        "CRS:",
        points.crs,
    )

    # ------------------------------------------------------------
    # Load original geomorphology
    # ------------------------------------------------------------

    print_header(
        "[2] Loading original categorical observations"
    )

    geom = pd.read_csv(
        GEOM_CSV
    )

    lulc = pd.read_csv(
        LULC_CSV
    )

    print(
        "Geomorphology rows:",
        len(geom),
    )

    print(
        "LUCC rows:",
        len(lulc),
    )

    # ------------------------------------------------------------
    # Identify coordinate columns
    # ------------------------------------------------------------

    geom_lat = find_column(
        geom,
        ["lat", "latitude"],
    )

    geom_lon = find_column(
        geom,
        ["lon", "longitude"],
    )

    lulc_lat = find_column(
        lulc,
        ["lat", "latitude"],
    )

    lulc_lon = find_column(
        lulc,
        ["lon", "longitude"],
    )

    if geom_lat is None or geom_lon is None:
        raise RuntimeError(
            "Could not identify geomorphology coordinates."
        )

    if lulc_lat is None or lulc_lon is None:
        raise RuntimeError(
            "Could not identify LUCC coordinates."
        )

    # ------------------------------------------------------------
    # Create coordinate keys
    # ------------------------------------------------------------

    points_work = points.copy()

    # GeoJSON geometry is authoritative.
    points_work["latitude"] = (
        points_work.geometry.y
    )

    points_work["longitude"] = (
        points_work.geometry.x
    )

    geom_work = geom.copy()

    geom_work["latitude"] = pd.to_numeric(
        geom_work[geom_lat]
    )

    geom_work["longitude"] = pd.to_numeric(
        geom_work[geom_lon]
    )

    lulc_work = lulc.copy()

    lulc_work["latitude"] = pd.to_numeric(
        lulc_work[lulc_lat]
    )

    lulc_work["longitude"] = pd.to_numeric(
        lulc_work[lulc_lon]
    )

    # Rounded coordinate key avoids floating-point
    # representation differences.
    points_work["coord_key"] = (
        points_work["latitude"].round(8).astype(str)
        + "_"
        + points_work["longitude"].round(8).astype(str)
    )

    geom_work["coord_key"] = (
        geom_work["latitude"].round(8).astype(str)
        + "_"
        + geom_work["longitude"].round(8).astype(str)
    )

    lulc_work["coord_key"] = (
        lulc_work["latitude"].round(8).astype(str)
        + "_"
        + lulc_work["longitude"].round(8).astype(str)
    )

    # ------------------------------------------------------------
    # Prepare expected geomorphology origin
    # ------------------------------------------------------------

    if "geomorph_origin" in geom_work.columns:

        geom_work["expected_geomorph_origin"] = (
            geom_work["geomorph_origin"]
        )

    elif "geomorphology" in geom_work.columns:

        def get_origin(value):

            if pd.isna(value):
                return np.nan

            value = str(value)

            prefixes = [
                "Structural Origin",
                "Fluvial Origin",
                "Denudational Origin",
                "Glacial Origin",
                "Water Bodies",
                "Lacustrine Origin",
            ]

            for prefix in prefixes:

                if value.startswith(prefix):
                    return prefix

            return np.nan

        geom_work[
            "expected_geomorph_origin"
        ] = geom_work[
            "geomorphology"
        ].apply(
            get_origin
        )

    else:

        raise RuntimeError(
            "Geomorphology file has neither "
            "'geomorph_origin' nor 'geomorphology'."
        )

    # ------------------------------------------------------------
    # Prepare expected LUCC
    # ------------------------------------------------------------

    if "Level_I" not in lulc_work.columns:

        raise RuntimeError(
            "LUCC file does not contain Level_I."
        )

    lulc_work["expected_lulc"] = (
        lulc_work["Level_I"]
    )

    # ------------------------------------------------------------
    # Merge expected observations
    # ------------------------------------------------------------

    print_header(
        "[3] Joining expected classes to training points"
    )

    comparison = points_work[
        [
            "coord_key",
            "latitude",
            "longitude",
        ]
    ].copy()

    comparison = comparison.merge(
        geom_work[
            [
                "coord_key",
                "expected_geomorph_origin",
            ]
        ],
        on="coord_key",
        how="left",
    )

    comparison = comparison.merge(
        lulc_work[
            [
                "coord_key",
                "expected_lulc",
            ]
        ],
        on="coord_key",
        how="left",
    )

    print(
        "Comparison rows:",
        len(comparison),
    )

    # ------------------------------------------------------------
    # Extract raster values
    # ------------------------------------------------------------

    print_header(
        "[4] Extracting raster values"
    )

    print(
        "Geomorphology raster:"
    )

    comparison[
        "geomorph_raster_code"
    ] = read_raster_points(
        GEOM_RASTER,
        points_work,
    )

    print(
        "LUCC raster:"
    )

    comparison[
        "lulc_raster_code"
    ] = read_raster_points(
        LULC_RASTER,
        points_work,
    )

    # ------------------------------------------------------------
    # Convert expected classes to codes
    # ------------------------------------------------------------

    geom_reverse = {
        value: key
        for key, value
        in GEOM_CODES.items()
    }

    lulc_reverse = {
        value: key
        for key, value
        in LULC_CODES.items()
    }

    comparison[
        "expected_geomorph_code"
    ] = comparison[
        "expected_geomorph_origin"
    ].map(
        geom_reverse
    )

    comparison[
        "expected_lulc_code"
    ] = comparison[
        "expected_lulc"
    ].map(
        lulc_reverse
    )

    # ------------------------------------------------------------
    # Compare geomorphology
    # ------------------------------------------------------------

    print_header(
        "[5] GEOMORPHOLOGY VALIDATION"
    )

    geom_expected = (
        comparison[
            "expected_geomorph_code"
        ]
        .fillna(0)
        .astype(int)
        .values
    )

    geom_predicted = (
        comparison[
            "geomorph_raster_code"
        ]
        .astype(int)
        .values
    )

    geom_result = calculate_validation(
        geom_expected,
        geom_predicted,
        GEOM_CODES,
        "geomorph_origin",
    )

    for key, value in geom_result.items():

        if key != "factor":
            print(
                f"{key}: {value}"
            )

    geom_table = make_confusion_table(
        geom_expected[
            np.isin(
                geom_expected,
                list(GEOM_CODES.keys()),
            )
            & np.isin(
                geom_predicted,
                list(GEOM_CODES.keys()),
            )
        ],
        geom_predicted[
            np.isin(
                geom_expected,
                list(GEOM_CODES.keys()),
            )
            & np.isin(
                geom_predicted,
                list(GEOM_CODES.keys()),
            )
        ],
        GEOM_CODES,
        "geomorph_origin",
    )

    # ------------------------------------------------------------
    # Compare LUCC
    # ------------------------------------------------------------

    print_header(
        "[6] LUCC VALIDATION"
    )

    lulc_expected = (
        comparison[
            "expected_lulc_code"
        ]
        .fillna(0)
        .astype(int)
        .values
    )

    lulc_predicted = (
        comparison[
            "lulc_raster_code"
        ]
        .astype(int)
        .values
    )

    lulc_result = calculate_validation(
        lulc_expected,
        lulc_predicted,
        LULC_CODES,
        "lulc_level1",
    )

    for key, value in lulc_result.items():

        if key != "factor":
            print(
                f"{key}: {value}"
            )

    lulc_table = make_confusion_table(
        lulc_expected[
            np.isin(
                lulc_expected,
                list(LULC_CODES.keys()),
            )
            & np.isin(
                lulc_predicted,
                list(LULC_CODES.keys()),
            )
        ],
        lulc_predicted[
            np.isin(
                lulc_expected,
                list(LULC_CODES.keys()),
            )
            & np.isin(
                lulc_predicted,
                list(LULC_CODES.keys()),
            )
        ],
        LULC_CODES,
        "lulc_level1",
    )

    # ------------------------------------------------------------
    # Point-level agreement flags
    # ------------------------------------------------------------

    comparison[
        "geomorph_agreement"
    ] = (
        comparison[
            "expected_geomorph_code"
        ]
        == comparison[
            "geomorph_raster_code"
        ]
    )

    comparison[
        "lulc_agreement"
    ] = (
        comparison[
            "expected_lulc_code"
        ]
        == comparison[
            "lulc_raster_code"
        ]
    )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    print_header(
        "[7] Saving validation outputs"
    )

    comparison.to_csv(
        POINT_OUT,
        index=False,
    )

    geom_table.to_csv(
        GEOM_OUT,
        index=False,
    )

    lulc_table.to_csv(
        LULC_OUT,
        index=False,
    )

    summary = pd.DataFrame(
        [
            geom_result,
            lulc_result,
        ]
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    # ------------------------------------------------------------
    # State-wise validation
    # ------------------------------------------------------------

    print_header(
        "[8] State-wise agreement"
    )

    # Try to identify state from points.
    state_col = find_column(
        points_work,
        [
            "state",
            "shapeName",
        ],
    )

    if state_col is not None:

        comparison["state"] = points_work[
            state_col
        ].values

        state_rows = []

        for state, group in comparison.groupby(
            "state",
            dropna=False,
        ):

            geom_comp = (
                group[
                    "expected_geomorph_code"
                ].notna()
                & group[
                    "geomorph_raster_code"
                ].isin(
                    list(GEOM_CODES.keys())
                )
            )

            lulc_comp = (
                group[
                    "expected_lulc_code"
                ].notna()
                & group[
                    "lulc_raster_code"
                ].isin(
                    list(LULC_CODES.keys())
                )
            )

            state_rows.append(
                {
                    "state": state,
                    "points": len(group),
                    "geomorph_comparable": int(
                        geom_comp.sum()
                    ),
                    "geomorph_agreement_percent": (
                        group.loc[
                            geom_comp,
                            "geomorph_agreement"
                        ].mean()
                        * 100
                        if geom_comp.sum() > 0
                        else np.nan
                    ),
                    "lulc_comparable": int(
                        lulc_comp.sum()
                    ),
                    "lulc_agreement_percent": (
                        group.loc[
                            lulc_comp,
                            "lulc_agreement"
                        ].mean()
                        * 100
                        if lulc_comp.sum() > 0
                        else np.nan
                    ),
                }
            )

        state_table = pd.DataFrame(
            state_rows
        )

        state_out = (
            OUT_DIR
            / "step64dF_statewise_agreement.csv"
        )

        state_table.to_csv(
            state_out,
            index=False,
        )

        print(
            state_table.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print_header(
        "STEP 64D-F COMPLETED"
    )

    print(
        "\nOverall validation:"
    )

    print(
        f"  Geomorphology agreement: "
        f"{geom_result['agreement_percent']:.2f}%"
    )

    print(
        f"  LUCC agreement: "
        f"{lulc_result['agreement_percent']:.2f}%"
    )

    print(
        "\nOutputs:"
    )

    print(
        GEOM_OUT
    )

    print(
        LULC_OUT
    )

    print(
        POINT_OUT
    )

    print(
        SUMMARY_OUT
    )

    print(
        "\nNo raster values were modified."
    )


if __name__ == "__main__":
    main()