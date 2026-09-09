from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "pilot_route_data"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "active_incidents.csv"
)


# ============================================================
# ACTIVE INCIDENT SCHEMA
# ============================================================

COLUMNS = [
    "incident_id",
    "latitude",
    "longitude",
    "timestamp",
    "incident_type",
    "severity",
    "blocking",
    "verification_status",
    "source",
    "description",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 4.1 — CREATE ACTIVE INCIDENT LAYER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CREATE EMPTY DATASET
    # --------------------------------------------------------

    incidents = pd.DataFrame(
        columns=COLUMNS
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    incidents.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\nActive incident dataset created.")

    print("\nColumns:")

    for column in COLUMNS:
        print(f"  {column}")

    print("\nCurrent incidents:")
    print(f"  {len(incidents)}")

    print("\nOutput:")
    print(OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("STEP 4.1 COMPLETE!")
    print("=" * 70)

    print("\nIMPORTANT:")
    print("This file is intentionally EMPTY.")
    print("No fake incident has been added.")
    print("Real incidents will be inserted later from")
    print("official reports, verified field reports,")
    print("sensors, or another real source.")


if __name__ == "__main__":
    main()