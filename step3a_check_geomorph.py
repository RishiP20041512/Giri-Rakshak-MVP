from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "processed"
    / "step69_production_8factor_rf.joblib"
)

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]

encoder = preprocessor.named_transformers_["categorical"]

print("=" * 60)
print("SAVED MODEL — GEOMORPHOLOGY ENCODER CHECK")
print("=" * 60)

print("\nEncoder:")
print(encoder)

print("\nCategories learned during training:")

for i, categories in enumerate(encoder.categories_):

    print(f"\nCategorical feature {i}:")
    print("Values:", categories)
    print("dtype :", categories.dtype)

    for value in categories:
        print(
            f"  value = {repr(value)}"
            f" | type = {type(value)}"
        )

print("\n" + "=" * 60)