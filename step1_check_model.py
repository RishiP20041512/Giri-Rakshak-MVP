from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "processed" / "step69_production_8factor_rf.joblib"

print("=" * 60)
print("GIRI-RAKSHAK — STEP 1 MODEL CHECK")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    print("\n❌ MODEL FILE NOT FOUND")
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("\n✅ MODEL LOADED")
print("Model type:")
print(type(model))

print("\nModel:")
print(model)

print("\nModel parameters:")
try:
    print(model.get_params())
except Exception as e:
    print("Could not read parameters:", e)

print("\nFeature information:")

if hasattr(model, "feature_names_in_"):
    print("feature_names_in_:")
    print(model.feature_names_in_)

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"  {name}: {type(step)}")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)