import joblib
import os

MODEL_PATH = r"processed\random_forest_model.joblib"

print("=" * 60)
print("GIRIRAKSHAK MODEL CHECK")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found!")
    print("Expected:")
    print(os.path.abspath(MODEL_PATH))
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully.")
print("Model type:", type(model).__name__)

print("\nNumber of features expected:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nFeature names stored in model:")
    for i, name in enumerate(model.feature_names_in_, start=1):
        print(f"{i}. {name}")
else:
    print("\nWARNING:")
    print("This model does not contain feature_names_in_.")
    print("We will use the known 8-factor order.")

print("\nClasses:", model.classes_)

print("\n" + "=" * 60)