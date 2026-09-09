import inspect

from dynamic.rainfall_trigger import (
    simplified_two_window_trigger,
    antecedent_wetness_index,
    combined_trigger_score,
    alert_state,
    evaluate_rainfall_trigger,
)

from dynamic.pipeline import run_dynamic_layer


print("=" * 70)
print("GIRI-RAKSHAK — DYNAMIC FUNCTION CHECK")
print("=" * 70)

functions = [
    simplified_two_window_trigger,
    antecedent_wetness_index,
    combined_trigger_score,
    alert_state,
    evaluate_rainfall_trigger,
    run_dynamic_layer,
]

for func in functions:
    print("\n" + "-" * 70)
    print(f"FUNCTION: {func.__name__}")
    print("-" * 70)
    print("Signature:")
    print(inspect.signature(func))

    print("\nSource:")
    try:
        print(inspect.getsource(func))
    except Exception as e:
        print("Could not read source:", e)

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)