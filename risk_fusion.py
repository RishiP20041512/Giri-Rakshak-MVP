"""
risk_fusion.py

Giri-Rakshak Risk Fusion Engine

Combines:
    S = Static Random Forest susceptibility score (0-1)
    T = Dynamic rainfall/wetness TriggerScore (0-1)

Final:
    RiskScore = 0.6*S + 0.4*T
"""

from dataclasses import dataclass


# ============================================================
# CONFIGURATION
# ============================================================

STATIC_WEIGHT = 0.60
DYNAMIC_WEIGHT = 0.40


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class RiskFusionResult:
    susceptibility_score: float
    trigger_score: float
    risk_score: float
    risk_level: str
    reasoning: list

    def as_dict(self):
        return {
            "susceptibility_score": self.susceptibility_score,
            "trigger_score": self.trigger_score,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "reasoning": self.reasoning,
        }


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(risk_score: float) -> str:

    if risk_score >= 0.75:
        return "CRITICAL"

    elif risk_score >= 0.50:
        return "HIGH"

    elif risk_score >= 0.25:
        return "MEDIUM"

    return "LOW"


# ============================================================
# FUSION FUNCTION
# ============================================================

def fuse_risk(
    susceptibility_score: float,
    trigger_score: float
) -> RiskFusionResult:

    # Keep both model outputs within 0-1
    S = max(0.0, min(1.0, float(susceptibility_score)))
    T = max(0.0, min(1.0, float(trigger_score)))

    # --------------------------------------------------------
    # FINAL EQUATION
    # --------------------------------------------------------

    risk_score = (
        STATIC_WEIGHT * S
        +
        DYNAMIC_WEIGHT * T
    )

    risk_score = round(risk_score, 4)

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    risk_level = classify_risk(risk_score)

    # --------------------------------------------------------
    # AUDITABLE REASONING
    # --------------------------------------------------------

    reasoning = [
        f"Static susceptibility score = {S:.3f}",
        f"Dynamic trigger score = {T:.3f}",
        f"Static contribution = {STATIC_WEIGHT} × {S:.3f} = {STATIC_WEIGHT*S:.3f}",
        f"Dynamic contribution = {DYNAMIC_WEIGHT} × {T:.3f} = {DYNAMIC_WEIGHT*T:.3f}",
        f"Final RiskScore = {risk_score:.3f}",
        f"Risk level = {risk_level}",
    ]

    return RiskFusionResult(
        susceptibility_score=round(S, 4),
        trigger_score=round(T, 4),
        risk_score=risk_score,
        risk_level=risk_level,
        reasoning=reasoning,
    )


# ============================================================
# FUSE DIRECTLY FROM FRIEND'S DYNAMIC RESULT
# ============================================================

def fuse_with_dynamic_result(
    susceptibility_score: float,
    dynamic_result
) -> RiskFusionResult:

    """
    Connects directly to the DynamicLayerResult produced
    by the friend's pipeline.py.
    """

    trigger_score = dynamic_result.rainfall.trigger_score

    result = fuse_risk(
        susceptibility_score=susceptibility_score,
        trigger_score=trigger_score,
    )

    # Add dynamic-state information to the audit trail
    result.reasoning.append(
        f"Dynamic layer state = {dynamic_result.final_state}"
    )

    if dynamic_result.escalation_reason:
        result.reasoning.append(
            f"Dynamic escalation = {dynamic_result.escalation_reason}"
        )

    return result


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GIRI-RAKSHAK RISK FUSION ENGINE")
    print("=" * 60)

    # Example only for software verification.
    # This does NOT represent real environmental data.

    S = 0.80
    T = 0.70

    result = fuse_risk(S, T)

    print("\nStatic Susceptibility :", result.susceptibility_score)
    print("Dynamic Trigger       :", result.trigger_score)
    print("Final Risk Score      :", result.risk_score)
    print("Risk Level            :", result.risk_level)

    print("\nAudit trail:")

    for line in result.reasoning:
        print(" -", line)

    print("\n" + "=" * 60)