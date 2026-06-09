# stage3_router.py
# Stage 3: Router — applies threshold logic to produce a final verdict and action.
#
# Input: final_score (blended from Stage 1 + Stage 2)
# Output: verdict, action, reasoning_summary
#
# Routing logic (matches the report exactly):
#   score < 0.30  → AUTO_APPROVE  (LIKELY_REAL)
#   score > 0.85  → AUTO_REJECT   (LIKELY_AI)
#   0.30 – 0.85   → HUMAN_REVIEW  (gray zone)

from config import STAGE1_THRESHOLDS


def route(final_score: float, stage1_result: dict, stage2_result: dict | None = None) -> dict:
    """
    Apply routing logic based on the final blended score.

    Returns:
        verdict: str   — LIKELY_REAL | UNCERTAIN | PROBABLY_AI | LIKELY_AI
        action: str    — AUTO_APPROVE | HUMAN_REVIEW | AUTO_REJECT
        confidence: str — HIGH | MEDIUM | LOW
        routing_reason: str — human-readable explanation
    """
    score = final_score

    # Determine verdict band
    if score < STAGE1_THRESHOLDS["likely_real"]:
        verdict = "LIKELY_REAL"
    elif score < STAGE1_THRESHOLDS["uncertain"]:
        verdict = "UNCERTAIN"
    elif score < STAGE1_THRESHOLDS["probably_ai"]:
        verdict = "PROBABLY_AI"
    else:
        verdict = "LIKELY_AI"

    # Determine action
    if score < 0.30:
        action = "AUTO_APPROVE"
        confidence = "HIGH"
        routing_reason = (
            f"Score {score:.2f} is below 0.30. Image shows strong real-camera signals. "
            "Automatically approved with high confidence."
        )
    elif score > 0.85:
        action = "AUTO_REJECT"
        confidence = "HIGH"
        routing_reason = (
            f"Score {score:.2f} exceeds 0.85. Strong AI-generation signals detected. "
            "Automatically rejected with high confidence."
        )
    else:
        action = "HUMAN_REVIEW"
        confidence = "LOW" if 0.45 <= score <= 0.65 else "MEDIUM"
        routing_reason = (
            f"Score {score:.2f} falls in the gray zone (0.30–0.85). "
            "Insufficient certainty for automatic decision. Queued for human review."
        )

    # Build a clean summary for the UI
    s1_score = stage1_result.get("stage1_ai_score", 0)
    s2_score = stage2_result.get("stage2_ai_score") if stage2_result else None
    s2_reasoning = (stage2_result or {}).get("stage2_reasoning", "")

    pipeline_summary = {
        "stage1_score": s1_score,
        "stage2_score": s2_score,
        "final_score": final_score,
        "verdict": verdict,
        "action": action,
        "confidence": confidence,
        "routing_reason": routing_reason,
        "llm_reasoning": s2_reasoning,
    }

    return pipeline_summary
