# pipeline.py
# Runs Stage 1 metadata analysis and returns a clean, self-contained result.
#
# Open-ended design:
#   result["next_stage_input"]  ← everything a future Stage 2 API needs
#   result["stage1_verdict"]    ← human-readable decision from Stage 1 alone
#   result["stage1_ai_score"]   ← 0.0–1.0 AI probability from Stage 1
#
# To wire in a future API (e.g. Hive, Google SafeSearch, custom model):
#   1. Import your stage 2 module
#   2. Pass result["next_stage_input"] to it
#   3. Blend / override result["stage1_ai_score"] with the API score
#   4. Re-run the verdict logic

import time
from datetime import datetime, timezone

import stage1_metadata
import logger
from config import STAGE1_THRESHOLDS


def _verdict_from_score(score: float) -> str:
    if score < STAGE1_THRESHOLDS["likely_real"]:
        return "LIKELY_REAL"
    elif score < STAGE1_THRESHOLDS["uncertain"]:
        return "UNCERTAIN"
    elif score < STAGE1_THRESHOLDS["probably_ai"]:
        return "PROBABLY_AI"
    else:
        return "LIKELY_AI"


def run(image_path: str) -> dict:
    """
    Run Stage 1 metadata analysis on an image.

    Returns a dict with:
      - All Stage 1 signal fields
      - stage1_verdict: human-readable label
      - next_stage_input: pre-packaged payload for a future Stage 2 API
      - pipeline_log: timestamped list of processing notes
    """
    start_ms = time.monotonic()
    ran_at = datetime.now(timezone.utc).isoformat()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    s1 = stage1_metadata.run(image_path)
    pipeline_log = list(s1.get("pipeline_notes", []))

    score = s1.get("stage1_ai_score", 0.5)
    verdict = _verdict_from_score(score)

    processing_time_ms = int((time.monotonic() - start_ms) * 1000)

    # ── next_stage_input: plug this into any future API ───────────────────────
    # Contains everything useful for making a follow-up API call.
    next_stage_input = {
        "image_path":           image_path,
        "stage1_ai_score":      score,
        "stage1_verdict":       verdict,
        "ai_software_detected": s1.get("ai_software_detected"),
        "ai_software_name":     s1.get("ai_software_name"),
        "sd_png_params_found":  s1.get("sd_png_params_found"),
        "xmp_ai_claim_found":   s1.get("xmp_ai_claim_found"),
        "has_c2pa":             s1.get("has_c2pa"),
        "has_exif":             s1.get("has_exif"),
        "real_camera_make":     s1.get("real_camera_make"),
        "exif_score":           s1.get("exif_score"),
        "format":               s1.get("format"),
        "width":                s1.get("width"),
        "height":               s1.get("height"),
        "file_size_kb":         s1.get("file_size_kb"),
        "signals":              s1.get("stage1_signals", []),
        # Suggested action hint for the future API:
        "recommended_action":   (
            "skip_api_call"   if s1.get("early_exit") else
            "call_api"
        ),
        "skip_reason":          s1.get("early_exit_verdict") if s1.get("early_exit") else None,
    }

    # ── Combined result ───────────────────────────────────────────────────────
    result = {
        # ── Summary ──────────────────────────────────────────────────────────
        "image_path":          image_path,
        "stage1_ai_score":     score,
        "stage1_verdict":      verdict,
        "processing_time_ms":  processing_time_ms,
        "ran_at":              ran_at,
        "pipeline_log":        pipeline_log,

        # ── Stage 1 signals (for UI / debugging) ─────────────────────────────
        "ai_software_detected": s1.get("ai_software_detected"),
        "ai_software_name":     s1.get("ai_software_name"),
        "sd_png_params_found":  s1.get("sd_png_params_found"),
        "xmp_ai_claim_found":   s1.get("xmp_ai_claim_found"),
        "has_c2pa":             s1.get("has_c2pa"),
        "has_exif":             s1.get("has_exif"),
        "exif_score":           s1.get("exif_score"),
        "camera_make":          s1.get("camera_make"),
        "camera_model":         s1.get("camera_model"),
        "real_camera_make":     s1.get("real_camera_make"),
        "has_gps":              s1.get("has_gps"),
        "file_size_kb":         s1.get("file_size_kb"),
        "format":               s1.get("format"),
        "width":                s1.get("width"),
        "height":               s1.get("height"),
        "stage1_signals":       s1.get("stage1_signals", []),
        "early_exit":           s1.get("early_exit"),
        "early_exit_verdict":   s1.get("early_exit_verdict"),
        "stage1_error":         s1.get("error"),

        # ── Open-ended hook for Stage 2 ───────────────────────────────────────
        "next_stage_input":    next_stage_input,
    }

    # ── Audit log ─────────────────────────────────────────────────────────────
    logger.log(result)

    return result
