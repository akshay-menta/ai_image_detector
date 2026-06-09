# stage2_winston.py
# Stage 2: AI Image Detection using Winston AI's image detection API.
#
# Winston AI API spec (from OpenAPI):
#   POST https://api.gowinston.ai/v2/image-detection
#   Content-Type: application/json
#   Authorization: Bearer <token>
#   Body: { "url": "<public image URL>", "version": "3" }
#
# The API requires a PUBLICLY accessible image URL.
# For local files, we upload to freeimage.host (free, public API) to get a temp public URL,
# then pass that URL to Winston AI.
#
# Response fields:
#   score            → 0–100 Human Score (100 = human, 0 = AI)
#   human_probability → 0–1
#   ai_probability   → 0–1  ← we use this directly as ai_score
#   c2pa             → C2PA metadata object (if present)

import os
import base64
import requests


# ── API endpoints ──────────────────────────────────────────────────────────────
_WINSTON_ENDPOINT = "https://api.gowinston.ai/v2/image-detection"
_FREEIMAGE_UPLOAD_URL = "https://freeimage.host/api/1/upload"

# freeimage.host public API key (works for POC)
_FREEIMAGE_API_KEY = "6d207e02198a847aa98d0a2a901485a5"


def _upload_to_freeimage(image_path: str) -> str | None:
    """
    Upload a local image to freeimage.host and return a public URL.
    Returns the image URL string, or None on failure.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = requests.post(
        _FREEIMAGE_UPLOAD_URL,
        data={"key": _FREEIMAGE_API_KEY, "source": b64, "format": "json"},
        timeout=30,
    )

    if resp.status_code != 200:
        return None

    data = resp.json()
    return data.get("image", {}).get("url")


def run(image_path: str, stage1_result: dict, api_key: str) -> dict:
    """
    Run Stage 2 Winston AI image detection.

    Args:
        image_path: Path to the local image file.
        stage1_result: Full result dict from Stage 1 (used for reasoning context).
        api_key: Winston AI Bearer token (from dev.gowinston.ai).

    Returns dict with:
        stage2_ai_score:       float 0.0–1.0  AI probability
        stage2_human_score:    float 0–100    raw Winston human score
        stage2_reasoning:      str            human-readable explanation
        stage2_model:          str
        stage2_raw:            dict           full Winston response
        stage2_error:          str or None
    """
    if not api_key:
        return _error("missing_api_key", "No Winston AI API key provided — Stage 2 skipped.")

    # ── Step 1: upload image to get a public URL ──────────────────────────────
    try:
        public_url = _upload_to_freeimage(image_path)
    except Exception as e:
        return _error("freeimage_upload_failed", f"Failed to upload image to freeimage.host: {e}")

    if not public_url:
        return _error("freeimage_no_url", "freeimage.host upload succeeded but returned no URL.")

    # ── Step 2: call Winston AI with the public URL ───────────────────────────
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": public_url,
        "version": "3",  # latest and most accurate Winston model
    }

    try:
        response = requests.post(
            _WINSTON_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return _error("timeout", "Winston AI API request timed out after 30s.")
    except Exception as e:
        return _error("request_failed", f"Winston AI API request failed: {e}")

    if response.status_code != 200:
        return _error(
            f"http_{response.status_code}",
            f"Winston AI returned HTTP {response.status_code}: {response.text[:300]}",
        )

    try:
        data = response.json()
    except Exception as e:
        return _error("json_parse_error", f"Could not parse Winston AI response: {e}")

    # ── Step 3: extract score ─────────────────────────────────────────────────
    # Prefer ai_probability (direct 0–1 value), fall back to inverting score.
    ai_prob = data.get("ai_probability")
    human_score = data.get("score")

    if ai_prob is not None:
        ai_score = round(float(ai_prob), 4)
    elif human_score is not None:
        ai_score = round(1.0 - float(human_score) / 100.0, 4)
    else:
        return _error("missing_score", f"Winston AI response missing score fields. Raw: {str(data)[:300]}")

    ai_score = max(0.0, min(1.0, ai_score))
    human_score_raw = float(human_score) if human_score is not None else round((1 - ai_score) * 100, 1)

    # ── Step 4: build reasoning ───────────────────────────────────────────────
    reasoning = _build_reasoning(ai_score, human_score_raw, data, stage1_result)

    return {
        "stage2_ai_score": ai_score,
        "stage2_human_score": human_score_raw,
        "stage2_reasoning": reasoning,
        "stage2_model": f"winston-ai/v2/image-detection (model v{data.get('version', '3')})",
        "stage2_raw": data,
        "stage2_error": None,
    }


def _error(code: str, message: str) -> dict:
    return {
        "stage2_ai_score": None,
        "stage2_human_score": None,
        "stage2_reasoning": message,
        "stage2_model": "winston-ai/v2/image-detection",
        "stage2_raw": None,
        "stage2_error": code,
    }


def _build_reasoning(ai_score: float, human_score: float, data: dict, stage1: dict) -> str:
    """Build a clean human-readable reasoning string from Winston AI's response."""
    if ai_score >= 0.85:
        verdict_text = "highly likely to be AI-generated"
    elif ai_score >= 0.60:
        verdict_text = "probably AI-generated"
    elif ai_score >= 0.30:
        verdict_text = "uncertain — could be AI or real"
    else:
        verdict_text = "likely a real photograph"

    lines = [
        f"Winston AI Human Score: {human_score:.1f}/100 "
        f"(AI probability: {ai_score:.0%}). The image is {verdict_text}."
    ]

    # Append C2PA info if Winston found it
    c2pa = data.get("c2pa")
    if c2pa and isinstance(c2pa, dict) and c2pa.get("active_manifest"):
        manifest = c2pa["active_manifest"]
        generator = manifest.get("claim_generator") or manifest.get("vendor") or "unknown"
        lines.append(f"C2PA content credential detected — generator: {generator}.")

    # Cross-reference with Stage 1
    s1_score = stage1.get("stage1_ai_score", 0.5)
    if abs(ai_score - s1_score) > 0.3:
        lines.append(
            f"Note: Stage 1 metadata score ({s1_score:.2f}) diverges from Winston AI "
            f"({ai_score:.2f}) — blended result used for final decision."
        )

    return " ".join(lines)
