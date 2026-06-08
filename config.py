# config.py
# All tunable values live here. No magic numbers anywhere else in the codebase.

# Stage 1 score thresholds (0.0 – 1.0)
# These control how the Stage 1 AI score is interpreted.
# Swap in any external API score here when you add a Stage 2.
STAGE1_THRESHOLDS = {
    "likely_real": 0.30,   # score < this → LIKELY_REAL
    "uncertain":   0.60,   # score < this → UNCERTAIN
    "probably_ai": 0.85,   # score < this → PROBABLY_AI
    # score >= probably_ai → LIKELY_AI
}

# Audit log (JSONL — one JSON object per line)
AUDIT_LOG_PATH = "audit_log.jsonl"
