# logger.py
# Audit log writer. Appends every pipeline result to a JSONL file.

import json
import sys
from datetime import datetime, timezone

from config import AUDIT_LOG_PATH


def log(result: dict) -> None:
    """
    Adds logged_at timestamp to result and appends it as a single JSON line
    to the audit log. Never raises an exception.
    """
    try:
        entry = dict(result)
        entry["logged_at"] = datetime.now(timezone.utc).isoformat()

        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        print(
            f"WARNING: could not write to audit log {AUDIT_LOG_PATH}: {exc}",
            file=sys.stderr,
        )
