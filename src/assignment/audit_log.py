"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from agents.security_boundary import contains_secret, normalize_for_security
from guardrails.output_guardrails import content_filter


def redact_for_audit(text: str) -> str:
    """Apply the output filter to both sides of the exchange before storage."""
    cleaned = content_filter(normalize_for_security(text))["redacted"]
    if any(contains_secret(part) for part in cleaned.split("[REDACTED]")):
        return "[REDACTED]"
    return cleaned


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[float, str, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the sanitized input and start time before running the plugins."""
        self._open[user_id, request_id] = (monotonic(), utc_now_iso(), redact_for_audit(text))

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Pair the decision with its input and record the elapsed time."""
        start = self._open.pop((user_id, request_id), None)
        started, timestamp, input_text = start if start else (None, None, None)
        elapsed = round((monotonic() - started) * 1000, 2) if started is not None else None
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp or utc_now_iso(),
            "input": input_text,
            "output": redact_for_audit(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": elapsed,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
