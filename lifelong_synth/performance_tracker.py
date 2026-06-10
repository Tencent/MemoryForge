from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Lightweight, failure-safe JSONL performance tracker."""

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = Path(output_path).resolve() if output_path else None
        if self.output_path:
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("PerformanceTracker init failed to prepare output dir")
                self.output_path = None

    @contextmanager
    def track(self, name: str, **tags: Any) -> Iterator[None]:
        start_ts = time.time()
        error: Optional[str] = None
        status = "success"
        try:
            yield
        except Exception as e:
            status = "error"
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            end_ts = time.time()
            record = {
                "name": name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_ms": round((end_ts - start_ts) * 1000, 3),
                "status": status,
                "error": error,
                "tags": tags,
            }
            self._append_record(record)

    def _append_record(self, record: Dict[str, Any]) -> None:
        if not self.output_path:
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"PerformanceTracker failed to write metric: {e}")

    def load_records(self) -> List[Dict[str, Any]]:
        if not self.output_path or not self.output_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            records.append(payload)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"PerformanceTracker failed to load metrics: {e}")
        return records

    def build_slowest_summary(
        self,
        top_n: int = 5,
        groups: Optional[List[Dict[str, str]]] = None,
    ) -> List[str]:
        records = self.load_records()
        if not records:
            return ["No performance metrics recorded."]

        groups = groups or [
            {"title": "Slowest operations", "prefix": ""},
        ]

        lines: List[str] = []
        for group in groups:
            title = group.get("title", "Slowest operations")
            prefix = group.get("prefix", "")
            filtered = [
                rec for rec in records
                if not prefix or str(rec.get("name", "")).startswith(prefix)
            ]
            if not filtered:
                continue
            filtered.sort(key=lambda rec: float(rec.get("duration_ms", 0.0)), reverse=True)
            lines.append(title + ":")
            for rec in filtered[:top_n]:
                tags = rec.get("tags") or {}
                tag_bits = []
                if isinstance(tags, dict):
                    for key in sorted(tags):
                        tag_bits.append(f"{key}={tags[key]}")
                tag_text = f" | {' '.join(tag_bits)}" if tag_bits else ""
                lines.append(
                    f"- {rec.get('name')} | {rec.get('duration_ms')} ms | {rec.get('status')}{tag_text}"
                )
        return lines or ["No matching performance metrics found."]
