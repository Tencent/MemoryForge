"""
Memory Manager — Memory Storage & Lifecycle Management
========================================================
Manages the creation, storage, and lifecycle of memory entries
produced during the simulation.

This module belongs to the P4 Memory Organiser layer and is responsible for:
  - Defining data models for the structured memory base
  - Creating memory entries from scene results
  - Storing and indexing memories for efficient retrieval
  - Managing memory salience decay over simulated time
  - Serializing memory history for persistence

Memory Base Structure
---------------------
The memory base follows a three-part structure:

1. **previous_life_period**: Historical life periods keyed by period_id.
   Each period contains a time_period string, a period_summary, and a
   list of event IDs that occurred during that period.

2. **previous_events**: Historical events keyed by event_id.
   Each event has a resolution_level ("high", "medium", or "low"),
   a time_period string, a summary, participants, languages_in_use, and
   interaction_details (populated only for high-resolution events).

3. **current_life_summary**: A string summarising the persona's current
   life state and recent developments.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from lifelong_synth.performance_tracker import PerformanceTracker
from lifelong_synth.simulation_p4_memory_organiser.definition import (
    VALID_TIME_SEGMENTS,
    VALID_LANGUAGES,
    VALID_RESOLUTION_LEVELS,
    LifePeriodRecord,
    InteractionDetail,
    EventRecord,
    MemoryBase,
    MemoryEntry,
)
from lifelong_synth.simulation_p4_memory_organiser.embedding_engine import EmbeddingEngine
from lifelong_synth.simulation_p4_memory_organiser.write_pipeline import WriteTimePipeline
from lifelong_synth.simulation_p4_memory_organiser.memory_retriever import MemoryRetriever

logger = logging.getLogger(__name__)

EVENT_STORAGE_STATUS_GENERAL_EVENT_READY = "general_event_ready"
EVENT_STORAGE_STATUS_SIMULATING = "simulating"
EVENT_STORAGE_STATUS_FINAL = "final"
EVENT_STORAGE_STATUS_FAILED = "failed"


# ================================================================
# Memory Manager
# ================================================================

class MemoryManager:
    """
    Manages the full lifecycle of memory entries and the structured
    memory base.

    Responsibilities:
      - Load / initialise the structured memory base
      - Add life periods, events, and interaction details
      - Query and retrieve historical data
      - Update the current life summary
      - Persist the memory base to disk
      - Maintain backward-compatible MemoryEntry operations

    Data flow:
      SceneDirector (P3) → MemoryManager → MemoryBase
                                         → MemoryRetriever (P4)
    """

    def __init__(self, memory_base_path: Optional[str] = None, llm_client=None, performance_tracker: Optional[PerformanceTracker] = None, embedding_model: str = "intfloat/multilingual-e5-base", embedding_device: str = "cpu"):
        """
        Initialise the MemoryManager.

        Args:
            memory_base_path: Optional path to a JSON file to load the
                initial memory base from.  If *None* or the file does not
                exist, an empty memory base is created.
            llm_client: Optional async LLM client for write-time summary
                generation. If None, only rule-based processing is used.
            embedding_model: Name of the local embedding model for STARE-E v2.
            embedding_device: Device for embedding inference ('cpu' or 'cuda').
        """
        self._memory_base: MemoryBase = MemoryBase()
        self._file_path: Optional[str] = memory_base_path
        self._performance_tracker = performance_tracker
        self._embedding_store_path: Optional[str] = None
        self._embedding_manifest_path: Optional[str] = None
        self._embedding_debug_path: Optional[str] = None
        self._default_run_id: str = ""

        # Legacy flat memory list (backward compatibility)
        self._memories: List[MemoryEntry] = []
        self._index: Dict[str, MemoryEntry] = {}  # memory_id -> MemoryEntry

        # 🆕 v6 / STARE-E v2: Embedding engine (local, zero API cost)
        self._embedding_engine = EmbeddingEngine(
            model_name=embedding_model,
            device=embedding_device,
        )

        # Write-time pipeline and retriever (with embedding support)
        # Must be initialized BEFORE load() since load() calls _lazy_enrich_on_load()
        self._write_pipeline = WriteTimePipeline(
            llm_client=llm_client,
            embedding_engine=self._embedding_engine,
        )
        self._retriever = MemoryRetriever(
            self._memory_base,
            embedding_engine=self._embedding_engine,
        )

        if memory_base_path and os.path.isfile(memory_base_path):
            self.load(memory_base_path)

    @property
    def default_file_path(self) -> Optional[str]:
        """Return the configured default persistence path, if any."""
        return self._file_path

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def memory_base(self) -> MemoryBase:
        """Return the structured memory base."""
        return self._memory_base

    @property
    def retriever(self) -> 'MemoryRetriever':
        """Public access to the retriever for P2/P3."""
        return self._retriever

    @property
    def write_pipeline(self) -> 'WriteTimePipeline':
        """Public access to the write pipeline."""
        return self._write_pipeline

    @property
    def memories(self) -> List[MemoryEntry]:
        """Return all stored legacy memories in chronological order."""
        return list(self._memories)

    @property
    def count(self) -> int:
        """Return the total number of stored legacy memories."""
        return len(self._memories)

    @property
    def period_count(self) -> int:
        """Return the number of life periods in the memory base."""
        return len(self._memory_base.previous_life_period)

    @property
    def event_count(self) -> int:
        """Return the number of events in the memory base."""
        return len(self._memory_base.previous_events)

    # ----------------------------------------------------------------
    # Life Period Operations
    # ----------------------------------------------------------------

    def add_life_period(
        self,
        period_id: str,
        time_period: str,
        period_summary: str,
        events: Optional[List[str]] = None,
    ) -> LifePeriodRecord:
        """
        Add or update a life period in the memory base.

        Args:
            period_id: Unique period identifier (e.g. 'LP1').
            time_period: Time span string (e.g. '1998-03-27 to 2002-03-26').
            period_summary: Brief narrative summary.
            events: Optional list of event IDs belonging to this period.

        Returns:
            The created or updated LifePeriodRecord.
        """
        record = LifePeriodRecord(
            time_period=time_period,
            period_summary=period_summary,
            events=events or [],
            period_summary_brief=self._write_pipeline.generate_period_summary_brief(period_summary),
        )
        self._memory_base.previous_life_period[period_id] = record
        logger.debug(f"Life period added/updated: {period_id}")
        return record

    def get_life_period(self, period_id: str) -> Optional[LifePeriodRecord]:
        """
        Retrieve a life period by its ID.

        Args:
            period_id: The period identifier.

        Returns:
            The LifePeriodRecord if found, else None.
        """
        return self._memory_base.previous_life_period.get(period_id)

    def get_all_life_periods(self) -> Dict[str, LifePeriodRecord]:
        """Return all life periods as a dict keyed by period_id."""
        return dict(self._memory_base.previous_life_period)

    def remove_life_period(self, period_id: str) -> bool:
        """
        Remove a life period and its associated events from the memory base.

        Args:
            period_id: The period identifier to remove.

        Returns:
            True if the period was found and removed, False otherwise.
        """
        record = self._memory_base.previous_life_period.pop(period_id, None)
        if record is None:
            return False
        # Also remove associated events
        for event_id in record.events:
            self._memory_base.previous_events.pop(event_id, None)
        logger.debug(
            f"Life period removed: {period_id} "
            f"(along with {len(record.events)} events)"
        )
        return True

    def update_period_summary(self, period_id: str, new_summary: str) -> bool:
        """
        Update the summary of an existing life period.

        Args:
            period_id: The period identifier.
            new_summary: The new summary text.

        Returns:
            True if the period was found and updated, False otherwise.
        """
        record = self._memory_base.previous_life_period.get(period_id)
        if record is None:
            return False
        record.period_summary = new_summary
        record.period_summary_brief = self._write_pipeline.generate_period_summary_brief(new_summary)
        return True

    # ----------------------------------------------------------------
    # Event Operations
    # ----------------------------------------------------------------

    def add_event(
        self,
        event_id: str,
        resolution_level: str,
        time_period: str,
        summary: str,
        participants: Optional[List[str]] = None,
        languages_in_use: Optional[List[str]] = None,
        interaction_details: Optional[List[Dict[str, Any]]] = None,
        period_id: Optional[str] = None,
    ) -> EventRecord:
        """
        Add or update an event in the memory base.

        If *period_id* is provided, the event ID is also appended to the
        corresponding life period's event list (if not already present).

        Args:
            event_id: Unique event identifier (e.g. 'LP3_E002').
            resolution_level: 'high-resolution', 'low-resolution', or 'outline'.
            time_period: Time span string.
            summary: Narrative summary of the event.
            participants: List of participant IDs.
            languages_in_use: Language codes (default ['eng']).
            interaction_details: List of interaction detail dicts
                (only for high-resolution events).
            period_id: Optional parent period to link this event to.

        Returns:
            The created or updated EventRecord.
        """
        details: List[InteractionDetail] = []
        if interaction_details:
            details = [
                InteractionDetail(**d) if isinstance(d, dict) else d
                for d in interaction_details
            ]

        record = EventRecord(
            resolution_level=resolution_level,
            time_period=time_period,
            summary=summary,
            participants=participants or [],
            languages_in_use=languages_in_use or ["eng"],
            interaction_details=details,
        )
        self._memory_base.previous_events[event_id] = record

        # Link to parent period if specified
        if period_id:
            period = self._memory_base.previous_life_period.get(period_id)
            if period and event_id not in period.events:
                period.events.append(event_id)

        logger.debug(
            f"Event added/updated: {event_id} "
            f"(resolution={resolution_level}, period={period_id})"
        )
        return record

    async def add_event_enriched(
        self,
        event_id: str,
        resolution_level: str,
        time_period: str,
        summary: str,
        participants: Optional[List[str]] = None,
        languages_in_use: Optional[List[str]] = None,
        interaction_details: Optional[List[Dict[str, Any]]] = None,
        period_id: Optional[str] = None,
        event_type: str = "",
        use_llm_summary: bool = False,
        **kwargs,
    ) -> EventRecord:
        """Add an event with write-time enrichment.

        This is the recommended API. Replaces direct add_event() calls.
        The original add_event() is preserved for backward compatibility.
        """
        existing = self.get_event(event_id)
        _ = kwargs

        # 1. Create base record (existing logic, unchanged)
        record = self.add_event(
            event_id=event_id,
            resolution_level=resolution_level,
            time_period=time_period,
            summary=summary,
            participants=participants,
            languages_in_use=languages_in_use,
            interaction_details=interaction_details,
            period_id=period_id,
        )
        object.__setattr__(record, "_period_id", period_id or "")

        # 2. Enrich with pre-computed fields (including embedding)
        record = await self._write_pipeline.enrich_event(
            record,
            event_type=event_type,
            period_theme=self._get_period_theme(period_id),
            existing_events={
                eid: ev
                for eid, ev in self._memory_base.previous_events.items()
                if eid != event_id
            },
            use_llm_summary=use_llm_summary,
            event_id=event_id,
        )

        # 3. Update retriever index
        if existing is not None:
            self._reset_retriever()
        else:
            self._retriever.on_event_added(event_id, record)

        return record

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """
        Retrieve an event by its ID.

        Args:
            event_id: The event identifier.

        Returns:
            The EventRecord if found, else None.
        """
        return self._memory_base.previous_events.get(event_id)

    def get_all_events(self) -> Dict[str, EventRecord]:
        """Return all events as a dict keyed by event_id."""
        return dict(self._memory_base.previous_events)

    def get_events_for_period(self, period_id: str) -> Dict[str, EventRecord]:
        """
        Retrieve all events belonging to a specific life period.

        Args:
            period_id: The period identifier.

        Returns:
            A dict of EventRecord objects keyed by event_id.
        """
        period = self._memory_base.previous_life_period.get(period_id)
        if period is None:
            return {}
        return {
            eid: self._memory_base.previous_events[eid]
            for eid in period.events
            if eid in self._memory_base.previous_events
        }

    def remove_event(self, event_id: str) -> bool:
        """
        Remove an event from the memory base and unlink it from its period.

        Args:
            event_id: The event identifier to remove.

        Returns:
            True if the event was found and removed, False otherwise.
        """
        if event_id not in self._memory_base.previous_events:
            return False
        self._memory_base.previous_events.pop(event_id)
        # Unlink from any period
        for period in self._memory_base.previous_life_period.values():
            if event_id in period.events:
                period.events.remove(event_id)
        logger.debug(f"Event removed: {event_id}")
        return True

    def get_events_by_resolution(
        self, resolution: str
    ) -> Dict[str, EventRecord]:
        """
        Retrieve all events of a given resolution level.

        Args:
            resolution: 'high-resolution' or 'low-resolution'.

        Returns:
            A dict of matching EventRecord objects keyed by event_id.
        """
        return {
            eid: rec
            for eid, rec in self._memory_base.previous_events.items()
            if rec.resolution_level == resolution
        }

    def get_events_by_participant(
        self, participant_id: str
    ) -> Dict[str, EventRecord]:
        """
        Retrieve all events involving a specific participant.

        Args:
            participant_id: The participant identifier.

        Returns:
            A dict of matching EventRecord objects keyed by event_id.
        """
        return {
            eid: rec
            for eid, rec in self._memory_base.previous_events.items()
            if participant_id in rec.participants
        }

    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------

    def _get_period_theme(self, period_id: Optional[str]) -> str:
        """Get the theme/summary of a period for embedding enrichment."""
        if not period_id:
            return ""
        period = self._memory_base.previous_life_period.get(period_id)
        if period:
            return period.period_summary_brief or period.period_summary
        return ""

    def set_run_id(self, run_id: str) -> None:
        """Set a default run_id used by subsequent upsert operations."""
        self._default_run_id = run_id or ""

    def _reset_retriever(self) -> None:
        """Rebuild retriever indices from the current in-memory state."""
        self._retriever = MemoryRetriever(
            self._memory_base,
            embedding_engine=self._embedding_engine,
        )

    @staticmethod
    def get_preferred_summary(event: Optional[EventRecord]) -> str:
        """Return the best available summary for display, retrieval, and prompts."""
        if event is None:
            return ""
        return event.refined_summary or event.summary or event.initial_summary

    def _resolve_summary_fields(
        self,
        existing: Optional[EventRecord],
        summary: str,
        initial_summary: str = "",
        refined_summary: str = "",
        summary_stage: str = "",
        summary_diff_note: str = "",
    ) -> Dict[str, str]:
        """Resolve dual-summary fields while preserving backward compatibility."""
        resolved_initial = (
            initial_summary
            or (existing.initial_summary if existing and existing.initial_summary else "")
            or summary
            or (existing.summary if existing else "")
        )
        resolved_refined = (
            refined_summary
            or (existing.refined_summary if existing and existing.refined_summary else "")
        )
        if resolved_refined:
            resolved_summary = resolved_refined
        else:
            resolved_summary = summary or (existing.summary if existing else "") or resolved_initial

        resolved_stage = summary_stage.strip()
        if not resolved_stage:
            if resolved_refined:
                resolved_stage = "refined"
            elif resolved_initial and resolved_summary == resolved_initial:
                resolved_stage = "initial"
            else:
                resolved_stage = existing.summary_stage if existing else "initial"

        resolved_diff_note = (
            summary_diff_note
            or (existing.summary_diff_note if existing else "")
        )

        return {
            "summary": resolved_summary,
            "initial_summary": resolved_initial,
            "refined_summary": resolved_refined,
            "summary_stage": resolved_stage,
            "summary_diff_note": resolved_diff_note,
        }

    def _backfill_event_summary_fields(self, event: EventRecord) -> None:
        """Backfill dual-summary fields for legacy records loaded from disk."""
        if not event.initial_summary:
            event.initial_summary = event.summary
        if event.refined_summary and not event.summary:
            event.summary = event.refined_summary
        if not event.summary:
            event.summary = event.initial_summary or event.refined_summary
        if not event.summary_stage:
            if event.refined_summary:
                event.summary_stage = "refined"
            elif event.initial_summary and event.summary == event.initial_summary:
                event.summary_stage = "initial"
            else:
                event.summary_stage = "initial"

    def mark_event_storage_status(self, event_id: str, status: str) -> EventRecord:
        """Update only the storage lifecycle status of an existing event."""
        record = self.get_event(event_id)
        if record is None:
            raise KeyError(f"Event '{event_id}' not found")
        record.storage_status = status
        self._memory_base.previous_events[event_id] = record
        return record

    def validate_linked_events(
        self,
        low_res_event_id: str,
        high_res_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate low/high-resolution event linkage and period registration."""
        low_res = self.get_event(low_res_event_id)
        if low_res is None:
            raise ValueError(f"Low-resolution event '{low_res_event_id}' not found")

        high_res = None
        resolved_high_res_event_id = high_res_event_id or low_res.linked_event_id or ""
        if resolved_high_res_event_id:
            high_res = self.get_event(resolved_high_res_event_id)
            if high_res is None:
                raise ValueError(
                    f"High-resolution event '{resolved_high_res_event_id}' not found"
                )
            if low_res.linked_event_id != resolved_high_res_event_id:
                raise ValueError(
                    f"Low-resolution event '{low_res_event_id}' is not linked to "
                    f"'{resolved_high_res_event_id}'"
                )
            if high_res.linked_event_id != low_res_event_id:
                raise ValueError(
                    f"High-resolution event '{resolved_high_res_event_id}' is not linked back to "
                    f"'{low_res_event_id}'"
                )
            if high_res.storage_status == EVENT_STORAGE_STATUS_FINAL and not high_res.interaction_details:
                raise ValueError(
                    f"Final high-resolution event '{resolved_high_res_event_id}' has empty interaction_details"
                )

        period_id = low_res_event_id.split("_")[0] if "_" in low_res_event_id else ""
        period = self.get_life_period(period_id) if period_id else None
        if period is None:
            raise ValueError(f"Life period '{period_id}' not found for event '{low_res_event_id}'")
        if low_res_event_id not in period.events:
            raise ValueError(
                f"Low-resolution event '{low_res_event_id}' is missing from period '{period_id}'"
            )
        if resolved_high_res_event_id and resolved_high_res_event_id not in period.events:
            raise ValueError(
                f"High-resolution event '{resolved_high_res_event_id}' is missing from period '{period_id}'"
            )

        return {
            "period_id": period_id,
            "low_res_event_id": low_res_event_id,
            "high_res_event_id": resolved_high_res_event_id,
            "low_res_status": low_res.storage_status,
            "high_res_status": high_res.storage_status if high_res else "",
        }

    def build_time_unit_commit_report(
        self,
        low_res_event_id: str,
        high_res_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a compact report for a fully committed time unit."""
        validation = self.validate_linked_events(low_res_event_id, high_res_event_id)
        low_res = self.get_event(low_res_event_id)
        high_res = self.get_event(high_res_event_id) if high_res_event_id else None
        return {
            **validation,
            "participants": list(low_res.participants) if low_res else [],
            "high_res_participants": list(high_res.participants) if high_res else [],
            "low_res_summary": self.get_preferred_summary(low_res),
            "high_res_summary": self.get_preferred_summary(high_res),
            "has_final_high_res": bool(
                high_res and high_res.storage_status == EVENT_STORAGE_STATUS_FINAL
            ),
        }

    def _derive_embedding_paths(self, memory_path: str) -> Tuple[str, str, str]:
        """Derive sidecar paths from the main memory JSON path."""
        base = Path(memory_path)
        return (
            str(base.with_name("memory_embeddings.npz")),
            str(base.with_name("embedding_manifest.json")),
            str(base.with_name("embedding_debug.json")),
        )

    def _compute_event_text_checksum(self, event: EventRecord) -> str:
        """Compute a stable checksum for the event text payload used by embeddings."""
        payload = {
            "summary": self.get_preferred_summary(event),
            "time_period": event.time_period,
            "participants": event.participants,
            "languages_in_use": event.languages_in_use,
            "interaction_count": len(event.interaction_details),
            "summary_for_prompt": event.summary_for_prompt,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    def _save_embeddings_sidecar(self, memory_path: str) -> None:
        """Persist embedding vectors to sidecar files next to memory_base.json."""
        embedding_path, manifest_path, debug_path = self._derive_embedding_paths(memory_path)
        self._embedding_store_path = embedding_path
        self._embedding_manifest_path = manifest_path
        self._embedding_debug_path = debug_path

        event_ids: List[str] = []
        embeddings: List[np.ndarray] = []
        manifest_records: Dict[str, Dict[str, Any]] = {}
        debug_records: List[Dict[str, Any]] = []
        updated_at = datetime.now().isoformat()

        for eid, event in self._memory_base.previous_events.items():
            if not event.embedding_vector:
                continue
            vector = np.array(event.embedding_vector, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                continue

            l2_norm = float(np.linalg.norm(vector))
            nonzero_count = int(np.count_nonzero(vector))
            is_all_zero = bool(nonzero_count == 0)
            min_value = float(np.min(vector)) if vector.size else 0.0
            max_value = float(np.max(vector)) if vector.size else 0.0
            mean_value = float(np.mean(vector)) if vector.size else 0.0

            event_ids.append(eid)
            embeddings.append(vector)
            manifest_records[eid] = {
                "text_checksum": self._compute_event_text_checksum(event),
                "updated_at": updated_at,
            }
            debug_records.append({
                "event_id": eid,
                "resolution_level": event.resolution_level,
                "summary": self.get_preferred_summary(event),
                "summary_for_prompt": event.summary_for_prompt,
                "vector_dim": int(vector.size),
                "is_all_zero": is_all_zero,
                "nonzero_count": nonzero_count,
                "l2_norm": l2_norm,
                "min_value": min_value,
                "max_value": max_value,
                "mean_value": mean_value,
                "vector": vector.tolist(),
            })

        if embeddings:
            embedding_matrix = np.vstack(embeddings).astype(np.float32)
            dimension = int(embedding_matrix.shape[1])
        else:
            dimension = int(getattr(self._embedding_engine, "_dim", 0) or 0)
            embedding_matrix = np.zeros((0, dimension), dtype=np.float32)

        np.savez_compressed(
            embedding_path,
            event_ids=np.array(event_ids, dtype=str),
            embeddings=embedding_matrix,
        )

        manifest = {
            "model_name": getattr(self._embedding_engine, "_model_name", ""),
            "dimension": dimension,
            "records": manifest_records,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        debug_payload = {
            "model_name": getattr(self._embedding_engine, "_model_name", ""),
            "dimension": dimension,
            "saved_at": updated_at,
            "record_count": len(debug_records),
            "all_zero_event_ids": [
                record["event_id"]
                for record in debug_records
                if record["is_all_zero"]
            ],
            "records": debug_records,
        }
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=4, ensure_ascii=False)

    def _load_embeddings_sidecar(self, memory_path: str) -> None:
        """Load embedding vectors from sidecar files when present."""
        embedding_path, manifest_path, debug_path = self._derive_embedding_paths(memory_path)
        self._embedding_store_path = embedding_path
        self._embedding_manifest_path = manifest_path
        self._embedding_debug_path = debug_path

        if not os.path.isfile(embedding_path):
            return

        try:
            payload = np.load(embedding_path, allow_pickle=True)
            event_ids = payload["event_ids"] if "event_ids" in payload else None
            embeddings = payload["embeddings"] if "embeddings" in payload else None
            if event_ids is None or embeddings is None:
                return
            ids_list = event_ids.tolist()
            for idx, raw_id in enumerate(ids_list):
                eid = str(raw_id)
                event = self._memory_base.previous_events.get(eid)
                if event is None or idx >= len(embeddings):
                    continue
                event.embedding_vector = embeddings[idx].astype(np.float32).tolist()
        except Exception as e:
            logger.warning(f"Failed to load embedding sidecar '{embedding_path}': {e}")

    def _extract_scene_result_payload(self, scene_result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize different scene_result layouts into a single payload."""
        metadata = scene_result.get("simulation_metadata", {}) or {}
        event_context = scene_result.get("event_context", {}) or {}
        duration = event_context.get("duration", {}) or {}
        interaction_sequence = scene_result.get("interaction_sequence", []) or scene_result.get("interaction_turns", []) or []

        # Prefer the top-level participants field (set by _build_simulation_result)
        # which preserves the full refined participant list.
        # Only fall back to extracting from interaction_sequence if truly empty.
        participants = scene_result.get("participants", []) or []
        if not participants:
            seen: List[str] = []
            for turn in interaction_sequence:
                speaker_id = turn.get("speaker_id", "")
                if speaker_id and speaker_id not in seen:
                    seen.append(speaker_id)
            participants = seen

        period_id = metadata.get("period_id", "")
        event_id = metadata.get("event_id", "")
        if not period_id and event_id and "_" in event_id:
            period_id = event_id.split("_")[0]

        return {
            "metadata": metadata,
            "event_context": event_context,
            "duration": duration,
            "interaction_sequence": interaction_sequence,
            "summary": (
                scene_result.get("refined_summary")
                or event_context.get("refined_summary")
                or event_context.get("summary")
                or scene_result.get("summary", "")
            ),
            "participants": participants,
            "languages_in_use": event_context.get("languages_in_use") or scene_result.get("languages_in_use") or ["eng"],
            "period_id": period_id,
        }

    @staticmethod
    def _format_scene_time_period(payload: Dict[str, Any]) -> str:
        """Build canonical time_period text from a normalized scene payload."""
        duration = payload.get("duration", {}) or {}
        raw = payload.get("event_context", {}) or {}
        start_date = duration.get("start_date") or raw.get("start_date", "")
        end_date = duration.get("end_date") or raw.get("end_date", "")
        start_token = (
            duration.get("precise_start_time")
            or raw.get("precise_start_time")
            or raw.get("start_time_segment")
            or raw.get("start_time")
            or ""
        )
        end_token = (
            duration.get("precise_end_time")
            or raw.get("precise_end_time")
            or raw.get("end_time_segment")
            or raw.get("end_time")
            or ""
        )
        if start_date and end_date and start_token and end_token:
            return f"from {start_date} {start_token} to {end_date} {end_token}"
        if start_date and end_date:
            return f"from {start_date} to {end_date}"
        return ""

    @staticmethod
    def _build_interaction_details_from_scene_result(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert interaction_sequence into EventRecord-compatible detail dicts."""
        details: List[Dict[str, Any]] = []
        for i, turn in enumerate(payload.get("interaction_sequence", []) or []):
            details.append({
                "turn_index": turn.get("turn_index", i),
                "speaker_id": turn.get("speaker_id", ""),
                "action_type": turn.get("action_type", "speak"),
                "content": turn.get("content", turn.get("utterance", "")),
                "internal_thought": turn.get("internal_thought", ""),
            })
        return details

    async def _upsert_event_record(
        self,
        event_id: str,
        resolution_level: str,
        time_period: str,
        summary: str,
        participants: Optional[List[str]] = None,
        languages_in_use: Optional[List[str]] = None,
        interaction_details: Optional[List[Dict[str, Any]]] = None,
        period_id: Optional[str] = None,
        event_type: str = "",
        use_llm_summary: bool = False,
        linked_event_id: str = "",
        source_event_id: str = "",
        storage_status: str = EVENT_STORAGE_STATUS_FINAL,
        run_id: str = "",
        version: Optional[int] = None,
        initial_summary: str = "",
        refined_summary: str = "",
        summary_stage: str = "",
        summary_diff_note: str = "",
    ) -> EventRecord:
        """Upsert an event while preserving versioned metadata."""
        existing = self.get_event(event_id)
        target_version = version if version is not None else ((existing.version + 1) if existing else 1)
        resolved_summaries = self._resolve_summary_fields(
            existing=existing,
            summary=summary,
            initial_summary=initial_summary,
            refined_summary=refined_summary,
            summary_stage=summary_stage,
            summary_diff_note=summary_diff_note,
        )

        record = await self.add_event_enriched(
            event_id=event_id,
            resolution_level=resolution_level,
            time_period=time_period,
            summary=resolved_summaries["summary"],
            participants=participants,
            languages_in_use=languages_in_use,
            interaction_details=interaction_details,
            period_id=period_id,
            event_type=event_type,
            use_llm_summary=use_llm_summary,
        )
        record.initial_summary = resolved_summaries["initial_summary"]
        record.refined_summary = resolved_summaries["refined_summary"]
        record.summary_stage = resolved_summaries["summary_stage"]
        record.summary_diff_note = resolved_summaries["summary_diff_note"]
        record.linked_event_id = linked_event_id or (existing.linked_event_id if existing else "")
        record.source_event_id = source_event_id or (existing.source_event_id if existing else event_id)
        record.storage_status = storage_status or (existing.storage_status if existing else EVENT_STORAGE_STATUS_FINAL)
        record.version = max(target_version, existing.version if existing else 0)
        record.run_id = run_id or (existing.run_id if existing else self._default_run_id)
        self._memory_base.previous_events[event_id] = record
        self._reset_retriever()
        return record

    async def upsert_low_res_event(
        self,
        event_id: str,
        time_period: str,
        summary: str,
        participants: Optional[List[str]] = None,
        languages_in_use: Optional[List[str]] = None,
        interaction_details: Optional[List[Dict[str, Any]]] = None,
        period_id: Optional[str] = None,
        event_type: str = "",
        linked_event_id: str = "",
        source_event_id: str = "",
        storage_status: str = EVENT_STORAGE_STATUS_FINAL,
        run_id: str = "",
        version: Optional[int] = None,
        use_llm_summary: bool = False,
        initial_summary: str = "",
        refined_summary: str = "",
        summary_stage: str = "",
        summary_diff_note: str = "",
    ) -> EventRecord:
        """Upsert a low-resolution event record."""
        return await self._upsert_event_record(
            event_id=event_id,
            resolution_level="low",
            time_period=time_period,
            summary=summary,
            participants=participants,
            languages_in_use=languages_in_use,
            interaction_details=interaction_details,
            period_id=period_id,
            event_type=event_type,
            use_llm_summary=use_llm_summary,
            linked_event_id=linked_event_id,
            source_event_id=source_event_id,
            storage_status=storage_status,
            run_id=run_id,
            version=version,
            initial_summary=initial_summary,
            refined_summary=refined_summary,
            summary_stage=summary_stage,
            summary_diff_note=summary_diff_note,
        )

    async def upsert_high_res_outline_event(
        self,
        event_id: str,
        time_period: str,
        summary: str,
        participants: Optional[List[str]] = None,
        languages_in_use: Optional[List[str]] = None,
        interaction_details: Optional[List[Dict[str, Any]]] = None,
        period_id: Optional[str] = None,
        event_type: str = "",
        linked_event_id: str = "",
        source_event_id: str = "",
        storage_status: str = EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
        run_id: str = "",
        version: Optional[int] = None,
        use_llm_summary: bool = False,
        initial_summary: str = "",
        refined_summary: str = "",
        summary_stage: str = "",
        summary_diff_note: str = "",
    ) -> EventRecord:
        """Upsert a high-resolution outline event record."""
        return await self._upsert_event_record(
            event_id=event_id,
            resolution_level="high",
            time_period=time_period,
            summary=summary,
            participants=participants,
            languages_in_use=languages_in_use,
            interaction_details=interaction_details,
            period_id=period_id,
            event_type=event_type,
            use_llm_summary=use_llm_summary,
            linked_event_id=linked_event_id,
            source_event_id=source_event_id,
            storage_status=storage_status,
            run_id=run_id,
            version=version,
            initial_summary=initial_summary,
            refined_summary=refined_summary,
            summary_stage=summary_stage,
            summary_diff_note=summary_diff_note,
        )

    async def upsert_high_res_event_from_scene_result(
        self,
        event_id: str,
        scene_result: Dict[str, Any],
        period_id: Optional[str] = None,
        event_type: str = "",
        linked_event_id: str = "",
        run_id: str = "",
        source_event_id: str = "",
        version: Optional[int] = None,
    ) -> EventRecord:
        """Upsert the final high-resolution event using the completed P3 scene result."""
        payload = self._extract_scene_result_payload(scene_result)
        existing = self.get_event(event_id)
        resolved_period_id = period_id or payload.get("period_id") or (event_id.split("_")[0] if "_" in event_id else None)
        resolved_summary = payload.get("summary") or (existing.summary if existing else "")
        resolved_participants = payload.get("participants") or (list(existing.participants) if existing else [])
        # If the scene_result provided participants, prefer them over existing
        # (they come from the refined participant list in _build_simulation_result).
        # But if existing has MORE participants (e.g. from refinement), keep the larger set
        # to avoid data loss from speaker_id fallback bugs.
        if existing and existing.participants:
            existing_set = set(existing.participants)
            resolved_set = set(resolved_participants)
            if existing_set - resolved_set:  # existing has participants not in resolved
                # Merge: keep all unique participants from both sources
                merged = list(existing.participants)
                for pid in resolved_participants:
                    if pid not in existing_set:
                        merged.append(pid)
                resolved_participants = merged
        resolved_languages = payload.get("languages_in_use") or (list(existing.languages_in_use) if existing else ["eng"])
        resolved_time_period = self._format_scene_time_period(payload) or (existing.time_period if existing else "")
        interaction_details = self._build_interaction_details_from_scene_result(payload)
        initial_summary = existing.initial_summary if existing and existing.initial_summary else resolved_summary

        return await self._upsert_event_record(
            event_id=event_id,
            resolution_level="high",
            time_period=resolved_time_period,
            summary=resolved_summary,
            participants=resolved_participants,
            languages_in_use=resolved_languages,
            interaction_details=interaction_details,
            period_id=resolved_period_id,
            event_type=event_type,
            use_llm_summary=False,
            linked_event_id=linked_event_id or (existing.linked_event_id if existing else ""),
            source_event_id=source_event_id or (existing.source_event_id if existing else event_id),
            storage_status=EVENT_STORAGE_STATUS_FINAL,
            run_id=run_id or (existing.run_id if existing else self._default_run_id),
            version=version,
            initial_summary=initial_summary,
            refined_summary=resolved_summary,
            summary_stage="refined",
        )

    async def update_low_res_refined_summary(
        self,
        event_id: str,
        refined_summary: str,
        run_id: str = "",
        version: Optional[int] = None,
        summary_diff_note: str = "",
    ) -> EventRecord:
        """Update only the refined summary payload of an existing low-resolution event."""
        existing = self.get_event(event_id)
        if existing is None:
            raise KeyError(f"Low-resolution event '{event_id}' not found")
        if existing.resolution_level != "low":
            raise ValueError(f"Event '{event_id}' is not a low-resolution event")

        target_summary = refined_summary or existing.summary or existing.initial_summary
        target_stage = "refined" if refined_summary else "final_same_as_initial"
        return await self._upsert_event_record(
            event_id=event_id,
            resolution_level=existing.resolution_level,
            time_period=existing.time_period,
            summary=target_summary,
            participants=list(existing.participants),
            languages_in_use=list(existing.languages_in_use),
            interaction_details=[detail.model_dump() for detail in existing.interaction_details],
            period_id=getattr(existing, "_period_id", None) or (event_id.split("_")[0] if "_" in event_id else None),
            event_type="",
            use_llm_summary=False,
            linked_event_id=existing.linked_event_id,
            source_event_id=existing.source_event_id or event_id,
            storage_status=existing.storage_status,
            run_id=run_id or existing.run_id or self._default_run_id,
            version=version,
            initial_summary=existing.initial_summary or existing.summary,
            refined_summary=refined_summary,
            summary_stage=target_stage,
            summary_diff_note=summary_diff_note or existing.summary_diff_note,
        )

    # ----------------------------------------------------------------
    # Current Life Summary
    # ----------------------------------------------------------------

    def get_current_life_summary(self) -> str:
        """Return the current life summary string."""
        return self._memory_base.current_life_summary

    def update_current_life_summary(self, summary: str) -> None:
        """
        Update the current life summary.

        Args:
            summary: The new summary text.
        """
        self._memory_base.current_life_summary = summary
        logger.debug("Current life summary updated")

    # ----------------------------------------------------------------
    # Bulk Import from Plan
    # ----------------------------------------------------------------

    def import_life_periods_from_plan(
        self, plan_dict: Dict[str, Any]
    ) -> int:
        """
        Bulk-import life periods from a life-period plan dict.

        Reads the ``life_periods`` list from the plan and creates
        LifePeriodRecord entries in the memory base.  Events are not
        imported (they are generated later by the simulation pipeline).

        Args:
            plan_dict: The full plan dict (as produced by LifePeriodPlanner).

        Returns:
            The number of periods imported.
        """
        periods = plan_dict.get("life_periods", [])
        count = 0
        for p in periods:
            pid = p.get("period_id", "")
            dr = p.get("period_date_range", {})
            start = dr.get("start_date", "")
            end = dr.get("end_date", "")
            time_period = f"{start} to {end}" if start and end else ""

            # Build a summary from available fields
            title = p.get("title", "")
            theme = p.get("dominant_theme", "")
            summary = f"{title}. {theme}" if theme else title

            self.add_life_period(
                period_id=pid,
                time_period=time_period,
                period_summary=summary,
            )
            count += 1

        logger.info(f"Imported {count} life periods from plan")
        return count

    # ----------------------------------------------------------------
    # Persistence — Load / Save
    # ----------------------------------------------------------------

    def load(self, file_path: str) -> None:
        """
        Load the memory base from a JSON file.

        Args:
            file_path: Path to the JSON file.
        """
        tracker = self._performance_tracker
        if tracker:
            with tracker.track("memory.load", file_path=file_path):
                self._load_impl(file_path)
            return
        self._load_impl(file_path)

    def _load_impl(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse previous_life_period
        raw_periods = data.get("previous_life_period", {})
        parsed_periods: Dict[str, LifePeriodRecord] = {}
        for pid, pdata in raw_periods.items():
            if isinstance(pdata, dict):
                parsed_periods[pid] = LifePeriodRecord(**pdata)

        # Parse previous_events
        raw_events = data.get("previous_events", {})
        parsed_events: Dict[str, EventRecord] = {}
        for eid, edata in raw_events.items():
            if isinstance(edata, dict):
                parsed = EventRecord(**edata)
                self._backfill_event_summary_fields(parsed)
                parsed_events[eid] = parsed

        self._memory_base = MemoryBase(
            previous_life_period=parsed_periods,
            previous_events=parsed_events,
            current_life_summary=data.get("current_life_summary", ""),
        )
        self._file_path = file_path
        self._load_embeddings_sidecar(file_path)

        # 🆕 v5: Lazy Enrichment — backfill empty pre-computed fields for old data
        self._lazy_enrich_on_load()

        # 🆕 v6: After loading, rebuild retriever indices (with embedding engine)
        self._reset_retriever()

        logger.info(
            f"Memory base loaded from {file_path}: "
            f"{len(parsed_periods)} periods, {len(parsed_events)} events"
        )

    def _lazy_enrich_on_load(self):
        """Backfill empty pre-computed fields for old data loaded from JSON.

        This runs ONCE at load time, not on every retrieval.
        After backfill, all events have complete pre-computed fields,
        so the retrieval path never needs fallback logic.

        v2: Also backfills embedding_vector for events that don't have it.
        """
        wp = self._write_pipeline
        enriched_count = 0

        for eid, ev in self._memory_base.previous_events.items():
            needs_enrich = False

            self._backfill_event_summary_fields(ev)
            preferred_summary = self.get_preferred_summary(ev)

            if not ev.summary_for_context:
                ev.summary_for_context = wp.generate_context_summary(preferred_summary, ev.participants)
                needs_enrich = True

            if not ev.summary_for_prompt:
                ev.summary_for_prompt = wp.generate_prompt_summary(preferred_summary, ev.participants)
                needs_enrich = True

            if not ev.summary_oneliner:
                ev.summary_oneliner = wp.generate_oneliner(preferred_summary)
                needs_enrich = True

            if not ev.event_date:
                ev.event_date = wp.normalize_event_date(ev.time_period)
                needs_enrich = True

            if ev.importance_score == 0.5:  # Default value, likely not computed
                ev.importance_score = wp.compute_importance(
                    summary=preferred_summary,
                    resolution_level=ev.resolution_level,
                    participants=ev.participants,
                )
                needs_enrich = True

            # 🆕 v2: Backfill embedding_vector (also handles all-zero vectors
            #   from previous runs where the model failed to load)
            if self._embedding_engine and self._embedding_engine.available:
                _needs_embedding = (
                    not ev.embedding_vector
                    or all(v == 0.0 for v in ev.embedding_vector)
                )
                if _needs_embedding:
                    ev.embedding_vector = wp.compute_embedding(preferred_summary)
                    needs_enrich = True

            if needs_enrich:
                enriched_count += 1

        # Also backfill period_summary_brief
        for pid, period in self._memory_base.previous_life_period.items():
            if not period.period_summary_brief:
                period.period_summary_brief = wp.generate_period_summary_brief(period.period_summary)

        if enriched_count > 0:
            logger.info(f"Lazy enrichment: backfilled {enriched_count} events with pre-computed fields")

    def save(self, file_path: Optional[str] = None) -> str:
        """
        Save the memory base to a JSON file.

        Args:
            file_path: Target path.  If *None*, uses the path from which
                the memory base was loaded (or raises ValueError).

        Returns:
            The path the file was saved to.
        """
        path = file_path or self._file_path
        if not path:
            raise ValueError(
                "No file path specified and no default path available. "
                "Pass file_path explicitly or initialise with memory_base_path."
            )
        tracker = self._performance_tracker
        if tracker:
            with tracker.track("memory.save", file_path=path, period_count=self.period_count, event_count=self.event_count):
                return self._save_impl(path)
        return self._save_impl(path)

    def _save_impl(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        self._save_embeddings_sidecar(path)
        self._file_path = path
        logger.info(f"Memory base saved to {path}")
        return path

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the entire memory base to a plain dict.

        Returns:
            A dict matching the canonical JSON structure.
        """
        return {
            "previous_life_period": {
                pid: rec.model_dump()
                for pid, rec in self._memory_base.previous_life_period.items()
            },
            "previous_events": {
                eid: rec.model_dump()
                for eid, rec in self._memory_base.previous_events.items()
            },
            "current_life_summary": self._memory_base.current_life_summary,
        }

    # ----------------------------------------------------------------
    # Legacy MemoryEntry Operations (backward compatibility)
    # ----------------------------------------------------------------

    def add_memory(self, memory: MemoryEntry) -> None:
        """
        Add a single legacy memory entry to the store.

        Args:
            memory: The MemoryEntry to store.
        """
        self._memories.append(memory)
        self._index[memory.memory_id] = memory

    def add_memories(self, memories: List[MemoryEntry]) -> None:
        """
        Add multiple legacy memory entries to the store.

        Args:
            memories: A list of MemoryEntry objects to store.
        """
        for m in memories:
            self.add_memory(m)

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific legacy memory by ID.

        Args:
            memory_id: The unique memory identifier.

        Returns:
            The MemoryEntry if found, else None.
        """
        return self._index.get(memory_id)

    def create_memory_from_scene(
        self,
        event_id: str,
        content: str,
        timestamp: str,
        salience: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """
        Create and store a new legacy memory entry from scene output.

        Args:
            event_id: The source event ID.
            content: Natural-language memory content.
            timestamp: ISO datetime string.
            salience: Initial salience score (0-1).
            tags: Optional semantic tags.

        Returns:
            The newly created MemoryEntry.
        """
        memory = MemoryEntry(
            memory_id=f"MEM_{uuid.uuid4().hex[:8]}",
            source_event_id=event_id,
            timestamp=timestamp,
            content=content,
            salience=salience,
            tags=tags or [],
        )
        self.add_memory(memory)
        return memory

    def extract_legacy_memories_from_scene_result(
        self,
        scene_result: Dict[str, Any],
        event_id: str,
    ) -> List[MemoryEntry]:
        """Extract only legacy MemoryEntry objects from a completed scene result."""
        if not scene_result:
            return []

        payload = self._extract_scene_result_payload(scene_result)
        summary = payload.get("summary", "")
        if not summary:
            return []

        duration = payload.get("duration", {}) or {}
        timestamp = (
            payload.get("metadata", {}).get("simulation_timestamp")
            or duration.get("start_date")
            or datetime.now().isoformat()
        )
        legacy_memory = self.create_memory_from_scene(
            event_id=event_id,
            content=summary,
            timestamp=timestamp,
            salience=scene_result.get("importance_score", 0.5),
            tags=scene_result.get("tags", []),
        )
        return [legacy_memory]

    def extract_memories_from_scene_result(
        self,
        scene_result: Dict[str, Any],
        event_id: str,
    ) -> List[MemoryEntry]:
        """Backward-compatible alias that now only extracts legacy memories."""
        return self.extract_legacy_memories_from_scene_result(
            scene_result=scene_result,
            event_id=event_id,
        )

    def get_recent_memories(self, n: int = 10) -> List[MemoryEntry]:
        """
        Return the N most recent legacy memories.

        Args:
            n: Number of recent memories to return.

        Returns:
            A list of the most recent MemoryEntry objects.
        """
        return self._memories[-n:] if self._memories else []

    def get_salient_memories(self, threshold: float = 0.7) -> List[MemoryEntry]:
        """
        Return legacy memories with salience above the threshold.

        Args:
            threshold: Minimum salience score.

        Returns:
            A list of high-salience MemoryEntry objects.
        """
        return [m for m in self._memories if m.salience >= threshold]

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all legacy memories to a list of dicts for persistence."""
        return [m.model_dump() for m in self._memories]

    def export_memory_forge_format(self) -> dict:
        """Export memory base in the paper's M_π = (L, G, E) three-layer format.

        Returns:
            dict with keys "L", "G", "E", "metadata" where:
              - L: Lifetime Period Summaries (first-person retrospective paragraphs)
              - G: General-Event Memories (structured outlines with first-person memory)
              - E: Event-Specific Experiences (full multi-turn interaction transcripts)
        """
        L = []  # Lifetime Period Summaries
        for period_id, period in sorted(self._memory_base.previous_life_period.items()):
            L.append({
                "period_id": period_id,
                "time_period": period.time_period,
                "summary": period.period_summary,
                "first_person_memory": getattr(period, "period_summary_first_person", ""),
            })

        G = []  # General-Event Memories (formerly "outline")
        E = []  # Event-Specific Experiences (high-resolution)

        for event_id, event in sorted(self._memory_base.previous_events.items()):
            base_entry = {
                "event_id": event_id,
                "period_id": event_id.split("_E")[0] if "_E" in event_id else "",
                "time_period": event.time_period,
                "summary": event.summary,
                "first_person_memory": getattr(event, "first_person_memory", ""),
                "participants": event.participants,
            }
            if event.resolution_level == "medium":
                base_entry["title"] = getattr(event, "general_event_title", "") or event.summary.split("[")[0].strip()
                base_entry["frequency"] = getattr(event, "general_event_frequency", "")
                G.append(base_entry)
            elif event.resolution_level == "high":
                base_entry["interaction_turns"] = [
                    turn.model_dump() for turn in event.interaction_details
                ]
                E.append(base_entry)
            # low-resolution events contribute to L via period summaries; skip here

        return {
            "L": L,
            "G": G,
            "E": E,
            "metadata": {
                "total_periods": len(L),
                "total_general_events": len(G),
                "total_specific_experiences": len(E),
            }
        }

    def clear(self) -> None:
        """Clear all stored data (both structured and legacy)."""
        self._memory_base = MemoryBase()
        self._memories.clear()
        self._index.clear()
        self._embedding_store_path = None
        self._embedding_manifest_path = None
        self._embedding_debug_path = None
        logger.debug("Memory manager cleared")
