"""
Memory Retriever — STARE-E v2: Structured Task-Adaptive Retrieval Engine with Embedding.

Design philosophy (v6 / STARE-E v2):
  - ALL operations are pure computation, ZERO LLM API calls
  - ZERO text processing in retrieval path (no smart_truncate, no [:N])
  - All text fields are pre-computed at write time
  - Five retrieval dimensions: recency × importance × semantic × participant × causal
  - Semantic relevance via local embedding model (zero API cost)
  - Task-adaptive weight configuration via TaskConfig
  - Backward compatible with v1 two-dimensional retrieval

Theoretical basis:
  - Autobiographical Memory theory (Conway, 2000): importance-priority retrieval
  - Ebbinghaus forgetting curve: exponential decay for recency
  - Social Network Analysis (Newman, 2004): co-occurrence as relationship proxy
  - Generative Agents (Park et al., 2023): multi-dimensional weighted retrieval
  - Multilingual E5 (Wang et al., 2024): asymmetric query/passage embedding
"""

import math
import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

from lifelong_synth.simulation_p4_memory_organiser.definition import (
    EventRecord, MemoryBase
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Module 5: DART — Dynamic Adaptive Retrieval for Trajectory Synthesis
# ═══════════════════════════════════════════════════════════════

class TaskComplexityLevel:
    """Task complexity classification for DART retrieval."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


def classify_task_complexity(
    event_type: str = "",
    has_key_event: str = "no",  # "no" | "key_event_detail" | "key_event_outline"
    period_transition: bool = False,
    participant_count: int = 0,
) -> str:
    """Rule-based task complexity classification (zero LLM cost).

    Inspired by Adaptive-RAG (Jeong et al., NAACL 2024).
    """
    if has_key_event != "no" or period_transition:
        return TaskComplexityLevel.COMPLEX
    if participant_count > 3 or event_type in ("career", "milestone", "academic"):
        return TaskComplexityLevel.MODERATE
    return TaskComplexityLevel.SIMPLE


DART_WEIGHT_PROFILES = {
    TaskComplexityLevel.SIMPLE: {
        "w_recency": 0.40, "w_importance": 0.15, "w_semantic": 0.20,
        "w_participant": 0.15, "w_causal": 0.10,
        "top_k": 4, "summary_field": "summary_oneliner",
    },
    TaskComplexityLevel.MODERATE: {
        "w_recency": 0.20, "w_importance": 0.25, "w_semantic": 0.30,
        "w_participant": 0.10, "w_causal": 0.15,
        "top_k": 6, "summary_field": "summary_for_context",
    },
    TaskComplexityLevel.COMPLEX: {
        "w_recency": 0.10, "w_importance": 0.20, "w_semantic": 0.25,
        "w_participant": 0.10, "w_causal": 0.35,
        "top_k": 10, "summary_field": "summary_for_prompt",
    },
}


# ═══════════════════════════════════════════════════════════════
# Task-Adaptive Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskConfig:
    """Configuration for task-adaptive retrieval. v2: 5 dimensions."""

    # Signal weights (must sum to 1.0)
    w_recency: float          # Signal 1: temporal recency
    w_importance: float       # Signal 2: event importance
    w_semantic: float         # Signal 3: semantic relevance (embedding)
    w_participant: float      # Signal 4: participant overlap
    w_causal: float           # Signal 5: causal proximity

    # Retrieval parameters
    top_k: int
    context_mode: str         # "general" | "period_focused"
    summary_field: str        # which pre-computed summary field to use


TASK_CONFIGS_V2 = {
    # ── P2 Step 1a: low-resolution event framework ──
    "p2_low_res_framework": TaskConfig(
        w_recency=0.20, w_importance=0.20, w_semantic=0.35,
        w_participant=0.05, w_causal=0.20,
        top_k=8, context_mode="general",
        summary_field="summary_for_context",
    ),
    # ── P2 Step 1b: high-resolution key event outline ──
    "p2_high_res_outline": TaskConfig(
        w_recency=0.10, w_importance=0.15, w_semantic=0.30,
        w_participant=0.10, w_causal=0.35,
        top_k=6, context_mode="period_focused",
        summary_field="summary_for_prompt",
    ),
    # ── P2 Period end: life summary generation ──
    "p2_life_summary": TaskConfig(
        w_recency=0.15, w_importance=0.40, w_semantic=0.20,
        w_participant=0.05, w_causal=0.20,
        top_k=10, context_mode="general",
        summary_field="summary_for_context",
    ),
    # ── P2 Period end: period summary rewrite ──
    "p2_period_summary_rewrite": TaskConfig(
        w_recency=0.10, w_importance=0.25, w_semantic=0.25,
        w_participant=0.05, w_causal=0.35,
        top_k=8, context_mode="period_focused",
        summary_field="summary_for_context",
    ),
    # ── P3: scene planning ──
    "p3_scene_planning": TaskConfig(
        w_recency=0.10, w_importance=0.10, w_semantic=0.25,
        w_participant=0.30, w_causal=0.25,
        top_k=6, context_mode="period_focused",
        summary_field="summary_for_prompt",
    ),
    # ── P3: character style generation ──
    "p3_character_style": TaskConfig(
        w_recency=0.10, w_importance=0.10, w_semantic=0.15,
        w_participant=0.50, w_causal=0.15,
        top_k=5, context_mode="general",
        summary_field="summary_for_context",
    ),
}


# ═══════════════════════════════════════════════════════════════
# Memory Retriever — STARE-E v2
# ═══════════════════════════════════════════════════════════════

class MemoryRetriever:
    """STARE-E v2: Zero-LLM, embedding-enhanced retrieval engine."""

    def __init__(self, memory_base: MemoryBase, embedding_engine=None):
        self._mb = memory_base
        self._emb = embedding_engine  # EmbeddingEngine or None

        # Existing indices
        self._participant_index: Dict[str, List[str]] = {}
        self._co_occurrence: Dict[Tuple[str, str], int] = {}

        # 🆕 STARE-E v2 indices
        self._embedding_matrix: Optional[np.ndarray] = None  # (N, dim)
        self._embedding_event_ids: List[str] = []
        self._causal_graph: Dict[str, List[str]] = {}  # event_id → parent_ids

        self._rebuild_indices()

    # ── Index Management ────────────────────────────────────────

    def _rebuild_indices(self):
        """Rebuild all indices from memory base."""
        self._participant_index.clear()
        self._co_occurrence.clear()
        self._causal_graph.clear()

        embeddings = []
        event_ids = []

        for eid, ev in self._mb.previous_events.items():
            self._index_event(eid, ev)

            # Build embedding matrix
            if ev.embedding_vector:
                embeddings.append(ev.embedding_vector)
                event_ids.append(eid)

            # Build causal graph
            if ev.causal_parent_ids:
                self._causal_graph[eid] = ev.causal_parent_ids

        if embeddings:
            self._embedding_matrix = np.array(embeddings, dtype=np.float32)
            self._embedding_event_ids = event_ids
        else:
            self._embedding_matrix = None
            self._embedding_event_ids = []

    def _index_event(self, event_id: str, event: EventRecord):
        """Index a single event for participant and co-occurrence lookup."""
        for pid in event.participants:
            self._participant_index.setdefault(pid, []).append(event_id)
        pids = event.participants
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                key = tuple(sorted([pids[i], pids[j]]))
                self._co_occurrence[key] = self._co_occurrence.get(key, 0) + 1

    def on_event_added(self, event_id: str, event: EventRecord):
        """Incrementally update indices when a new event is added."""
        self._index_event(event_id, event)

        # Update embedding matrix
        if event.embedding_vector:
            new_emb = np.array([event.embedding_vector], dtype=np.float32)
            if self._embedding_matrix is not None:
                self._embedding_matrix = np.vstack(
                    [self._embedding_matrix, new_emb]
                )
            else:
                self._embedding_matrix = new_emb
            self._embedding_event_ids.append(event_id)

        # Update causal graph
        if event.causal_parent_ids:
            self._causal_graph[event_id] = event.causal_parent_ids

    # ── Signal Computation ──────────────────────────────────────

    def _compute_all_semantic_scores(
        self, query_embedding: np.ndarray
    ) -> Dict[str, float]:
        """Compute semantic relevance scores for ALL events via matrix multiply.

        Uses vectorized dot product for O(1) batch computation.
        For 200 events × 768-dim: ~0.1ms.
        """
        if self._embedding_matrix is None or len(self._embedding_event_ids) == 0:
            return {}

        similarities = self._embedding_matrix @ query_embedding
        scores = {}
        for i, eid in enumerate(self._embedding_event_ids):
            # Normalize cosine sim from [-1, 1] to [0, 1]
            scores[eid] = (float(similarities[i]) + 1.0) / 2.0
        return scores

    def _compute_participant_overlap(
        self, event: EventRecord, task_participant_ids: List[str]
    ) -> float:
        """Compute participant overlap. Identical to v1 STARE."""
        if not task_participant_ids or not event.participants:
            return 0.5  # neutral

        event_pids = set(event.participants)
        task_pids = set(task_participant_ids)

        overlap = event_pids & task_pids
        if not overlap:
            max_co_occurrence = 0
            for ep in event_pids:
                for tp in task_pids:
                    key = tuple(sorted([ep, tp]))
                    co = self._co_occurrence.get(key, 0)
                    max_co_occurrence = max(max_co_occurrence, co)
            return min(max_co_occurrence / 10.0, 0.5) * 0.5

        return len(overlap) / len(task_pids)

    def _compute_causal_proximity(
        self, event_id: str, causal_context_ids: List[str]
    ) -> float:
        """Compute causal proximity score.

        Higher score if the event is causally connected to context events.
        """
        if not causal_context_ids:
            return 0.5  # neutral

        # Check if event is a direct causal parent of any context event
        for ctx_id in causal_context_ids:
            parents = self._causal_graph.get(ctx_id, [])
            if event_id in parents:
                return 1.0  # direct causal parent

        # Check if event shares causal parents with context events
        event_parents = set(self._causal_graph.get(event_id, []))
        if event_parents:
            for ctx_id in causal_context_ids:
                ctx_parents = set(self._causal_graph.get(ctx_id, []))
                if event_parents & ctx_parents:
                    return 0.7  # shared causal ancestry

        # Check if event is in the same period as context events
        event_period = event_id.split("_")[0] if "_" in event_id else ""
        for ctx_id in causal_context_ids:
            ctx_period = ctx_id.split("_")[0] if "_" in ctx_id else ""
            if event_period and event_period == ctx_period:
                return 0.6  # same period

        return 0.3  # no causal connection

    def _build_query_text(
        self,
        task_type: str,
        topic_hints: Optional[List[str]] = None,
        keyword_hints: Optional[Set[str]] = None,
        period_theme: str = "",
        event_type: str = "",
    ) -> str:
        """Build query text for embedding from structured task hints."""
        parts = []

        if event_type:
            parts.append(f"Event type: {event_type}")
        if period_theme:
            parts.append(f"Period theme: {period_theme}")
        if topic_hints:
            parts.append(f"Related topics: {', '.join(topic_hints)}")
        if keyword_hints:
            parts.append(f"Keywords: {', '.join(keyword_hints)}")

        if not parts:
            task_descriptions = {
                "p2_low_res_framework": "Generate low-resolution event framework, need macro-narrative coherence",
                "p2_high_res_outline": "Generate high-resolution key event outline, need causal chain and emotional details",
                "p2_life_summary": "Generate current life summary, need global perspective covering important events",
                "p2_period_summary_rewrite": "Rewrite period summary, need to integrate plan with actual",
                "p3_scene_planning": "Plan scene details, need history highly relevant to current event",
                "p3_character_style": "Generate character style, need specific character's historical interactions",
            }
            parts.append(task_descriptions.get(task_type, "Retrieve relevant historical events"))

        return ". ".join(parts)

    def _get_period_theme(self, period_id: str) -> str:
        """Get the theme/summary of a period for query construction."""
        period = self._mb.previous_life_period.get(period_id)
        if period:
            return period.period_summary_brief or period.period_summary
        return ""

    # ── Core: STARE-E Five-Dimensional Retrieval ────────────────

    def retrieve_stare_e(
        self,
        task_type: str,
        current_date: Optional[date] = None,
        period_id: str = "",
        topic_hints: Optional[List[str]] = None,
        participant_hints: Optional[List[str]] = None,
        causal_context_ids: Optional[List[str]] = None,
        keyword_hints: Optional[Set[str]] = None,
        exclude_ids: Optional[set] = None,
    ) -> List[Tuple[str, EventRecord, float]]:
        """STARE-E: Structured Task-Adaptive Retrieval Engine with Embedding.

        Five-dimensional weighted scoring with local embedding semantic relevance.
        Zero LLM API cost. Embedding computed locally.
        """
        config = TASK_CONFIGS_V2.get(task_type)
        if config is None:
            return self.retrieve_weighted(current_date=current_date, top_k=8)

        events = self._mb.previous_events
        if not events:
            return []

        exclude = exclude_ids or set()

        # Step 1: Compute query embedding (one-time, ~30ms on CPU)
        semantic_scores: Dict[str, float] = {}
        if self._emb and self._emb.available and config.w_semantic > 0:
            query_text = self._build_query_text(
                task_type=task_type,
                topic_hints=topic_hints,
                keyword_hints=keyword_hints,
                period_theme=self._get_period_theme(period_id),
            )
            query_emb = self._emb.encode_query(query_text)
            # Step 2: Batch compute semantic scores (vectorized, ~0.1ms)
            semantic_scores = self._compute_all_semantic_scores(query_emb)

        # Step 3: Score each event
        scored = []
        _participant_hints = participant_hints or []
        _causal_context_ids = causal_context_ids or []

        for eid, ev in events.items():
            if eid in exclude:
                continue

            # Signal 1: Recency
            if current_date and ev.event_date:
                try:
                    ev_date = date.fromisoformat(ev.event_date)
                    days_ago = max((current_date - ev_date).days, 0)
                except (ValueError, TypeError):
                    days_ago = 365
            else:
                days_ago = 365
            s_recency = math.exp(-0.005 * days_ago)

            # Signal 2: Importance
            s_importance = ev.importance_score

            # Signal 3: Semantic Relevance
            s_semantic = semantic_scores.get(eid, 0.5)

            # Signal 4: Participant Overlap
            s_participant = self._compute_participant_overlap(
                ev, _participant_hints
            )

            # Signal 5: Causal Proximity
            s_causal = self._compute_causal_proximity(
                eid, _causal_context_ids
            )

            # Weighted fusion
            score = (
                config.w_recency * s_recency
                + config.w_importance * s_importance
                + config.w_semantic * s_semantic
                + config.w_participant * s_participant
                + config.w_causal * s_causal
            )

            scored.append((eid, ev, round(score, 4)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[: config.top_k]

    # ── Legacy: Two-Dimensional Weighted Retrieval ──────────────

    def retrieve_weighted(
        self,
        current_date: Optional[date] = None,
        top_k: int = 8,
        alpha: float = 0.35,
        beta: float = 0.65,
        exclude_ids: Optional[set] = None,
    ) -> List[Tuple[str, EventRecord, float]]:
        """Retrieve top-k events using two-dimensional weighted scoring.

        score = alpha × recency + beta × importance
        Backward compatible with v1.
        """
        events = self._mb.previous_events
        if not events:
            return []

        exclude = exclude_ids or set()
        scored = []

        for eid, ev in events.items():
            if eid in exclude:
                continue

            if current_date and ev.event_date:
                try:
                    ev_date = date.fromisoformat(ev.event_date)
                    days_ago = max((current_date - ev_date).days, 0)
                except (ValueError, TypeError):
                    days_ago = 365
            else:
                days_ago = 365
            recency = math.exp(-0.005 * days_ago)
            importance = ev.importance_score

            score = alpha * recency + beta * importance
            scored.append((eid, ev, round(score, 4)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    # ── Period-Based Retrieval ──────────────────────────────────

    def retrieve_by_period(
        self, period_id: str, top_k: int = 10,
    ) -> List[Tuple[str, EventRecord]]:
        """Retrieve events for a specific period, sorted by importance."""
        period = self._mb.previous_life_period.get(period_id)
        if not period:
            return []
        events = []
        for eid in period.events:
            ev = self._mb.previous_events.get(eid)
            if ev:
                events.append((eid, ev))
        events.sort(key=lambda x: x[1].importance_score, reverse=True)
        return events[:top_k]

    # ── Participant-Based Retrieval ─────────────────────────────

    def retrieve_by_participant(
        self, participant_id: str, top_k: int = 5,
    ) -> List[Tuple[str, EventRecord]]:
        """Retrieve events involving a participant, sorted by importance."""
        event_ids = self._participant_index.get(participant_id, [])
        events = []
        for eid in event_ids:
            ev = self._mb.previous_events.get(eid)
            if ev:
                events.append((eid, ev))
        events.sort(key=lambda x: x[1].importance_score, reverse=True)
        return events[:top_k]

    def get_top_connections(
        self, pid: str, top_k: int = 5
    ) -> List[Tuple[str, int]]:
        """Get most frequently co-occurring participants."""
        connections = {}
        for (p1, p2), count in self._co_occurrence.items():
            if p1 == pid:
                connections[p2] = count
            elif p2 == pid:
                connections[p1] = count
        return sorted(connections.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

    # ── Full Event Access ───────────────────────────────────────

    def get_full_event(self, event_id: str) -> Optional[EventRecord]:
        """Get the complete EventRecord for on-demand full detail access."""
        return self._mb.previous_events.get(event_id)

    # ── Context Building (Main API for P2/P3) ───────────────────

    def build_context(
        self,
        task_type: str = "",
        mode: str = "general",
        current_date: Optional[date] = None,
        period_id: str = "",
        max_events: int = 8,
        topic_hints: Optional[List[str]] = None,
        participant_hints: Optional[List[str]] = None,
        causal_context_ids: Optional[List[str]] = None,
        keyword_hints: Optional[Set[str]] = None,
    ) -> str:
        """Build formatted memory context for LLM prompts.

        Main API for P2/P3. Supports both STARE-E v2 (task_type) and
        legacy v1 (mode) retrieval.

        If task_type is provided and recognized, uses STARE-E v2.
        Otherwise falls back to legacy two-dimensional retrieval.
        """
        # Try STARE-E v2 if task_type is provided
        config = TASK_CONFIGS_V2.get(task_type) if task_type else None

        if config is not None:
            return self._build_context_stare_e(
                task_type=task_type,
                config=config,
                current_date=current_date,
                period_id=period_id,
                topic_hints=topic_hints,
                participant_hints=participant_hints,
                causal_context_ids=causal_context_ids,
                keyword_hints=keyword_hints,
            )

        # Fallback to legacy mode
        return self._build_context_legacy(
            mode=mode,
            current_date=current_date,
            period_id=period_id,
            max_events=max_events,
        )

    def _build_context_stare_e(
        self,
        task_type: str,
        config: TaskConfig,
        current_date: Optional[date] = None,
        period_id: str = "",
        topic_hints: Optional[List[str]] = None,
        participant_hints: Optional[List[str]] = None,
        causal_context_ids: Optional[List[str]] = None,
        keyword_hints: Optional[Set[str]] = None,
    ) -> str:
        """Build context using STARE-E v2 retrieval."""
        parts = []

        # 1. Core Memory: current_life_summary
        summary = self._mb.current_life_summary
        if summary:
            parts.append(f"Current life summary: {summary}")

        # 2. Period structure overview
        if self._mb.previous_life_period:
            parts.append("Life periods:")
            for pid, rec in self._mb.previous_life_period.items():
                brief = rec.period_summary_brief or rec.period_summary
                parts.append(f"  - {pid}: {rec.time_period} — {brief}")

        # 3. STARE-E retrieval
        events = self.retrieve_stare_e(
            task_type=task_type,
            current_date=current_date,
            period_id=period_id,
            topic_hints=topic_hints,
            participant_hints=participant_hints,
            causal_context_ids=causal_context_ids,
            keyword_hints=keyword_hints,
        )

        if events:
            summary_field = config.summary_field

            if config.context_mode == "period_focused" and period_id:
                current_period_events = [
                    (eid, ev, s)
                    for eid, ev, s in events
                    if eid.startswith(period_id)
                ]
                cross_period_events = [
                    (eid, ev, s)
                    for eid, ev, s in events
                    if not eid.startswith(period_id)
                ]

                if current_period_events:
                    parts.append(f"Current period ({period_id}) related events:")
                    for eid, ev, score in current_period_events:
                        text = getattr(
                            ev, summary_field, ev.summary_for_context
                        )
                        parts.append(f"  [{eid}] {text}")

                if cross_period_events:
                    parts.append("Cross-period related events:")
                    for eid, ev, score in cross_period_events:
                        text = getattr(
                            ev, summary_field, ev.summary_for_context
                        )
                        parts.append(f"  [{eid}] {text}")
            else:
                parts.append("Related historical events:")
                for eid, ev, score in events:
                    text = getattr(
                        ev, summary_field, ev.summary_for_context
                    )
                    parts.append(f"  [{eid}] {text}")

        return "\n".join(parts) if parts else "Memory bank is empty (life just begun)"

    # ── Module 5: DART — Dynamic Adaptive Retrieval ─────────────

    def build_context_dart(
        self,
        task_complexity: str,
        task_type: str = "",
        current_date: Optional[date] = None,
        period_id: str = "",
        topic_hints: Optional[List[str]] = None,
        participant_hints: Optional[List[str]] = None,
        causal_context_ids: Optional[List[str]] = None,
        keyword_hints: Optional[Set[str]] = None,
    ) -> str:
        """DART: Dynamic Adaptive Retrieval for Trajectory Synthesis.

        Adapts retrieval strategy based on task complexity:
        - SIMPLE: Only period summaries + life summary (no individual events)
        - MODERATE: Period summaries + top-k events with compact summaries
        - COMPLEX: Full STARE-E retrieval with detailed summaries

        Falls back to build_context() if complexity is not recognized.
        """
        profile = DART_WEIGHT_PROFILES.get(task_complexity)
        if profile is None:
            return self.build_context(
                task_type=task_type, current_date=current_date,
                period_id=period_id, topic_hints=topic_hints,
                participant_hints=participant_hints,
                causal_context_ids=causal_context_ids,
                keyword_hints=keyword_hints,
            )

        parts = []

        # 1. Core Memory: current_life_summary (always included)
        summary = self._mb.current_life_summary
        if summary:
            parts.append(f"Current life summary: {summary}")

        if task_complexity == TaskComplexityLevel.SIMPLE:
            # Only semantic layer: period summaries, no individual events
            if self._mb.previous_life_period:
                parts.append("Life periods:")
                for pid, rec in self._mb.previous_life_period.items():
                    brief = rec.period_summary_brief or rec.period_summary
                    parts.append(f"  - {pid}: {rec.time_period} — {brief}")
            return "\n".join(parts) if parts else "Memory bank is empty (life just begun)"

        # MODERATE and COMPLEX: include period summaries
        if self._mb.previous_life_period:
            parts.append("Life periods:")
            for pid, rec in self._mb.previous_life_period.items():
                brief = rec.period_summary_brief or rec.period_summary
                parts.append(f"  - {pid}: {rec.time_period} — {brief}")

        # Build a temporary TaskConfig from DART profile
        dart_config = TaskConfig(
            w_recency=profile["w_recency"],
            w_importance=profile["w_importance"],
            w_semantic=profile["w_semantic"],
            w_participant=profile["w_participant"],
            w_causal=profile["w_causal"],
            top_k=profile["top_k"],
            context_mode="period_focused" if period_id else "general",
            summary_field=profile["summary_field"],
        )

        # Use STARE-E retrieval with DART weights
        events = self._mb.previous_events
        if not events:
            return "\n".join(parts) if parts else "Memory bank is empty (life has just begun)"

        # Compute semantic scores
        semantic_scores: Dict[str, float] = {}
        if self._emb and self._emb.available and dart_config.w_semantic > 0:
            query_text = self._build_query_text(
                task_type=task_type, topic_hints=topic_hints,
                keyword_hints=keyword_hints,
                period_theme=self._get_period_theme(period_id),
            )
            query_emb = self._emb.encode_query(query_text)
            semantic_scores = self._compute_all_semantic_scores(query_emb)

        # Score events with DART weights
        scored = []
        _participant_hints = participant_hints or []
        _causal_context_ids = causal_context_ids or []

        for eid, ev in events.items():
            if current_date and ev.event_date:
                try:
                    ev_date = date.fromisoformat(ev.event_date)
                    days_ago = max((current_date - ev_date).days, 0)
                except (ValueError, TypeError):
                    days_ago = 365
            else:
                days_ago = 365

            s_recency = math.exp(-0.005 * days_ago)
            s_importance = ev.importance_score
            s_semantic = semantic_scores.get(eid, 0.5)
            s_participant = self._compute_participant_overlap(ev, _participant_hints)
            s_causal = self._compute_causal_proximity(eid, _causal_context_ids)

            score = (
                dart_config.w_recency * s_recency
                + dart_config.w_importance * s_importance
                + dart_config.w_semantic * s_semantic
                + dart_config.w_participant * s_participant
                + dart_config.w_causal * s_causal
            )
            scored.append((eid, ev, round(score, 4)))

        scored.sort(key=lambda x: x[2], reverse=True)
        top_events = scored[:dart_config.top_k]

        if top_events:
            summary_field = dart_config.summary_field
            parts.append("Related historical events:")
            for eid, ev, score in top_events:
                text = getattr(ev, summary_field, ev.summary_for_context)
                parts.append(f"  [{eid}] {text}")

        return "\n".join(parts) if parts else "Memory bank is empty (life just begun)"

    def _build_context_legacy(
        self,
        mode: str = "general",
        current_date: Optional[date] = None,
        period_id: str = "",
        max_events: int = 8,
    ) -> str:
        """Build context using legacy two-dimensional retrieval.

        Backward compatible with v1 build_context().
        """
        parts = []

        # 1. Always include current_life_summary
        summary = self._mb.current_life_summary
        if summary:
            parts.append(f"Current life summary: {summary}")

        # 2. Period structure overview
        if self._mb.previous_life_period:
            parts.append("Life periods:")
            for pid, rec in self._mb.previous_life_period.items():
                brief = rec.period_summary_brief or rec.period_summary
                parts.append(f"  - {pid}: {rec.time_period} — {brief}")

        # 3. Mode-specific event retrieval
        if mode == "general":
            events = self.retrieve_weighted(
                current_date=current_date,
                top_k=max_events,
                alpha=0.35,
                beta=0.65,
            )
            if events:
                parts.append("Related historical events (sorted by importance × time):")
                for eid, ev, score in events:
                    parts.append(f"  [{eid}] {ev.summary_for_context}")

        elif mode == "period_focused":
            if period_id:
                period_events = self.retrieve_by_period(period_id, top_k=5)
                if period_events:
                    parts.append(f"Current period ({period_id}) important events:")
                    for eid, ev in period_events:
                        parts.append(f"  [{eid}] {ev.summary_for_context}")

                period_event_ids = {eid for eid, _ in period_events}
                cross_events = self.retrieve_weighted(
                    current_date=current_date,
                    top_k=3,
                    alpha=0.15,
                    beta=0.85,
                    exclude_ids=period_event_ids,
                )
                cross_events = [
                    (eid, ev, s)
                    for eid, ev, s in cross_events
                    if not eid.startswith(period_id)
                ]
                if cross_events:
                    parts.append("Cross-period important events:")
                    for eid, ev, _ in cross_events:
                        parts.append(f"  [{eid}] {ev.summary_for_context}")
            else:
                return self._build_context_legacy(
                    "general", current_date, "", max_events
                )

        return "\n".join(parts) if parts else "Memory bank is empty (life just begun)"
