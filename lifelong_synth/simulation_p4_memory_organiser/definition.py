"""
P4 Memory Organiser — Data Model Definitions
==============================================
Centralised data model definitions for the P4 Memory Organiser module.

This file contains all Pydantic data models and constants used across
the memory management subsystem, including:
  - LifePeriodRecord: a single historical life period
  - InteractionDetail: a single interaction within a high-res event
  - EventRecord: a single historical event
  - MemoryBase: top-level structured memory base
  - MemoryEntry: legacy flat memory record (backward compatibility)
"""

from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ================================================================
# Constants
# ================================================================

VALID_TIME_SEGMENTS = [
    "early morning",   # 00:00–06:00
    "morning",         # 06:00–12:00
    "afternoon",       # 12:00–18:00
    "night",           # 18:00–24:00
]

VALID_LANGUAGES = ["zho", "eng", "jpn", "kor", "fra", "deu", "spa"]

VALID_RESOLUTION_LEVELS = ["low", "medium", "high"]


# ================================================================
# Data Models — Structured Memory Base
# ================================================================

class LifePeriodRecord(BaseModel):
    """
    A record for a single historical life period stored in the memory base.

    Corresponds to one entry under ``previous_life_period`` in the memory
    base JSON.
    """

    time_period: str = Field(
        ...,
        description=(
            "Time span of this life period, e.g. "
            "'1998-03-27 to 2002-03-26'"
        ),
    )
    period_summary: str = Field(
        ...,
        description="Brief narrative summary of what happened during this period",
    )
    events: List[str] = Field(
        default_factory=list,
        description="Ordered list of event IDs that occurred in this period",
    )

    # === 🆕 New: pre-computed brief summary ===
    period_summary_brief: str = Field(
        default="",
        description=(
            "Brief period summary (<=120 chars) for build_context() display. "
            "Generated at period creation/update time."
        ),
    )

    # === Paper alignment: first-person retrospective summary (ℓ_i) ===
    period_summary_first_person: str = Field(
        default="",
        description=(
            "First-person retrospective summary of this life period. "
            "Written as if the protagonist is recalling this era of their life. "
            "Uses 'I' as subject. This is the paper's Lifetime Period Summary (ℓ_i)."
        ),
    )


class InteractionDetail(BaseModel):
    """
    A single interaction record within a high-resolution event.

    Captures one action/reaction in chronological order.
    """

    turn_index: int = Field(..., description="0-based chronological index")
    speaker_id: str = Field(..., description="Participant ID of the actor")
    action_type: str = Field(
        default="speak",
        description="Type of action: speak / act / think / observe",
    )
    content: str = Field(..., description="What the participant said or did")
    internal_thought: str = Field(
        default="",
        description="Internal monologue of the target persona (if applicable)",
    )


class EventRecord(BaseModel):
    """
    A record for a single historical event stored in the memory base.

    Corresponds to one entry under ``previous_events`` in the memory base
    JSON.
    """

    resolution_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Resolution level of this event record: high (Event-Specific Experience, P3 simulated with interaction_details), medium (General-Event Memory, habitual/recurring pattern with title+frequency), low (Lifetime Period level summary)",
    )
    time_period: str = Field(
        ...,
        description=(
            "Time span of this event. Format: "
            "'from [YYYY-MM-DD] [segment] to [YYYY-MM-DD] [segment]'. "
            "Segments: early morning (00:00–06:00), morning (06:00–12:00), "
            "afternoon (12:00–18:00), night (18:00–24:00)"
        ),
    )
    summary: str = Field(
        ...,
        description="Narrative summary of the event",
    )
    first_person_memory: str = Field(
        default="",
        description=(
            "First-person perspective memory of this event from the protagonist's viewpoint. "
            "Uses 'I' as subject. Populated from UPER reflection for P_TARGET. "
            "This is the primary retrieval target for the paper's M_π memory base."
        ),
    )
    initial_summary: str = Field(
        default="",
        description="Initial summary captured when the event is first written to memory",
    )
    refined_summary: str = Field(
        default="",
        description="Refined summary produced after high-resolution simulation finalization",
    )
    summary_stage: str = Field(
        default="initial",
        description="Current summary stage: initial | refined | final_same_as_initial",
    )
    summary_diff_note: str = Field(
        default="",
        description="Optional note describing the delta between initial and refined summaries",
    )
    participants: List[str] = Field(
        default_factory=list,
        description="List of participant IDs involved in this event",
    )
    linked_event_id: str = Field(
        default="",
        description="Paired low-resolution/high-resolution event ID",
    )
    source_event_id: str = Field(
        default="",
        description="Original source event ID before later refinement or upsert",
    )
    storage_status: str = Field(
        default="committed",
        description="Storage lifecycle status: pending | committed | superseded",
    )
    version: int = Field(
        default=1,
        description="Monotonic version used for safe overwrite/upsert",
    )
    run_id: str = Field(
        default="",
        description="Owning run ID for this event record",
    )
    languages_in_use: List[str] = Field(
        default_factory=lambda: ["eng"],
        description=(
            "Language codes used in the simulation. "
            "Only supports ['zho'], ['eng'], or ['zho', 'eng']"
        ),
    )
    interaction_details: List[InteractionDetail] = Field(
        default_factory=list,
        description=(
            "Chronological interaction records. "
            "Populated for high-resolution events; empty for low-resolution"
        ),
    )

    # === 🆕 New: scene-oriented pre-computed retrieval fields ===

    summary_for_context: str = Field(
        default="",
        description=(
            "Summary optimized for build_context() display (100-120 chars). "
            "Generated at write time using extractive summarization. "
            "Preserves core event + causal relationships."
        ),
    )

    summary_for_prompt: str = Field(
        default="",
        description=(
            "Summary optimized for P3 scene planning prompt injection (200-300 chars). "
            "Generated at write time. Preserves time, location, participants, causality."
        ),
    )

    summary_oneliner: str = Field(
        default="",
        description=(
            "One-line summary (<=50 chars), generated at write time. "
            "Used for compact listings and period overview."
        ),
    )

    importance_score: float = Field(
        default=0.5,
        description=(
            "Importance score 0-1, computed at write time via rule-based scoring. "
            "Combines event_type weight + resolution boost + participant count bonus."
        ),
    )

    event_date: str = Field(
        default="",
        description=(
            "Normalized date string YYYY-MM-DD, parsed from time_period at write time. "
            "Used for temporal sorting and recency calculation in retrieval."
        ),
    )

    # === 🆕 STARE-E v2: embedding and causal fields ===

    embedding_vector: list = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Pre-computed embedding vector from local model (multilingual-e5-base). "
            "768 dimensions, L2-normalized. Computed at write time from "
            "enriched text (event_type + period_theme + summary) with 'passage:' prefix. "
            "Used for semantic relevance scoring in retrieval."
        ),
    )

    causal_parent_ids: list = Field(
        default_factory=list,
        description=(
            "Event IDs of causal predecessors. "
            "Computed at write time using structural heuristics + embedding similarity."
        ),
    )

    # v6: behavioral style extraction from high-res events
    behavioral_style: Optional[dict] = Field(
        default=None,
        description=(
            "Behavioral style markers extracted from high-res interaction details. "
            "Contains sample_utterances, style_markers, and interaction_turn_count."
        ),
    )

    # v8: event theme category for diversity constraint in _rank_outlines_for_detail()
    event_theme_category: Optional[str] = Field(
        default=None,
        description=(
            "Coarse-grained theme category for diversity constraint. "
            "One of: career_transition, eval_debugging, meeting, social, "
            "academic, milestone, daily_routine, other. "
            "Generated by LLM (merged with summary call) or rule-based fallback."
        ),
    )

    # General-Event (medium) specific fields — paper's "title + frequency" format
    general_event_title: str = Field(
        default="",
        description="Title of the habitual/recurring event pattern (e.g., 'Morning lab routine')",
    )
    general_event_frequency: str = Field(
        default="",
        description="How often this event occurs (e.g., 'daily', 'weekly', 'every semester')",
    )

    @field_validator("languages_in_use")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        valid_langs = set(VALID_LANGUAGES)
        for lang in v:
            if lang not in valid_langs:
                raise ValueError(
                    f"Unsupported language code '{lang}'. "
                    f"Valid codes: {VALID_LANGUAGES}"
                )
        return v


class MemoryBase(BaseModel):
    """
    Top-level structured memory base for a single persona.

    This is the canonical in-memory representation that mirrors the JSON
    format defined in ``memory_base_test.json``.
    """

    previous_life_period: Dict[str, LifePeriodRecord] = Field(
        default_factory=dict,
        description="Historical life periods keyed by period_id (e.g. 'LP1')",
    )
    previous_events: Dict[str, EventRecord] = Field(
        default_factory=dict,
        description="Historical events keyed by event_id (e.g. 'LP3_E002')",
    )
    current_life_summary: str = Field(
        default="",
        description="Summary of the persona's current life state",
    )


# ================================================================
# Legacy Data Model (kept for backward compatibility)
# ================================================================

class MemoryEntry(BaseModel):
    """A single memory record stored after an event (legacy format)."""

    memory_id: str = Field(..., description="Unique memory identifier")
    source_event_id: str = Field(..., description="Event that produced this memory")
    timestamp: str = Field(..., description="ISO datetime when the memory was formed")
    content: str = Field(..., description="Natural-language memory content")
    salience: float = Field(default=0.5, description="Memory salience score 0-1")
    tags: List[str] = Field(default_factory=list, description="Semantic tags for retrieval")
