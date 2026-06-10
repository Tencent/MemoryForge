"""
P2 Event Organiser — Data Model Definitions
=============================================
Centralised data model definitions for the P2 Event Organiser module.

This file contains all Pydantic data models and constants used across
the event organiser subsystem, including:
  - NewParticipantInfo: new participant to add to the pool
  - LowResEventOutput: LLM output for low-resolution events
  - InteractionTurnOutput: a single interaction turn in a high-res event
  - HighResEventOutput: LLM output for high-resolution event scenes
  - InteractionHistoryEntry / InteractionHistoryBatch: interaction history records
  - LowResolutionEvent / HighResolutionEvent / InteractionTurn: legacy models
    (kept for backward compatibility with P3 and configs modules)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ================================================================
# Constants
# ================================================================

# DEPRECATED: This mapping is no longer the primary source of time_unit.
# New code uses compute_period_density() from temporal_context.py.
# Kept for backward compatibility with process_period() wrapper.
DENSITY_TO_UNIT: Dict[str, str] = {
    "none": "year",
    "around once a year": "year",
    "once per season": "season",
    "around once a month": "month",
    "around once a week": "month",  # week removed, mapped to month
}

TIME_SEGMENTS = ["early morning", "morning", "afternoon", "night"]


# ================================================================
# LLM Structured-Output Models (used by EventOrganiser)
# ================================================================

class NewParticipantInfo(BaseModel):
    """LLM output: a new participant to add to the pool."""
    persona_name_text: str = Field(..., description="Full name of the new participant")
    relationship_towards_the_main_character: str = Field(
        ..., description="Relationship description, e.g. 'college roommate'"
    )
    persona_brief_text: str = Field(
        ..., description="One-paragraph persona description"
    )
    date_of_birth_year: Optional[int] = Field(None, description="Birth year")
    date_of_birth_month: Optional[int] = Field(None, description="Birth month 1-12")
    date_of_birth_day: Optional[int] = Field(None, description="Birth day 1-31")


class LowResEventOutput(BaseModel):
    """LLM output for a single low-resolution time-unit event."""
    value_for_target: str = Field(
        ..., description="How this time unit contributes to building the target persona"
    )
    value_for_period: str = Field(
        ..., description="How this time unit contributes to the life-period goals"
    )
    summary: str = Field(
        ..., description="Narrative summary of the main content during this time unit"
    )
    participant_ids: List[str] = Field(
        default_factory=list,
        description="List of existing participant_ids involved in this time unit"
    )
    new_participants: List[NewParticipantInfo] = Field(
        default_factory=list,
        description="New participants to add (if any existing ones are insufficient)"
    )
    has_key_event: Literal["no", "key_event_detail", "key_event_general"] = Field(
        default="no",
        description=(
            "Whether this time unit contains a key event. "
            "'no': no key event; "
            "'key_event_detail': key event worth full P3 simulation; "
            "'key_event_general': key event worth recording as general-event memory"
        )
    )
    key_event_value_for_target: Optional[str] = Field(
        None, description="(If has_key_event != 'no') How the key event contributes to building the target persona"
    )
    key_event_motivation: Optional[str] = Field(
        None, description="(If has_key_event) What triggered / caused the key event"
    )
    key_event_outcome: Optional[str] = Field(
        None, description="(If has_key_event) The outcome / result of the key event"
    )
    key_event_start_date: Optional[str] = Field(
        None, description="(If has_key_event) Start date YYYY-MM-DD"
    )
    key_event_start_time: Optional[str] = Field(
        None, description="(If has_key_event) Start time segment: early morning / morning / afternoon / night"
    )
    key_event_end_date: Optional[str] = Field(
        None, description="(If has_key_event) End date YYYY-MM-DD"
    )
    key_event_end_time: Optional[str] = Field(
        None, description="(If has_key_event) End time segment: early morning / morning / afternoon / night"
    )
    key_event_participant_ids: Optional[List[str]] = Field(
        None, description="(If has_key_event) Participant IDs involved in the key event"
    )
    key_event_summary: Optional[str] = Field(
        None, description="(If has_key_event) Detailed summary of the key event"
    )


class InteractionTurnOutput(BaseModel):
    """LLM output for a single interaction turn in a high-res event."""
    turn_index: int = Field(..., description="0-based turn index")
    speaker_id: str = Field(..., description="participant_id of the speaker/actor")
    action_type: str = Field(
        default="speak", description="speak / act / think / observe"
    )
    content: str = Field(..., description="What the participant said or did")
    internal_thought: str = Field(
        default="",
        description="Internal monologue of the target persona (only for P_TARGET turns)"
    )


class HighResEventOutput(BaseModel):
    """LLM output for a high-resolution event scene."""
    detailed_summary: str = Field(
        ..., description="Rich narrative summary of the scene"
    )
    interaction_turns: List[InteractionTurnOutput] = Field(
        ..., description="Chronological multi-turn interaction transcript"
    )


class InteractionHistoryEntry(BaseModel):
    """LLM output for a participant's interaction history record."""
    participant_id: str = Field(..., description="participant_id")
    summary: str = Field(
        ...,
        description=(
            "First-person perspective summary of the interaction from this participant's viewpoint. "
            "MUST use 'I' as the subject. "
            "Correct: 'I remember sitting across from Jake and noticing he seemed distracted.' "
            "Wrong: 'Jake sat across from the table looking distracted.' "
            "Wrong: 'Across this period, Jake slides from post-dropout avoidance...'"
        )
    )


class InteractionHistoryBatch(BaseModel):
    """LLM output: batch of interaction history entries for all participants."""
    entries: List[InteractionHistoryEntry] = Field(
        ..., description="One entry per participant involved in the event"
    )


class CurrentLifeSummaryOutput(BaseModel):
    """LLM output: current life summary generated at the end of a life period."""
    current_life_summary: str = Field(
        ...,
        description=(
            "A concise third-person summary of the target persona's life "
            "up to the current point, covering key achievements, relationships, "
            "personality development, and current situation (3-6 sentences, <=500 chars)"
        ),
    )


# ================================================================
# Multi-Step Event Generation Models (Steps 1-5)
# ================================================================

# ── Step 1a: Low-Resolution Event Framework ──

class LowResFrameworkOutput(BaseModel):
    """Step 1a LLM output: focused low-resolution event framework."""
    event_type: str = Field(
        ..., description="Type of event, e.g. 'daily_routine', 'milestone', 'social', 'academic', 'career'"
    )
    event_theme: str = Field(
        ..., description="Core theme of the event"
    )
    event_location: str = Field(
        default="", description="Where the event takes place"
    )
    value_for_target: str = Field(
        ..., description="How this event contributes to building the target persona"
    )
    value_for_period: str = Field(
        ..., description="How this event contributes to the life-period goals"
    )
    summary: str = Field(
        ..., description="Narrative summary of the event (2-4 sentences)"
    )
    # v8: has_key_event is no longer an LLM output field    # to "no" after the LLM call and later updated by segment-level logic.
    # Kept as a regular field with default so it can be read/written on
    # Pydantic instances.  key_event_brief_reason is fully removed.
    has_key_event: Literal["no", "key_event_detail", "key_event_general"] = Field(
        default="no",
        description="Internal routing flag; always initialised to 'no' after Step 1a. "
                    "Segment-level logic (Step 1b) may promote this to "
                    "'key_event_detail' or 'key_event_general'.",
    )
    # Fields used by _step2_unified_pool_for_lr_and_outlines() — default to empty
    # so that LowResFrameworkOutput instances created without these fields remain valid.
    required_role_types: List[str] = Field(
        default_factory=list,
        description="List of required participant role types (e.g. ['family', 'friend']). "
                    "Used by participant pool matching.",
    )
    required_relationship_hints: List[str] = Field(
        default_factory=list,
        description="Hints about required relationship types for participant selection.",
    )
    estimated_participant_count: int = Field(
        default=2,
        description="Estimated number of participants needed for this event.",
    )


# ── Step 1b: High-Resolution Key Event Outline ──

class HighResOutlineOutput(BaseModel):
    """Step 1b LLM output: focused high-resolution key event outline."""
    key_event_theme: str = Field(
        ..., description="Specific theme of the key event"
    )
    key_event_summary: str = Field(
        ..., description="Detailed summary of the key event (3-5 sentences)"
    )
    # Precise time
    key_event_start_date: str = Field(
        ..., description="Start date in YYYY-MM-DD format"
    )
    key_event_start_time: str = Field(
        ..., description="Start time segment: early morning / morning / afternoon / night"
    )
    key_event_precise_start_time: str = Field(
        ..., description="Precise start time in HH:MM:SS format, e.g. '09:30:00'"
    )
    key_event_end_date: str = Field(
        ..., description="End date in YYYY-MM-DD format"
    )
    key_event_end_time: str = Field(
        ..., description="End time segment: early morning / morning / afternoon / night"
    )
    key_event_precise_end_time: str = Field(
        ..., description="Precise end time in HH:MM:SS format, e.g. '11:45:00'"
    )
    # Causal chain
    key_event_motivation: str = Field(
        ..., description="What triggered / caused the key event"
    )
    key_event_value_for_target: str = Field(
        ..., description="Impact on target persona development"
    )
    key_event_outcome: str = Field(
        ..., description="Result and consequences of the key event (1-2 sentences)"
    )
    # Scene details (optional but recommended)
    key_event_setting: Optional[str] = Field(
        None, description="Specific physical setting and atmosphere"
    )
    key_event_turning_point: Optional[str] = Field(
        None, description="The critical turning point moment"
    )
    required_participant_roles: List[str] = Field(
        default_factory=list,
        description=(
            "Role types explicitly needed for this key event scene, "
            "e.g. ['teacher', 'classmate', 'parent']. "
            "Step 3 MUST ensure these roles are covered."
        )
    )


# ── Legacy: Combined EventFrameworkOutput (kept for backward compatibility) ──

class EventFrameworkOutput(BaseModel):
    """Step 1 LLM output: base event framework WITHOUT participant selection.
    
    DEPRECATED: This model is kept for backward compatibility.
    New code should use LowResFrameworkOutput + HighResOutlineOutput instead.
    """
    event_type: str = Field(
        ..., description="Type of event, e.g. 'daily_routine', 'milestone', 'social', 'academic', 'career'"
    )
    event_theme: str = Field(
        ..., description="Core theme of the event"
    )
    event_location: str = Field(
        default="", description="Where the event takes place"
    )
    value_for_target: str = Field(
        ..., description="How this event contributes to building the target persona"
    )
    value_for_period: str = Field(
        ..., description="How this event contributes to the life-period goals"
    )
    summary: str = Field(
        ..., description="Narrative summary of the event (2-4 sentences)"
    )
    has_key_event: Literal["no", "key_event_detail", "key_event_general"] = Field(
        default="no",
        description=(
            "Internal routing flag; always initialised to 'no' after Step 1a. "
            "'no': no key event; "
            "'key_event_detail': key event worth full P3 simulation; "
            "'key_event_general': key event worth recording as general-event memory"
        ),
    )
    key_event_value_for_target: Optional[str] = Field(None)
    key_event_motivation: Optional[str] = Field(None)
    key_event_outcome: Optional[str] = Field(None)
    key_event_start_date: Optional[str] = Field(None)
    key_event_start_time: Optional[str] = Field(None)
    key_event_end_date: Optional[str] = Field(None)
    key_event_end_time: Optional[str] = Field(None)
    key_event_precise_start_time: Optional[str] = Field(
        None,
        description="Precise start time in HH:MM:SS format, e.g. '09:30:00'"
    )
    key_event_precise_end_time: Optional[str] = Field(
        None,
        description="Precise end time in HH:MM:SS format, e.g. '11:45:00'"
    )
    key_event_summary: Optional[str] = Field(None)


class ParticipantCandidate(BaseModel):
    """A single candidate participant selected for an event."""
    participant_id: str = Field(..., description="participant_id from the pool")
    selection_reason: str = Field(
        ..., description="Why this participant is suitable for this event"
    )


class ParticipantSelectionOutput(BaseModel):
    """Step 2 LLM output: selected participants for an event."""
    selected_participants: List[ParticipantCandidate] = Field(
        ..., description="List of selected participants from the existing pool"
    )
    needs_new_participants: bool = Field(
        default=False,
        description="Whether additional new participants are needed"
    )
    new_participant_suggestions: List[str] = Field(
        default_factory=list,
        description="Descriptions of new participants needed, e.g. ['college roommate, male, same major']"
    )


class PersonaValidationResult(BaseModel):
    """Validation result for a single participant's persona consistency."""
    participant_id: str = Field(...)
    is_consistent: bool = Field(
        ..., description="Whether the current persona is consistent with target and current time"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of inconsistency issues found"
    )
    suggested_current_persona_brief: Optional[str] = Field(
        None,
        description="Updated current_persona_brief_text if inconsistencies found"
    )


class PersonaValidationOutput(BaseModel):
    """Step 3 LLM output: persona validation for a single participant."""
    is_consistent: bool = Field(...)
    issues: List[str] = Field(default_factory=list)
    updated_current_persona_brief_text: Optional[str] = Field(
        None,
        description="Updated persona brief reflecting current time state, if update needed"
    )


class NewParticipantRequirement(BaseModel):
    """Detailed requirement for a single new participant to be generated."""
    role_type: str = Field(
        ..., description="Role type, e.g. 'family', 'friend', 'colleague', 'acquaintance', 'teacher'"
    )
    relationship_to_main_character: str = Field(
        ..., description="Specific relationship to the main character, e.g. 'kindergarten teacher', 'neighborhood friend'"
    )
    persona_requirements: str = Field(
        ..., description="Detailed persona requirements including age range, personality traits, background, etc."
    )
    reason_needed: str = Field(
        ..., description="Why this specific participant is needed for the event"
    )


class ParticipantSufficiencyOutput(BaseModel):
    """Step 4 LLM output: assessment of participant count sufficiency."""
    is_sufficient: bool = Field(
        ..., description="Whether current participants are enough for the event"
    )
    reason: str = Field(
        default="", description="Overall explanation of the assessment"
    )
    missing_role_types: List[str] = Field(
        default_factory=list,
        description="Specific role types that are missing, e.g. ['teacher', 'classmate']"
    )
    insufficiency_analysis: str = Field(
        default="",
        description="Detailed analysis of why current participants are insufficient (empty if sufficient)"
    )
    additional_needed: int = Field(
        default=0, description="Number of additional participants needed"
    )
    new_participant_requirements: List[NewParticipantRequirement] = Field(
        default_factory=list,
        description="Detailed requirements for each new participant to be generated"
    )
    additional_descriptions: List[str] = Field(
        default_factory=list,
        description="Simple text descriptions of additional participants needed (legacy fallback)"
    )


class PostEventPersonaUpdateOutput(BaseModel):
    """Post-event LLM output: updated persona brief for a participant after an event."""
    updated_current_persona_brief_text: str = Field(
        ...,
        description=(
            "Updated current_persona_brief_text reflecting the participant's state "
            "after the event. Should maintain consistency with target_persona_brief_text "
            "while accurately reflecting current time state evolution."
        ),
    )


class PeriodSummaryOutput(BaseModel):
    """LLM output: period summary based on plan + actual events."""
    period_summary: str = Field(
        ...,
        description=(
            "Rewritten period summary integrating the original period plan "
            "with actual simulated events. Should capture both planned goals "
            "and actual outcomes."
        ),
    )
    period_summary_first_person: str = Field(
        default="",
        description=(
            "First-person retrospective summary of this life period (3-5 sentences). "
            "Written as the protagonist recalling this era. MUST use 'I' as subject. "
            "Example: 'I remember those years as a time of intense growth...'"
        ),
    )


# ================================================================
# Legacy Data Models (kept for backward compatibility with P3 / configs)
# ================================================================

class LowResolutionEvent(BaseModel):
    """A coarse-grained event summary within a life period (legacy model)."""
    event_id: str = Field(..., description="Unique event identifier, e.g. 'LP3_E002'")
    period_id: str = Field(..., description="Parent life-period identifier, e.g. 'LP3'")
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    start_time: str = Field(default="08:00", description="HH:MM")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_time: str = Field(default="18:00", description="HH:MM")
    participants: List[str] = Field(default_factory=list, description="List of participant_ids")
    summary: str = Field(..., description="Brief narrative summary of the event")
    importance_score: float = Field(default=0.0, description="Event importance score 0-1")


class InteractionTurn(BaseModel):
    """A single turn in a multi-turn interaction scene (legacy model)."""
    turn_index: int = Field(..., description="0-based turn index")
    speaker_id: str = Field(..., description="participant_id of the speaker")
    utterance: str = Field(..., description="What the speaker said or did")
    internal_thought: str = Field(
        default="",
        description="Internal monologue of the target persona (if applicable)",
    )


class HighResolutionEvent(BaseModel):
    """A fine-grained event with detailed interaction scenes (legacy model)."""
    event_id: str = Field(..., description="Unique event identifier, e.g. 'LP6_E001_HR'")
    parent_low_res_event_id: str = Field(..., description="Corresponding low-res event id")
    period_id: str = Field(..., description="Parent life-period identifier")
    locale: str = Field(default="zh-CN", description="Language/locale for this scene")
    start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    start_time: str = Field(default="09:00:00", description="Precise start time HH:MM:SS (24h)")
    end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    end_time: str = Field(default="10:00:00", description="Precise end time HH:MM:SS (24h)")
    start_time_segment: str = Field(
        default="morning",
        description="Time segment: early morning / morning / afternoon / night"
    )
    end_time_segment: str = Field(
        default="morning",
        description="Time segment: early morning / morning / afternoon / night"
    )
    participants: List[str] = Field(default_factory=list, description="List of participant_ids")
    detailed_summary: str = Field(..., description="Rich narrative summary of the scene")
    interaction_turns: List[InteractionTurn] = Field(
        default_factory=list, description="Multi-turn interaction transcript"
    )


# ================================================================
# Module 1: Unified Participant Pool Output (replaces Step 2+4+5)
# ================================================================

class SelectedParticipantEntry(BaseModel):
    """A participant selected for the unified pool."""
    participant_id: str = Field(..., description="participant_id from the pool")
    role_in_event: str = Field(..., description="This participant's role in the event")
    selection_reason: str = Field(..., description="Why this participant is selected")
    is_essential_for_high_res: bool = Field(
        default=False,
        description="Whether this participant is essential for the high-res key event"
    )


class NewParticipantRequirementEntry(BaseModel):
    """Requirement for a new participant that doesn't exist in the pool."""
    role_type: str = Field(
        ..., description="Role type: family / friend / colleague / teacher / etc."
    )
    relationship_to_main_character: str = Field(
        ..., description="Specific relationship, e.g. 'kindergarten teacher'"
    )
    reason_needed: str = Field(
        ..., description="Why this participant is needed"
    )
    persona_sketch: str = Field(
        ...,
        description="Brief persona description for the new character"
    )


class UnifiedParticipantPoolOutput(BaseModel):
    """Step 2 unified output: participant selection + sufficiency + new participant requirements.
    
    Replaces the sequential Step 2 (ParticipantSelectionOutput) + Step 4
    (ParticipantSufficiencyOutput) + Step 5 input with a single LLM call.
    """
    selected_participants: List[SelectedParticipantEntry] = Field(
        ..., description="Participants selected from the existing pool"
    )
    is_sufficient: bool = Field(
        ..., description="Whether the selected participants are sufficient for all events"
    )
    sufficiency_reason: str = Field(
        default="", description="Explanation of sufficiency assessment"
    )
    new_participant_requirements: List[NewParticipantRequirementEntry] = Field(
        default_factory=list,
        description="Requirements for new participants (only when is_sufficient=False)"
    )


# ================================================================
# Module 2: Batch Persona Validation Output
# ================================================================

class BatchPersonaValidationEntry(BaseModel):
    """Validation result for one participant in a batch."""
    participant_id: str = Field(..., description="participant_id")
    is_consistent: bool = Field(
        ..., description="Whether the current persona is consistent"
    )
    updated_persona: str = Field(
        default="",
        description="Updated persona brief if inconsistent; empty if consistent"
    )


class BatchPersonaValidationOutput(BaseModel):
    """Batch validation output for all participants in a single LLM call."""
    validations: List[BatchPersonaValidationEntry] = Field(
        ..., description="Validation results for each participant"
    )


# ================================================================
# Module 3: UPER — Unified Post-Event Reflection Output
# ================================================================

class ParticipantReflection(BaseModel):
    """Combined reflection output for one participant after an event."""
    participant_id: str = Field(
        ...,
        description=(
            "The exact participant_id string as shown in the participant list above, "
            "e.g. 'P_TARGET', 'P_001', 'P_002'. "
            "MUST be the ID code, NOT the character's name. "
            "Copy it verbatim from the '### P_XXX: Name' header."
        )
    )
    memory: str = Field(
        ...,
        description=(
            "First-person colloquial memory of the event (1-2 sentences). "
            "MUST use 'I' as the subject — this is a personal memory, not a biography. "
            "✓ Correct: 'I remember feeling nervous when the professor called on me.' "
            "✓ Correct: 'I finally walked that stage and got my diploma.' "
            "✗ Wrong: 'Jake felt nervous during the class.' (third person) "
            "✗ Wrong: 'Tyler Johnson is raised at home...' (third person biography) "
            "✗ Wrong: 'During this period, the protagonist struggled with...' (narrator voice) "
            "✗ Wrong: 'He/She/They did...' (any third-person pronoun)"
        )
    )
    updated_persona: str = Field(
        ...,
        description="Updated current_persona_brief_text after the event"
    )


class BatchReflectionOutput(BaseModel):
    """UPER batch output: interaction histories + persona updates for all participants."""
    reflections: List[ParticipantReflection] = Field(
        ..., description="One reflection per participant"
    )


class BatchLROutput(BaseModel):
    """Batch LR output: multiple LowResFrameworkOutput + habitual events from a single prompt.

    Used by _step1a_batch_generate_all_lr() and _step1a_batch_generate_all_lr_multi_period()
    to return all LR events and habitual event seeds for a batch of consecutive time units.
    """
    lr_frameworks: List[LowResFrameworkOutput] = Field(
        ..., description="One LowResFrameworkOutput per time unit in the batch"
    )
    batch_size: int = Field(
        ..., description="Number of time units in this batch"
    )
    habitual_events_per_unit: List[List["HabitualEventSeed"]] = Field(
        default_factory=list,
        description=(
            "For each time unit in the batch, a list of 2-3 habitual/recurring event seeds "
            "(General-Event layer). Outer list length must equal batch_size."
        )
    )


class BatchUnifiedParticipantPoolOutput(BaseModel):
    """Batch participant pool output for LR + outline unified pool.

    Used by _step2_unified_participant_pool_batch() to return participant
    selections for all time units in a batch simultaneously.
    """
    pool_outputs: List[UnifiedParticipantPoolOutput] = Field(
        ..., description="One UnifiedParticipantPoolOutput per time unit in the batch"
    )


class BatchUPEROutput(BaseModel):
    """Batch UPER output: reflections for multiple events in a batch.

    Used by _uper_reflect_batch_period() to return UPER results
    for multiple events within a single time period.
    """
    batch_reflections: List[BatchReflectionOutput] = Field(
        ..., description="One BatchReflectionOutput per event in the batch"
    )


# ================================================================
# M4+M5: LR+Outline merged generation models
# ================================================================

class OutlineInLR(BaseModel):
    """Outline embedded within LR generation output.

    Used by _step1a_generate_lr_with_outlines() to return outlines
    alongside the LR framework in a single LLM call.
    """
    key_event_theme: str = Field(
        ..., description="Core theme of the key event outline"
    )
    key_event_summary: str = Field(
        ..., description="Summary of the key event (1-2 sentences)"
    )
    key_event_start_date: str = Field(
        ..., description="Start date of the key event (ISO format)"
    )
    key_event_end_date: str = Field(
        ..., description="End date of the key event (ISO format)"
    )
    key_event_start_time: str = Field(
        default="morning", description="Start time segment"
    )
    key_event_end_time: str = Field(
        default="afternoon", description="End time segment"
    )
    key_event_precise_start_time: Optional[str] = Field(
        default=None,
        description="Precise start time in HH:MM:SS format, e.g. '09:30:00'"
    )
    key_event_precise_end_time: Optional[str] = Field(
        default=None,
        description="Precise end time in HH:MM:SS format, e.g. '11:45:00'"
    )
    key_event_motivation: str = Field(
        default="", description="Motivation/cause of the key event"
    )
    key_event_outcome: str = Field(
        default="", description="Expected outcome of the key event"
    )
    key_event_setting: str = Field(
        default="", description="Scene setting description"
    )
    key_event_turning_point: str = Field(
        default="", description="Turning point description"
    )
    key_event_value_for_target: str = Field(
        default="", description="Impact on the target character"
    )
    required_participant_roles: List[str] = Field(
        default_factory=list,
        description="Required participant roles for this key event outline"
    )


class HabitualEventSeed(BaseModel):
    """A habitual/recurring event pattern (General-Event / Medium resolution).

    Represents a repeated activity from Script Theory (Schank & Abelson 1978).
    These become the G layer (General-Event Memories) in the paper's M_π format.
    """
    habitual_event_title: str = Field(
        ..., description="Title of the habitual pattern (e.g., 'Morning lab routine')"
    )
    habitual_event_frequency: str = Field(
        ..., description="Frequency: daily/weekly/monthly/every semester/etc."
    )
    habitual_event_summary: str = Field(
        ..., description="1-2 sentence description of the routine"
    )


class LRWithOutlinesOutput(BaseModel):
    """Single-prompt output: LR framework + k outlines + habitual event seeds.

    Used by _step1a_generate_lr_with_outlines() to return both
    the LR event, its outlines, and habitual event seeds in one LLM call.
    """
    lr_framework: LowResFrameworkOutput = Field(
        ..., description="The low-resolution event framework"
    )
    outlines: List[OutlineInLR] = Field(
        default_factory=list,
        description="Key event outlines embedded in this LR generation"
    )
    outline_count: int = Field(
        default=0, description="Number of outlines generated"
    )
    habitual_events: List[HabitualEventSeed] = Field(
        default_factory=list,
        description="2-3 habitual/recurring event patterns for this time unit (General-Event seeds)"
    )


class OutlinePoolEntry(BaseModel):
    """Participant pool entry for a single outline.

    Used by _step2_unified_pool_for_lr_and_outlines() to return
    the participant pool for each outline alongside the LR pool.
    """
    outline_index: int = Field(
        ..., description="Index of the outline in the outlines list"
    )
    selected_participants: List[SelectedParticipantEntry] = Field(
        default_factory=list,
        description="Selected participants for this outline"
    )
    is_sufficient: bool = Field(
        default=True,
        description="Whether selected participants are sufficient"
    )
    new_participant_requirements: List[NewParticipantRequirementEntry] = Field(
        default_factory=list,
        description="Required new participants if insufficient"
    )


class LRAndOutlinePoolOutput(BaseModel):
    """Unified pool output for LR + outlines from a single prompt.

    Used by _step2_unified_pool_for_lr_and_outlines() to return
    both LR participant pool and outline participant pools.
    """
    lr_pool: UnifiedParticipantPoolOutput = Field(
        ..., description="Participant pool for the LR event"
    )
    outline_pools: List[OutlinePoolEntry] = Field(
        default_factory=list,
        description="Participant pools for each outline"
    )


# Resolve forward references in BatchLROutput (HabitualEventSeed defined after it)
BatchLROutput.model_rebuild()
