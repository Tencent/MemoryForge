"""
P3 High-Resolution Simulation — Data Model Definitions
========================================================
Data model definitions for the four-step cascaded participant refinement flow
in high-resolution event simulation.

Includes:
  - Step 1: Refine from low-resolution pre-selected list
  - Step 2: Supplement from participant pool
  - Step 3: New participant suggestions
  - Step 4: Persona update detection
  - Aggregated output: ParticipantRefinementOutput
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ================================================================
# Step 1 output: refine core subset from the low-resolution pre-selected list
# ================================================================

class ParticipantRefinementEntry(BaseModel):
    """Refinement decision for a single participant."""
    participant_id: str = Field(..., description="Participant ID")
    included: bool = Field(..., description="Whether to include in high-res simulation")
    reason: str = Field(..., description="Brief reason for inclusion/exclusion")


class Step1RefinementOutput(BaseModel):
    """Step 1 output: refine core subset from the low-resolution pre-selected list."""
    selected_from_lr: List[str] = Field(
        ..., description="Participant IDs selected from the low-res list (must include P_TARGET)"
    )
    refinement_decisions: List[ParticipantRefinementEntry] = Field(
        ..., description="Decision details for each original participant"
    )
    missing_role_types: List[str] = Field(
        default_factory=list,
        description="Role types explicitly needed by the scene but absent from the low-res list"
    )
    refinement_reasoning: str = Field(
        ..., description="Overall reasoning for the refinement decisions"
    )


# ================================================================
# Step 2 output: supplement from participant pool
# ================================================================

class Step2PoolSupplementOutput(BaseModel):
    """Step 2 output: supplement missing roles from the participant pool."""
    added_from_pool: List[str] = Field(
        default_factory=list,
        description="Participant IDs pulled from the pool to fill missing roles"
    )
    pool_insufficient_reason: str = Field(
        default="",
        description="If pool has no suitable candidates, explain why (empty if supplement succeeded)"
    )
    supplement_reasoning: str = Field(
        ..., description="Reasoning for pool supplement decisions"
    )


# ================================================================
# Step 3 output: new participant suggestions
# ================================================================

class NewParticipantSuggestion(BaseModel):
    """Creation spec for a brand-new participant."""
    role_type: str = Field(..., description="Role type: family/friend/colleague/stranger/etc.")
    relationship_to_main_character: str = Field(..., description="Relationship description")
    persona_requirements: str = Field(..., description="Key persona traits needed for the scene")
    reason_needed: str = Field(..., description="Why this new participant is needed")


# ================================================================
# Step 4 output: persona update detection
# ================================================================

class PersonaUpdateSuggestion(BaseModel):
    """Persona update suggestion for a single participant."""
    participant_id: str = Field(..., description="Participant ID")
    needs_update: bool = Field(..., description="Whether current_persona_brief_text needs updating")
    updated_brief: str = Field(
        default="",
        description="The updated persona brief text (empty if needs_update=false)"
    )
    update_reason: str = Field(
        default="",
        description="Brief explanation of what changed (empty if needs_update=false)"
    )


class Step4PersonaCheckOutput(BaseModel):
    """Step 4 output: persona consistency check results for all final participants."""
    persona_updates: List[PersonaUpdateSuggestion] = Field(
        ..., description="Update suggestions for each final participant"
    )


# ================================================================
# Aggregated output: combined result of the four-step cascaded flow
# ================================================================

class ParticipantRefinementOutput(BaseModel):
    """Aggregated output of the four-step cascaded participant refinement flow."""
    # Step 1 result
    step1: Step1RefinementOutput
    # Step 2 result (populated only when triggered)
    step2: Optional[Step2PoolSupplementOutput] = None
    # Step 3 new participant suggestions (populated only when triggered)
    new_participant_suggestions: List[NewParticipantSuggestion] = Field(
        default_factory=list
    )
    # Step 4 persona update detection result
    step4: Step4PersonaCheckOutput
    # Final participant list (deduplicated merge of Step 1+2+3)
    final_participants: List[str] = Field(
        ..., description="Final deduplicated participant ID list for high-res simulation"
    )


# ================================================================
# M3: Full-Script Generation + Review + Correction Models
# ================================================================

class ScriptTurn(BaseModel):
    """A single turn in the generated script."""
    turn_index: int = Field(..., description="0-based turn index")
    speaker_id: str = Field(..., description="Participant ID of the speaker")
    speaker_name: str = Field(..., description="Display name of the speaker")
    action_type: str = Field(
        default="speak",
        description=(
            "One of: 'speak' (dialogue), 'act' (physical action), 'narrate' (omniscient narrator). "
            "IMPORTANT: 'narrate' MUST use speaker_id='NARRATOR', speaker_name='narration'. "
            "Do NOT assign 'narrate' to a specific character."
        )
    )
    content: str = Field(
        ...,
        description=(
            "Pure dialogue, action, or narration text. "
            "MUST NOT include the speaker's name as a prefix. "
            "Correct: 'Dude, have you seen this?' "
            "Wrong: 'Jake: Dude, have you seen this?'"
        )
    )
    body_language: str = Field(default="", description="Body language description")
    internal_thought: str = Field(default="", description="Internal monologue (only for P_TARGET)")
    emotional_state: str = Field(default="", description="Current emotional state")
    scene_time: str = Field(default="", description="In-scene time marker")
    scene_phase: str = Field(default="rising", description="Scene phase: setup/rising/climax/resolution/closing")


class FullScriptOutput(BaseModel):
    """M3-a: Output of one-shot full script generation."""
    turns: List[ScriptTurn] = Field(
        ..., description="Complete list of interaction turns"
    )
    scene_summary: str = Field(..., description="Brief summary of the complete scene")
    beats_covered: List[str] = Field(
        default_factory=list,
        description="List of narrative beats covered in the script"
    )
    emotional_arc: str = Field(default="", description="Emotional trajectory of the scene")


class ScriptIssue(BaseModel):
    """A single issue found during script review."""
    turn_index: int = Field(..., description="Index of the problematic turn")
    issue_type: str = Field(
        ...,
        description="Type: inconsistency / out_of_character / anachronism / "
                    "logic_error / pacing / missing_beat / other"
    )
    description: str = Field(..., description="What is wrong")
    suggested_fix: str = Field(default="", description="Suggested fix for this turn")


class ScriptReviewOutput(BaseModel):
    """M3-b: Output of script review."""
    issues: List[ScriptIssue] = Field(
        default_factory=list,
        description="List of issues found in the script"
    )
    overall_quality: str = Field(
        default="acceptable",
        description="Overall quality: good / acceptable / needs_revision"
    )
    review_summary: str = Field(default="", description="Brief review summary")


class CorrectedTurn(BaseModel):
    """M3-c: A corrected version of a single turn."""
    turn_index: int = Field(..., description="Index of the corrected turn")
    content: str = Field(
        ...,
        description=(
            "Corrected pure dialogue/action text. "
            "MUST NOT include speaker name prefix. "
            "Correct: 'That doesn't look right.' "
            "Wrong: 'Jake: That doesn't look right.'"
        )
    )
    body_language: str = Field(default="", description="Corrected body language")
    internal_thought: str = Field(default="", description="Corrected internal thought")
    emotional_state: str = Field(default="", description="Corrected emotional state")
    correction_reason: str = Field(default="", description="Why this correction was made")


class ScriptCorrectionOutput(BaseModel):
    """M3-c: Output of script correction for specific issues."""
    corrected_turns: List[CorrectedTurn] = Field(
        default_factory=list,
        description="Corrected versions of problematic turns"
    )


# ================================================================
# Parallel Protagonist Mode — Fixed 10-turn (5 beat × 2) Structure
# ================================================================

class BeatPlan(BaseModel):
    """Director's plan for one narrative beat: narrator content + protagonist slot."""
    beat_index: int = Field(..., description="0-based beat index (0–4)")
    beat_label: str = Field(default="", description="Short label identifying this beat")
    # Narrator turn (Turn A of this beat)
    narrator_content: str = Field(
        ...,
        description=(
            "3-5 sentences of omniscient narration covering this beat: "
            "environment changes, other characters' actions/speech, sensory details. "
            "Written in present tense, third-person. NO protagonist actions here."
        )
    )
    # Protagonist slot (Turn B of this beat) — filled by parallel generation
    protagonist_action_type: str = Field(
        default="speak",
        description="'speak' for dialogue, 'act' for physical action"
    )
    protagonist_direction_hint: str = Field(
        ...,
        description=(
            "Specific, colloquial instruction for what the protagonist says/does in this beat. "
            "E.g.: 'blurts out \"I'm not asking for a bailout—rent's gonna bounce\"' "
            "or 'yanks Jason back by the sleeve and hisses low, keep it moving'. "
            "Be concrete — no vague instructions like 'react' or 'express concern'."
        )
    )
    protagonist_scene_context: str = Field(
        default="",
        description=(
            "2-3 sentences of what the protagonist directly perceives right now: "
            "what they see, hear, feel in their body. At least 2 concrete sensory details."
        )
    )
    scene_phase: str = Field(
        default="rising",
        description="Scene phase for this beat: setup/rising/climax/resolution/closing"
    )
    scene_time: str = Field(default="", description="In-scene time marker, e.g. '10:25'")
    protagonist_emotional_state: str = Field(
        default="neutral",
        description="Expected emotional state of protagonist during this beat (one word)"
    )


class BeatSkeletonOutput(BaseModel):
    """Director's one-shot output: 5 beat plans covering the full scene."""
    beats: List[BeatPlan] = Field(
        ...,
        description="Exactly 5 beat plans in chronological order"
    )
    scene_summary: str = Field(..., description="1-2 sentence summary of the complete scene arc")
    emotional_arc: str = Field(default="", description="Emotional trajectory across all 5 beats")


class ProtagonistTurnResult(BaseModel):
    """Result of parallel protagonist generation for a single beat's protagonist turn."""
    turn_index: int = Field(..., description="Which beat index (0–4) this fills")
    content: str = Field(
        ...,
        description=(
            "What the protagonist says/does — pure text, no name prefix. "
            "Must be colloquial and natural. 2-4 sentences."
        )
    )
    internal_thought: str = Field(
        default="",
        description="Stream-of-consciousness inner monologue (2-3 fragmented thoughts)"
    )
    emotional_state: str = Field(
        default="neutral",
        description="Protagonist's emotional state at this moment (one precise word)"
    )
    body_language: str = Field(
        default="",
        description="2-3 specific physical micro-cues: eye movement, posture, hands, face"
    )
