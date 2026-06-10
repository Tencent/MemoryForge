"""
High-Resolution Event Simulator
=================================
Implements multi-agent interaction simulation for high-resolution events.

Borrows key design patterns from Concordia (DeepMind):
  - Per-agent personalized observation (MakeObservation)
  - Event resolution / validation (EventResolution)
  - Scene tracking with beat progression (SceneTracker)
  - Environment state management (WorldState)

This module orchestrates the complete simulation flow:
  1. Environment Director plans scene outline with concrete beats
  2. Director selects the next actor, generates personalized observation
  3. The selected agent generates a response based on persona, memory & observation
  4. Director resolves/validates the action against scene logic
  5. Environment state updates (time, positions, emotions)
  6. Loop continues until all beats are covered
  7. Post-scene memories generated for all participants

Data flow:
  high_resolution_event_test.json + persona_pool.json + memory_base_test.json
  → HighResEventSimulator → demo.json (complete interaction transcript)
"""

import json
import logging
import os
import re
import asyncio
from contextlib import nullcontext
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from pydantic import BaseModel, Field

from lifelong_synth.performance_tracker import PerformanceTracker
from llm.client import AsyncLLMClient
from lifelong_synth.persona_extensions_formatter import format_persona_extensions
from lifelong_synth.simulation_p4_memory_organiser.fragment_scorer import (
    score_memory_fragment,
    score_role_type_importance,
)
from lifelong_synth.simulation_p3_multi_resolution_simulation.definition import (
    ParticipantRefinementEntry,
    Step1RefinementOutput,
    Step2PoolSupplementOutput,
    NewParticipantSuggestion,
    PersonaUpdateSuggestion,
    Step4PersonaCheckOutput,
    ParticipantRefinementOutput,
    # M3: Full-Script Generation + Review + Correction
    ScriptTurn,
    FullScriptOutput,
    ScriptIssue,
    ScriptReviewOutput,
    CorrectedTurn,
    ScriptCorrectionOutput,
)

logger = logging.getLogger(__name__)


# ================================================================
# T01: Speaker prefix stripping utility
# ================================================================

_SPEAKER_PREFIX_RE = re.compile(r'^[A-Za-z][A-Za-z\s\-\']{0,29}(?:\s*\([^)]{0,20}\))?\s*:\s+')


def _strip_speaker_prefix(content: str, speaker_name: str = "") -> str:
    """Remove speaker name prefix from content if present.

    Examples:
        _strip_speaker_prefix("Jake: Hello there", "Jake") -> "Hello there"
        _strip_speaker_prefix("Angela Carter: She walks in", "Angela Carter") -> "She walks in"
        _strip_speaker_prefix("Hello there", "Jake") -> "Hello there"
    """
    if not content:
        return content
    if speaker_name:
        prefix = f"{speaker_name}:"
        if content.startswith(prefix):
            return content[len(prefix):].lstrip()
        if content.lower().startswith(prefix.lower()):
            return content[len(prefix):].lstrip()
        # Also handle with space before colon
        prefix_space = f"{speaker_name} :"
        if content.startswith(prefix_space):
            return content[len(prefix_space):].lstrip()
    return _SPEAKER_PREFIX_RE.sub('', content, count=1)


# ================================================================
# Structured Output Models for LLM Calls
# ================================================================

class SceneOutlinePlan(BaseModel):
    """Environment director's initial plan for the scene structure."""
    scene_setting: str = Field(
        ..., description="Physical setting with sensory details"
    )
    opening_action: str = Field(
        ..., description="How the scene opens — a concrete first action/line of dialogue"
    )
    key_beats: List[str] = Field(
        default_factory=list,
        description="5-8 key narrative beats that MUST happen, each a concrete micro-event"
    )
    expected_turn_count: int = Field(
        default=15,
        description="Expected number of interaction turns (12-25)"
    )
    emotional_arc: str = Field(
        default="",
        description="The emotional trajectory, e.g. 'nervous → focused → relieved → proud'"
    )
    time_markers: List[str] = Field(
        default_factory=list,
        description="Key time points within the event, e.g. ['09:30 exam starts', '10:15 break', '11:00 results']"
    )


class DirectorActionSelection(BaseModel):
    """Environment director's decision on the next action in the scene."""
    next_speaker_id: str = Field(
        ..., description="participant_id of the next actor"
    )
    action_type: str = Field(
        default="speak",
        description="Action type: speak / act / think / observe / narrate / react"
    )
    direction_hint: str = Field(
        default="",
        description="Concrete, colloquial hint for what the actor should do"
    )
    personalized_observation: str = Field(
        default="",
        description="What this specific character notices/perceives right now from their unique vantage point"
    )
    scene_phase: str = Field(
        default="development",
        description="Current phase: opening / development / turning_point / resolution / closing"
    )
    current_beat_index: int = Field(
        default=0,
        description="Index of the current narrative beat being addressed (0-based)"
    )
    should_end_scene: bool = Field(
        default=False,
        description="Whether the scene should end after this turn"
    )
    scene_time: str = Field(
        default="",
        description="Current in-scene time, e.g. '09:45'"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for this selection"
    )
    # Module 4: Uncertainty-driven selective resolution
    needs_resolution: bool = Field(
        default=False,
        description=(
            "Set to true ONLY if: "
            "1) The action involves physical interaction unrealistic for the character's age, "
            "2) The action references technology/culture that might not exist in this era, "
            "3) The action could significantly alter the scene trajectory unexpectedly, "
            "4) You are unsure whether the character would realistically do this. "
            "Set to false if the action is a natural continuation of the scene."
        )
    )
    resolution_reason: str = Field(
        default="",
        description="If needs_resolution=true, briefly explain why (1 sentence)"
    )


class AgentActionResponse(BaseModel):
    """An agent's response to a scene action."""
    content: str = Field(
        ..., description="What the agent says, does, or thinks — must be colloquial and natural"
    )
    internal_thought: str = Field(
        default="",
        description="Internal monologue (only meaningful for P_TARGET) — stream-of-consciousness style"
    )
    emotional_state: str = Field(
        default="neutral",
        description="Current emotional state in one word (use the scene's primary language)"
    )
    body_language: str = Field(
        default="",
        description="Subtle physical cues: fidgeting, eye movement, posture shifts"
    )
    # Module 4: Uncertainty-driven selective resolution
    needs_resolution: bool = Field(
        default=False,
        description=(
            "Set to true ONLY if: "
            "1) You are unsure if this response is consistent with the character's established persona, "
            "2) The response references specific knowledge/technology that might be anachronistic, "
            "3) The emotional reaction might be inconsistent with the character's personality, "
            "4) The physical action described might be unrealistic. "
            "Set to false if you are confident this is a natural, in-character response."
        )
    )
    resolution_reason: str = Field(
        default="",
        description="If needs_resolution=true, briefly explain why (1 sentence)"
    )


class EventResolutionResult(BaseModel):
    """Director's validation of an agent's action against scene logic."""
    is_valid: bool = Field(
        default=True,
        description="Whether the action is consistent with the scene context and character"
    )
    resolved_event: str = Field(
        default="",
        description="The canonical description of what actually happened (may adjust the raw action)"
    )
    environment_changes: str = Field(
        default="",
        description="Any changes to the physical environment caused by this action"
    )
    other_reactions: str = Field(
        default="",
        description="Brief note on how bystanders/environment react (ambient reactions)"
    )
    time_elapsed: str = Field(
        default="0 min",
        description="How much scene time this action consumed"
    )


class PostSceneMemory(BaseModel):
    """A single memory entry generated after the scene."""
    participant_id: str = Field(..., description="Who this memory belongs to")
    summary: str = Field(
        ..., description="First-person colloquial memory (2-3 sentences)"
    )
    emotional_tone: str = Field(
        default="neutral",
        description="Emotional tone: warm / joyful / proud / nervous / etc."
    )
    sensory_detail: str = Field(
        default="",
        description="One vivid sensory detail this person would remember (a sound, smell, image)"
    )


class PostSceneMemoryBatch(BaseModel):
    """Batch of post-scene memories for all participants."""
    memories: List[PostSceneMemory] = Field(
        default_factory=list,
        description="One memory per participant"
    )


class EraContextGuide(BaseModel):
    """LLM-generated era and cultural context guide for the simulation."""

    technology_constraints: str = Field(
        ...,
        description=(
            "What technology exists and doesn't exist in this era and location. "
            "Be specific: communication devices, entertainment, transportation, "
            "payment methods. Include what people WOULD use instead."
        )
    )
    social_norms: str = Field(
        ...,
        description=(
            "Social norms, etiquette, and behavioral expectations for this "
            "time/place/setting. How do people greet each other? What's "
            "considered polite/rude? Power dynamics?"
        )
    )
    cultural_details: str = Field(
        ...,
        description=(
            "Cultural context: local customs, common expressions, popular "
            "culture references, food, clothing styles that would naturally "
            "appear in this scene."
        )
    )
    physical_environment_hints: str = Field(
        ...,
        description=(
            "What the physical environment would realistically look like: "
            "architecture, furniture, decorations, lighting, common objects. "
            "Include sensory details: typical sounds, smells, textures."
        )
    )
    anachronism_warnings: str = Field(
        ...,
        description=(
            "Specific things that MUST NOT appear because they didn't exist "
            "yet or weren't common. Be explicit about what to avoid."
        )
    )
    era_specific_flavor: str = Field(
        ...,
        description=(
            "2-3 era-specific 'flavor' details: a popular song, a specific "
            "brand/product, a news event people might reference, seasonal details."
        )
    )


class CharacterStyleGuide(BaseModel):
    """LLM-generated personalized style guide for a single character."""

    speech_style: str = Field(
        ...,
        description=(
            "How this person speaks: sentence length, vocabulary level, "
            "favorite expressions, verbal tics, tone. Include 2-3 example "
            "phrases this person would typically say (in quotes)."
        )
    )
    behavioral_patterns: str = Field(
        ...,
        description=(
            "Typical physical behaviors and habits: nervous tics, comfort "
            "gestures, how they show emotions through body language. "
            "Give 2-3 specific examples."
        )
    )
    emotional_expression: str = Field(
        ...,
        description=(
            "How this person expresses emotions: do they suppress or amplify? "
            "What's their default emotional register? How do they show joy, "
            "anxiety, anger?"
        )
    )
    interaction_style: str = Field(
        ...,
        description=(
            "How this person interacts with others: deferential/authoritative/"
            "casual/awkward? How do they address the main character?"
        )
    )
    forbidden_expressions: str = Field(
        ...,
        description=(
            "Expressions or behaviors OUT OF CHARACTER for this person. "
            "Things they would NEVER say given their age, personality, background."
        )
    )
    memory_voice: str = Field(
        ...,
        description=(
            "How this person would retell this event later: what details "
            "would they focus on? What tone? Dramatic/understated/factual?"
        )
    )


# ================================================================
# Configuration Constants
# ================================================================

DEFAULT_TEMPERATURES = {
    "era_context": 0.0,
    "character_style": 0.0,
    "scene_outline": 0.0,
    "director_select": 0.0,
    "agent_action": 0.0,
    "event_resolution": 0.0,
    "post_memory": 0.0,
    "forced_resolution": 0.0,
    "refined_summary": 0.0,
}

DEFAULT_STYLE_GEN_CONFIG = {
    "max_concurrent": 3,         # Max parallel LLM calls for style generation
    "max_retries": 1000,         # Max retries per character on failure (high to avoid LLM-induced failures)
    "base_retry_delay": 1.0,     # Base delay for exponential backoff (seconds)
    "retry_backoff_factor": 2.0, # Multiplier for each retry
}


# ================================================================
# Pydantic models for anti-moralization guard (T30)
# ================================================================

class _AttitudeClassificationItem(BaseModel):
    topic: str = Field(description="Attitude topic, echoed back verbatim from input.")
    is_identity_core_negative: bool = Field(
        description=(
            "True only if this attitude is BOTH (a) negative in valence AND (b) central to the "
            "persona's identity or worldview (e.g. an Atheist's rejection of religion; a hardliner's "
            "hostility to their political out-group; a recovering addict's disgust at drug culture). "
            "Mild preferences or habits-only negatives (e.g. 'dislikes exercise', 'doesn't enjoy "
            "cooking') must return False."
        ),
    )
    rationale: str = Field(
        default="",
        description="One English sentence explaining the decision, at most 30 words.",
    )


class _NegativeAttitudeClassification(BaseModel):
    attitudes: List[_AttitudeClassificationItem] = Field(
        default_factory=list,
        description="One _AttitudeClassificationItem per input attitude, in the same order.",
    )


# ================================================================
# Core Simulator
# ================================================================

class HighResEventSimulator:
    """
    Orchestrates multi-agent interaction simulation for a high-resolution event.

    Design patterns borrowed from Concordia:
      - Per-agent observation: each character sees the scene from their own vantage point
      - Event resolution: every action is validated/adjusted by the director
      - Beat tracking: director tracks which narrative beats have been covered
      - Environment state: time, positions, ambient details are tracked and evolve

    The simulation follows a director-agent loop:
      1. Director plans the scene outline with concrete beats
      2. Director selects next actor + generates personalized observation for them
      3. Agent generates response based on persona, memory, observation, and direction
      4. Director resolves/validates the action (concordia's resolve step)
      5. Environment state updates; loop back to step 2
      6. After scene ends, generate post-scene memories with sensory details
    """

    # ── Module 7: Class-level Phase 0 caches ─────────────────────
    # These persist across instances within the same process.
    # Key format: "<year>" for era context, "<participant_id>_<year>" for styles.
    _era_context_cache: Dict[str, "EraContextGuide"] = {}
    _character_style_cache: Dict[str, "CharacterStyleGuide"] = {}

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all Phase 0 caches. Call at period transitions."""
        cls._era_context_cache.clear()
        cls._character_style_cache.clear()
        logger.info("[P3 Cache] All Phase 0 caches cleared")

    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Return cache sizes for monitoring."""
        return {
            "era_context_entries": len(cls._era_context_cache),
            "character_style_entries": len(cls._character_style_cache),
        }

    def _compute_adaptive_max_turns(self, base_max_turns: int = 22) -> int:
        """Module 8: Compute adaptive max turns based on AM weight + event importance.
        
        Uses AutobiographicalMemoryDistributionModel weight as primary driver,
        with event importance as secondary factor.
        
        Returns:
            Adjusted max_turns value.
        """
        # Primary: AM weight (from event dict, passed by P2)
        am_weight = self.event.get("am_weight", None)
        if am_weight is not None:
            # AM weight drives TelescopingMode selection
            if am_weight >= 0.7:
                return base_max_turns  # deep mode
            elif am_weight >= 0.35:
                return max(int(base_max_turns * 0.72), 12)  # standard mode
            else:
                return max(int(base_max_turns * 0.44), 8)  # telescoped mode

        # Fallback: event importance (backward compat when AM weight not available)
        VALUE_SCORES = {
            "critical": 1.0, "high": 0.8, "medium": 0.6,
            "low": 0.4, "minimal": 0.2,
        }
        value_target = self.event.get("value_for_target", "medium")
        value_period = self.event.get("value_for_period", "medium")
        importance = max(
            VALUE_SCORES.get(value_target, 0.6),
            VALUE_SCORES.get(value_period, 0.6),
        )

        if importance >= 0.8:
            return base_max_turns
        elif importance >= 0.6:
            return max(int(base_max_turns * 0.65), 10)
        else:
            return max(int(base_max_turns * 0.4), 8)

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        persona_pool: Dict[str, Any],
        memory_base: Dict[str, Any],
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
        high_res_event: Dict[str, Any],
        temperature_config: Optional[Dict[str, float]] = None,
        style_gen_config: Optional[Dict[str, Any]] = None,
        retriever=None,
        performance_tracker: Optional[PerformanceTracker] = None,
        run_id: str = "",
        event_id: str = "",
        period_id: str = "",
        pre_selected_participants: Optional[List[str]] = None,
    ):
        self.llm = llm_client
        self.persona_pool = persona_pool
        self.memory_base = memory_base
        self.life_plan = life_plan
        self.persona_config = persona_config
        self.retriever = retriever
        self.event = high_res_event
        self._performance_tracker = performance_tracker
        self._run_id = run_id or high_res_event.get("run_id", "")
        self._event_id = event_id or high_res_event.get("event_id", "")
        self._period_id = period_id or high_res_event.get("period_id", "")
        self._pre_selected_participants = pre_selected_participants  # M4+M5-e: Skip Step 1/2 if provided

        # Temperature configuration (allow override)
        self._temperatures = {**DEFAULT_TEMPERATURES}
        if temperature_config:
            self._temperatures.update(temperature_config)

        # Style generation concurrency configuration
        self._style_gen_config = {**DEFAULT_STYLE_GEN_CONFIG}
        if style_gen_config:
            self._style_gen_config.update(style_gen_config)

        # Pre-generated context caches (populated in run_simulation Phase 0)
        self._era_context: Optional[EraContextGuide] = None
        self._character_style_guides: Dict[str, CharacterStyleGuide] = {}

        # Build participant lookup: participant_id -> participant data
        self._participants: Dict[str, Dict[str, Any]] = {}
        raw_participants = persona_pool.get("participants", [])
        if isinstance(raw_participants, dict):
            participant_iterable = raw_participants.values()
        else:
            participant_iterable = raw_participants
        for p in participant_iterable:
            if not isinstance(p, dict):
                continue
            pid = p.get("participant_id", "")
            if pid:
                self._participants[pid] = p

        # Compute event year and character ages
        self._event_year: Optional[int] = None
        self._event_date_str: str = ""
        try:
            duration = self.event.get("duration", {})
            self._event_date_str = duration.get("start_date", "")
            if self._event_date_str:
                self._event_year = int(self._event_date_str.split("-")[0])
        except (ValueError, IndexError):
            pass

        # Scene state
        self._interaction_turns: List[Dict[str, Any]] = []
        self._scene_outline: Optional[Dict[str, Any]] = None
        self._current_phase: str = "opening"
        self._completed_beats: List[int] = []  # indices of completed beats
        self._current_beat_index: int = 0
        self._scene_time: str = self.event.get("duration", {}).get("precise_start_time", "09:30:00")[:5]
        self._environment_state: Dict[str, Any] = {
            "ambient_details": [],
            "character_positions": {},
            "recent_environment_changes": [],
        }
        self._phase_history: List[str] = ["opening"]

        # ── SQE components (initialized in run_simulation) ──
        self._simulation_plan: Optional[Any] = None
        self._turn_directive_computer: Optional[Any] = None
        self._repetition_guard: Optional[Any] = None
        self._beat_tracker: Optional[Any] = None
        self._temporal_context: Optional[Any] = None

        # ── Key Life Path data (from P2) ──
        self._key_life_path_data: Optional[Dict[str, Any]] = None

    # ── Anti-moralization guard helpers (T30, T14) ─────────────────────────

    async def _llm_classify_negative_attitudes(
        self,
        items: List[Dict[str, Any]],
    ) -> _NegativeAttitudeClassification:
        """Classify each attitude as identity-core-negative (must not soften) or not.
        Cached on self._anti_moralization_cache for the lifetime of the simulator instance.
        """
        cached = getattr(self, "_anti_moralization_cache", None)
        if cached is not None:
            return cached

        persona_brief = (self.persona_config.get("persona_brief_text") or "").strip()
        import json as _json
        items_json = _json.dumps(items, ensure_ascii=False)

        user_prompt = (
            "Classify each attitude below as identity-core-negative or not, per the schema. "
            "Use the persona brief as context for what 'identity-core' means for this specific "
            "person. Return one item per input attitude, in the same order. English only; each "
            "rationale at most 30 words.\n\n"
            f"## Persona brief\n{persona_brief}\n\n"
            f"## Attitudes (JSON list)\n{items_json}"
        )
        try:
            result: _NegativeAttitudeClassification = await self.llm.generate_structured(
                prompt=user_prompt,
                system_prompt=(
                    "You are a persona-simulation consistency analyst. Only flag an attitude as "
                    "identity-core-negative when the persona brief makes it central to who the "
                    "character is. When in doubt, return False to allow natural evolution."
                ),
                response_model=_NegativeAttitudeClassification,
                temperature=0.0,
                task_type="p3_attitude_classification",
            )
        except Exception as e:
            logger.warning(f"[MultiResSimulator::HighRes] _llm_classify_negative_attitudes failed: {e}; defaulting to empty classification.")
            result = _NegativeAttitudeClassification(attitudes=[])
        self._anti_moralization_cache = result
        return result

    async def _anti_moralization_guard(self) -> str:
        """Return a system-prompt suffix that prevents the LLM from softening
        or redeeming identity-core negative persona attitudes.

        Delegates "is this attitude identity-core?" to an LLM classifier whose output is cached on
        self._anti_moralization_cache. Returns an empty string when no identity-core negative
        attitudes are found — in that case the three call sites append nothing.
        """
        ext = self.persona_config.get("persona_extensions") or {}
        attitudes_block = ext.get("specific_attitudes") or {}
        items = attitudes_block.get("items", []) if isinstance(attitudes_block, dict) else []
        items = [it for it in items if isinstance(it, dict) and it.get("topic")]
        if not items:
            return ""

        classification = await self._llm_classify_negative_attitudes(items)
        locked_topics = [
            c.topic for c in classification.attitudes
            if c.is_identity_core_negative
        ]
        if not locked_topics:
            return ""
        joined = ", ".join(locked_topics)
        return (
            "\n\n## ⚠️ Character Consistency Guard\n"
            f"This character has identity-core negative attitudes toward: {joined}.\n"
            "Do NOT introduce redemptive arcs, moral growth, or softening of these specific "
            "attitudes. These are FIXED identity markers, not flaws to be resolved. Simulate the "
            "character as they ARE, not as they 'should' be. Do NOT add moments of doubt, regret, "
            "or growth regarding these listed topics. (Other attitudes not in this list may evolve "
            "naturally over the course of the simulation.)"
        )

    def _track_perf(self, name: str, **tags: Any):
        """Performance tracking helper with auto-injected run/event/period IDs."""
        tracker = self._performance_tracker
        if self._run_id and "run_id" not in tags:
            tags["run_id"] = self._run_id
        if self._event_id and "event_id" not in tags:
            tags["event_id"] = self._event_id
        if self._period_id and "period_id" not in tags:
            tags["period_id"] = self._period_id
        if tracker:
            return tracker.track(name, **tags)
        return nullcontext()

    # ── Helper: Character info & age ─────────────────────────────

    def _get_participant_info(self, pid: str) -> Dict[str, Any]:
        """Get participant data by ID, with fallback."""
        return self._participants.get(pid, {
            "participant_id": pid,
            "persona_name_text": f"Unknown ({pid})",
            "persona_brief_text": "",
            "current_persona_brief_text": "",
            "relationship_towards_the_main_character": "unknown",
        })

    def _get_participant_name(self, pid: str) -> str:
        info = self._get_participant_info(pid)
        return info.get("persona_name_text", pid)

    def _get_character_age(self, pid: str) -> Optional[int]:
        """Compute character's precise age at the time of the event.

        Uses compute_exact_age_safe() which considers whether birthday has passed.
        Falls back to year-difference if only birth year is available.
        """
        from lifelong_synth.configs.temporal_context import compute_exact_age_safe

        if not self._event_date_str:
            return None
        info = self._get_participant_info(pid)
        dob = info.get("date_of_birth") or {}
        if not isinstance(dob, dict):
            # Handle Pydantic model or other non-dict types
            try:
                dob = dob.model_dump() if hasattr(dob, "model_dump") else dict(dob)
            except Exception:
                dob = {}

        # Try full date first
        birth_year = dob.get("year")
        birth_month = dob.get("month")
        birth_day = dob.get("day")

        if birth_year and birth_month and birth_day:
            birth_date_str = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"
            age, _ = compute_exact_age_safe(birth_date_str, self._event_date_str)
            return age
        elif birth_year:
            age, _ = compute_exact_age_safe(
                None, self._event_date_str, fallback_birth_year=birth_year
            )
            return age
        return None

    def _get_age_speech_style(self, pid: str) -> str:
        """
        Get personalized speech style guidance for a participant.

        REFACTORED in v2: Returns the LLM-generated style guide from cache
        instead of hardcoded age-bracket rules.
        """
        guide = self._character_style_guides.get(pid)
        if not guide:
            age = self._get_character_age(pid)
            if age is None:
                return ""
            return f"[Character aged {age}] Speak and act naturally according to the cognitive and expressive abilities of this age group."

        return (
            f"## Personalized Style Guide\n"
            f"### Speech Style\n{guide.speech_style}\n\n"
            f"### Behavioral Patterns\n{guide.behavioral_patterns}\n\n"
            f"### Emotional Expression\n{guide.emotional_expression}\n\n"
            f"### Interaction Style\n{guide.interaction_style}\n\n"
            f"### Forbidden Expressions (must never appear)\n{guide.forbidden_expressions}"
        )

    def _build_present_participants_text(self) -> str:
        """Build a text listing all participants currently in the scene."""
        participant_ids = self.event.get("participants", [])
        lines = []
        for pid in participant_ids:
            name = self._get_participant_name(pid)
            age = self._get_character_age(pid)
            relationship = self._get_participant_info(pid).get(
                "relationship_towards_the_main_character", ""
            )
            age_str = f", age {age}" if age else ""
            lines.append(f"- {name} ({pid}): {relationship}{age_str}")
        return "\n".join(lines) if lines else "No other participants"

    def _get_age_cognitive_constraints(self, age: Optional[int]) -> str:
        """
        Return age-specific cognitive and language constraints for the agent prompt.

        Based on developmental psychology (Piaget's stages) to ensure
        age-appropriate language and thought patterns.
        """
        if age is None:
            return ""

        if age <= 2:
            return (
                "\n[Toddler cognitive constraints (0-2 years)]\n"
                "- Can only produce single words or two-word phrases (e.g., 'mama', 'want', 'no')\n"
                "- No causal reasoning ability; only immediate sensory reactions\n"
                "- internal_thought can only be sensory impressions (e.g., 'bright', 'soft', 'warm'), max 5 words\n"
                "- Cannot use any compound sentences or conditional clauses"
            )
        elif age <= 3:
            return (
                "\n[Toddler cognitive constraints (2-3 years)]\n"
                "- Primarily simple short phrases, max 3-5 words (e.g., 'I want that', 'mama hold')\n"
                "- No causal reasoning; only immediate emotions and sensory reactions\n"
                "- Frequently uses wrong words, invents words, has disordered word order\n"
                "- internal_thought can only be sensory impressions and immediate emotions (e.g., 'scared', 'big', 'want mama'), max 10 words\n"
                "- Cannot use conditional clauses, rhetorical questions, or compound sentences\n"
                "- Refers to self by own name instead of 'I'"
            )
        elif age <= 4:
            return (
                "\n[Toddler cognitive constraints (3-4 years)]\n"
                "- Primarily simple sentences, max 5-8 words\n"
                "- Beginning to use simple 'because...so...' but often incorrectly\n"
                "- Attention span approximately 3-5 minutes before being attracted to new things\n"
                "- internal_thought primarily sensory impressions and immediate emotions, can have simple associations, max 15 words\n"
                "- Cannot have long causal reasoning chains (e.g., 'if I behave -> teacher will praise -> so I should keep doing this')\n"
                "- Frequently repeats what adults have said but doesn't fully understand the meaning\n"
                "- Mixes use of first person and own name"
            )
        elif age <= 6:
            return (
                "\n[Young child cognitive constraints (4-6 years)]\n"
                "- Can form compound sentences but logic often jumps around\n"
                "- Attention span approximately 5-8 minutes\n"
                "- internal_thought can have simple causality (e.g., 'teacher praised me, happy'), max 20 words\n"
                "- Cannot have adult-style metacognition (e.g., 'I realize that...')\n"
                "- Still many grammatical errors and invented vocabulary\n"
                "- Thinking is fragmented and jumpy; cannot sustain long coherent reasoning"
            )
        elif age <= 9:
            return (
                "\n[Child cognitive constraints (6-9 years)]\n"
                "- Sentences more complete, but colloquial speech still has many grammar errors\n"
                "- Beginning concrete operational thinking, but abstract reasoning limited\n"
                "- internal_thought can have simple planning and causal reasoning, max 30 words\n"
                "- Cannot use adult-register written-style expressions"
            )
        elif age <= 12:
            return (
                "\n[Child cognitive constraints (9-12 years)]\n"
                "- Language ability approaching adult level but still colloquial characteristics\n"
                "- Beginning abstract thinking ability\n"
                "- internal_thought can have more complex reasoning, but still primarily concrete"
            )
        else:
            return ""

    # ── Phase 0: Pre-generation Methods ──────────────────────────

    async def generate_era_context(self) -> EraContextGuide:
        """
        [Step 1] Use LLM to generate era and cultural context based on event metadata.
        Called once at simulation start, result cached in self._era_context.

        Replaces hardcoded constraints like "real environment of Chinese elementary school in 2003".
        """
        duration = self.event.get("duration", {})
        event_date = duration.get("start_date", "unknown")
        event_time = duration.get("precise_start_time", "unknown")

        # Location info from persona config — stage-aware:
        # use current_living_location for work stages, growing_up_location for childhood/education
        period_meta = self.event.get("period", {}) or {}
        _is_work = period_meta.get("is_work_stage", False)
        _is_edu  = period_meta.get("is_education_stage", False)
        if _is_work and not _is_edu:
            _loc_src = self.persona_config.get("current_living_location") or \
                       self.persona_config.get("growing_up_location", {})
        else:
            _loc_src = self.persona_config.get("growing_up_location", {})
        location_parts = [
            _loc_src.get("country", ""),
            _loc_src.get("province", ""),
            _loc_src.get("city", ""),
        ]
        location_str = ", ".join(filter(None, location_parts)) or "unknown location"

        # Participant age summary
        participant_ages = []
        for pid in self.event.get("participants", []):
            name = self._get_participant_name(pid)
            age = self._get_character_age(pid)
            if age:
                participant_ages.append(f"{name} (age {age})")

        system_prompt = (
            "You are a historical and cultural consultant and scene design expert.\n"
            "Based on the given time, location, and event information, generate a detailed "
            "era background and cultural context guide.\n\n"
            "Requirements:\n"
            "1. All content must be based on real historical and cultural knowledge\n"
            "2. Details must be specific enough to be used directly in scene descriptions\n"
            "3. Pay special attention to technology levels — what existed, what did not\n"
            "4. Account for regional differences — the same era can be very different across locations\n"
            "5. IMPORTANT: Accurately reflect the SPECIFIC country and city provided. "
            "Do NOT default to Chinese or American cultural norms unless explicitly specified. "
            "A scene set in Auckland, New Zealand in 2005 should reflect New Zealand culture, "
            "not Chinese or American culture.\n"
            "6. Keep each field to 80-150 words; be concise and useful"
        )

        user_prompt = (
            f"Generate an era background and cultural context guide for the following scene:\n\n"
            f"## Basic Information\n"
            f"- Date: {event_date}\n"
            f"- Time: {event_time}\n"
            f"- Location/cultural background: {location_str}\n"
            f"- Event year: {self._event_year or 'unknown'}\n\n"
            f"## Event Summary\n"
            f"- Event description: {self.event.get('summary', 'unknown')}\n"
            f"- Event motivation: {self.event.get('motivation', 'unknown')}\n\n"
            f"## Participant Ages\n"
            f"- {', '.join(participant_ages) if participant_ages else 'unknown'}\n\n"
            f"## Life Stage Background\n"
            f"{self._get_period_context()}\n\n"
            f"Generate a complete era background guide. Ensure all cultural details accurately "
            f"reflect {location_str} in {self._event_year or 'the given year'}, not a generic "
            f"or different country's cultural context."
        )

        try:
            era_context: EraContextGuide = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=EraContextGuide,
                system_prompt=system_prompt,
                task_type="era_context",
                temperature=self._temperatures.get("era_context", 0.5),
            )
            self._era_context = era_context
            logger.info("[Step 1] Era context generated successfully")
            return era_context
        except Exception:
            logger.exception("[Step 1] Era context generation failed; using fallback context")
            # Minimal fallback — still provides useful context
            self._era_context = EraContextGuide(
                technology_constraints=(
                    f"The scene takes place in {location_str} in {self._event_year or 'unknown year'}. "
                    "Infer appropriate technology level for this era and location."
                ),
                social_norms="Infer appropriate social norms based on the event background and location.",
                cultural_details="Infer appropriate cultural details based on the event background and location.",
                physical_environment_hints="Infer appropriate physical environment details based on the event description.",
                anachronism_warnings="Avoid including objects or behaviors inconsistent with this era and location.",
                era_specific_flavor="Add details characteristic of this era and location.",
            )
            return self._era_context

    def _format_key_life_path_for_prompt(self) -> str:
        """Format key life path anchors for prompt injection.

        Injects deterministic life path facts (education stage, exams, key events)
        between TemporalContext and EraContextGuide in all 6 prompt locations.
        This ensures every LLM call in the simulation loop respects the
        character's actual life path.
        """
        if not self._key_life_path_data:
            return ""

        life_path = self._key_life_path_data
        parts = []
        parts.append("## 🎯 Key Life Path Anchors (Definitive Facts, HIGHEST PRIORITY)")

        # Education status
        edu = life_path.get("education_status", {})
        if edu:
            grade_label = edu.get("grade_label", "")
            institution = edu.get("institution", "")
            research_dir = edu.get("research_direction")

            edu_line = f"- Current education stage: {grade_label}"
            if institution:
                edu_line += f" ({institution})"
            if research_dir:
                edu_line += f", research direction: {research_dir}"
            parts.append(edu_line)

        # Exams
        exams = life_path.get("exams_this_year", [])
        if exams:
            parts.append("- Important exams this year:")
            for exam in exams:
                parts.append(
                    f"  · {exam.get('exam_name', '')} ({exam.get('date_range', '')}): "
                    f"{exam.get('description', '')}"
                )

        # Key events
        events = life_path.get("key_events", [])
        if events:
            parts.append("- Key life events:")
            for evt in events:
                transition_tag = " ⚡stage transition" if evt.get("is_stage_transition") else ""
                parts.append(
                    f"  · {evt.get('event_name', '')} ({evt.get('date', '')}): "
                    f"{evt.get('description', '')}{transition_tag}"
                )

        # Career status
        career = life_path.get("career_status", {})
        if career and career.get("is_working"):
            career_desc = (
                f"- Career status: {career.get('job_title', '')}, "
                f"{career.get('employer', '')} ({career.get('industry', '')})"
            )
            if career.get("description"):
                career_desc += f"\n  {career['description']}"
            parts.append(career_desc)

        # Pathway notes
        notes = life_path.get("pathway_notes", "")
        if notes:
            parts.append(f"- Path notes: {notes}")

        parts.append(
            "⚠️ The above are definitive factual anchors. All behaviors, dialogue, and events "
            "in the scene MUST strictly conform to these facts. "
            "Do NOT include plot elements inconsistent with the above path (e.g., a character who "
            "has already graduated cannot appear in a student context for that grade; a direct-PhD "
            "student cannot have master's-related plot elements; exams cannot be completed before "
            "their scheduled date)."
        )

        return "\n".join(parts)

    def _format_era_context_for_prompt(self) -> str:
        """Format temporal context + life path + era context for prompt injection.

        Information hierarchy (by priority):
        1. TemporalContext — system-computed temporal facts (age, grade, semester)
        2. Key Life Path — deterministic personal facts (exams, key events, stage transitions)
        3. EraContextGuide — society/culture background (technology, norms, environment)

        This order ensures personal path facts sit between system-computed temporal info
        and society-level era context, creating a natural information flow.
        """
        parts = []

        # Part 1: Temporal context (system-computed, highest priority)
        if self._temporal_context:
            parts.append(self._temporal_context.format_for_prompt(language="en"))

        # Part 2: Key Life Path (deterministic personal facts, HIGHEST PRIORITY for facts)
        life_path_text = self._format_key_life_path_for_prompt()
        if life_path_text:
            parts.append(life_path_text)

        # Part 3: Era context (society/culture background)
        if self._era_context:
            ctx = self._era_context
            parts.append(
                f"## Era Background & Cultural Context (auto-generated from event metadata)\n"
                f"### Technology Constraints\n{ctx.technology_constraints}\n\n"
                f"### Social Norms\n{ctx.social_norms}\n\n"
                f"### Cultural Details\n{ctx.cultural_details}\n\n"
                f"### Physical Environment\n{ctx.physical_environment_hints}\n\n"
                f"### Anachronism Warnings (must never appear)\n{ctx.anachronism_warnings}\n\n"
                f"### Era-Specific Flavor\n{ctx.era_specific_flavor}"
            )

        return "\n\n".join(parts) if parts else ""

    def _format_persona_extensions_for_prompt(self) -> str:
        """Format persona_extensions as a human-readable section for prompt injection.

        Uses the benchmark-agnostic formatter from persona_extensions_formatter.
        Returns empty string if persona_extensions is missing or empty.
        """
        extensions = self.persona_config.get("persona_extensions")
        if not extensions:
            return ""
        text = format_persona_extensions(extensions, stage="simulation")
        if not text:
            return ""
        # Add specific_attitudes guidance if present
        attitude_hint = ""
        if extensions.get("specific_attitudes"):
            attitude_hint = (
                "\n> **Attitude Guidance**: The 'Specific Attitudes' above are confirmed character traits. "
                "Reflect them naturally in the character's reactions and choices during this scene."
            )
        return f"\n{text}{attitude_hint}"

    def _format_milestone_constraint_for_prompt(self) -> str:
        """Format milestone skeleton constraints for scene outline prompt injection.

        If the high-res event has an attached milestone_skeleton from P2,
        inject its turning_point_hint, emotional_arc_hint, and mandatory_beats
        as hard constraints for the scene outline.
        """
        milestone_skel = self.event.get("milestone_skeleton")
        if not milestone_skel:
            return ""

        ms_name = milestone_skel.get("milestone_name", "")
        ms_turning = milestone_skel.get("turning_point_hint", "")
        ms_emotional = milestone_skel.get("emotional_arc_hint", "")
        ms_beats = milestone_skel.get("mandatory_beats", [])
        ms_value = milestone_skel.get("value_for_target", "")

        beats_text = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(ms_beats)) if ms_beats else "  (none)"

        return f"""
## 🎯 Pre-planned Milestone Constraints (MUST FOLLOW)
This event is a pre-planned key life milestone: **{ms_name}**

### Turning Point (must occur in the scene)
{ms_turning}

### Emotional Arc (scene must follow)
{ms_emotional}

### Mandatory Beats (each must be reflected in key_beats)
{beats_text}

### Value for the Protagonist
{ms_value}

**IMPORTANT**: Your key_beats MUST cover all mandatory beats listed above. You may add additional detail beats, but you cannot omit any mandatory beat.
"""

    def _format_turn_directive_for_prompt(self, turn_count: int) -> str:
        """Compute and format TurnDirective for injection into director prompt.

        Returns an empty string if TurnDirectiveComputer is not available.
        """
        if not self._turn_directive_computer:
            return ""
        try:
            # Gather state for TurnDirective computation
            current_beat_index = (
                self._beat_tracker.current_index if self._beat_tracker else 0
            )
            completed_beats = (
                self._beat_tracker.completed if self._beat_tracker else []
            )
            recent_turns = self._interaction_turns[-5:] if self._interaction_turns else []
            forbidden_patterns = (
                self._repetition_guard.detected_patterns
                if self._repetition_guard else []
            )

            directive = self._turn_directive_computer.compute(
                turn_index=turn_count,
                current_beat_index=current_beat_index,
                completed_beats=completed_beats,
                recent_turns=recent_turns,
                forbidden_patterns=forbidden_patterns,
            )
            return directive.format_for_prompt(language="en")
        except Exception as e:
            logger.warning(f"[SQE] Failed to compute TurnDirective: {e}")
            return ""

    async def _generate_single_character_style(
        self,
        pid: str,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[str, CharacterStyleGuide]:
        """
        Generate style guide for a single character with concurrency control
        and exponential backoff retry.

        Args:
            pid: Participant ID
            semaphore: Shared semaphore for concurrency limiting

        Returns:
            Tuple of (participant_id, CharacterStyleGuide)
        """
        info = self._get_participant_info(pid)
        name = info.get("persona_name_text", pid)
        age = self._get_character_age(pid)
        relationship = info.get("relationship_towards_the_main_character", "")
        persona_brief = (
            info.get("current_persona_brief_text", "")
            or info.get("persona_brief_text", "")
        )

        # Get recent memories for context
        memories = self._get_relevant_memories(pid, max_count=2)
        memories_text = "\n".join(memories) if memories else "None"

        target_name = self.persona_config.get("persona_name_text", "protagonist")
        era_context_text = self._format_era_context_for_prompt()

        system_prompt = (
            "You are a character design expert and behavioral psychology consultant.\n"
            "Your task is to generate a highly personalized style guide based on the character's complete attributes.\n\n"
            "Core principles:\n"
            "1. The style must be fully consistent with the character's age, personality, and cultural background\n"
            "2. Speech style must be specific enough to imitate directly — must provide example phrases\n"
            "3. Behavioral patterns must be specific enough to describe directly — give concrete actions\n"
            "4. Forbidden expressions must be clear — what this person would absolutely never say\n"
            "5. All content must conform to the era/cultural context constraints\n"
            "6. Memory voice must reflect this person's cognitive level and expression habits"
        )

        user_prompt = (
            f"Generate a personalized style guide for the following character.\n\n"
            f"## Event Context\n"
            f"- Event: {self.event.get('summary', 'unknown')}\n"
            f"- Protagonist: {target_name}\n\n"
            f"## Era and Cultural Context\n"
            f"{era_context_text if era_context_text else 'Infer from event information'}\n\n"
            f"## Character Information\n"
            f"- Name: {name}\n"
            f"- Age: {age or 'unknown'}\n"
            f"- Relationship to protagonist: {relationship}\n"
            f"- Persona brief: {persona_brief}\n"
            f"- Recent memories: {memories_text}\n\n"
            f"## Requirements\n"
            f"- speech_style must include 2-3 example phrases this person would say (in quotes)\n"
            f"- behavioral_patterns must include 2-3 specific body language/micro-gestures\n"
            f"- forbidden_expressions must explicitly list the types of things this person would absolutely never say\n"
            f"- memory_voice should reflect this person's cognitive level and expression habits"
        )

        config = self._style_gen_config
        max_retries = config["max_retries"]
        base_delay = config["base_retry_delay"]
        backoff_factor = config["retry_backoff_factor"]

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    guide: CharacterStyleGuide = await self.llm.generate_structured(
                        prompt=user_prompt,
                        response_model=CharacterStyleGuide,
                        system_prompt=system_prompt,
                        task_type="character_style",
                        temperature=self._temperatures.get("character_style", 0.6),
                    )
                    logger.info(
                        f"[Step 2] Style guide generated for {name} ({pid})"
                        f"{f' [retry {attempt}]' if attempt > 0 else ''}"
                    )
                    return (pid, guide)

            except Exception as e:
                delay = base_delay * (backoff_factor ** attempt)
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[Step 2] Style generation failed for {name} ({pid}), "
                        f"attempt {attempt + 1}/{max_retries}, "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        f"[Step 2] Style generation failed for {name} ({pid}) "
                        f"after {max_retries} attempts; using fallback style guide"
                    )

        # All retries exhausted — return fallback
        return (pid, self._build_fallback_style_guide(pid))

    async def generate_character_styles_parallel(
        self, participant_ids: Optional[List[str]] = None,
    ) -> Dict[str, CharacterStyleGuide]:
        """
        [Step 2] Generate personalized style guides for ALL participants
        using parallel LLM calls with concurrency control.

        Key design:
        - Each character gets its own dedicated LLM call (better quality)
        - asyncio.Semaphore limits concurrent calls (API protection)
        - Exponential backoff handles rate limiting
        - Failed characters get fallback guides (graceful degradation)

        Called once at simulation start, results cached in
        self._character_style_guides[participant_id].

        :param participant_ids: Optional list of participant IDs to generate styles for.
                                If None, generates for all event participants.
                                Used by Module 7 caching to only generate missing styles.
        """
        if participant_ids is None:
            participant_ids = self.event.get("participants", [])

        if not participant_ids:
            logger.warning("[Step 2] No participants found, skipping style generation")
            self._character_style_guides = {}
            return {}

        max_concurrent = self._style_gen_config["max_concurrent"]
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"[Step 2] Generating style guides for {len(participant_ids)} characters "
            f"(max_concurrent={max_concurrent})"
        )

        # Launch all character style generations in parallel
        tasks = [
            self._generate_single_character_style(pid, semaphore)
            for pid in participant_ids
        ]

        results: List[Tuple[str, CharacterStyleGuide]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        # Collect results into dict
        guides = {}
        success_count = 0
        fallback_count = 0
        for pid, guide in results:
            guides[pid] = guide
            # Check if it's a fallback (heuristic: fallback guides have generic content)
            if "Speak naturally according to" in guide.speech_style:
                fallback_count += 1
            else:
                success_count += 1

        logger.info(
            f"[Step 2] Style generation complete: "
            f"{success_count} success, {fallback_count} fallback"
        )

        self._character_style_guides = guides
        return guides

    def _build_fallback_style_guide(self, pid: str) -> CharacterStyleGuide:
        """Build a minimal fallback style guide when LLM generation fails."""
        age = self._get_character_age(pid)
        name = self._get_participant_name(pid)
        age_hint = f"age {age}" if age else "unknown age"
        return CharacterStyleGuide(
            speech_style=f"Speak naturally according to {name} ({age_hint})'s character attributes",
            behavioral_patterns="Act naturally according to the character's attributes",
            emotional_expression="Express emotions naturally according to character attributes",
            interaction_style="Interact naturally according to character relationships",
            forbidden_expressions="Avoid expressions inconsistent with character's age and identity",
            memory_voice="Recall naturally according to character traits",
        )

    def _validate_pregen_results(self) -> Dict[str, Any]:
        """
        [Step 3] Validate pre-generation results and report quality metrics.

        Checks:
        1. Era context exists and has non-empty fields
        2. All participants have style guides
        3. Style guides have substantive content (not just fallback)
        """
        report = {
            "era_context_valid": False,
            "total_participants": 0,
            "style_guides_generated": 0,
            "style_guides_fallback": 0,
            "quality_score": 0.0,
            "warnings": [],
        }

        # Validate era context
        if self._era_context:
            era_fields = [
                self._era_context.technology_constraints,
                self._era_context.social_norms,
                self._era_context.cultural_details,
                self._era_context.physical_environment_hints,
                self._era_context.anachronism_warnings,
                self._era_context.era_specific_flavor,
            ]
            non_empty = sum(1 for f in era_fields if f and len(f) > 20)
            report["era_context_valid"] = non_empty >= 4
            if non_empty < 4:
                report["warnings"].append(
                    f"Era context quality low: only {non_empty}/6 fields are substantive"
                )
        else:
            report["warnings"].append("Era context is None — using fallback")

        # Validate character style guides
        participant_ids = self.event.get("participants", [])
        report["total_participants"] = len(participant_ids)

        for pid in participant_ids:
            guide = self._character_style_guides.get(pid)
            if not guide:
                report["warnings"].append(f"Missing style guide for {pid}")
                continue

            if "Speak naturally according to" in guide.speech_style:
                report["style_guides_fallback"] += 1
            else:
                report["style_guides_generated"] += 1

        # Calculate quality score
        total = report["total_participants"]
        if total > 0:
            era_score = 1.0 if report["era_context_valid"] else 0.3
            style_score = report["style_guides_generated"] / total
            report["quality_score"] = era_score * 0.3 + style_score * 0.7

        # Log report
        logger.info(
            f"[Step 3] Validation report: "
            f"era={'✓' if report['era_context_valid'] else '✗'}, "
            f"styles={report['style_guides_generated']}/{total} generated, "
            f"{report['style_guides_fallback']}/{total} fallback, "
            f"quality={report['quality_score']:.1%}"
        )
        for warning in report["warnings"]:
            logger.warning(f"[Step 3] {warning}")

        return report

    def _build_fallback_outline(self) -> Dict[str, Any]:
        """Build a generic fallback scene outline from event metadata."""
        summary = self.event.get("summary", "a daily event")
        motivation = self.event.get("motivation", "")
        outcome = self.event.get("outcome", "")

        setting = f"Scene: {summary}"

        beats = []
        if motivation:
            beats.append(f"Event trigger: {motivation}")
        beats.append("Event begins to unfold")
        beats.append("Main interaction occurs")
        beats.append("Key turning point")
        if outcome:
            beats.append(f"Event outcome: {outcome}")
        else:
            beats.append("Event wraps up")
        while len(beats) < 5:
            beats.insert(-1, "Further development")

        return {
            "scene_setting": setting,
            "opening_action": "Scene begins",
            "key_beats": beats,
            "expected_turn_count": 15,
            "emotional_arc": "calm → engaged → tense → resolved",
            "time_markers": [],
        }

    def _get_relevant_memories(self, pid: str, max_count: int = 5) -> List[str]:
        """Get recent interaction memories for a participant, filtered by relevance."""
        info = self._get_participant_info(pid)
        history = info.get("interactions_history_with_the_main_character", [])

        # Filter: prefer memories from the same or adjacent life periods
        event_period = self.event.get("period_id", "")
        relevant = []
        other = []
        for m in history:
            eid = m.get("event_id", "")
            if eid.startswith(event_period) or eid.startswith("INIT_MEM"):
                relevant.append(m)
            else:
                other.append(m)

        # Take relevant first, then fill with others
        selected = relevant[-max_count:]
        remaining = max_count - len(selected)
        if remaining > 0:
            selected = other[-remaining:] + selected

        return [
            f"[{m.get('date', '?')}] {m.get('summary', '')}"
            for m in selected
        ]

    def _get_event_context(self) -> str:
        """Build event context string."""
        duration = self.event.get("duration", {})
        return (
            f"Event ID: {self.event.get('event_id', '?')}\n"
            f"Date: {duration.get('start_date', '?')}\n"
            f"Time: {duration.get('precise_start_time', '?')} ~ {duration.get('precise_end_time', '?')}\n"
            f"Event summary: {self.event.get('summary', '')}\n"
            f"Event motivation: {self.event.get('motivation', '')}\n"
            f"Low-resolution context: {self.event.get('low_res_context', '')}\n"
            f"Expected outcome: {self.event.get('outcome', '')}\n"
            f"Value for protagonist: {self.event.get('value_for_target', '')}"
        )

    def _get_period_context(self) -> str:
        """Build life period context string."""
        period_id = self.event.get("period_id", "")
        for period in self.life_plan.get("life_periods", []):
            if period.get("period_id") == period_id:
                return (
                    f"Life stage: {period.get('title', '')}\n"
                    f"Theme: {period.get('dominant_theme', '')}\n"
                    f"Developmental tasks: {', '.join(period.get('developmental_tasks', []))}\n"
                    f"Stage goals: {', '.join(period.get('stage_goals', []))}\n"
                    f"Time range: {period.get('period_date_range', {}).get('start_date', '')} ~ "
                    f"{period.get('period_date_range', {}).get('end_date', '')}"
                )
        return f"Life stage: {period_id}"

    def _build_retriever_context(self) -> str:
        """Build memory context using the unified retriever API.

        Falls back to legacy memory_base dict access if retriever is not available.
        """
        if self.retriever is not None:
            from datetime import date as _date
            current_date = None
            if self._event_date_str:
                try:
                    current_date = _date.fromisoformat(self._event_date_str)
                except (ValueError, TypeError):
                    pass
            return self.retriever.build_context(
                mode="period_focused",
                current_date=current_date,
                period_id=self.event.get("period_id", ""),
            )
        # Legacy fallback: direct dict access
        import json as _json
        parts = []
        life_summary = self.memory_base.get('current_life_summary', 'None')
        if life_summary:
            parts.append(f"Current life summary: {life_summary}")
        period_id = self.event.get("period_id", "XX")
        related = [
            {"event_id": eid, "summary": edata.get("summary", "")}
            for eid, edata in self.memory_base.get("previous_events", {}).items()
            if eid.startswith(period_id)
        ]
        if related:
            parts.append(f"Previous related events:\n{_json.dumps(related, ensure_ascii=False, indent=2)}")
        return "\n".join(parts) if parts else "Memory bank is empty"

    def _build_turn_history_text(self, max_turns: int = 10) -> str:
        """Build a text representation of recent interaction turns with body language."""
        if not self._interaction_turns:
            return "(Scene just started, no interaction records yet)"

        recent = self._interaction_turns[-max_turns:]
        lines = []
        for turn in recent:
            name = turn.get("speaker_name", turn.get("speaker_id", "?"))
            action_type = turn.get("action_type", "speak")
            content = turn.get("content", "")
            body = turn.get("body_language", "")
            time_str = turn.get("scene_time", "")
            time_prefix = f"[{time_str}] " if time_str else ""

            if action_type == "speak":
                body_note = f" {body}" if body else ""
                lines.append(f"{time_prefix}{name}{body_note}：「{content}」")
            elif action_type == "act":
                lines.append(f"{time_prefix}[{name} {content}]")
            elif action_type == "think":
                lines.append(f"{time_prefix}({name} thinks: {content})")
            elif action_type == "observe":
                lines.append(f"{time_prefix}[{name} notices: {content}]")
            elif action_type == "narrate":
                lines.append(f"{time_prefix}【{content}】")
            elif action_type == "react":
                lines.append(f"{time_prefix}[Surrounding reaction: {content}]")
            else:
                lines.append(f"{time_prefix}{name}：{content}")

        return "\n".join(lines)

    def _get_beat_status_text(self) -> str:
        """Build a text showing which beats are completed and which remain."""
        if not self._scene_outline or not self._scene_outline.get("key_beats"):
            return "No beat plan"
        beats = self._scene_outline["key_beats"]
        lines = []
        for i, beat in enumerate(beats):
            status = "✅ Done" if i in self._completed_beats else "⬜ Pending"
            marker = "→ " if i == self._current_beat_index and i not in self._completed_beats else "  "
            lines.append(f"{marker}{i+1}. [{status}] {beat}")
        return "\n".join(lines)

    def _get_environment_state_text(self) -> str:
        """Build environment state summary."""
        parts = [f"Current scene time: {self._scene_time}"]
        if self._environment_state.get("ambient_details"):
            parts.append("Ambient details: " + "; ".join(self._environment_state["ambient_details"][-3:]))
        if self._environment_state.get("recent_environment_changes"):
            parts.append("Recent changes: " + "; ".join(self._environment_state["recent_environment_changes"][-2:]))
        return "\n".join(parts)

    # ── Step 1: Scene Outline Planning ───────────────────────────

    async def plan_scene_outline(self) -> Dict[str, Any]:
        """
        Environment director plans the overall scene structure.
        Generates concrete, time-anchored beats with sensory details.
        """
        participant_ids = self.event.get("participants", [])
        participant_descriptions = []
        for pid in participant_ids:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            age = self._get_character_age(pid)
            relationship = info.get("relationship_towards_the_main_character", "")
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            age_str = f", age {age} at the time" if age else ""
            participant_descriptions.append(
                f"- {name} ({pid}): {relationship}{age_str}\n  Persona: {brief}"
            )
        participants_text = "\n".join(participant_descriptions)

        target_name = self.persona_config.get("persona_name_text", "protagonist")
        target_age = self._get_character_age("P_TARGET")
        target_age_str = f" (age {target_age} at the time)" if target_age else ""

        duration = self.event.get("duration", {})
        start_time = duration.get("precise_start_time", "09:30:00")
        end_time = duration.get("precise_end_time", "15:45:00")

        system_prompt = (
            "You are the Screenwriter Agent of MemoryForge's High-Resolution Simulator.\n"
            "Your task is to plan a detailed scene outline for an Event-Specific Experience.\n\n"
            "[Core Principles — Be as real as a documentary]\n"
            "1. Scene settings must have five-sense details: what is seen, heard, smelled, felt\n"
            "2. All dialogue must be purely colloquial, like secretly recorded real conversation\n"
            "3. People of different ages/identities speak in completely different ways\n"
            "4. Include everyday 'filler talk' and trivial details — real life is not all meaningful\n"
            "5. Key beats must be specific to one action/one line, not abstract summaries\n"
            "6. Time markers must be realistic — consider how long each step actually takes\n"
            "7. Strictly forbidden: literary phrases like 'felt relieved', 'full of anticipation', 'boosted confidence'\n"
            "8. Include surprises and imperfections — things don't go perfectly in real life\n"
            "9. IMPORTANT: All dialogue, internal thoughts, and narration MUST be in English. "
            "rather than full non-English sentences."
            "9. IMPORTANT: All dialogue, internal thoughts, and narration MUST be in English. "
            "rather than full non-English sentences."
            "9. IMPORTANT: All dialogue, internal thoughts, and narration MUST be in English. "
            "rather than full non-English sentences."
            "\n[Narrative Coherence — MANDATORY]\n"
            "The scene outline MUST be coherent with the pre-confirmed event framework:\n"
            "- The scene setting, beats, and emotional arc must align with the 'Event summary' and 'Low-resolution context' in the Event Information above\n"
            "- The key beats must lead to the 'Expected outcome' stated in the Event Information\n"
            "- Do NOT invent a different event type, location, or plot direction than what is described in the Event Information\n"
        )

        user_prompt = f"""## Event Information
{self._get_event_context()}

## Time Range
From {start_time} to {end_time}

## Life Stage Background
{self._get_period_context()}

## Participants
{participants_text}

## Protagonist Information
- Name: {target_name}{target_age_str}
- Persona: {self.persona_config.get('persona_brief_text', '')}
{self._format_persona_extensions_for_prompt()}

## Memory Context
{self._build_retriever_context()}

## Task
Plan an extremely detailed, authentic scene outline for this high-resolution event.

Requirements:
1. scene_setting: Describe the specific physical scene using five senses (200-400 words)
   - Visual: spatial layout, lighting, arrangement of nearby objects, character positions
   - Auditory: environmental sounds (voices, object sounds, natural sounds)
   - Other: temperature, smells, tactile sensory details
   
2. opening_action: The first specific action of the scene (not a description — one person did what/said what)
   - Good: a specific person performs a specific action, with sensory details and casual speech
   - Bad: 'someone enters, announces the activity begins' (too abstract, lacks detail)

3. key_beats: 5-8 key narrative beats, each a specific micro-event
   - Good: '{target_name} gets stuck doing something, expression changes, nearby person notices' (specific to action/expression/sound)
   - Bad: '{target_name} encounters difficulty but ultimately overcomes it' (too abstract, lacks visual quality)
   - Must include: at least one small surprise/incident, at least one environmental change, at least one emotional turning point
   
4. expected_turn_count: Estimated interaction turns (recommended 15-22)

5. emotional_arc: Concrete emotional trajectory, connected with arrows

6. time_markers: Key time points (based on the time range {start_time} to {end_time})
   - Example: ['09:30 students gradually arrive', '09:45 start answering', '10:30 mid-break', '11:00 results announced']

{self._format_era_context_for_prompt()}
{self._format_milestone_constraint_for_prompt()}
Note:
- Protagonist {target_name} was {target_age} years old at the time; all behaviors must conform to the cognitive and expressive abilities of this age group
- Include real small incidents and surprises — things do not go perfectly in real life
- All scene details must strictly conform to the era background constraints above
"""

        logger.info("[HighRes::Screenwriter] Planning scene outline...")

        try:
            outline: SceneOutlinePlan = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=SceneOutlinePlan,
                system_prompt=system_prompt,
                task_type="scene_outline",
                temperature=self._temperatures["scene_outline"],
            )

            self._scene_outline = outline.model_dump()

            # Merge mandatory beats from milestone skeleton
            milestone_skel = self.event.get("milestone_skeleton")
            if milestone_skel and milestone_skel.get("mandatory_beats"):
                mandatory = milestone_skel["mandatory_beats"]
                existing_beats = self._scene_outline.get("key_beats", [])
                # Prepend mandatory beats that are not already covered
                merged_beats = list(mandatory)
                for b in existing_beats:
                    if b not in merged_beats:
                        merged_beats.append(b)
                self._scene_outline["key_beats"] = merged_beats
                logger.info(
                    f"[Director] Milestone mandatory beats injected: "
                    f"{len(mandatory)} mandatory + {len(existing_beats)} LLM = "
                    f"{len(merged_beats)} total beats"
                )

            # Initialize environment state from outline
            self._environment_state["ambient_details"] = [outline.scene_setting]

            logger.info(
                f"[Director] Scene outline planned: "
                f"{len(outline.key_beats)} beats, "
                f"{outline.expected_turn_count} expected turns, "
                f"{len(outline.time_markers)} time markers"
            )
            return self._scene_outline

        except Exception:
            logger.exception("[Director] Scene outline planning failed; using fallback outline")
            self._scene_outline = self._build_fallback_outline()
            return self._scene_outline

    # ── Step 1.5: Beat Narration Generation ─────────────────────

    async def generate_beat_narration(
        self,
        beat_index: int,
        beat_text: str,
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a single omniscient narrator paragraph for a beat.

        This narration covers:
        - Scene/environment setup or transition
        - Background actions of non-protagonist characters
        - Time passage and routine activities
        - Sensory atmosphere details

        It replaces multiple low-value turns (observe/react/narrate) with one
        rich paragraph, freeing the simulation loop to focus on the protagonist's
        direct speech, actions, and thoughts.

        Args:
            beat_index: Index of the current beat (0-based).
            beat_text: The beat description from the scene outline.
            is_opening: True if this is the very first beat (scene establishment).
            is_closing: True if this is the final beat (scene wrap-up).

        Returns:
            A turn_data dict with action_type="narrate".
        """
        target_name = self.persona_config.get("persona_name_text", "protagonist")

        # Build participant context for narration
        participant_ids = self.event.get("participants", [])
        participant_lines = []
        for pid in participant_ids:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            role = info.get("role", "unknown")
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            participant_lines.append(f"- {name} ({pid}): {role} — {brief[:80]}")
        participants_text = "\n".join(participant_lines)

        # Recent turns for context
        recent_turns_text = self._build_turn_history_text(max_turns=5)

        # Era context
        era_hints = ""
        if self._era_context:
            era_hints = (
                f"Technology: {self._era_context.technology_constraints}\n"
                f"Social norms: {self._era_context.social_norms}\n"
                f"Anachronism warning: {self._era_context.anachronism_warnings}"
            )

        total_beats = len(self._scene_outline.get("key_beats", [])) if self._scene_outline else 1
        scene_setting = self._scene_outline.get("scene_setting", "") if self._scene_outline else ""

        if is_opening:
            narration_role = (
                "This is the OPENING narration. Establish the scene vividly:\n"
                "- Describe the physical environment with sensory details (sight, sound, smell, touch)\n"
                "- Introduce the protagonist's starting position and initial state\n"
                "- Set the atmosphere and time of day\n"
                "- Mention other characters' presence naturally\n"
                "Length: 3-5 sentences."
            )
        elif is_closing:
            narration_role = (
                "This is the CLOSING narration. Wrap up the scene:\n"
                "- Describe how the scene winds down\n"
                "- Note the protagonist's final state or small action\n"
                "- Capture the lingering atmosphere\n"
                "Length: 2-3 sentences."
            )
        else:
            narration_role = (
                f"This is a TRANSITION narration for beat {beat_index + 1}/{total_beats}.\n"
                "Cover in ONE paragraph (3-5 sentences):\n"
                "- Time passage since the last beat (how much time elapsed, what happened in between)\n"
                "- Background activities of non-protagonist characters (what they were doing)\n"
                "- Any environmental changes (light, sound, atmosphere shifts)\n"
                "- The protagonist's routine actions leading into this beat\n"
                "Do NOT include the protagonist's key speech or decisions — those come in the simulation turns."
            )

        system_prompt = (
            "You are an omniscient narrator for a life simulation. "
            "Write in plain, documentary prose — no literary flourishes, no emotional editorializing. "
            "Describe what happened as if you were a fly on the wall with a camera. "
            "Use past tense. Be specific about objects, sounds, and small physical details. "
            "Forbidden: 'felt relieved', 'full of anticipation', 'boosted confidence', any abstract emotion summary. "
            "All narration must be in English. "
        )

        user_prompt = f"""## Event
{self._get_event_context()}

## Scene Setting
{scene_setting[:400] if scene_setting else 'See event context'}

## Current Beat ({beat_index + 1}/{total_beats})
{beat_text}

## Participants
{participants_text}

## What Just Happened (recent turns)
{recent_turns_text if recent_turns_text else 'Scene is just beginning.'}

## Current Scene Time
{self._scene_time}

## Era Context
{era_hints}

## Your Task
{narration_role}

Write a single narration paragraph. Plain prose, no bullet points, no headers.
Protagonist name: {target_name}
"""

        try:
            # Use a simple text generation (not structured) for narration
            # We'll use generate_structured with a simple wrapper model
            from pydantic import BaseModel as _BaseModel

            class _NarrationOutput(_BaseModel):
                narration: str = Field(
                    ...,
                    description=(
                        "The narration paragraph. 3-5 sentences, plain prose, past tense. "
                        "No markdown, no bullet points, no headers, no character name prefixes. "
                        "Output plain text only."
                    )
                )

            result = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=_NarrationOutput,
                system_prompt=system_prompt,
                task_type="beat_narration",
                temperature=self._temperatures.get("forced_resolution", 0.5),
            )
            narration_text = result.narration

        except Exception:
            logger.exception(f"[Narration] Beat {beat_index} narration generation failed; using fallback")
            narration_text = (
                f"{target_name} continued working on the task. "
                f"Time passed as the scene moved into the next phase."
            )

        turn_data = {
            "turn_index": len(self._interaction_turns),
            "speaker_id": "NARRATOR",
            "speaker_name": "narration",
            "action_type": "narrate",
            "content": narration_text,
            "body_language": "",
            "internal_thought": "",
            "emotional_state": "",
            "direction_hint": beat_text,
            "personalized_observation": "",
            "scene_time": self._scene_time,
            "scene_phase": self._current_phase,
            "timestamp": datetime.now().isoformat(),
        }
        self._interaction_turns.append(turn_data)

        logger.info(
            f"[Narration] Beat {beat_index + 1}: {narration_text[:100]}..."
            if len(narration_text) > 100
            else f"[Narration] Beat {beat_index + 1}: {narration_text}"
        )
        return turn_data

    # ── Step 2: Director Action Selection (with personalized observation) ──

    async def director_select_action(self) -> DirectorActionSelection:
        """
        Environment director selects the next actor and action type.
        
        Key improvement over basic version: generates a PERSONALIZED observation
        for the selected character (concordia's make_observation pattern).
        Each character sees the scene from their own vantage point.
        """
        participant_ids = self.event.get("participants", [])
        participant_info_lines = []
        for pid in participant_ids:
            name = self._get_participant_name(pid)
            age = self._get_character_age(pid)
            relationship = self._get_participant_info(pid).get("relationship_towards_the_main_character", "")
            age_str = f", age {age}" if age else ""
            # Put ID first and bold to encourage LLM to return pure ID
            participant_info_lines.append(f"- **{pid}** {name}: {relationship}{age_str}")

        turn_count = len(self._interaction_turns)
        expected_turns = (
            self._scene_outline.get("expected_turn_count", 15)
            if self._scene_outline else 15
        )

        # Determine phase based on progress
        progress_ratio = turn_count / max(expected_turns, 1)
        total_beats = len(self._scene_outline.get("key_beats", [])) if self._scene_outline else 5
        beats_done = len(self._completed_beats)

        system_prompt = (
            "You are the Modulator Agent of MemoryForge's High-Resolution Simulator.\n"
            "You evaluate the current scene state and decide the next action.\n"
            "Your core responsibilities:\n"
            "1. Select the next actor and action type to advance the current beat\n"
            "2. Generate a [personalized observation] for the selected character\n"
            "3. Focus exclusively on the PROTAGONIST (P_TARGET) and key dialogue moments\n\n"
            "[CRITICAL DESIGN PRINCIPLE]\n"
            "This simulation uses a BEAT-DRIVEN structure:\n"
            "- Each beat already has a narration (旁白) that covers background, transitions, and other characters' routine actions\n"
            "- Your job here is ONLY to generate the protagonist's direct speech/action/thought within this beat\n"
            "- Other characters only appear when they have KEY dialogue with the protagonist\n\n"
            "Available action types (ONLY these three):\n"
            "- speak: talking — the protagonist or a key character says something important\n"
            "- act: physical action — the protagonist does something concrete that advances the beat\n"
            "- think: internal monologue — P_TARGET only, reveals inner state at a key moment\n\n"
            "[Speaker Selection Rules]\n"
            "- DEFAULT: next_speaker_id = P_TARGET (protagonist speaks/acts/thinks)\n"
            "- Only choose a non-P_TARGET speaker when:\n"
            "  · The beat requires a specific response FROM another character to the protagonist\n"
            "  · A non-P_TARGET character initiates a key dialogue that triggers protagonist's reaction\n"
            "  · Maximum 1 non-P_TARGET turn per beat (then return to P_TARGET)\n\n"
            "[direction_hint requirements]\n"
            "Must be specific and colloquial — what exactly does this person say or do:\n"
            "✓ Good: 'mutters \"wait, that doesn't look right\" and leans closer to the screen'\n"
            "✓ Good: 'asks Jason directly: \"does this repo name make sense to you?\"'\n"
            "✗ Bad: 'express enthusiasm', 'demonstrate inner growth', 'react to the situation'\n\n"
            "[Personalized Observation]\n"
            "What does this specific character see/hear/feel right now from their position:\n"
            "- Include one concrete sensory detail (visual, sound, or touch)\n"
            f"- Must conform to era: {self._era_context.anachronism_warnings if self._era_context else 'avoid anachronisms'}\n\n"
            "[Scene Phase Rules]\n"
            "- opening: protagonist settles in, first action\n"
            "- development: protagonist works through the beat, key exchanges\n"
            "- turning_point: protagonist faces the key moment of this beat\n"
            "- resolution: protagonist reacts to outcome\n"
            "- closing: protagonist's final action/thought\n\n"
            "[should_end_scene]\n"
            "Set to true ONLY when: all beats are complete AND current phase is closing.\n\n"
            "[Narrative Coherence — MANDATORY]\n"
            "Every turn you generate MUST stay coherent with:\n"
            "1. The 'Event summary' and 'Low-resolution context' in the Event Information — do not contradict the established plot\n"
            "2. The current beat in '## Narrative Beat Progress' — each turn must advance the current beat, not skip or contradict it\n"
            "3. The 'Emotional arc' in '## Scene Outline (MUST FOLLOW)' — the protagonist's emotional trajectory must follow the arc\n"
            "If a beat or the event summary says X happens, do NOT generate turns where X does not happen.\n\n"
            "[Language Rule] All direction_hints must be in English. "
            "If a character would naturally speak in another language, "
            "render the hint in English with cultural flavor, not in the original language. "

        )
        system_prompt += await self._anti_moralization_guard()

        user_prompt = f"""## Event Information
{self._get_event_context()}

## Scene Outline (MUST FOLLOW — do not contradict)
- Scene setting: {self._scene_outline.get('scene_setting', 'unknown') if self._scene_outline else 'unknown'}
- Emotional arc: {self._scene_outline.get('emotional_arc', 'unknown') if self._scene_outline else 'unknown'}
- Time markers: {', '.join(self._scene_outline.get('time_markers', [])) if self._scene_outline else 'none'}
- Key event summary (anchor): {self.event.get('summary', self.event.get('refined_summary', ''))}

## Narrative Beat Progress
{self._get_beat_status_text()}

## Environment State
{self._get_environment_state_text()}

## Available Actors
{chr(10).join(participant_info_lines)}
""" + (f"""

### Character Behavioral Profile
{format_persona_extensions(self.persona_config.get('persona_extensions') or {}, stage='simulation', heading='', char_budget=1000)}
Ensure the character's dialogue and behavior are consistent with this profile.
If "Specific Attitudes" are listed, reflect them naturally when the scene topic overlaps with an attitude target (e.g., a character who hates a subject should show reluctance/avoidance when that subject comes up).
""" if self.persona_config.get('persona_extensions') else "") + f"""
## Current Progress
- Completed turns: {turn_count}/{expected_turns}
- Current phase: {self._current_phase}
- Completed beats: {beats_done}/{total_beats}
- Progress ratio: {progress_ratio:.0%}

## Recent Interaction History
{self._build_turn_history_text(max_turns=10)}

## Task
Select the next actor and generate a personalized scene observation for them.

Requirements:
1. next_speaker_id: MUST be a pure participant ID (e.g., P_TARGET, P_001, P_020), do NOT include names
2. action_type: ONLY speak / act / think (no narrate, no observe, no react)
3. direction_hint: specific, colloquial guidance — what exactly does this person say or do (1-2 sentences)
4. personalized_observation: what does this character see/hear/feel right now (1-2 sentences, one concrete sensory detail)
5. scene_phase: update based on progress ratio
6. current_beat_index: index of the beat currently being advanced
7. should_end_scene: true ONLY when all beats complete AND phase is closing
8. scene_time: current in-scene time
9. reasoning: brief explanation

Notes:
- DEFAULT speaker is P_TARGET — choose non-P_TARGET only for key dialogue moments
- Non-P_TARGET speakers: max 1 consecutive turn, then return to P_TARGET
- P_TARGET should have think actions at emotional turning points
- Scene phase must advance with progress; don't stay in opening indefinitely
- When all beats are complete, enter closing phase
- If the last 2 turns have the same speaker and similar content, MUST switch speaker or advance beat
- next_speaker_id MUST be in pure ID format (e.g., P_TARGET, P_001), absolutely do NOT write names

{self._format_turn_directive_for_prompt(turn_count)}

{self._format_era_context_for_prompt()}
"""

        try:
            selection: DirectorActionSelection = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=DirectorActionSelection,
                system_prompt=system_prompt,
                task_type="director_selection",
                temperature=self._temperatures["director_select"],
            )

            # Validate speaker_id with fuzzy matching
            if selection.next_speaker_id not in participant_ids:
                # Try fuzzy match: LLM may return "Zhang San (P_020)" instead of "P_020"
                matched_id = None
                raw = selection.next_speaker_id
                for pid in participant_ids:
                    if pid in raw or raw == self._get_participant_name(pid):
                        matched_id = pid
                        break
                if matched_id:
                    logger.info(f"[HighRes::Modulator] Fuzzy-matched speaker_id '{raw}' → '{matched_id}'")
                    selection.next_speaker_id = matched_id
                else:
                    logger.warning(
                        f"[Director] Invalid speaker_id '{raw}', "
                        f"falling back to round-robin"
                    )
                    # Round-robin instead of always first participant
                    idx = turn_count % len(participant_ids)
                    selection.next_speaker_id = participant_ids[idx]

            # SQE: Use SimulationPlan for hard phase progression (system-level, not LLM-dependent)
            if self._simulation_plan:
                forced_phase = self._simulation_plan.get_phase_for_turn(turn_count)
                if forced_phase != selection.scene_phase:
                    logger.info(
                        f"[SQE] Phase override: LLM said '{selection.scene_phase}' "
                        f"→ system forces '{forced_phase}' at turn {turn_count}"
                    )
                    selection.scene_phase = forced_phase
            else:
                # Fallback: original progress-based phase forcing
                if progress_ratio > 0.9 and selection.scene_phase not in ("closing",):
                    selection.scene_phase = "closing"
                elif progress_ratio > 0.75 and selection.scene_phase in ("opening", "development"):
                    selection.scene_phase = "resolution"
                elif progress_ratio > 0.5 and selection.scene_phase == "opening":
                    selection.scene_phase = "development"
                elif progress_ratio > 0.15 and selection.scene_phase == "opening":
                    selection.scene_phase = "development"

            self._current_phase = selection.scene_phase
            if selection.scene_phase not in self._phase_history:
                self._phase_history.append(selection.scene_phase)

            # Update scene time
            if selection.scene_time:
                self._scene_time = selection.scene_time

            # Track beat progression
            if selection.current_beat_index >= 0:
                self._current_beat_index = selection.current_beat_index

            logger.info(
                f"[HighRes::Modulator] Turn {turn_count + 1} [{self._scene_time}] "
                f"({selection.scene_phase}): "
                f"{self._get_participant_name(selection.next_speaker_id)} "
                f"({selection.action_type}) — {selection.direction_hint[:60]}"
            )
            return selection

        except Exception:
            logger.exception("[Director] Action selection failed; using fallback selection")
            idx = turn_count % len(participant_ids)
            return DirectorActionSelection(
                next_speaker_id=participant_ids[idx],
                action_type="speak",
                direction_hint="Continue advancing the scene",
                personalized_observation="",
                scene_phase=self._current_phase,
                current_beat_index=self._current_beat_index,
                should_end_scene=(turn_count >= expected_turns),
                scene_time=self._scene_time,
                reasoning="fallback selection",
            )

    # ── Step 3: Agent Action Generation ──────────────────────────

    async def agent_generate_action(
        self,
        speaker_id: str,
        action_type: str,
        direction_hint: str,
        personalized_observation: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an agent's action/response for the current turn.
        
        Key improvement: receives a personalized observation that tells the agent
        what they specifically see/hear/feel from their vantage point.
        """
        info = self._get_participant_info(speaker_id)
        name = info.get("persona_name_text", speaker_id)
        relationship = info.get("relationship_towards_the_main_character", "")
        current_brief = (
            info.get("current_persona_brief_text", "")
            or info.get("persona_brief_text", "")
        )
        age = self._get_character_age(speaker_id)
        age_style = self._get_age_speech_style(speaker_id)

        # Get relevant memories
        memories = self._get_relevant_memories(speaker_id, max_count=4)
        memories_text = "\n".join(memories) if memories else "No relevant memories"

        target_name = self.persona_config.get("persona_name_text", "protagonist")
        is_target = (speaker_id == "P_TARGET")

        # Action type instructions with concrete examples
        action_instructions = {
            "speak": (
                "Generate what this character would say right now.\n"
                "[Iron rule] Must be words this person would truly blurt out in real life:\n"
                "- Has natural filler words and verbal tics\n"
                "- Has catchphrases and habitual expressions\n"
                "- Sentences can be incomplete, repetitive, with '...' pauses\n"
                "- Can include 'you know', 'like', 'well...' type filler words\n"
                "- Absolutely no written-language expressions: 'I feel that', 'This makes me', 'I realize', 'filled with'"
            ),
            "act": (
                "Describe this character's physical actions right now.\n"
                "[Must use first-person perspective] — you are this character, describe what you are doing:\n"
                "✓ 'I lower my head and chew on the pencil eraser, brow furrowed, left hand absent-mindedly rubbing an eraser corner'\n"
                "✗ 'They lowered their head' (don't describe yourself in third person)\n"
                "✗ 'Thinking seriously about the problem'"
            ),
            "think": (
                "Generate this character's inner monologue right now.\n"
                "Like a stream of consciousness, fragmented thoughts going wherever they go:\n"
                "✓ 'Oh no oh no how do I do this question...wait, I think I practiced this before...no wait, that was addition...'\n"
                "✗ 'I felt a wave of anxiety, but then remembered my previous practice'"
            ),
            "observe": (
                "Describe one specific detail this character notices right now.\n"
                "[Must use first-person perspective] — seen through your eyes:\n"
                "✓ 'The girl with the ponytail in front has been shaking her leg, making the whole desk wobble'\n"
                "✗ 'The classmates around me are all focused on their work'\n"
                "✗ Don't describe yourself in third person; use 'I see...'"
            ),
            "narrate": (
                "Environmental narration, describing the scene in plain detail.\n"
                "Like a documentary voice-over, only describing what can be seen and heard:\n"
                "✓ 'The classroom is so quiet only the scratch of pen on paper remains; occasionally someone turns a page with a rustle'\n"
                "✗ 'The classroom is filled with a tense and focused atmosphere'"
            ),
            "react": (
                "Describe the natural reactions of the surrounding environment or bystanders.\n"
                "Background noise and small gestures in a real scene:\n"
                "✓ 'A guy in the back row sneezed; a few people nearby looked up then lowered their heads to keep writing'\n"
                "✗ 'The classmates expressed concern about this'"
            ),
        }

        age_str = f", age {age}" if age else ""
        age_line = f"- Age: {age}\n" if age else ""

        system_prompt = (
            f"You are playing \u300c{name}\u300d{age_str}, participating in a realistic life simulation scene.\n"
            f"You must speak and act exactly as this person would in real life, "
            f"BUT all dialogue and internal thoughts must be in English. "
            f"If the character would naturally use code-switching (e.g., Hinglish), "
            f"you may include occasional loan-words or cultural expressions in quotes, "
            f"but the overall sentence must be English. "
            f"Example: 'Bas—enough of this. From today, one format only, understand?' "
            f"NOT: 'Bas, bas—yeh roz ka hai. Aaj se ek hi format mein aayega, samjhe?'\n\n"
            f"## Character Information\n"
            f"- Name: {name}\n"
            f"{age_line}"
            f"- Relationship to protagonist ({target_name}): {relationship}\n"
            f"- Current status: {current_brief}\n\n"
            + (
                f"## Character Behavioral Profile\n"
                f"{format_persona_extensions(self.persona_config.get('persona_extensions') or {}, stage='simulation', heading='', char_budget=1000)}\n"
                f"Ensure the character's dialogue and behavior are consistent with this profile.\n\n"
                if self.persona_config.get('persona_extensions') else ""
            )
            + f"## Action Requirements\n"
            f"{action_instructions.get(action_type, 'Generate an appropriate action.')}\n\n"
            f"## Speech Style\n"
            f"{age_style}\n\n"
            f"## Absolutely Forbidden Expressions (any of these = failure)\n"
            f"- 'I feel that...' / 'This makes me...' / 'I realize...' / 'filled with...'\n"
            f"- 'This strengthened my...' / 'I deeply experienced...' / 'In the process of...'\n"
            f"- Any written-language sentence resembling a news report, essay, or summary report\n"
            f"- Any expression too formal or literary for the character's age and identity\n"
            f"{f'- Character-specific taboos: {self._character_style_guides[speaker_id].forbidden_expressions}' if speaker_id in self._character_style_guides else ''}\n\n"
            f"## Currently Present Participants (you can only interact with these people)\n"
            f"{self._build_present_participants_text()}\n"
            f"Do not assume all characters mentioned in the event summary are present in the current scene. Only interact with the above listed participants.\n\n"
            f"## Remember\n"
            f"Real people speak casually, imperfectly, emotionally, with filler words.\n"
            f"A {'child' if age and age <= 12 else 'person'} of {age} years old would not say things beyond their age.\n"
            f"{self._get_age_cognitive_constraints(age)}"
        )
        system_prompt += await self._anti_moralization_guard()

        # Build observation context
        observation_text = ""
        if personalized_observation:
            observation_text = f"\n## What you see/feel right now (from your perspective)\n{personalized_observation}\n"

        user_prompt = f"""## Current Event
{self._get_event_context()}

## Scene Setting
{self._scene_outline.get('scene_setting', 'unknown') if self._scene_outline else 'unknown'}
{observation_text}
## Your Recent Memories
{memories_text}

## Previous Interactions
{self._build_turn_history_text(max_turns=8)}

## Environment State
{self._get_environment_state_text()}

## Director Guidance
{direction_hint}

{self._format_era_context_for_prompt()}

## Task
As [{name}], generate your action at this moment ({action_type}).

Output requirements:
- content: What you say/do/think (1-3 sentences, must be colloquial and natural).
  Do NOT prefix with your own name (e.g., write "I don't get it." NOT "{name}: I don't get it.").
- body_language: Your body language or micro-gesture at this moment (one concrete detail, e.g., 'fingers unconsciously tapping the table')
{f'- internal_thought: What is really going on in your head (like an inner stream of consciousness)' if is_target else '- internal_thought: leave empty'}
- emotional_state: Your emotional state at this moment (one word)
"""

        try:
            response: AgentActionResponse = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=AgentActionResponse,
                system_prompt=system_prompt,
                task_type="agent_action",
                temperature=self._temperatures["agent_action"],
            )

            turn_data = {
                "turn_index": len(self._interaction_turns),
                "speaker_id": speaker_id,
                "speaker_name": name,
                "action_type": action_type,
                "content": response.content,
                "body_language": response.body_language,
                "internal_thought": response.internal_thought if is_target else "",
                "emotional_state": response.emotional_state,
                "direction_hint": direction_hint,
                "personalized_observation": personalized_observation,
                "scene_time": self._scene_time,
                "scene_phase": self._current_phase,
                "timestamp": datetime.now().isoformat(),
            }

            self._interaction_turns.append(turn_data)

            logger.info(
                f"  [{name}] ({action_type}) {response.content[:80]}..."
                if len(response.content) > 80
                else f"  [{name}] ({action_type}) {response.content}"
            )

            return turn_data

        except Exception:
            logger.exception(f"[Agent {name}] Action generation failed; using fallback turn")
            turn_data = {
                "turn_index": len(self._interaction_turns),
                "speaker_id": speaker_id,
                "speaker_name": name,
                "action_type": action_type,
                "content": f"({name} fell silent for a moment)",
                "body_language": "",
                "internal_thought": "",
                "emotional_state": "neutral",
                "direction_hint": direction_hint,
                "personalized_observation": personalized_observation,
                "scene_time": self._scene_time,
                "scene_phase": self._current_phase,
                "timestamp": datetime.now().isoformat(),
            }
            self._interaction_turns.append(turn_data)
            return turn_data

    # ── Step 4: Event Resolution (concordia pattern) ─────────────

    async def resolve_action(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and resolve an agent's action against scene logic.
        
        This is the concordia 'resolve' step: the director checks whether
        the action is consistent with the scene context, adjusts if needed,
        and determines environmental consequences.
        """
        name = turn_data.get("speaker_name", "?")
        action_type = turn_data.get("action_type", "speak")
        content = turn_data.get("content", "")
        age = self._get_character_age(turn_data.get("speaker_id", ""))

        # Skip resolution for narrate/react (these are director-generated)
        if action_type in ("narrate", "react"):
            return {
                "is_valid": True,
                "resolved_event": content,
                "environment_changes": "",
                "other_reactions": "",
                "time_elapsed": "1 min",
            }

        beats = self._scene_outline.get("key_beats", []) if self._scene_outline else []
        current_beat = beats[self._current_beat_index] if self._current_beat_index < len(beats) else "None"

        system_prompt = (
            "You are the event parser for the scene. Your task is:\n"
            "1. Validate whether the character's action fits the scene logic and character's age\n"
            "2. Judge whether this action advances the current narrative beat\n"
            "3. Describe the environmental changes caused by this action\n"
            "4. Describe the natural reactions of surrounding people/environment\n"
            "5. Estimate how much scene time this action consumed"
        )

        user_prompt = f"""## Action to Parse
- Character: {name} (age {age})
- Action type: {action_type}
- Content: {content}

## Current Scene State
- Scene time: {self._scene_time}
- Scene phase: {self._current_phase}
- Current beat: {current_beat}

## Recent Interactions
{self._build_turn_history_text(max_turns=5)}

{self._format_era_context_for_prompt()}

## Task
1. is_valid: Is this action reasonable? (consider age, scene, character relationships, era context)
2. resolved_event: What actually happened (can slightly adjust the original action to make it more reasonable)
3. environment_changes: What changes happened to the environment (if any)
4. other_reactions: Natural reactions of surrounding people/environment (brief, 1 sentence)
5. time_elapsed: How much time was consumed (e.g., '2 min', '30 sec')

Note: If the action is basically reasonable, use the original content for resolved_event. Only adjust when clearly unreasonable.
"""

        try:
            resolution: EventResolutionResult = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=EventResolutionResult,
                system_prompt=system_prompt,
                task_type="event_resolution",
                temperature=self._temperatures["event_resolution"],
            )

            # Update environment state
            if resolution.environment_changes:
                self._environment_state["recent_environment_changes"].append(
                    resolution.environment_changes
                )
                # Keep only recent changes
                self._environment_state["recent_environment_changes"] = \
                    self._environment_state["recent_environment_changes"][-5:]

            # Check if current beat is completed
            if resolution.is_valid and self._current_beat_index < len(beats):
                if self._current_beat_index not in self._completed_beats:
                    # Simple heuristic: mark beat as done after a few turns on it
                    turns_on_beat = sum(
                        1 for t in self._interaction_turns[-4:]
                        if t.get("scene_phase") == self._current_phase
                    )
                    if turns_on_beat >= 2:
                        self._completed_beats.append(self._current_beat_index)
                        if self._current_beat_index + 1 < len(beats):
                            self._current_beat_index += 1
                        logger.info(
                            f"[Resolution] Beat {self._current_beat_index} completed. "
                            f"Progress: {len(self._completed_beats)}/{len(beats)}"
                        )

            # If action was adjusted, update the turn data
            if not resolution.is_valid:
                logger.warning(
                    f"[Resolution] Action by {name} was adjusted: {resolution.resolved_event[:80]}"
                )
                turn_data["content"] = resolution.resolved_event
                # Update the stored turn
                if self._interaction_turns:
                    self._interaction_turns[-1]["content"] = resolution.resolved_event

            return resolution.model_dump()

        except Exception:
            logger.exception("[Resolution] Failed; using fallback resolution result")
            return {
                "is_valid": True,
                "resolved_event": content,
                "environment_changes": "",
                "other_reactions": "",
                "time_elapsed": "1 min",
            }

    # ── Step 5: Post-Scene Memory Generation ─────────────────────

    async def generate_post_scene_memories(self) -> List[Dict[str, Any]]:
        """
        Generate first-person memories for all participants after the scene.
        Enhanced with sensory details and colloquial style.
        """
        participant_ids = self.event.get("participants", [])

        # Build full interaction transcript
        transcript_lines = []
        for turn in self._interaction_turns:
            name = turn.get("speaker_name", "?")
            action_type = turn.get("action_type", "speak")
            content = turn.get("content", "")
            body = turn.get("body_language", "")
            time_str = turn.get("scene_time", "")
            time_prefix = f"[{time_str}] " if time_str else ""

            if action_type == "speak":
                body_note = f" ({body})" if body else ""
                transcript_lines.append(f"{time_prefix}{name}{body_note}：「{content}」")
            elif action_type == "act":
                transcript_lines.append(f"{time_prefix}[{name} {content}]")
            elif action_type == "think":
                transcript_lines.append(f"{time_prefix}({name} thinks: {content})")
            elif action_type == "narrate":
                transcript_lines.append(f"{time_prefix}[narration: {content}]")
            elif action_type == "react":
                transcript_lines.append(f"{time_prefix}[surroundings: {content}]")
            else:
                transcript_lines.append(f"{time_prefix}{name}：{content}")
        transcript = "\n".join(transcript_lines)

        # Build participant info
        participant_info_lines = []
        for pid in participant_ids:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            age = self._get_character_age(pid)
            relationship = info.get("relationship_towards_the_main_character", "")
            age_str = f", age {age}" if age else ""
            participant_info_lines.append(f"- {name} ({pid}): {relationship}{age_str}")
        participants_text = "\n".join(participant_info_lines)

        duration = self.event.get("duration", {})
        event_date = duration.get("start_date", "?")

        system_prompt = (
            "You are a memory generator. Based on the scene interaction record,\n"
            "generate one first-person perspective memory for each participant.\n\n"
            "[Memory characteristics]\n"
            "1. Written in spoken language, like casually telling a friend about it\n"
            "2. Contains one specific sensory detail (one image, one sound, one tactile sensation)\n"
            "3. Different ages remember differently:\n"
            "   - Children's memories: simple and direct, remember specific images and emotions\n"
            "   - Adults' memories: more reflection and feelings\n"
            "4. Absolutely no written language: 'achieved excellent results', 'performed outstandingly', 'felt gratified', etc.\n\n"
            "Good memory examples (general):\n"
            "- Young person/child: 'By the time I was halfway through, my palms were sweating so much I almost dropped my pen'\n"
            "- Parent: 'When I was there with them, I was even more nervous than they were, couldn't sit still'\n"
            "- Other participant: 'I remember at one point they froze for a second, and my heart sank'\n\n"
            "IMPORTANT: All memory summaries must be written in English. "
            "If the character would naturally think in another language, "
            "translate the thought into English while preserving the tone and cultural flavor."
        )

        # Build character-specific memory style hints from pre-generated guides
        memory_style_hints = []
        for pid in participant_ids:
            guide = self._character_style_guides.get(pid)
            name = self._get_participant_name(pid)
            if guide and guide.memory_voice and "Speak naturally" not in guide.memory_voice:
                memory_style_hints.append(f"- {name}: {guide.memory_voice}")
        if memory_style_hints:
            memory_style_text = "\n".join(memory_style_hints)
            system_prompt += (
                f"\n\nMemory style guides for each character:\n{memory_style_text}"
            )

        user_prompt = f"""## Event Information
{self._get_event_context()}

## Participants
{participants_text}

## Full Interaction Record
{transcript}

## Task
Generate one memory for each participant.

Requirements:
- summary: First-person colloquial memory (2-3 sentences)
- emotional_tone: Emotional tone
- sensory_detail: One specific sensory detail (an image/sound/tactile sensation this person would remember)
- participant_id: Must exactly match the participant list

Participant list:
{chr(10).join(f'- {pid}: {self._get_participant_name(pid)}' for pid in participant_ids)}
"""

        logger.info("[Memory] Generating post-scene memories...")

        try:
            batch: PostSceneMemoryBatch = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=PostSceneMemoryBatch,
                system_prompt=system_prompt,
                task_type="post_scene_memory",
                temperature=self._temperatures["post_memory"],
            )

            memories = []
            for mem in batch.memories:
                memory_entry = {
                    "participant_id": mem.participant_id,
                    "event_id": self.event.get("event_id", "?"),
                    "date": event_date,
                    "summary": mem.summary,
                    "emotional_tone": mem.emotional_tone,
                    "sensory_detail": mem.sensory_detail,
                }
                memories.append(memory_entry)
                logger.info(
                    f"  [{self._get_participant_name(mem.participant_id)}] "
                    f"{mem.summary[:60]}..."
                    if len(mem.summary) > 60
                    else f"  [{self._get_participant_name(mem.participant_id)}] {mem.summary}"
                )

            return memories

        except Exception:
            logger.exception("[Memory] Post-scene memory generation failed; returning empty memory list")
            return []

    # ================================================================
    # Participant Refinement: Four-step cascaded flow
    # ================================================================

    async def refine_participants(self) -> ParticipantRefinementOutput:
        """
        Four-step cascaded participant refinement flow (optimized: Step 4 and Step 2 run in parallel).

        Step 1: Refine core subset from low-resolution pre-selected list (mandatory)
        Step 2: Supplement missing roles from participant pool (conditional)
        Step 3: Generate new participant suggestions (conditional, currently DISABLED)
        Step 4: Persona consistency check (mandatory)

        Parallelization strategy:
        - Step 1 must run first (determines current_ids and missing_role_types)
        - If Step 2 is NOT needed: Step 4 runs on Step 1's current_ids directly
        - If Step 2 IS needed: Step 2 and Step 4 (for Step 1 ids) run in parallel,
          then a supplemental Step 4 check runs for any newly added ids from Step 2

        Returns:
            ParticipantRefinementOutput: Aggregated refinement results
        """
        with self._track_perf(
            "event.P3.refine_participants",
            resolution_level="high",
        ):
            lr_participants = self.event.get("participants", [])
            original_lr = self.event.get("original_low_res_participants", lr_participants)

            # ★ M4+M5-e: If pre-selected participants provided (from outline pool),
            # skip Step 1 and Step 2 entirely — just run Step 4 (persona currency check)
            if self._pre_selected_participants:
                logger.info(
                    f"[Refine] M4+M5-e: Pre-selected participants provided: "
                    f"{self._pre_selected_participants}. Skipping Step 1 & Step 2."
                )
                current_ids = list(self._pre_selected_participants)
                # Ensure P_TARGET is included
                if "P_TARGET" not in current_ids:
                    current_ids.insert(0, "P_TARGET")
                # Filter to only existing participants
                current_ids = [
                    pid for pid in current_ids
                    if pid == "P_TARGET" or pid in self._participants
                ]

                # Run Step 4 (persona currency check) only
                step4_result = await self._step4_check_persona_currency(current_ids)

                # Apply participant cap
                MAX_P3_PARTICIPANTS = 8
                if len(current_ids) > MAX_P3_PARTICIPANTS:
                    current_ids = current_ids[:MAX_P3_PARTICIPANTS]

                result = ParticipantRefinementOutput(
                    step1=Step1RefinementOutput(
                        selected_from_lr=current_ids,
                        refinement_decisions=[
                            ParticipantRefinementEntry(
                                participant_id=pid, included=True,
                                reason="Pre-selected via M4+M5-e shortcut; skipping Step 1 LLM call"
                            )
                            for pid in current_ids
                        ],
                        missing_role_types=[],
                        refinement_reasoning="M4+M5-e shortcut: participants were pre-selected during outline phase, Step 1 LLM refinement skipped.",
                    ),
                    step2=None,
                    new_participant_suggestions=[],
                    step4=step4_result,
                    final_participants=current_ids,
                )
                logger.info(
                    f"[Refine] M4+M5-e complete: final_participants={current_ids}"
                )
                return result

            logger.info(f"[Refine] Starting 4-step participant refinement (parallel-optimized)")
            logger.info(f"[Refine] Low-res participants: {lr_participants}")

            # ── Step 1 ────────────────────────────────────────────────
            step1_result = await self._step1_refine_from_lr(lr_participants)
            current_ids = list(step1_result.selected_from_lr)
            logger.info(f"[Refine] Step 1 complete: selected {current_ids}")

            step2_result = None
            new_suggestions: List[NewParticipantSuggestion] = []

            # ── Determine if Step 2 is needed ─────────────────────────
            needs_supplement = (
                len([p for p in current_ids if p != "P_TARGET"]) < 1
                or bool(step1_result.missing_role_types)
            )

            if needs_supplement:
                # ── Parallel: Step 2 + Step 4 (for Step 1 ids) ────────
                logger.info(
                    f"[Refine] Step 2 + Step 4 running in parallel "
                    f"(Step 2: supplementing from pool, "
                    f"Step 4: checking {len(current_ids)} Step 1 ids)"
                )
                step2_coro = self._step2_supplement_from_pool(
                    current_ids, step1_result.missing_role_types
                )
                step4_coro = self._step4_check_persona_currency(current_ids)

                step2_result, step4_partial = await asyncio.gather(
                    step2_coro, step4_coro
                )

                # Merge Step 2 results into current_ids
                newly_added_ids = [
                    pid for pid in step2_result.added_from_pool
                    if pid not in current_ids
                ]
                current_ids = list(dict.fromkeys(current_ids + step2_result.added_from_pool))
                logger.info(f"[Refine] Step 2 complete: current_ids={current_ids}")
                logger.info(f"[Refine] Step 4 (partial) complete for Step 1 ids")

                # ── Supplemental Step 4 for newly added ids from Step 2 ──
                if newly_added_ids:
                    logger.info(
                        f"[Refine] Step 4 (supplemental): checking "
                        f"{len(newly_added_ids)} newly added ids: {newly_added_ids}"
                    )
                    step4_supplement = await self._step4_check_persona_currency(newly_added_ids)
                    # Merge persona updates from both Step 4 runs
                    merged_updates = list(step4_partial.persona_updates) + list(step4_supplement.persona_updates)
                    step4_result = Step4PersonaCheckOutput(persona_updates=merged_updates)
                    logger.info(f"[Refine] Step 4 (supplemental) complete")
                else:
                    step4_result = step4_partial
            else:
                logger.info(f"[Refine] Step 2 skipped: participants sufficient")

                # ── Step 3（DISABLED by Module 1.4 — pool constraint）────
                still_needs_new = False  # Step 2 was not needed
                if still_needs_new:
                    logger.warning(
                        f"[Refine] Step 3 DISABLED (Module 1.4 pool constraint): "
                        f"missing roles {step1_result.missing_role_types} cannot be filled."
                    )

                # ── Step 4 (full): run on final current_ids ──────────
                logger.info(f"[Refine] Step 4: checking persona currency for {current_ids}")
                step4_result = await self._step4_check_persona_currency(current_ids)
                logger.info(f"[Refine] Step 4 complete")

            # ── Step 3 logging (when Step 2 was triggered) ────────────
            if needs_supplement:
                still_needs_new = (
                    bool(step1_result.missing_role_types)
                    and step2_result is not None
                    and not step2_result.added_from_pool
                )
                if still_needs_new:
                    logger.warning(
                        f"[Refine] Step 3 DISABLED (Module 1.4 pool constraint): "
                        f"missing roles {step1_result.missing_role_types} cannot be filled. "
                        f"New participants must be created via P2 unified pool, not P3."
                    )
                    new_suggestions = []
                else:
                    logger.info(f"[Refine] Step 3 skipped")

            # ── Hard participant cap (v3 optimization) ──
            # Cap at 8 participants to reduce per-turn context length in P3.
            # P_TARGET is always first; keep the first 8 (most relevant).
            MAX_P3_PARTICIPANTS = 8
            if len(current_ids) > MAX_P3_PARTICIPANTS:
                logger.info(
                    f"[Refine] Participant cap: {len(current_ids)} -> {MAX_P3_PARTICIPANTS} "
                    f"(dropped: {current_ids[MAX_P3_PARTICIPANTS:]})"
                )
                current_ids = current_ids[:MAX_P3_PARTICIPANTS]

            result = ParticipantRefinementOutput(
                step1=step1_result,
                step2=step2_result,
                new_participant_suggestions=new_suggestions,
                step4=step4_result,
                final_participants=current_ids,
            )
            logger.info(f"[Refine] Refinement complete: final_participants={current_ids}")
            return result

    # ================================================================
    # M3: One-Shot Full Script Generation + Review + Correction
    # ================================================================

    def _normalize_script_turns(self, script: FullScriptOutput) -> FullScriptOutput:
        """Post-process script turns: strip speaker prefixes, fix narrate/speaker mismatch.

        T04: Called by _generate_full_script() after LLM generation.
        """
        normalized_turns = []
        for t in script.turns:
            content = _strip_speaker_prefix(t.content, t.speaker_name)
            action_type = t.action_type
            speaker_id = t.speaker_id
            speaker_name = t.speaker_name
            # Fix narrate/speaker mismatch: if narrate but not NARRATOR, convert to act
            if action_type == "narrate" and speaker_id not in ("NARRATOR", "NARRATION", ""):
                action_type = "act"
            normalized_turns.append(ScriptTurn(
                turn_index=t.turn_index,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                action_type=action_type,
                content=content,
                body_language=getattr(t, 'body_language', ''),
                internal_thought=getattr(t, 'internal_thought', ''),
                emotional_state=getattr(t, 'emotional_state', ''),
                scene_time=getattr(t, 'scene_time', ''),
                scene_phase=getattr(t, 'scene_phase', ''),
            ))
        return FullScriptOutput(
            turns=normalized_turns,
            scene_summary=script.scene_summary,
            beats_covered=script.beats_covered,
            emotional_arc=getattr(script, 'emotional_arc', ''),
        )

    async def _generate_full_script(
        self,
        scene_outline: Dict[str, Any],
        participants: List[str],
        max_turns: int,
    ) -> FullScriptOutput:
        """M3-a: Generate complete interaction script in a single LLM call.

        Instead of the N-turn Director→Agent loop (2N LLM calls),
        generate the entire script at once (1 LLM call).

        Args:
            scene_outline: Scene outline from plan_scene_outline().
            participants: List of participant IDs in this scene.
            max_turns: Target number of turns.

        Returns:
            FullScriptOutput with all turns, beats covered, and emotional arc.
        """
        target_name = self.persona_config.get("persona_name_text", "protagonist")

        # Build participant details
        participant_details = []
        for pid in participants:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            role = info.get("role", "unknown")
            rel = info.get("relationship_towards_the_main_character", "")
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            style = ""
            if pid in self._character_style_guides:
                s = self._character_style_guides[pid]
                style = f"Speech style: {s.speech_style}, Behavioral patterns: {s.behavioral_patterns}"
            participant_details.append(
                f"- ID: {pid}, Name: {name}, Role: {role}, "
                f"Relationship: {rel}\n"
                f"  Current status: {brief}\n"
                f"  {style}"
            )
        pool_context = "\n".join(participant_details)

        # Era context
        era_hints = ""
        if self._era_context:
            era_hints = (
                f"Technology: {self._era_context.technology_constraints}\n"
                f"Social norms: {self._era_context.social_norms}\n"
                f"Cultural: {self._era_context.cultural_details}\n"
                f"Anachronism warning: {self._era_context.anachronism_warnings}"
            )

        # Temporal context
        tc_hints = ""
        if self._temporal_context:
            tc_hints = (
                f"Character age: {self._temporal_context.exact_age}\n"
                f"Life stage: {self._temporal_context.stage_title}\n"
            )

        # Scene outline
        key_beats = scene_outline.get("key_beats", [])
        beats_text = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(key_beats))

        # Retrieve relevant memories for grounding
        memory_context = self._build_memory_context(participants)

        system_prompt = (
            "You are a screenwriter generating a complete interaction script for a life simulation. "
            "Write ALL dialogue and narration for the entire scene in one go. "
            "Each turn should be spoken by one participant (or be narration). "
            "The protagonist (P_TARGET) must be actively involved throughout. "
            "Ensure the dialogue feels natural, with each character speaking in their own voice. "
            "Cover all the key narrative beats. The scene should have a clear emotional arc.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. The `content` field must contain ONLY the pure dialogue/action/narration text. "
            "Do NOT prefix content with the character's name (e.g., write 'Dude, have you seen this?' "
            "NOT 'Jake: Dude, have you seen this?'). "
            "The speaker is already identified by speaker_id and speaker_name fields.\n\n"
            "2. NARRATION RULE:\n"
            "   - For scene-level narration (describing environment, transitions, or multiple characters' actions), "
            "use action_type='narrate', speaker_id='NARRATOR', speaker_name='narration'.\n"
            "   - Do NOT assign narrate to a specific character.\n"
            "   - If a character performs a physical action, use action_type='act' with that character's speaker_id.\n"
            "   - action_type='narrate' is ONLY for the omniscient narrator perspective."
        )

        user_prompt = f"""## Event
- Summary: {self.event.get('summary', '')}
- Setting: {scene_outline.get('scene_setting', '')}
- Opening: {scene_outline.get('opening_action', '')}
- Emotional arc: {scene_outline.get('emotional_arc', '')}
- Expected turns: {max_turns}

## Key Narrative Beats
{beats_text}

## Participants
{pool_context}

## Protagonist: {target_name} (P_TARGET, must appear frequently)

## Era Context
{era_hints}

## Temporal Context
{tc_hints}

## Relevant Memories
{memory_context}

## Instructions
Generate a complete script with approximately {max_turns} turns. For each turn:
- Assign a speaker (use participant IDs like P_TARGET, P_001, etc.)
- Write natural dialogue or narration
- Include body language and emotional state
- Track the scene phase (setup → rising → climax → resolution → closing)
- Cover all narrative beats in order
- Make sure P_TARGET has internal thoughts in at least 30% of their turns

Return the complete script with all turns, a list of beats covered, and the emotional arc.
"""

        result: FullScriptOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=FullScriptOutput,
            system_prompt=system_prompt,
            temperature=self._temperatures.get("scene_outline", 0.7),
            task_type="m3_generate_full_script",
        )
        return self._normalize_script_turns(result)

    # ================================================================
    # Parallel Protagonist Mode — Fixed 10-turn structure (5 beats × 2)
    # Step 1: Director generates BeatSkeletonOutput (1 LLM call)
    # Step 2: 5 protagonist turns generated in parallel (5 LLM calls)
    # Step 3: Merge into 10 interleaved turns (no LLM call)
    # ================================================================

    async def _generate_beat_skeleton(
        self,
        scene_outline: Dict[str, Any],
        participants: List[str],
    ) -> "BeatSkeletonOutput":
        """
        Parallel Protagonist Mode Step 1:
        Director generates exactly 5 beat plans in one LLM call.

        Each beat plan contains:
          - narrator_content: 3-5 sentence narration (env + other chars)
          - protagonist_direction_hint + protagonist_action_type + protagonist_scene_context
        """
        from .definition import BeatSkeletonOutput

        target_name = self.persona_config.get("persona_name_text", "protagonist")

        # Build participant details
        participant_details = []
        for pid in participants:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            role = info.get("role", "unknown")
            rel = info.get("relationship_towards_the_main_character", "")
            age_at_event = self._get_character_age(pid)

            # Use age-at-event to pick the most appropriate brief.
            # current_persona_brief_text may be stale (computed at an earlier LP);
            # prefer it only when its age matches age_at_event, otherwise fall back
            # to persona_brief_text (the target/adult state) so the skeleton Director
            # always gets a temporally correct description.
            current_brief = info.get("current_persona_brief_text", "") or ""
            target_brief = info.get("persona_brief_text", "") or ""

            # Heuristic: if current_brief mentions an age that differs by >3 years
            # from age_at_event, consider it stale and use target_brief instead.
            brief = current_brief
            if age_at_event and current_brief:
                import re as _re
                # Look for "X-year-old" or "age X" patterns in current_brief
                age_matches = _re.findall(r'\b(\d{1,2})[- ]?year[s]?[- ]old|\bage[:\s]+(\d{1,2})\b', current_brief, flags=_re.I)
                if age_matches:
                    found_ages = [int(a or b) for a, b in age_matches if (a or b)]
                    if found_ages and abs(found_ages[0] - age_at_event) > 3:
                        # current_brief describes a very different age — use target_brief
                        brief = target_brief
                        logger.debug(
                            f"[BeatSkeleton] {pid} current_brief age mismatch "
                            f"({found_ages[0]} vs event age {age_at_event}); using target_brief"
                        )

            # Compose context line: always include computed age
            age_str = f"{age_at_event}" if age_at_event else "unknown age"
            participant_details.append(
                f"- {pid} / {name} (age {age_str}, {role}): {rel}\n  {brief[:200]}"
            )
        pool_context = "\n".join(participant_details)

        key_beats = scene_outline.get("key_beats", [])
        # Ensure exactly 5 beats — pad or trim
        while len(key_beats) < 5:
            key_beats.append(f"Beat {len(key_beats)+1}: scene continues")
        key_beats = key_beats[:5]

        era_hints = ""
        if self._era_context:
            era_hints = (
                f"Technology: {self._era_context.technology_constraints}\n"
                f"Anachronism warning: {self._era_context.anachronism_warnings}"
            )

        memory_context = self._build_memory_context(participants)

        beats_text = "\n".join(
            f"Beat {i+1} ({self._beat_phase(i)}): {b}"
            for i, b in enumerate(key_beats)
        )

        system_prompt = (
            "You are a scene director planning a fixed 5-beat scene skeleton for a life simulation.\n\n"
            "## Structure\n"
            "You MUST output exactly 5 beat plans (beat_index 0-4), covering the full scene arc:\n"
            "  Beat 0 → setup\n"
            "  Beat 1 → rising\n"
            "  Beat 2 → climax\n"
            "  Beat 3 → resolution\n"
            "  Beat 4 → closing\n\n"
            "## For each beat, provide TWO things:\n\n"
            "### 1. narrator_content (the scene background)\n"
            "- 3-5 sentences of omniscient narration\n"
            "- Describe: environment changes, sensory details (sound/smell/light), other characters' "
            "actions/speech, time passing, physical objects\n"
            "- Must NOT describe what the protagonist says or does (that's in the protagonist slot)\n"
            "- Written in vivid present tense\n\n"
            "### 2. Protagonist slot\n"
            "- protagonist_action_type: 'speak' (dialogue) or 'act' (physical action)\n"
            "- protagonist_direction_hint: SPECIFIC, COLLOQUIAL instruction for what the protagonist "
            "says/does in response to this beat's context\n"
            "  GOOD: 'blurts out \"wait, that's not — rent's gonna bounce\" and shoves the phone in his pocket'\n"
            "  BAD: 'reacts to the news', 'expresses frustration'\n"
            "- protagonist_scene_context: 2-3 sentences of what the protagonist directly perceives "
            "(what they see, hear, feel in their body) — at least 2 concrete sensory details\n\n"
            "## CRITICAL RULES\n"
            "- Beat 4 (closing) MUST show the final outcome and the protagonist's exit/last action\n"
            "- scene_phase values: setup/rising/climax/resolution/closing (one per beat, in order)\n"
            "- All 5 key beats from the scene plan must be addressed"
        )

        user_prompt = f"""## Event
- Summary: {self.event.get('summary', '')}
- Setting: {scene_outline.get('scene_setting', '')}
- Opening action: {scene_outline.get('opening_action', '')}
- Emotional arc: {scene_outline.get('emotional_arc', '')}
- Time markers: {', '.join(scene_outline.get('time_markers', []))}

## 5 Key Beats to Cover (one per beat plan)
{beats_text}

## Participants in Scene
{pool_context}

## Protagonist: {target_name} (P_TARGET)
The protagonist is the focus. Their turns will be generated separately in parallel.
You provide: direction_hint (what they should specifically do) + scene_context (what they perceive).

## Era / Cultural Context
{era_hints}

## Relevant Memories
{memory_context}

## Task
Generate exactly 5 BeatPlan items (beat_index 0–4).
Each beat covers ONE key beat from the list above, in order.
scene_phase must progress: setup → rising → climax → resolution → closing.
"""

        result: BeatSkeletonOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=BeatSkeletonOutput,
            system_prompt=system_prompt,
            temperature=self._temperatures.get("scene_outline", 0.7),
            task_type="parallel_protagonist_skeleton",
        )
        return result

    @staticmethod
    def _beat_phase(beat_index: int) -> str:
        """Map fixed beat index to scene phase."""
        return ["setup", "rising", "climax", "resolution", "closing"][min(beat_index, 4)]

    async def _generate_protagonist_turns_parallel(
        self,
        beat_skeleton: "BeatSkeletonOutput",
        scene_outline: Dict[str, Any],
    ) -> Dict[int, "ProtagonistTurnResult"]:
        """
        Parallel Protagonist Mode Step 2:
        Generate all 5 protagonist turns in parallel using asyncio.gather().

        Each call receives the beat's narrator_content as surrounding context.

        Returns:
            Dict mapping beat_index (0-4) → ProtagonistTurnResult
        """
        from .definition import ProtagonistTurnResult

        target_name = self.persona_config.get("persona_name_text", "protagonist")
        info = self._get_participant_info("P_TARGET")
        current_brief = (
            info.get("current_persona_brief_text", "")
            or info.get("persona_brief_text", "")
        )
        age = self._get_character_age("P_TARGET")
        age_style = self._get_age_speech_style("P_TARGET")
        memories = self._get_relevant_memories("P_TARGET", max_count=4)
        memories_text = "\n".join(memories) if memories else "No relevant memories"

        persona_ext_text = ""
        if self.persona_config.get("persona_extensions"):
            persona_ext_text = (
                f"## Character Behavioral Profile\n"
                f"{format_persona_extensions(self.persona_config.get('persona_extensions') or {}, stage='simulation', heading='', char_budget=800)}\n"
                f"Ensure dialogue and behavior are consistent with this profile.\n\n"
            )

        system_prompt = (
            f"You are playing \u300c{target_name}\u300d, age {age}, in a realistic life simulation scene.\n"
            f"You must speak and act exactly as this person would in real life.\n\n"
            f"## Character\n"
            f"- Name: {target_name}, Age: {age}\n"
            f"- Status: {current_brief}\n\n"
            f"{persona_ext_text}"
            f"## Speech Style\n{age_style}\n\n"
            f"## Forbidden\n"
            f"- 'I feel that...' / 'I realize...' / 'This made me...' / 'filled with...'\n"
            f"- Any formal/literary language; news-report style sentences\n\n"
            f"## Rule\nReal people speak casually, imperfectly, with filler words.\n"
            f"{self._get_age_cognitive_constraints(age)}"
        )
        system_prompt += await self._anti_moralization_guard()

        action_instructions = {
            "speak": (
                "Generate the protagonist's exact dialogue (2-4 sentences).\n"
                "Natural filler words, incomplete sentences, verbal tics.\n"
                "No written-language expressions."
            ),
            "act": (
                "Describe the protagonist's physical action (2-3 specific micro-actions).\n"
                "First-person: 'I lower my head...' not 'They lowered...'\n"
                "Include one environmental detail they interact with."
            ),
        }

        async def _generate_one(beat: "BeatPlan") -> ProtagonistTurnResult:
            action_type = beat.protagonist_action_type or "speak"
            surrounding = beat.narrator_content  # what just happened before protagonist acts

            user_prompt = f"""## Event: {self.event.get('summary', '')}
## Scene: {scene_outline.get('scene_setting', '')[:300]}
## Emotional arc: {scene_outline.get('emotional_arc', '')}

## Your Memories
{memories_text}

## What Just Happened (narrator context)
{surrounding}

## What You Perceive Right Now
{beat.protagonist_scene_context or 'No specific perception noted.'}

## Director's Instruction
{beat.protagonist_direction_hint}

## Scene Time: {beat.scene_time} | Phase: {beat.scene_phase}
## Expected emotional state: {beat.protagonist_emotional_state}
## Action type: {action_type}
{action_instructions.get(action_type, 'Generate an appropriate action.')}

## Task — as [{target_name}], generate your response:
- content: {action_type} (2-4 sentences, colloquial). NO name prefix.
- internal_thought: 2-3 fragmented stream-of-consciousness thoughts
- emotional_state: one precise word
- body_language: 2-3 specific micro-cues (eyes, hands, posture, face)
"""
            result: ProtagonistTurnResult = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=ProtagonistTurnResult,
                system_prompt=system_prompt,
                task_type="parallel_protagonist_turn",
                temperature=self._temperatures.get("agent_action", 0.7),
            )
            result.turn_index = beat.beat_index
            return result

        beats = beat_skeleton.beats[:5]
        logger.info(f"[ParallelProtagonist] Launching {len(beats)} protagonist turns in parallel")
        tasks = [_generate_one(b) for b in beats]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[int, ProtagonistTurnResult] = {}
        for i, res in enumerate(results_list):
            beat_idx = beats[i].beat_index
            if isinstance(res, Exception):
                logger.warning(
                    f"[ParallelProtagonist] Beat {beat_idx} failed: {res}; "
                    f"using direction_hint as fallback"
                )
                results[beat_idx] = ProtagonistTurnResult(
                    turn_index=beat_idx,
                    content=beats[i].protagonist_direction_hint or "(protagonist paused)",
                    internal_thought="",
                    emotional_state="neutral",
                    body_language="",
                )
            else:
                results[beat_idx] = res

        logger.info(
            f"[ParallelProtagonist] {len(results)}/{len(beats)} protagonist turns generated"
        )
        return results

    def _merge_beats_into_turns(
        self,
        beat_skeleton: "BeatSkeletonOutput",
        protagonist_results: Dict[int, "ProtagonistTurnResult"],
    ) -> List[Dict[str, Any]]:
        """
        Parallel Protagonist Mode Step 3:
        Interleave narrator and protagonist turns for each beat.

        Output: 10 turns in order:
          [beat0_narrator, beat0_protagonist, beat1_narrator, beat1_protagonist, ...]
        """
        target_name = self.persona_config.get("persona_name_text", "P_TARGET")
        merged: List[Dict[str, Any]] = []
        turn_idx = 0

        for beat in beat_skeleton.beats[:5]:
            bi = beat.beat_index
            phase = beat.scene_phase or self._beat_phase(bi)

            # Turn A: Narrator
            merged.append({
                "turn_index": turn_idx,
                "speaker_id": "NARRATOR",
                "speaker_name": "narration",
                "action_type": "narrate",
                "content": beat.narrator_content,
                "body_language": "",
                "internal_thought": "",
                "emotional_state": "",
                "direction_hint": "",
                "personalized_observation": "",
                "scene_time": beat.scene_time,
                "scene_phase": phase,
                "timestamp": datetime.now().isoformat(),
            })
            turn_idx += 1

            # Turn B: Protagonist
            pr = protagonist_results.get(bi)
            if pr:
                merged.append({
                    "turn_index": turn_idx,
                    "speaker_id": "P_TARGET",
                    "speaker_name": target_name,
                    "action_type": beat.protagonist_action_type or "speak",
                    "content": pr.content,
                    "body_language": pr.body_language,
                    "internal_thought": pr.internal_thought,
                    "emotional_state": pr.emotional_state,
                    "direction_hint": beat.protagonist_direction_hint,
                    "personalized_observation": beat.protagonist_scene_context,
                    "scene_time": beat.scene_time,
                    "scene_phase": phase,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                # Fallback: empty protagonist turn
                merged.append({
                    "turn_index": turn_idx,
                    "speaker_id": "P_TARGET",
                    "speaker_name": target_name,
                    "action_type": beat.protagonist_action_type or "speak",
                    "content": beat.protagonist_direction_hint or "",
                    "body_language": "",
                    "internal_thought": "",
                    "emotional_state": beat.protagonist_emotional_state or "neutral",
                    "direction_hint": beat.protagonist_direction_hint,
                    "personalized_observation": beat.protagonist_scene_context,
                    "scene_time": beat.scene_time,
                    "scene_phase": phase,
                    "timestamp": datetime.now().isoformat(),
                })
            turn_idx += 1

        return merged

    async def _run_parallel_protagonist_mode(
        self,
        scene_outline: Dict[str, Any],
        participants: List[str],
        max_turns: int,
    ) -> None:
        """
        Parallel Protagonist Mode: fixed 10-turn pipeline.

        Step 1: Director generates 5 beat plans (1 LLM call)
        Step 2: 5 protagonist turns in parallel (5 concurrent LLM calls)
        Step 3: Merge into 10 interleaved turns (no LLM call)
        Step 4: Sync _completed_beats and _phase_history
        """
        logger.info("[ParallelProtagonist] Step 1: Generating 5-beat skeleton...")
        beat_skeleton = await self._generate_beat_skeleton(
            scene_outline=scene_outline,
            participants=participants,
        )
        logger.info(
            f"[ParallelProtagonist] Skeleton complete: {len(beat_skeleton.beats)} beats"
        )

        logger.info("[ParallelProtagonist] Step 2: Parallel protagonist generation (5 turns)...")
        protagonist_results = await self._generate_protagonist_turns_parallel(
            beat_skeleton=beat_skeleton,
            scene_outline=scene_outline,
        )

        logger.info("[ParallelProtagonist] Step 3: Merging into 10 turns...")
        self._interaction_turns = self._merge_beats_into_turns(beat_skeleton, protagonist_results)

        # Step 4: Sync state for downstream compatibility
        for td in self._interaction_turns:
            phase = td.get("scene_phase", "rising")
            if phase not in self._phase_history:
                self._phase_history.append(phase)

        # All 5 beats are always considered completed (fixed structure)
        beats = scene_outline.get("key_beats", []) if scene_outline else []
        self._completed_beats = list(range(min(5, len(beats)) if beats else 5))

        logger.info(
            f"[ParallelProtagonist] Complete: {len(self._interaction_turns)} turns, "
            f"phases={self._phase_history}, beats_completed={len(self._completed_beats)}"
        )


    async def _review_script(
        self,
        script: FullScriptOutput,
        scene_outline: Dict[str, Any],
    ) -> ScriptReviewOutput:
        """M3-b: Review the generated script for issues.

        Checks for: character consistency, anachronisms, logic errors,
        pacing problems, missing beats, and out-of-character behavior.

        Args:
            script: The generated full script.
            scene_outline: Original scene outline for comparison.

        Returns:
            ScriptReviewOutput with list of issues found.
        """
        target_name = self.persona_config.get("persona_name_text", "protagonist")

        # Build script text for review
        script_text_parts = []
        for t in script.turns:
            script_text_parts.append(
                f"[{t.turn_index}] {t.speaker_name} ({t.speaker_id}) "
                f"[{t.action_type}, {t.scene_phase}]: {t.content}"
                + (f"\n  (internal: {t.internal_thought})" if t.internal_thought else "")
                + (f"\n  (body: {t.body_language})" if t.body_language else "")
            )
        script_text = "\n".join(script_text_parts)

        # Build character profiles for consistency check
        participants = self.event.get("participants", [])
        char_profiles = []
        for pid in participants:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            role = info.get("role", "unknown")
            rel = info.get("relationship_towards_the_main_character", "")
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            char_profiles.append(f"- {pid} ({name}): role={role}, rel={rel}, status={brief}")
        profiles_text = "\n".join(char_profiles)

        # Era context
        era_hints = ""
        if self._era_context:
            era_hints = (
                f"Technology: {self._era_context.technology_constraints}\n"
                f"Anachronism warning: {self._era_context.anachronism_warnings}"
            )

        key_beats = scene_outline.get("key_beats", [])
        beats_text = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(key_beats))

        system_prompt = (
            "You are a script reviewer for a life simulation. Analyze the script for issues:\n"
            "1. Character consistency: Does each character act and speak consistently with their profile?\n"
            "2. Anachronisms: Are there items, technology, or language out of the era?\n"
            "3. Logic errors: Are there contradictions or impossible sequences?\n"
            "4. Pacing: Does the scene flow naturally through beats?\n"
            "5. Missing beats: Were any key narrative beats skipped?\n"
            "6. Out-of-character: Does anyone behave contrary to their established persona?\n\n"
            "Be strict but practical. Only flag real issues that affect scene quality."
        )

        user_prompt = f"""## Script to Review ({len(script.turns)} turns)

{script_text}

## Character Profiles
{profiles_text}

## Required Narrative Beats
{beats_text}

## Era Context
{era_hints}

## Review Instructions
Analyze the script and report any issues. For each issue, specify:
- Which turn(s) are affected
- The type of issue
- A description of what's wrong
- A suggested fix (if applicable)

Also provide an overall quality assessment.
"""

        result: ScriptReviewOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=ScriptReviewOutput,
            system_prompt=system_prompt,
            temperature=0.0,
            task_type="m3_review_script",
        )
        return result

    async def _correct_script(
        self,
        script: FullScriptOutput,
        review: ScriptReviewOutput,
    ) -> FullScriptOutput:
        """M3-c: Correct specific turns identified in the review.

        Only corrects turns with issues. All other turns remain unchanged.

        Args:
            script: The original script.
            review: The review output with issues.

        Returns:
            Updated FullScriptOutput with corrected turns.
        """
        if not review.issues:
            return script

        # Collect unique turn indices that need correction
        turn_indices = sorted(set(issue.turn_index for issue in review.issues))
        max_idx = len(script.turns) - 1
        turn_indices = [i for i in turn_indices if 0 <= i <= max_idx]

        if not turn_indices:
            return script

        # Build context for the turns that need correction
        issues_by_turn = {}
        for issue in review.issues:
            issues_by_turn.setdefault(issue.turn_index, []).append(issue)

        turns_context = []
        for i in turn_indices:
            t = script.turns[i]
            issues = issues_by_turn.get(i, [])
            issues_text = "; ".join(f"{iss.issue_type}: {iss.description}" for iss in issues)
            turns_context.append(
                f"Turn {i} ({t.speaker_name}, {t.action_type}): {t.content}\n"
                f"  Issues: {issues_text}\n"
                f"  Suggested fix: {'; '.join(iss.suggested_fix for iss in issues if iss.suggested_fix)}"
            )

        # Surrounding context (2 turns before and after each problem turn)
        context_turns = set()
        for i in turn_indices:
            for j in range(max(0, i - 2), min(len(script.turns), i + 3)):
                context_turns.add(j)

        surrounding = []
        for j in sorted(context_turns):
            t = script.turns[j]
            marker = " >>> PROBLEM" if j in turn_indices else ""
            surrounding.append(f"[{j}] speaker={t.speaker_name} | type={t.action_type} | content: {t.content[:100]}{marker}")
        surrounding_text = "\n".join(surrounding)

        # Character profiles
        participants = self.event.get("participants", [])
        profiles = []
        for pid in participants:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            profiles.append(f"- {pid} ({name}): {brief}")
        profiles_text = "\n".join(profiles)

        system_prompt = (
            "You are a script corrector. Fix the specific turns identified as problematic. "
            "Only rewrite the turns that have issues. Keep all other turns unchanged. "
            "Make minimal changes to address the issues while preserving the overall flow.\n\n"
            "CRITICAL: The `content` field must contain ONLY pure dialogue/action text. "
            "Do NOT prefix with the speaker's name. The speaker is identified by speaker_id."
        )

        user_prompt = f"""## Problem Turns to Fix
{chr(10).join(turns_context)}

## Surrounding Context
{surrounding_text}

## Character Profiles
{profiles_text}

## Instructions
For each problematic turn, provide a corrected version that addresses the issues.
Make minimal changes — only fix what's wrong.
"""

        correction: ScriptCorrectionOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=ScriptCorrectionOutput,
            system_prompt=system_prompt,
            temperature=0.0,
            task_type="m3_correct_script",
        )

        # Apply corrections to the script
        corrected_turns_map = {ct.turn_index: ct for ct in correction.corrected_turns}
        updated_turns = []
        for t in script.turns:
            if t.turn_index in corrected_turns_map:
                ct = corrected_turns_map[t.turn_index]
                updated_turn = ScriptTurn(
                    turn_index=t.turn_index,
                    speaker_id=t.speaker_id,
                    speaker_name=t.speaker_name,
                    action_type=t.action_type,
                    content=_strip_speaker_prefix(ct.content, speaker_name=t.speaker_name),
                    body_language=ct.body_language or t.body_language,
                    internal_thought=ct.internal_thought or t.internal_thought,
                    emotional_state=ct.emotional_state or t.emotional_state,
                    scene_time=t.scene_time,
                    scene_phase=t.scene_phase,
                )
                updated_turns.append(updated_turn)
            else:
                updated_turns.append(t)

        return FullScriptOutput(
            turns=updated_turns,
            scene_summary=script.scene_summary,
            beats_covered=script.beats_covered,
            emotional_arc=script.emotional_arc,
        )

    def _build_memory_context(self, participant_ids: List[str]) -> str:
        """Build relevant memory context for script generation grounding."""
        try:
            if not self.retriever:
                return "No memory retriever available."

            event_summary = self.event.get("summary", "")

            # Use the correct MemoryRetriever.build_context() API
            from datetime import date as _date
            current_date = None
            if self._event_date_str:
                try:
                    current_date = _date.fromisoformat(self._event_date_str)
                except (ValueError, TypeError):
                    pass

            return self.retriever.build_context(
                task_type="p3_scene_planning",
                mode="period_focused",
                current_date=current_date,
                period_id=self.event.get("period_id", ""),
                topic_hints=[event_summary] if event_summary else None,
                participant_hints=participant_ids,
                max_events=8,
            )
        except Exception as e:
            logger.error(f"[M3] Failed to build memory context: {e} — LLM will generate dialogue without memory context, severely degrading quality")
            return "Memory context unavailable."

    async def _step1_refine_from_lr(
        self, lr_participants: List[str]
    ) -> Step1RefinementOutput:
        """
        Step 1: Refine core subset from low-resolution pre-selected list.
        P_TARGET must be retained. Recommend 2-5 people, but flexible if scene authenticity requires.
        """
        # Build participant details
        participant_details_lines = []
        for pid in lr_participants:
            info = self._get_participant_info(pid)
            name = info.get("persona_name_text", pid)
            role = info.get("role", "unknown")
            rel = info.get("relationship_towards_the_main_character", "")
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            participant_details_lines.append(
                f"- ID: {pid}, name: {name}, role: {role}, "
                f"relationship: {rel}, persona summary: {brief}"
            )
        participant_details = "\n".join(participant_details_lines)

        # Build scene information
        duration = self.event.get("duration", {})
        scene_info = (
            f"- Event ID: {self.event.get('event_id', '?')}\n"
            f"- Scene summary: {self.event.get('summary', '?')}\n"
            f"- Scene setting: {self.event.get('setting', 'N/A')}\n"
            f"- Turning point: {self.event.get('turning_point', 'N/A')}\n"
            f"- Motivation: {self.event.get('motivation', 'N/A')}\n"
            f"- Expected outcome: {self.event.get('outcome', 'N/A')}\n"
            f"- Time: {duration.get('start_date', '?')} {duration.get('precise_start_time', '?')} ~ "
            f"{duration.get('end_date', '?')} {duration.get('precise_end_time', '?')}"
        )

        system_prompt = (
            "You are a scene participant refiner for life simulation.\n"
            "Based on the specific description of the high-resolution scene, select the most suitable core subset from the low-resolution event's pre-selected participants.\n"
            "Consider when refining:\n"
            "1. The specific theme and context of the scene\n"
            "2. Each participant's relevance to this specific scene (not to the entire low-resolution event)\n"
            "3. Clarity and manageability of the dialogue simulation (recommend 2-5 people, but flexible if scene authenticity requires more)\n"
            "4. Whether there are role types clearly required by the scene but completely absent from the low-resolution list"
        )

        user_prompt = (
            f"## High-Resolution Scene Information\n{scene_info}\n\n"
            f"## Low-Resolution Event Context\n"
            f"- Low-resolution summary: {self.event.get('low_res_context', 'N/A')}\n"
            f"- Belongs to stage: {self.event.get('period_id', '?')}\n\n"
            f"## Low-Resolution Pre-Selected Participants\n{participant_details}\n\n"
            f"## Task\n"
            f"1. P_TARGET must be retained\n"
            f"2. Select the core participants most relevant to this high-resolution scene from the pre-selected list"
            f"(recommend 2-5 people, can be increased if scene authenticity requires)\n"
            f"3. Provide brief reasoning for each selection/exclusion decision\n"
            f"4. If the pre-selected list is missing role types clearly required by the scene, note them in missing_role_types\n"
            f"   (Note: only flag when explicitly mentioned in the scene description, do not speculate)"
        )

        try:
            result = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Step1RefinementOutput,
                system_prompt=system_prompt,
                task_type="p3_participant_refinement",
                temperature=0.0,
            )
            # Ensure P_TARGET is in the selected list
            if "P_TARGET" not in result.selected_from_lr:
                result.selected_from_lr.insert(0, "P_TARGET")
            return result
        except Exception as e:
            logger.exception("[Refine] Step 1 LLM call failed; using all LR participants")
            return Step1RefinementOutput(
                selected_from_lr=lr_participants,
                refinement_decisions=[
                    ParticipantRefinementEntry(
                        participant_id=pid, included=True,
                        reason="Fallback: LLM call failed, keeping all"
                    )
                    for pid in lr_participants
                ],
                missing_role_types=[],
                refinement_reasoning=f"Fallback due to LLM error: {e}",
            )

    async def _step2_supplement_from_pool(
        self,
        current_ids: List[str],
        missing_role_types: List[str],
    ) -> Step2PoolSupplementOutput:
        """
        Step 2: Supplement missing roles from participant pool (conditional).
        """
        # Build selected participant details
        selected_details = []
        for pid in current_ids:
            info = self._get_participant_info(pid)
            selected_details.append(
                f"- {pid}: {info.get('persona_name_text', pid)} "
                f"({info.get('relationship_towards_the_main_character', '')})"
            )

        # 构建可用池（排除已选）
        excluded_set = set(current_ids)
        available_pool = []
        raw_participants = self.persona_pool.get("participants", [])
        if isinstance(raw_participants, dict):
            participant_iterable = raw_participants.values()
        else:
            participant_iterable = raw_participants
        for p in participant_iterable:
            if not isinstance(p, dict):
                continue
            pid = p.get("participant_id", "")
            if pid and pid not in excluded_set:
                brief = p.get("current_persona_brief_text", "") or p.get("persona_brief_text", "")
                available_pool.append(
                    f"- ID: {pid}, name: {p.get('persona_name_text', pid)}, "
                    f"role: {p.get('role', '?')}, "
                    f"relationship: {p.get('relationship_towards_the_main_character', '')}, "
                    f"persona summary: {brief}"
                )

        if not available_pool:
            return Step2PoolSupplementOutput(
                added_from_pool=[],
                pool_insufficient_reason="No other available roles in participant pool",
                supplement_reasoning="Pool is empty after excluding current participants",
            )

        duration = self.event.get("duration", {})
        scene_info = (
            f"- Scene summary: {self.event.get('summary', '?')}\n"
            f"- Scene setting: {self.event.get('setting', 'N/A')}\n"
            f"- Turning point: {self.event.get('turning_point', 'N/A')}"
        )

        system_prompt = (
            "You are a participant pool matcher for life simulation.\n"
            "Based on the scene requirements, find the most suitable supplementary roles from the existing participant pool."
        )

        user_prompt = (
            f"## High-Resolution Scene Information\n{scene_info}\n\n"
            f"## Confirmed Participants (Step 1 Refinement Results)\n"
            + "\n".join(selected_details) + "\n\n"
            f"## Role Types to Supplement\n"
            + "\n".join(f"- {rt}" for rt in missing_role_types) + "\n\n"
            f"## Participant Pool (Available List After Excluding Selected)\n"
            + "\n".join(available_pool) + "\n\n"
            f"## Task\n"
            f"Select the participants from the pool that best satisfy the missing role requirements.\n"
            f"1. Prioritize those with the most matching relationship and role type\n"
            f"2. If no suitable roles in pool, leave returned_ids empty and explain in pool_insufficient_reason\n"
            f"3. Minimize supplementation (only add what is needed for the scene)"
        )

        try:
            result = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Step2PoolSupplementOutput,
                system_prompt=system_prompt,
                task_type="p3_pool_supplement",
                temperature=0.0,
            )
            return result
        except Exception as e:
            logger.exception("[Refine] Step 2 LLM call failed; returning empty supplement result")
            return Step2PoolSupplementOutput(
                added_from_pool=[],
                pool_insufficient_reason=f"LLM call failed: {e}",
                supplement_reasoning=f"Fallback due to LLM error: {e}",
            )

    async def _step3_generate_new_participant_specs(
        self,
        missing_role_types: List[str],
    ) -> List[NewParticipantSuggestion]:
        """
        Step 3: Generate new participant suggestions (conditional).
        Add at most 2 new participants, avoiding role bloat.
        """
        event_summary = self.event.get('summary', '?')

        # Score and sort by importance (no simple slicing)
        scored_roles = [
            (rt, score_role_type_importance(rt, event_summary))
            for rt in missing_role_types
        ]
        scored_roles.sort(key=lambda x: x[1], reverse=True)

        # Take top-3 by importance (was top-2 by order)
        _MAX_NEW_ROLE_TYPES = 3
        selected_roles = [rt for rt, _score in scored_roles[:_MAX_NEW_ROLE_TYPES]]

        scene_info = (
            f"Scene summary: {event_summary}\n"
            f"Scene setting: {self.event.get('setting', 'N/A')}\n"
            f"Missing role types (by priority): {', '.join(selected_roles)}"
        )

        system_prompt = (
            "You are a character creator for life simulation.\n"
            "Based on the scene requirements, generate creation specs for new participants for the missing role types.\n"
            f"Generate at most {_MAX_NEW_ROLE_TYPES} new characters, avoiding role bloat."
        )

        user_prompt = (
            f"## Scene Information\n{scene_info}\n\n"
            f"## Task\n"
            f"Generate one new participant spec for each of the following missing role types:\n"
            + "\n".join(f"- {rt}" for rt in selected_roles) + "\n\n"
            f"Each new participant must include: role_type, relationship_to_main_character, "
            f"persona_requirements, reason_needed"
        )

        suggestions = []
        for rt in selected_roles:
            suggestions.append(NewParticipantSuggestion(
                role_type=rt,
                relationship_to_main_character=f"a {rt} in the scene",
                persona_requirements=f"A {rt} role suitable for: {self.event.get('summary', 'the scene')}",
                reason_needed=f"Scene clearly requires a {rt} role but no suitable match exists in the participant pool",
            ))

        return suggestions

    async def _step4_check_persona_currency(
        self, final_participant_ids: List[str]
    ) -> Step4PersonaCheckOutput:
        """
        Step 4: Persona consistency check (mandatory).
        Check whether participants' current personas are consistent with the scene's time point.

        Improvement: pass DOB-derived age_at_event to the LLM so it can definitively
        detect stale briefs (e.g., a 5-year-old child description for a 25-year-old adult).
        Also updates simulator._participants in-memory so subsequent turns use the
        corrected brief immediately.
        """
        # Build participant details — include computed age so LLM detects mismatches
        participants_details = []
        for pid in final_participant_ids:
            info = self._get_participant_info(pid)
            age_at_event = self._get_character_age(pid)
            brief = info.get("current_persona_brief_text", "") or info.get("persona_brief_text", "")
            age_str = f"{age_at_event}" if age_at_event else "unknown"
            participants_details.append(
                f"- ID: {pid}, name: {info.get('persona_name_text', pid)}, "
                f"role: {info.get('role', '?')}, "
                f"ACTUAL AGE AT SCENE DATE: {age_str} years old, "
                f"current persona on file: {brief}"
            )

        duration = self.event.get("duration", {})
        scene_date = duration.get("start_date", "unknown")
        period_id = self.event.get("period_id", "?")

        system_prompt = (
            "You are a persona consistency reviewer for life simulation.\n"
            "Each participant has an ACTUAL AGE AT SCENE DATE computed from their date of birth. "
            "Your primary job is to detect when the 'current persona on file' describes a person "
            "at a completely different life stage than their actual age at the scene date, "
            "and rewrite it to match the actual age and scene context.\n\n"
            "## CRITICAL RULE\n"
            "If the 'current persona on file' describes someone as a child/student/young adult "
            "but ACTUAL AGE AT SCENE DATE shows they are significantly older (or vice versa), "
            "you MUST set needs_update=true and write an updated_brief that accurately reflects "
            "their state at the scene date (age, likely occupation, life stage).\n"
            "Example: if actual age = 25 but persona says '5-year-old attending preschool', "
            "that is a MANDATORY update — rewrite as a 25-year-old adult with appropriate context."
        )

        user_prompt = (
            f"## Scene Time Point\n{scene_date} ({period_id})\n\n"
            f"## Scene Summary\n{self.event.get('summary', '?')}\n\n"
            f"## Participants to Check\n"
            + "\n".join(participants_details) + "\n\n"
            f"## Task\n"
            f"For each participant:\n"
            f"1. Compare ACTUAL AGE AT SCENE DATE against the age implied by 'current persona on file'\n"
            f"2. If there is a life-stage mismatch (difference > 3 years, or wrong life stage), "
            f"set needs_update=true and write an updated_brief describing them at their actual age\n"
            f"3. The updated_brief should be 1-3 sentences: who they are NOW at the scene date, "
            f"their approximate life situation, and their relationship to the protagonist\n"
            f"4. If persona is consistent with actual age, mark needs_update=false"
            + (
                f"\n\n## Character Behavioral Attributes (must remain consistent)\n"
                f"{format_persona_extensions(self.persona_config.get('persona_extensions') or {}, stage='consistency', heading='', char_budget=800)}"
                if self.persona_config.get('persona_extensions') else ""
            )
        )

        try:
            result = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Step4PersonaCheckOutput,
                system_prompt=system_prompt,
                task_type="step3_persona_validation",
                temperature=0.0,
            )
            # Apply updates to simulator._participants immediately so subsequent
            # turns (skeleton generation, protagonist fill) use corrected briefs
            for update in result.persona_updates:
                if update.needs_update and update.updated_brief:
                    pid = update.participant_id
                    if pid in self._participants:
                        self._participants[pid]["current_persona_brief_text"] = update.updated_brief
                        logger.info(
                            f"[Step4] In-memory persona updated: {pid} → {update.updated_brief[:80]}"
                        )
            return result
        except Exception as e:
            logger.exception("[Refine] Step 4 LLM call failed; returning no-op persona updates")
            return Step4PersonaCheckOutput(
                persona_updates=[
                    PersonaUpdateSuggestion(
                        participant_id=pid,
                        needs_update=False,
                        update_reason=f"Fallback: LLM call failed ({e})",
                    )
                    for pid in final_participant_ids
                ]
            )

    # ── Main Simulation Loop ─────────────────────────────────────

    async def run_simulation(self, max_turns: int = 22) -> Dict[str, Any]:
        """
        Execute the complete high-resolution event simulation.

        The loop follows concordia's sequential engine pattern:
          1. Plan scene outline
          2. For each turn:
             a. Director selects actor + generates personalized observation
             b. Agent generates action
             c. Director resolves/validates action
             d. Environment state updates
          3. Generate post-scene memories
        """
        logger.info("=" * 60)
        logger.info("[Simulation] Starting high-resolution event simulation")
        logger.info(f"  Event: {self.event.get('event_id', '?')}")
        logger.info(f"  Summary: {self.event.get('summary', '?')}")
        logger.info(f"  Time: {self.event.get('duration', {}).get('precise_start_time', '?')} ~ "
                     f"{self.event.get('duration', {}).get('precise_end_time', '?')}")
        logger.info("=" * 60)

        # ── Phase 0: Pre-generation (with Module 7 caching) ─────
        year_key = str(self._event_year) if self._event_year else "unknown"

        # Step 1: Era context — prefer P2 enrichment, fallback to LLM generation
        year_enrichment = self.event.get("year_enrichment")
        if year_enrichment:
            # P2-2: Use enrichment from P2 (no LLM call needed)
            self._era_context = EraContextGuide(
                technology_constraints=year_enrichment.get("technology_media", ""),
                social_norms=year_enrichment.get("cultural_atmosphere", ""),
                cultural_details=year_enrichment.get("cultural_atmosphere", ""),
                physical_environment_hints=year_enrichment.get("stage_environment", ""),
                anachronism_warnings=(
                    f"The scene takes place in {self._event_year or 'unknown'}, "
                    "avoid items, technology, or behaviors inconsistent with that era"
                ),
                era_specific_flavor=year_enrichment.get("stage_environment", ""),
            )
            HighResEventSimulator._era_context_cache[year_key] = self._era_context
            logger.info(
                f"[Simulation] Phase 0, Step 1: Era context built from P2 enrichment "
                f"(year={year_key}, no LLM call)"
            )
        else:
            # Fallback: use cached or generate via LLM (backward compat)
            cached_era = HighResEventSimulator._era_context_cache.get(year_key)
            if cached_era is not None:
                self._era_context = cached_era
                logger.info(f"[Simulation] Phase 0, Step 1: Era context loaded from cache (year={year_key})")
            else:
                logger.info("[Simulation] Phase 0, Step 1: No P2 enrichment available, generating era context via LLM...")
                await self.generate_era_context()
                if self._era_context is not None:
                    HighResEventSimulator._era_context_cache[year_key] = self._era_context
                    logger.info(f"[Simulation] Phase 0, Step 1: Era context cached (year={year_key})")

        # Step 2: Generate character styles (check cache, only generate missing)

        # ── Key Life Path data from P2 ──
        self._key_life_path_data = self.event.get("key_life_path")
        if self._key_life_path_data:
            logger.info(
                f"[Simulation] Key life path data loaded for year {year_key}"
            )
        else:
            logger.info(
                f"[Simulation] No key life path data available for this event"
            )
        participants = self.event.get("participants", [])
        # Always ensure P_TARGET is included in the participants list for style guide generation.
        # P2 may omit it from the event's participants field in some edge cases.
        if "P_TARGET" not in participants:
            logger.warning(
                "[Simulation] P_TARGET not in event participants list; "
                "inserting at position 0 to ensure style guide is generated"
            )
            participants = ["P_TARGET"] + list(participants)

        # ── TC: Compute temporal context for this event ──
        try:
            from lifelong_synth.configs.temporal_context import TemporalContextComputer
            life_periods = self.life_plan.get("life_periods", [])
            social_context = self.life_plan.get("global_summary", {}).get("social_context", {})
            birth_date_str = self.persona_config.get("date_of_birth_str", "")
            if not birth_date_str:
                dob = self.persona_config.get("date_of_birth", {})
                if isinstance(dob, dict):
                    birth_date_str = f"{dob.get('year', 2000)}-{dob.get('month', 1):02d}-{dob.get('day', 1):02d}"
            if birth_date_str and self._event_date_str:
                tc_computer = TemporalContextComputer(birth_date_str, social_context, life_periods)
                # Find the current period for this event to pass stage type tags
                current_period = tc_computer._find_current_period(self._event_date_str)
                self._temporal_context = tc_computer.compute(self._event_date_str, current_period)
                logger.info(
                    f"[TC] Computed temporal context: age={self._temporal_context.exact_age}, "
                    f"edu={self._temporal_context.is_education_stage}, "
                    f"work={self._temporal_context.is_work_stage}, "
                    f"label={self._temporal_context.formatted_education_label or 'N/A'}"
                )
        except Exception as e:
            logger.warning(f"[TC] Failed to compute temporal context: {e}")
            self._temporal_context = None
        cached_count = 0
        missing_pids = []
        for pid in participants:
            style_key = f"{pid}_{year_key}"
            cached_style = HighResEventSimulator._character_style_cache.get(style_key)
            if cached_style is not None:
                self._character_style_guides[pid] = cached_style
                cached_count += 1
            else:
                missing_pids.append(pid)

        if missing_pids:
            logger.info(
                f"[Simulation] Phase 0, Step 2: Generating {len(missing_pids)} character styles "
                f"(parallel), {cached_count} loaded from cache..."
            )
            await self.generate_character_styles_parallel(participant_ids=missing_pids)
            # Cache newly generated styles
            for pid in missing_pids:
                if pid in self._character_style_guides:
                    style_key = f"{pid}_{year_key}"
                    HighResEventSimulator._character_style_cache[style_key] = self._character_style_guides[pid]
        else:
            logger.info(
                f"[Simulation] Phase 0, Step 2: All {cached_count} character styles loaded from cache"
            )

        # Step 3: Validate and report quality
        logger.info("[Simulation] Phase 0, Step 3: Validating pre-generation results...")
        pregen_report = self._validate_pregen_results()

        logger.info(
            f"[Simulation] Phase 0 complete: quality={pregen_report['quality_score']:.1%}"
        )

        # Phase 1: Plan scene outline (now uses cached era_context + style_guides)
        scene_outline = await self.plan_scene_outline()

        # ── SQE: Initialize SimulationPlan ──
        from lifelong_synth.configs.simulation_quality import (
            SimulationPlanBuilder,
            TurnDirectiveComputer,
            RepetitionGuard,
            BeatTracker,
        )
        max_turns = self._compute_adaptive_max_turns(max_turns)

        # Use AM weight to drive TelescopingMode selection if available
        am_weight = self.event.get("am_weight", None)
        plan_builder = SimulationPlanBuilder()
        if am_weight is not None:
            plan = plan_builder.build(am_weight)
            logger.info(
                f"[SQE] SimulationPlan (AM-driven): am_weight={am_weight:.2f}, "
                f"{plan.total_turns} turns, max_beats={plan.max_beats}, "
                f"mode={plan.telescoping_mode}"
            )
        else:
            plan = plan_builder.build_from_turns(max_turns)
            logger.info(
                f"[SQE] SimulationPlan (turns-based fallback): {plan.total_turns} turns, "
                f"max_beats={plan.max_beats}, mode={plan.telescoping_mode}"
            )
        self._simulation_plan = plan

        # ── SQE: Initialize turn-level components ──
        beats = scene_outline.get("key_beats", [])
        # Trim beats to plan.max_beats to ensure mathematical guarantee
        if len(beats) > plan.max_beats:
            logger.info(f"[SQE] Trimming beats from {len(beats)} to {plan.max_beats}")
            beats = beats[:plan.max_beats]
            scene_outline["key_beats"] = beats

        self._beat_tracker = BeatTracker(beats)
        self._repetition_guard = RepetitionGuard()
        self._turn_directive_computer = TurnDirectiveComputer(plan, beats)

        logger.info(f"[Simulation] Adaptive max_turns = {max_turns} (SQE simulation_turns = {plan.simulation_turns})")

        # ★ Mode selection: Parallel Protagonist > M3 > Beat-driven
        use_parallel_protagonist_mode = self.event.get("use_parallel_protagonist_mode", True)
        use_m3_mode = self.event.get("use_m3_script_mode", True)

        if use_parallel_protagonist_mode:
            # ── Parallel Protagonist Path (fastest): Skeleton + Parallel Fill ──
            logger.info(
                "[ParallelProtagonist] Using parallel protagonist mode "
                "(1 skeleton call + N parallel protagonist calls)"
            )
            try:
                await self._run_parallel_protagonist_mode(
                    scene_outline=scene_outline,
                    participants=participants,
                    max_turns=max_turns,
                )
                resolution_log = []
                turn_count = len(self._interaction_turns)
                logger.info(
                    f"[ParallelProtagonist] Complete: {turn_count} turns"
                )
            except Exception:
                logger.exception(
                    "[ParallelProtagonist] Failed; falling back to M3 mode"
                )
                self._interaction_turns = []
                self._phase_history = []
                use_parallel_protagonist_mode = False
                use_m3_mode = True  # fallback to M3

        if not use_parallel_protagonist_mode and use_m3_mode:
            # ── M3 Path: One-Shot Script Generation + Review + Correction ──
            logger.info("[M3] Using one-shot script generation mode (1+1+1 LLM calls vs 2N)")

            try:
                # M3-a: Generate full script in one call
                logger.info("[M3-a] Generating full script...")
                script = await self._generate_full_script(
                    scene_outline=scene_outline,
                    participants=participants,
                    max_turns=max_turns,
                )
                logger.info(f"[M3-a] Full script generated: {len(script.turns)} turns")

                # Populate _interaction_turns from script
                for t in script.turns:
                    turn_data = {
                        "turn_index": t.turn_index,
                        "speaker_id": t.speaker_id,
                        "speaker_name": t.speaker_name,
                        "action_type": t.action_type,
                        "content": t.content,
                        "body_language": t.body_language,
                        "internal_thought": t.internal_thought,
                        "emotional_state": t.emotional_state,
                        "scene_time": t.scene_time,
                        "scene_phase": t.scene_phase,
                        "direction_hint": "",
                        "personalized_observation": "",
                        "timestamp": datetime.now().isoformat(),
                    }
                    self._interaction_turns.append(turn_data)

                # Track beats from script
                if self._beat_tracker and script.beats_covered:
                    for beat in script.beats_covered:
                        try:
                            await self._beat_tracker.check_and_advance_async(
                                turn_content=beat,
                                recent_turns=[],
                            )
                        except Exception:
                            pass

                # M3-b: Review script for issues
                logger.info("[M3-b] Reviewing script...")
                review = await self._review_script(
                    script=script,
                    scene_outline=scene_outline,
                )
                logger.info(
                    f"[M3-b] Review complete: quality={review.overall_quality}, "
                    f"issues={len(review.issues)}"
                )

                # M3-c: Correct script if issues found
                if review.issues and review.overall_quality != "good":
                    logger.info(f"[M3-c] Correcting {len(review.issues)} issues...")
                    corrected_script = await self._correct_script(
                        script=script,
                        review=review,
                    )

                    # Update _interaction_turns with corrected content
                    corrected_map = {t.turn_index: t for t in corrected_script.turns}
                    for i, turn_data in enumerate(self._interaction_turns):
                        if i in corrected_map:
                            ct = corrected_map[i]
                            turn_data["content"] = ct.content
                            turn_data["body_language"] = ct.body_language
                            turn_data["internal_thought"] = ct.internal_thought
                            turn_data["emotional_state"] = ct.emotional_state
                    logger.info("[M3-c] Script corrected")
                else:
                    logger.info("[M3-c] No correction needed (quality is good or no issues)")

                # Track phase history for SQE compatibility
                phase_sequence = ["setup", "rising", "climax", "resolution", "closing"]
                for turn_data in self._interaction_turns:
                    phase = turn_data.get("scene_phase", "rising")
                    if phase not in self._phase_history:
                        self._phase_history.append(phase)

                resolution_log = []  # M3 mode doesn't use resolution_log
                turn_count = len(self._interaction_turns)

                logger.info(
                    f"[M3] Script generation complete: {turn_count} turns, "
                    f"beats covered: {len(script.beats_covered)}"
                )

            except Exception:
                logger.exception("[M3] One-shot script generation failed; falling back to turn-by-turn loop")
                # Reset state for fallback
                self._interaction_turns = []
                self._phase_history = []
                use_m3_mode = False

        if not use_m3_mode:
            # Phase 2-4: Beat-driven Director-Agent interaction loop
            # Structure: for each beat → [narrate] + [2-4 P_TARGET speak/act/think turns]
            turn_count = 0
            resolution_log = []
            beats = scene_outline.get("key_beats", [])
            total_beats = len(beats)

            # Compute turns per beat (excluding reserved turns)
            turns_per_beat = max(2, plan.simulation_turns // max(total_beats, 1))

            logger.info(
                f"[Beat-Driven] {total_beats} beats × ~{turns_per_beat} turns/beat "
                f"= ~{total_beats * turns_per_beat} agent turns + {total_beats} narrations"
            )

            for beat_idx, beat_text in enumerate(beats):
                is_opening = (beat_idx == 0)
                is_closing = (beat_idx == total_beats - 1)

                # Determine phase for this beat
                beat_progress = beat_idx / max(total_beats - 1, 1)
                if beat_progress < 0.15:
                    beat_phase = "opening"
                elif beat_progress < 0.5:
                    beat_phase = "development"
                elif beat_progress < 0.75:
                    beat_phase = "turning_point"
                elif beat_progress < 0.9:
                    beat_phase = "resolution"
                else:
                    beat_phase = "closing"
                self._current_phase = beat_phase
                if beat_phase not in self._phase_history:
                    self._phase_history.append(beat_phase)

                logger.info(
                    f"[Beat {beat_idx + 1}/{total_beats}] Phase={beat_phase}: {beat_text[:80]}"
                )

                # ── Step A: Generate beat narration (旁白) ──
                await self.generate_beat_narration(
                    beat_index=beat_idx,
                    beat_text=beat_text,
                    is_opening=is_opening,
                    is_closing=is_closing,
                )

                # ── Step B: Agent turns for this beat ──
                beat_turn_count = 0
                max_beat_turns = turns_per_beat

                while beat_turn_count < max_beat_turns:
                    # Director selects next action (speak/act/think only, P_TARGET preferred)
                    selection = await self.director_select_action()

                    # Force phase to match beat phase
                    selection.scene_phase = beat_phase
                    self._current_phase = beat_phase

                    # If director wants to end scene, only allow on last beat
                    if selection.should_end_scene:
                        if beat_idx == total_beats - 1:
                            logger.info(
                                f"[Director] Scene ending at beat {beat_idx + 1}, "
                                f"turn {turn_count + 1}"
                            )
                            if selection.direction_hint:
                                turn_data = await self.agent_generate_action(
                                    speaker_id=selection.next_speaker_id,
                                    action_type=selection.action_type,
                                    direction_hint=selection.direction_hint,
                                    personalized_observation=selection.personalized_observation,
                                )
                                resolution_log.append({
                                    "is_valid": True,
                                    "resolved_event": turn_data.get("content", ""),
                                    "environment_changes": "",
                                    "other_reactions": "",
                                    "time_elapsed": "1 min",
                                })
                                turn_count += 1
                            break
                        else:
                            logger.info(
                                f"[SQE] Director wants to end scene but beat {beat_idx + 1}/{total_beats} "
                                f"— overriding, continuing to next beat."
                            )

                    # Generate agent action (speak/act/think)
                    turn_data = await self.agent_generate_action(
                        speaker_id=selection.next_speaker_id,
                        action_type=selection.action_type,
                        direction_hint=selection.direction_hint,
                        personalized_observation=selection.personalized_observation,
                    )

                    # ── SQE: Repetition check ──
                    if self._repetition_guard:
                        content = turn_data.get("content", "") if isinstance(turn_data, dict) else ""
                        recent = [t.get("content", "") for t in self._interaction_turns[-5:]]
                        self._repetition_guard.check(content, recent)

                        if self._repetition_guard.should_force_skip:
                            logger.warning(
                                f"[SQE] Repetition guard triggered at beat {beat_idx + 1}, "
                                f"turn {beat_turn_count}. Force-advancing to next beat."
                            )
                            self._beat_tracker.force_advance()
                            self._repetition_guard.reset()
                            break  # Move to next beat

                    # ── SQE: Beat tracking ──
                    if self._beat_tracker:
                        content = turn_data.get("content", "") if isinstance(turn_data, dict) else ""
                        recent_contents = [t.get("content", "") for t in self._interaction_turns[-3:]]
                        await self._beat_tracker.check_and_advance_async(
                            turn_content=content,
                            recent_turns=recent_contents,
                        )

                    # Selective resolution (uncertainty-driven)
                    needs_resolution = getattr(selection, 'needs_resolution', False)
                    if needs_resolution:
                        resolution = await self.resolve_action(turn_data)
                        resolution_log.append(resolution)
                    else:
                        resolution_log.append({
                            "is_valid": True,
                            "resolved_event": turn_data.get("content", "") if isinstance(turn_data, dict) else "",
                            "environment_changes": "",
                            "other_reactions": "",
                            "time_elapsed": "1 min",
                        })

                    turn_count += 1
                    beat_turn_count += 1

                    # Check global turn limit
                    if turn_count >= plan.simulation_turns:
                        logger.info(
                            f"[Beat-Driven] Global turn limit reached ({turn_count}), "
                            f"stopping at beat {beat_idx + 1}/{total_beats}"
                        )
                        break

                if turn_count >= plan.simulation_turns:
                    break

            logger.info(f"[Simulation] Beat-driven loop complete: {len(self._interaction_turns)} total turns "
                        f"({total_beats} narrations + {turn_count} agent turns)")
            logger.info(f"[Simulation] Beats completed: "
                        f"{len(self._beat_tracker.completed if self._beat_tracker else self._completed_beats)}"
                        f"/{total_beats}")
            logger.info(f"[Simulation] Phases traversed: {' → '.join(self._phase_history)}")

            # ── SQE Layer 3: System-reserved turns (resolution + closing guaranteed) ──
            logger.info("[SQE] Generating system-reserved resolution + closing turns...")
            await self._generate_reserved_turns()

            # ── SQE: Sanity Check (lightweight) ──
            if "closing" not in self._phase_history:
                logger.warning("[SQE] Sanity check: closing phase never reached after reserved turns")

        # Phase 5: Generate post-scene memories
        post_memories = await self.generate_post_scene_memories()

        # ── T14: state sync for M3 / beat-driven paths ──────────────────
        # PP mode already set _completed_beats in _run_parallel_protagonist_mode().
        # Only run the inference logic for M3 / beat-driven (which lack beat tracking).
        if use_parallel_protagonist_mode:
            # PP mode: _completed_beats already set to list(range(5)) — preserve it.
            pass
        elif self._beat_tracker:
            self._completed_beats = list(self._beat_tracker.completed)
        else:
            # T-D2: M3 path has no beat_tracker. Infer beat completion from
            # scene phases traversed (_phase_history). M3 generates a full script
            # so if the scene reached resolution/closing, all beats are done.
            beats = scene_outline.get("key_beats", []) if scene_outline else []
            if beats and self._phase_history:
                phases_seen = set(self._phase_history)
                if phases_seen & {"resolution", "closing"}:
                    # Scene reached a proper ending — all beats completed
                    self._completed_beats = list(range(len(beats)))
                elif "climax" in phases_seen:
                    # Scene reached climax — ~80% of beats completed
                    self._completed_beats = list(range(max(1, int(len(beats) * 0.8))))
                elif "rising" in phases_seen:
                    # Scene reached rising action — ~40% of beats completed
                    self._completed_beats = list(range(max(1, int(len(beats) * 0.4))))
                # else: opening only — leave _completed_beats as []

        if self._phase_history:
            self._current_phase = self._phase_history[-1]

        if self._interaction_turns:
            last_turn = self._interaction_turns[-1]
            if isinstance(last_turn, dict) and last_turn.get('scene_time'):
                self._scene_time = last_turn['scene_time']  # T-D1: fix T14 bug — was _current_scene_time

        if hasattr(self, '_environment_history') and self._environment_history:
            self._environment_state = self._environment_history[-1]

        # Build complete result
        result = await self._build_simulation_result(scene_outline, post_memories, resolution_log)

        logger.info("=" * 60)
        logger.info("[Simulation] High-resolution event simulation complete")
        logger.info("=" * 60)

        return result

    async def _generate_reserved_turns(self) -> None:
        """Generate system-reserved resolution + closing turns.

        These turns are guaranteed to exist regardless of the simulation loop outcome.
        SQE Layer 3: ensures every scene has a proper ending.
        """
        target_id = "P_TARGET"
        target_name = self.persona_config.get("persona_name_text", "protagonist")

        # Turn N-1: Resolution
        if "resolution" not in self._phase_history:
            resolution_prompt = (
                f"The scene must now reach its RESOLUTION. "
                f"As {target_name}, generate a brief moment that reveals the outcome "
                f"or impact of what just happened. Do NOT introduce new conflicts. "
                f"1-2 sentences, colloquial style."
            )
            await self.agent_generate_action(
                speaker_id=target_id,
                action_type="think",
                direction_hint=resolution_prompt,
                personalized_observation="",
            )
            self._current_phase = "resolution"
            if "resolution" not in self._phase_history:
                self._phase_history.append("resolution")
            logger.info("[SQE] Reserved turn: resolution generated")

        # Turn N: Closing
        if "closing" not in self._phase_history:
            closing_prompt = (
                f"This is the FINAL moment of the scene. "
                f"As {target_name}, generate one brief closing gesture, look, or thought "
                f"that captures the emotional residue of this scene. "
                f"1 sentence only. No new content."
            )
            await self.agent_generate_action(
                speaker_id=target_id,
                action_type="think",
                direction_hint=closing_prompt,
                personalized_observation="",
            )
            self._current_phase = "closing"
            if "closing" not in self._phase_history:
                self._phase_history.append("closing")
            logger.info("[SQE] Reserved turn: closing generated")

    async def _generate_refined_summary(
        self,
        base_summary: str,
        opening_beat: str = "",
        outcome: str = "",
        memory_fragments: Optional[List[str]] = None,
        linked_low_res_summary: str = "",
    ) -> str:
        """Generate a refined event summary via LLM, replacing concatenation + truncation.

        Combines all available information into a coherent narrative summary.
        Length is controlled implicitly via prompt instruction (<=300 chars),
        with NO post-processing truncation.

        v4: Added linked_low_res_summary for broader time period context,
        and weight instructions (HR 70% + LR 30%).
        """
        # If no extra info beyond base, skip LLM call
        has_extra = opening_beat or outcome or memory_fragments or linked_low_res_summary
        if not has_extra:
            return base_summary

        # Build context sections
        context_parts = []
        if base_summary:
            context_parts.append(f"Key event summary: {base_summary}")
        if linked_low_res_summary and linked_low_res_summary.strip() != base_summary.strip():
            context_parts.append(f"Broader time period context (background): {linked_low_res_summary}")
        if opening_beat:
            context_parts.append(f"Scene opening: {opening_beat}")
        if outcome:
            context_parts.append(f"Event outcome: {outcome}")
        if memory_fragments:
            context_parts.append(f"Participant memories: {'; '.join(memory_fragments)}")

        context_text = "\n".join(context_parts)

        system_prompt = (
            "You are an event summary generator for a character simulation system. "
            "Generate a coherent, detailed summary of a key event based on the simulation results. "
            "You MUST respond in English, regardless of the language of the input event summary."
        )

        user_prompt = (
            f"Generate a refined summary of this key event based on the simulation results.\n\n"
            f"{context_text}\n\n"
            f"Requirements:\n"
            f"- Focus on the key event details (~70%), with the broader context as background (~30%)\n"
            f"- Include cause, process, and outcome of the key event\n"
            f"- Maintain narrative completeness — no half-sentences or incomplete descriptions\n"
            f"- 2-4 sentences, ≤300 chars\n"
            f"- Use a colloquial narrative style matching the original summary's language\n"
            f"- Output the summary text directly, no prefixes or labels"
        )

        class _SummaryOutput(BaseModel):
            summary: str = Field(
                ...,
                description=(
                    "Merged event summary. 2-4 sentences, ≤300 chars. "
                    "Plain text only — no markdown, no labels, no prefixes."
                )
            )

        MAX_SUMMARY_RETRIES = 1000
        for attempt in range(MAX_SUMMARY_RETRIES):
            try:
                result: _SummaryOutput = await self.llm.generate_structured(
                    prompt=user_prompt,
                    response_model=_SummaryOutput,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.0,
                    task_type="refined_summary",
                )
                if result.summary.strip():
                    return result.summary.strip()
            except Exception as e:
                logger.warning(
                    f"[Summary] Attempt {attempt + 1}/{MAX_SUMMARY_RETRIES} failed: {e}"
                )
                if attempt < MAX_SUMMARY_RETRIES - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

        logger.warning("[Summary] All retries exhausted; falling back to base_summary")
        return base_summary

    async def _build_simulation_result(
        self,
        scene_outline: Dict[str, Any],
        post_memories: List[Dict[str, Any]],
        resolution_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the complete simulation result for JSON output."""
        duration = self.event.get("duration", {})
        participant_ids = self.event.get("participants", [])

        # Build initial states for all participants (with age)
        initial_states = {}
        for pid in participant_ids:
            info = self._get_participant_info(pid)
            age = self._get_character_age(pid)
            # Use same age-based brief selection as _generate_beat_skeleton
            current_brief = info.get("current_persona_brief_text", "") or ""
            target_brief = info.get("persona_brief_text", "") or ""
            brief = current_brief
            if age and current_brief:
                import re as _re
                age_matches = _re.findall(r'\b(\d{1,2})[- ]?year[s]?[- ]old|\bage[:\s]+(\d{1,2})\b', current_brief, flags=_re.I)
                if age_matches:
                    found_ages = [int(a or b) for a, b in age_matches if (a or b)]
                    if found_ages and abs(found_ages[0] - age) > 3:
                        brief = target_brief
            initial_states[pid] = {
                "participant_id": pid,
                "name": info.get("persona_name_text", pid),
                "age_at_event": age,
                "role": info.get("role", "unknown"),
                "relationship": info.get("relationship_towards_the_main_character", ""),
                "current_persona_brief": brief or target_brief,
            }

        # ── Build refined_summary via LLM (no post-processing truncation) ──
        base_summary = (self.event.get("summary", "") or "").strip()
        opening = scene_outline.get("opening_beat", "") if scene_outline else ""
        outcome = (self.event.get("outcome", "") or "").strip()
        # Select memory fragments by importance-weighted scoring (no truncation)
        # post_memories are same-scene in-memory data, so recency is uniform;
        # scoring focuses on protagonist priority, emotional intensity, and content richness.
        # See fragment_scorer.py for theoretical basis.
        scored_memories = [
            (mem, score_memory_fragment(mem))
            for mem in (post_memories or [])
        ]
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        first_memories = [
            (mem.get("summary", "") or "").strip()
            for mem, _score in scored_memories
            if (mem.get("summary", "") or "").strip()
        ]

        refined_summary = await self._generate_refined_summary(
            base_summary=base_summary,
            opening_beat=opening,
            outcome=outcome,
            memory_fragments=first_memories,
            linked_low_res_summary=self.event.get("linked_low_res_summary", "")
                or self.event.get("low_res_context", ""),
        )
        if not refined_summary:
            refined_summary = base_summary

        # T-B1: Collect only participants who actually spoke or acted in the scene.
        # self._participants contains ALL loaded participants (full pool), not just
        # those physically present. Use _interaction_turns to find actual speakers.
        active_speaker_ids: set = set()
        for turn in self._interaction_turns:
            sid = turn.get("speaker_id", "")
            if sid and sid not in ("NARRATOR", "NARRATION", ""):
                active_speaker_ids.add(sid)
        # Always include P_TARGET (protagonist is always present)
        active_speaker_ids.add("P_TARGET")
        # Preserve original insertion order from self._participants
        ordered_active = [
            pid for pid in self._participants.keys()
            if pid in active_speaker_ids
        ]

        return {
            "participants": ordered_active,
            "simulation_metadata": {
                "event_id": self.event.get("event_id", ""),
                "parent_low_res_event_id": self.event.get("parent_low_res_event_id", ""),
                "period_id": self.event.get("period_id", ""),
                "resolution_level": "high",
                "simulation_timestamp": datetime.now().isoformat(),
                "total_turns": len(self._interaction_turns),
                "beats_completed": len(self._completed_beats),
                "total_beats": len(scene_outline.get("key_beats", [])),
                "design_patterns_used": [
                    "personalized_observation (concordia MakeObservation)",
                    "event_resolution (concordia EventResolution)",
                    "beat_tracking (concordia SceneTracker)",
                    "environment_state_management (concordia WorldState)",
                ],
            },
            "summary": refined_summary,
            "refined_summary": refined_summary,
            "event_context": {
                "summary": refined_summary,
                "refined_summary": refined_summary,
                "motivation": self.event.get("motivation", ""),
                "outcome": self.event.get("outcome", ""),
                "value_for_target": self.event.get("value_for_target", ""),
                "duration": {
                    "start_date": duration.get("start_date", ""),
                    "precise_start_time": duration.get("precise_start_time", ""),
                    "end_date": duration.get("end_date", ""),
                    "precise_end_time": duration.get("precise_end_time", ""),
                },
"languages_in_use": self.event.get("languages_in_use", ["eng"]),
            },
            "scene_outline": scene_outline,
            "participant_initial_states": initial_states,
            "interaction_sequence": self._interaction_turns,
            "resolution_log": resolution_log,
            "post_scene_memories": post_memories,
            "environment_state_changes": {
                "scene_phases_traversed": self._phase_history,
                "final_phase": self._current_phase,
                "beats_completed_indices": self._completed_beats,
                "final_scene_time": self._scene_time,
                "environment_state": self._environment_state,
            },
        }


# ================================================================
# Demo Entry Point
# ================================================================

async def _demo_main() -> None:
    """Run the high-resolution event simulation demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load all input data
    data_paths = {
        "persona_pool": f"{base_dir}/simulation_p1_initialisation/persona_pool.json",
        "memory_base": f"{base_dir}/simulation_p4_memory_organiser/memory_base_test.json",
        "life_plan": f"{base_dir}/simulation_p1_initialisation/test_plan.json",
        "persona_config": f"{base_dir}/simulation_p0_persona_settings/test_sample.json",
        "high_res_event": f"{base_dir}/simulation_p2_event_organiser/high_resolution_event_test.json",
    }

    data = {}
    for key, path in data_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data[key] = json.load(f)
        logger.info(f"Loaded {key}: {path}")

    # Initialize LLM client
    llm_client = AsyncLLMClient(
        default_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_base=os.environ.get("OPENAI_API_BASE", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Create simulator
    simulator = HighResEventSimulator(
        llm_client=llm_client,
        persona_pool=data["persona_pool"],
        memory_base=data["memory_base"],
        life_plan=data["life_plan"],
        persona_config=data["persona_config"],
        high_res_event=data["high_res_event"],
    )

    # Run simulation
    result = await simulator.run_simulation(max_turns=25)

    # Save result
    output_path = f"{base_dir}/simulation_p3_multi_resolution_simulation/demo.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Simulation result saved to: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Event: {result['simulation_metadata']['event_id']}")
    logger.info(f"Total turns: {result['simulation_metadata']['total_turns']}")
    logger.info(f"Beats completed: {result['simulation_metadata']['beats_completed']}/{result['simulation_metadata']['total_beats']}")
    logger.info(f"Phases: {' → '.join(result['environment_state_changes']['scene_phases_traversed'])}")
    logger.info(f"Participants: {len(result['participant_initial_states'])}")
    logger.info(f"Post-scene memories: {len(result['post_scene_memories'])}")
    logger.info(f"\nScene outline:")
    logger.info(f"  Setting: {result['scene_outline'].get('scene_setting', 'N/A')[:150]}")
    logger.info(f"  Emotional arc: {result['scene_outline'].get('emotional_arc', 'N/A')}")
    logger.info(f"  Time markers: {result['scene_outline'].get('time_markers', [])}")
    logger.info(f"\nFirst 5 interaction turns:")
    for turn in result["interaction_sequence"][:5]:
        name = turn.get("speaker_name", "?")
        action = turn.get("action_type", "?")
        content = turn.get("content", "")[:100]
        time_str = turn.get("scene_time", "")
        body = turn.get("body_language", "")
        time_prefix = f"[{time_str}] " if time_str else ""
        body_note = f" ({body})" if body else ""
        logger.info(f"  {time_prefix}[{name}] ({action}){body_note} {content}")
    if len(result["interaction_sequence"]) > 5:
        logger.info(f"  ... and {len(result['interaction_sequence']) - 5} more turns")
    logger.info(f"\nPost-scene memories:")
    for mem in result["post_scene_memories"]:
        name = mem.get("participant_id", "?")
        summary = mem.get("summary", "")[:100]
        sensory = mem.get("sensory_detail", "")
        logger.info(f"  [{name}] {summary}")
        if sensory:
            logger.info(f"    Sensory detail: {sensory}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(_demo_main())
