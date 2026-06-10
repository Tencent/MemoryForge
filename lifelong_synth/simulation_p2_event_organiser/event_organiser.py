"""
Event Organiser — Multi-Step Density-Aware Event Generation Pipeline
=====================================================================
Generates structured event data for each life period based on the
configured memory density level.

Multi-Step Event Generation Flow
---------------------------------
  Step 1: Generate base event framework (type, theme, location, summary)
          WITHOUT participant selection
  Step 2: Browse participant pool and select suitable participants based on
          role, relationship, current persona, and target persona
  Step 3: Validate and update each participant's persona for consistency
          with target persona and current time state
  Step 4: Assess whether participant count is sufficient for the event
  Step 5: Generate new participants if needed (using P1 standard flow)

Post-Event Processing
---------------------
  - Update all participants' current_persona_brief_text after each event
  - Rewrite period_summary based on original plan + actual events

Density → time-unit mapping
----------------------------
  none              → 1 year
  around once a year → 1 year
  once per season    → 1 season (3 months)
  around once a month → 1 month
  around once a week  → 1 week

Collaborates with:
  - MemoryManager   (P4) for memory read / write
  - ParticipantPool (P1) for participant lookup and creation
  - AsyncLLMClient  (llm) for all generative calls
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import asyncio
from contextlib import nullcontext
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from lifelong_synth.performance_tracker import PerformanceTracker
from llm.client import AsyncLLMClient
from lifelong_synth.persona_extensions_formatter import format_persona_extensions
from lifelong_synth.simulation_p1_initialisation.definition import (
    Participant,
    DateOfBirth,
)
from lifelong_synth.simulation_p1_initialisation.participant_pool import (
    ParticipantPoolManager,
)
from lifelong_synth.simulation_p3_multi_resolution_simulation.high_res_event_simulator import (
    HighResEventSimulator,
)
from lifelong_synth.simulation_p4_memory_organiser.memory_manager import (
    EVENT_STORAGE_STATUS_FAILED,
    EVENT_STORAGE_STATUS_FINAL,
    EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
    EVENT_STORAGE_STATUS_SIMULATING,
    MemoryManager,
)
from lifelong_synth.simulation_p4_memory_organiser.fragment_scorer import (
    score_memory_fragment,
)
from lifelong_synth.simulation_p2_event_organiser.definition import (
    DENSITY_TO_UNIT,
    TIME_SEGMENTS,
    NewParticipantInfo,
    NewParticipantRequirement,
    LowResEventOutput,
    InteractionTurnOutput,
    HighResEventOutput,
    InteractionHistoryEntry,
    InteractionHistoryBatch,
    CurrentLifeSummaryOutput,
    EventFrameworkOutput,
    LowResFrameworkOutput,
    HighResOutlineOutput,
    ParticipantSelectionOutput,
    PersonaValidationOutput,
    ParticipantSufficiencyOutput,
    PostEventPersonaUpdateOutput,
    PeriodSummaryOutput,
    # Module 1: Unified Participant Pool
    UnifiedParticipantPoolOutput,
    # Module 2: Batch Persona Validation
    BatchPersonaValidationOutput,
    # Module 3: UPER
    BatchReflectionOutput,
    # M4+M5: LR+Outline merged generation
    OutlineInLR,
    LRWithOutlinesOutput,
    OutlinePoolEntry,
    LRAndOutlinePoolOutput,
    # Batch LR generation
    BatchLROutput,
)

logger = logging.getLogger(__name__)


# ================================================================
# Helper: Time-Unit Iteration
# ================================================================

def _iter_time_units(
    start_date: date,
    end_date: date,
    unit: str,
) -> List[tuple[date, date]]:
    """
    Yield (unit_start, unit_end) pairs covering [start_date, end_date].

    Each pair is a closed interval.  The last unit may be shorter than
    the nominal length if the period boundary is reached.
    """
    units: List[tuple[date, date]] = []
    cursor = start_date

    while cursor <= end_date:
        if unit == "year":
            try:
                next_cursor = cursor.replace(year=cursor.year + 1)
            except ValueError:
                next_cursor = cursor.replace(year=cursor.year + 1, day=28)
        elif unit == "season":
            # Season = 3 months
            month = cursor.month + 3
            year = cursor.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            try:
                next_cursor = cursor.replace(year=year, month=month)
            except ValueError:
                from calendar import monthrange
                max_day = monthrange(year, month)[1]
                next_cursor = cursor.replace(year=year, month=month, day=min(cursor.day, max_day))
        elif unit == "month":
            month = cursor.month + 1
            year = cursor.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            try:
                next_cursor = cursor.replace(year=year, month=month)
            except ValueError:
                from calendar import monthrange
                max_day = monthrange(year, month)[1]
                next_cursor = cursor.replace(year=year, month=month, day=min(cursor.day, max_day))
        elif unit in ("period", "module"):
            # Special case: entire period/module as one unit
            next_cursor = end_date + timedelta(days=1)
        else:
            raise ValueError(f"Unknown time unit: {unit}. Supported: year, season, month, period, module")

        unit_end = min(next_cursor - timedelta(days=1), end_date)
        if unit_end < cursor:
            unit_end = cursor
        units.append((cursor, unit_end))
        cursor = next_cursor

    return units


def _format_time_period(
    start: date,
    start_seg: str,
    end: date,
    end_seg: str,
) -> str:
    """Format a time period string in the canonical format."""
    return (
        f"from {start.isoformat()} {start_seg} "
        f"to {end.isoformat()} {end_seg}"
    )


# ================================================================
# Module 6: SNAP — Social Network-Aware Participant Filtering
# ================================================================

class SNAPFilter:
    """Social Network-Aware Participant filtering.

    Pre-filters participant pool to top-k most relevant candidates
    using multi-signal scoring. Reduces prompt length by 50-70%.

    Inspired by:
    - Social Network Analysis (Newman, 2004)
    - Concordia (Vezhnevets et al., 2023)
    - Park et al. (2025) — social distance decay
    """

    def __init__(self, pool, memory):
        self._pool = pool
        self._memory = memory

    def filter_relevant_participants(
        self,
        event_theme: str,
        event_type: str,
        period_id: str,
        current_date: date,
        max_candidates: int = 8,
    ) -> List[str]:
        """Return top-k most relevant participant IDs for the current event."""
        import math

        all_participants = self._pool.list_all()
        scored = []

        for p in all_participants:
            pid = p.participant_id
            if pid == "P_TARGET":
                continue  # Always included separately

            # Signal 1: Recency — when was this participant last active?
            # Use h.date (end-date anchor) for recency; fall back to extracting
            # end date from time_period if date is not set.
            last_date_str = ""
            if p.interactions_history_with_the_main_character:
                dates = []
                for h in p.interactions_history_with_the_main_character:
                    anchor = h.date or _extract_end_date_from_period(h.time_period)
                    if anchor:
                        dates.append(anchor)
                if dates:
                    last_date_str = max(dates)

            if last_date_str:
                try:
                    last_d = date.fromisoformat(last_date_str)
                    days_ago = max((current_date - last_d).days, 0)
                except (ValueError, TypeError):
                    days_ago = 3650
            else:
                days_ago = 3650
            s_recency = math.exp(-0.005 * days_ago)

            # Signal 2: Co-occurrence with P_TARGET (interaction frequency)
            interaction_count = len(p.interactions_history_with_the_main_character)
            s_social = min(interaction_count / 10.0, 1.0)

            # Signal 3: Role relevance — keyword overlap
            theme_words = set(event_theme.lower().split()) if event_theme else set()
            persona_words = set(
                (p.current_persona_brief_text or "").lower().split()
            )
            if theme_words:
                s_role = len(theme_words & persona_words) / max(len(theme_words), 1)
            else:
                s_role = 0.3

            # Signal 4: Period activity — T26: Hard filter for appear_period
            appear = p.appear_period
            if appear and appear not in ("LP_UNKNOWN", "LP0"):
                try:
                    appear_num = int(appear.replace("LP", ""))
                    period_num = int(period_id.replace("LP", ""))
                    if appear_num > period_num:
                        continue  # Hard filter: character hasn't appeared yet
                except ValueError:
                    pass
            s_period = 1.0 if period_id in (appear or "") else 0.5

            # Signal 5: Social distance decay (Park et al. 2025)
            recent_interactions = len([
                h for h in p.interactions_history_with_the_main_character
                if (h.date or _extract_end_date_from_period(h.time_period))
                and _safe_days_ago(
                    h.date or _extract_end_date_from_period(h.time_period),
                    current_date
                ) < 365
            ])
            s_social_distance = 1.0 - math.exp(-0.5 * recent_interactions)

            score = (
                0.25 * s_recency
                + 0.20 * s_social
                + 0.15 * s_role
                + 0.15 * s_period
                + 0.25 * s_social_distance
            )
            scored.append((pid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = ["P_TARGET"]
        for pid, _ in scored:
            if len(result) >= max_candidates:
                break
            result.append(pid)
        return result


def _safe_days_ago(date_str: str, current_date: date) -> int:
    """Safely compute days between date_str and current_date."""
    try:
        d = date.fromisoformat(date_str)
        return max((current_date - d).days, 0)
    except (ValueError, TypeError):
        return 99999


def _extract_end_date_from_period(time_period: str) -> Optional[str]:
    """Extract the END date from a time_period string for recency calculations.

    Supported formats:
      - 'YYYY-MM-DD' → returns as-is
      - 'YYYY-MM-DD to YYYY-MM-DD' → returns the part after ' to '
      - 'YYYY-MM' → returns 'YYYY-MM-28' (end of month approximation)
      - 'YYYY' → returns 'YYYY-12-31' (end of year)

    Returns None if the format cannot be parsed.
    """
    if not time_period:
        return None
    time_period = time_period.strip()

    # Range format: 'YYYY-MM-DD to YYYY-MM-DD'
    if ' to ' in time_period:
        end_part = time_period.split(' to ')[-1].strip()
        try:
            date.fromisoformat(end_part)
            return end_part
        except ValueError:
            return None

    # Full date: 'YYYY-MM-DD'
    if len(time_period) == 10 and time_period[4] == '-' and time_period[7] == '-':
        try:
            date.fromisoformat(time_period)
            return time_period
        except ValueError:
            return None

    # Year-month: 'YYYY-MM'
    if len(time_period) == 7 and time_period[4] == '-':
        try:
            year, month = int(time_period[:4]), int(time_period[5:7])
            if 1 <= month <= 12:
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                return f"{year:04d}-{month:02d}-{last_day:02d}"
        except (ValueError, IndexError):
            return None

    # Year only: 'YYYY'
    if len(time_period) == 4 and time_period.isdigit():
        return f"{time_period}-12-31"

    return None


# ================================================================
# Pydantic models for LLM-first persona context signals (T29)
# ================================================================

class _PersonaContextSignals(BaseModel):
    """LLM-derived signals for period-level persona context construction.

    Returned by _llm_persona_context_signals() and consumed by
    _build_structured_persona_context(). Replaces hardcoded country sets
    and brief truncation with LLM-based semantic judgement.
    """
    cultural_grounding_needed: bool = Field(
        description=(
            "True if the persona is rooted in a cultural context whose daily life, "
            "workplace norms, family structures, social expectations, or milestones "
            "would differ meaningfully from mainstream US/UK/Canada/Australia defaults. "
            "Base this judgement on growing_up_location, current_living_location, and "
            "persona_brief_text taken together — NOT on a fixed country whitelist."
        ),
    )
    cultural_grounding_instruction: str = Field(
        default="",
        description=(
            "If cultural_grounding_needed is True, a 40-to-80-word English instruction "
            "telling the event generator which specific cultural dynamics to reflect "
            "(e.g. prayer-time scheduling, multi-generational household norms, seasonal "
            "festivals, post-colonial workplace dynamics). Empty string otherwise."
        ),
    )
    occupation_depth_needed: bool = Field(
        description=(
            "True if the persona brief names an occupation or vocation whose day-to-day "
            "practice is distinctive enough that it should anchor a specific episodic memory "
            "(e.g. truck driver, chef, monk, athlete, surgeon, luthier). False for generic "
            "white-collar roles with no distinctive daily practice."
        ),
    )
    occupation_depth_instruction: str = Field(
        default="",
        description=(
            "If occupation_depth_needed is True, a 40-to-80-word English instruction naming "
            "the occupation and two or three kinds of concrete, signature episodes that would "
            "authentically demonstrate domain expertise for this persona. Empty otherwise."
        ),
    )
    identity_salience_needed: bool = Field(
        default=False,
        description=(
            "True if the persona has one or more explicit identity traits "
            "(sexual_orientation, gender_identity, religious_identity, racial_ethnic, "
            "disability, or political_identity) that are marked as 'primary' salience "
            "and would meaningfully shape the kinds of life events this person experiences. "
            "False if no such traits are present or all traits are 'background' salience."
        ),
    )
    identity_salience_instruction: str = Field(
        default="",
        description=(
            "If identity_salience_needed is True, a 40-to-80-word English instruction "
            "telling the event generator HOW each identity trait should manifest in events "
            "for THIS specific persona — naming the traits explicitly and giving 1-2 concrete "
            "event types per trait (e.g. 'As a homosexual woman, include at least one event "
            "involving a same-sex relationship milestone or coming-out moment per relevant period'). "
            "Must be persona-specific, not generic. Empty string otherwise."
        ),
    )


class _OutlineSalienceScore(BaseModel):
    """LLM-derived salience score for a single event outline.

    Returned by _llm_score_outline_salience() and consumed by _rank_outlines_for_detail().
    Replaces hardcoded keyword lists with LLM-based semantic judgement.
    """
    event_id: str = Field(description="The outline's event_id, echoed back verbatim.")
    importance_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Semantic importance in [0,1]. 0.90+ for phase-transition milestones "
            "(graduation, first-job, marriage, divorce, major identity-defining moments); "
            "0.75-0.85 for identity-exploration, cultural/religious practice, or adversity/hardship "
            "moments that would be central to PersonaGym-style evaluation; 0.80-0.90 for "
            "formative historical events explicitly witnessed by elderly personas (age>=70); "
            "0.55-0.65 for distinctive but routine daily events; 0.30-0.50 for filler."
        ),
    )
    identity_relevance: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description=(
            "How directly this outline's theme relates to the persona's explicit identity traits "
            "(sexual_orientation, gender_identity, religious_identity, racial_ethnic, disability, "
            "political_identity). Score 1.0 if the event is primarily about an identity trait "
            "(e.g. coming-out moment, religious practice, racial discrimination experience); "
            "0.6-0.9 if identity is a significant secondary element; 0.1-0.5 if identity is "
            "incidentally present; 0.0 if the event has no connection to the persona's identity "
            "traits. Only non-zero when the persona has identity_salience_needed=True; "
            "always 0.0 otherwise."
        ),
    )
    rationale: str = Field(
        default="",
        description="One-sentence English rationale (<=40 words).",
    )


class _OutlineSalienceBatchResult(BaseModel):
    scores: List[_OutlineSalienceScore] = Field(
        default_factory=list,
        description="One _OutlineSalienceScore per input outline, in the same order.",
    )


# ================================================================
# Event Organiser — Multi-Step Pipeline
# ================================================================

class EventOrganiser:
    """
    Multi-step density-aware event generation pipeline.

    For each life period, iterates over time units determined by the
    period's ``default_memory_density`` and generates events via a
    structured multi-step flow.

    Multi-Step Workflow per time unit:
      Step 1: Generate event framework
      Step 2: Select participants
      Step 3: Validate & update personas
      Step 4: Assess participant sufficiency
      Step 5: Generate new participants if needed
      Post:   Interaction histories, memory update, persona updates
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        memory_manager: MemoryManager,
        participant_pool: ParticipantPoolManager,
        target_persona: Dict[str, Any],
        participant_pool_path: Optional[str] = None,
        memory_base_path: Optional[str] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        run_id: str = "",
        milestone_plan: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm_client
        self.memory = memory_manager
        self.pool = participant_pool
        self.persona = target_persona
        self._event_counters: Dict[str, int] = {}
        self._pool_save_path = participant_pool_path or getattr(self.pool, '_save_path', None)
        self._memory_save_path = memory_base_path
        self._performance_tracker = performance_tracker
        self._run_id = run_id or ""
        self._milestone_plan: Dict[str, Any] = milestone_plan or {}
        self._restore_event_counters_from_memory()

        # ── TC Engine: Bridge SocialContextProfile from P1 ──
        self._social_context: Dict[str, Any] = {}
        self._tc_computer: Optional[Any] = None
        self._enrichment_cache: Dict[int, Dict[str, Any]] = {}
        # Semaphore to limit concurrent enrichment API calls (avoid overwhelming proxy)
        self._enrichment_semaphore: asyncio.Semaphore = asyncio.Semaphore(10)

        # Track last event date for sub-step TC injection
        self._last_event_date_str: str = ""

        # ── Enrichment cache persistence ──
        # Derive the enrichment cache path from memory_base_path
        # e.g. /output/run_xxx/memory_base.json → /output/run_xxx/year_society_enrichment_cache.json
        self._enrichment_cache_path: Optional[str] = None
        if memory_base_path:
            cache_dir = os.path.dirname(memory_base_path)
            # New name; fall back to old name for backward compatibility
            new_path = os.path.join(cache_dir, "year_society_enrichment_cache.json")
            old_path = os.path.join(cache_dir, "year_enrichment_cache.json")
            if os.path.isfile(new_path):
                self._enrichment_cache_path = new_path
            elif os.path.isfile(old_path):
                self._enrichment_cache_path = old_path
                logger.info(
                    f"[TC-Enrichment] Using legacy cache file: {old_path}. "
                    f"Consider renaming to year_society_enrichment_cache.json"
                )
            else:
                # No existing cache; use new name for future saves
                self._enrichment_cache_path = new_path
        self._load_enrichment_cache()

        # ── Key Life Path cache persistence ──
        self._key_life_path_cache: Dict[int, Dict[str, Any]] = {}
        self._key_life_path_cache_path: Optional[str] = None
        if memory_base_path:
            cache_dir = os.path.dirname(memory_base_path)
            self._key_life_path_cache_path = os.path.join(
                cache_dir, "year_enrichment_key_life_path_cache.json"
            )
        self._load_key_life_path_cache()

        # ── Simulation end date (for target_persona_brief_text generation) ──
        # Priority: pool._simulation_end_date (correctly read from timeline_anchor in P1)
        # > milestone_plan["reference_date"] > target_persona["reference_date"]
        # Note: milestone_plan and target_persona do NOT have a "reference_date" field in
        # practice; the correct value lives in life_plan.global_summary.timeline_anchor,
        # which participant_pool already parses during P1 initialisation.
        self.simulation_end_date: Optional[str] = (
            getattr(participant_pool, "_simulation_end_date", "") or None
        )
        if not self.simulation_end_date:
            if milestone_plan and milestone_plan.get("reference_date"):
                self.simulation_end_date = milestone_plan["reference_date"]
            elif target_persona and target_persona.get("reference_date"):
                self.simulation_end_date = target_persona["reference_date"]

        # ── Simulation start date (for Phase-3 profile extension of dynamic participants) ──
        # Prefer the value already stored in the participant pool (set during P1 initialisation).
        # Fall back to target_persona / milestone_plan if the pool hasn't been initialised yet.
        self.simulation_start_date: Optional[str] = (
            getattr(participant_pool, "_simulation_start_date", "") or ""
        )
        if not self.simulation_start_date:
            self.simulation_start_date = (
                target_persona.get("simulation_start_date", "")
                if target_persona
                else ""
            )

        # ── T16: Period-level persona current state cache (populated by P1d) ──
        self._persona_current_state_cache: Dict[str, Dict] = {}

    # ── LLM helpers for persona context signals (T29) ────────────────────────

    async def _llm_persona_context_signals(self) -> _PersonaContextSignals:
        """Return cached _PersonaContextSignals for self.persona.

        Runs at most once per persona — the result is cached on self._persona_context_signals_cache.
        Now also analyses identity_traits (if present) to produce identity_salience_instruction.
        """
        cached = getattr(self, "_persona_context_signals_cache", None)
        if cached is not None:
            return cached

        persona_brief = (self.persona.get("persona_brief_text") or "").strip()
        cur_loc = self.persona.get("current_living_location") or {}
        grew_loc = self.persona.get("growing_up_location") or {}

        # Serialise locations as JSON so the LLM sees all fields verbatim.
        cur_loc_str = json.dumps(cur_loc, ensure_ascii=False)
        grew_loc_str = json.dumps(grew_loc, ensure_ascii=False)

        # Extract identity_traits from persona_extensions (if present)
        ext = self.persona.get("persona_extensions") or {}
        identity_traits_items = (ext.get("identity_traits") or {}).get("items") or []
        primary_traits = [
            item for item in identity_traits_items
            if item.get("salience") == "primary"
        ]
        identity_traits_str = ""
        if primary_traits:
            lines = []
            for item in primary_traits:
                lines.append(
                    f"- {item.get('trait_category', 'unknown')}: {item.get('trait_value', 'unknown')}"
                )
            identity_traits_str = (
                "\n\n## identity_traits (primary salience only)\n"
                + "\n".join(lines)
                + "\n\nFor each identity trait above, decide whether it would meaningfully shape "
                "the kinds of life events this person experiences (e.g. coming-out moments, "
                "discrimination, religious conflict, community belonging). If yes, set "
                "identity_salience_needed=true and write a concrete, persona-specific instruction "
                "in identity_salience_instruction naming each trait and 1-2 event types per trait."
            )

        user_prompt = (
            "Analyse the following persona for event-generation context.\n\n"
            f"## persona_brief_text\n{persona_brief}\n\n"
            f"## current_living_location (JSON)\n{cur_loc_str}\n\n"
            f"## growing_up_location (JSON)\n{grew_loc_str}"
            f"{identity_traits_str}\n\n"
            "Return a structured judgement with all fields described in the schema. "
            "Cap each instruction at 80 words. Write instructions in English, grounded in the "
            "persona's own context — do not prescribe WEIRD defaults when the persona is not WEIRD."
        )
        try:
            result: _PersonaContextSignals = await self.llm.generate_structured(
                prompt=user_prompt,
                system_prompt=(
                    "You are a sociocultural simulation analyst. Decide whether a given persona "
                    "requires cultural grounding, occupation-specific episodic depth, or "
                    "identity-salience guidance, and compose concise English instructions for "
                    "downstream event generators. Do not invent facts."
                ),
                response_model=_PersonaContextSignals,
                temperature=0.0,
                task_type="p2_persona_context_signals",
            )
        except Exception as e:
            logger.warning(f"[P2] _llm_persona_context_signals failed: {e}; returning inert signals.")
            result = _PersonaContextSignals(
                cultural_grounding_needed=False,
                occupation_depth_needed=False,
                identity_salience_needed=False,
            )
        self._persona_context_signals_cache = result
        return result

    async def _llm_score_outline_salience(
        self,
        outlines: List[Tuple],  # List of (event_id, HighResOutlineOutput) or (event_id, outline, unit_idx)
        identity_salience_instruction: str = "",  # persona-specific identity instruction, if any
    ) -> Dict[str, Tuple[float, float]]:
        """Ask the LLM to score a batch of outline themes for salience.

        Returns a dict {event_id: (importance_score, identity_relevance)}.
        Falls back to (0.5, 0.0) for any event_id that is missing from the LLM output,
        so _rank_outlines_for_detail() stays robust.
        One LLM call per batch — not one per outline.
        """
        if not outlines:
            return {}

        # Collect one-line summaries for the LLM, without truncation.
        items_block = []
        for item in outlines:
            if len(item) == 3:
                event_id, outline, _unit_idx = item
            else:
                event_id, outline = item
            theme = outline.key_event_theme or ""
            start = getattr(outline, "key_event_start_date", "") or ""
            items_block.append(f"- event_id: {event_id} | start: {start} | theme: {theme}")
        items_text = "\n".join(items_block)

        # Persona-side context (no truncation).
        persona_brief = (self.persona.get("persona_brief_text") or "").strip()
        try:
            target_age = int(self.persona.get("target_age_exact") or 0)
        except (TypeError, ValueError):
            target_age = 0

        # Build identity salience block for the prompt (only when relevant)
        identity_block = ""
        if identity_salience_instruction:
            identity_block = (
                f"\n## Identity Salience Instruction\n{identity_salience_instruction}\n"
                "For each outline, set identity_relevance > 0.0 if the theme directly or "
                "significantly relates to the identity traits described above. "
                "Set identity_relevance=1.0 for events that are primarily about an identity "
                "trait (coming-out, religious practice, discrimination experience, etc.)."
            )

        user_prompt = (
            "Score each outline's salience for a persona role-play evaluation (PersonaGym style). "
            "Prioritise identity-exploration, cultural/religious practice, adversity, and — for "
            "personas aged >=70 — formative distant historical events. Return a list of scores in the "
            "SAME order as the input outlines. Each rationale must be in English and <=40 words.\n\n"
            f"## Persona brief\n{persona_brief}\n\n"
            f"## Persona target age\n{target_age}\n"
            f"{identity_block}\n"
            f"## Outlines\n{items_text}"
        )
        try:
            result: _OutlineSalienceBatchResult = await self.llm.generate_structured(
                prompt=user_prompt,
                system_prompt=(
                    "You are a persona-simulation salience scorer. Use the scoring guide in the "
                    "schema's field description verbatim. Do not invent outlines not in the input."
                ),
                response_model=_OutlineSalienceBatchResult,
                temperature=0.0,
                task_type="p2_outline_salience",
            )
        except Exception as e:
            logger.warning(f"[P2] _llm_score_outline_salience failed: {e}; defaulting all scores to 0.5.")
            return {}

        return {
            s.event_id: (float(s.importance_score), float(s.identity_relevance))
            for s in result.scores
        }

    # ── Milestone plan helpers ───────────────────────────────────

    def get_milestone_for_period(self, period_id: str) -> Optional[Dict[str, Any]]:
        """Return the pre-planned milestone skeleton for a period, or None."""
        skel = self._milestone_plan.get(period_id)
        if skel is None:
            return None
        # Accept both dict and pydantic model
        if hasattr(skel, "model_dump"):
            return skel.model_dump()
        return skel

    def milestone_matches_time_unit(
        self, period_id: str, unit_start: "date", unit_end: "date"
    ) -> bool:
        """Check if the pre-planned milestone's expected date falls within a time unit."""
        skel = self.get_milestone_for_period(period_id)
        if not skel:
            return False
        date_range = skel.get("expected_date_range", "")
        if not date_range:
            return False
        # Parse the first date token from expected_date_range
        # Formats: "2016-06-07 to 2016-06-08", "2016-06-07~08", "2016-06"
        first_token = date_range.split(" ")[0].split("~")[0].strip()
        try:
            from datetime import date as _date
            if len(first_token) == 7:  # YYYY-MM
                first_token += "-15"  # approximate mid-month
            milestone_date = _date.fromisoformat(first_token)
            return unit_start <= milestone_date <= unit_end
        except (ValueError, TypeError):
            return False

    def _get_tc_prompt_for_event(self, event_date: str, period: Optional[Dict[str, Any]] = None) -> str:
        """Compute and format temporal context for injection into LLM prompts.

        Returns an empty string if TC is not available.
        """
        if not self._tc_computer:
            return ""
        try:
            tc = self._tc_computer.compute(event_date, period)
            return "\n" + tc.format_for_prompt(language="en") + "\n"
        except Exception as e:
            logger.warning(f"[TC] Failed to compute TC for {event_date}: {e}")
            return ""

    # ── Enrichment formatting for prompt injection ───────────────

    def _format_key_life_path_for_prompt(self, year: int) -> str:
        """Format key life path data for prompt injection.

        Uses the unified full-content approach — no truncation, no detail-level grading.
        Data length is controlled at generation time (Phase 1).

        Args:
            year: Calendar year to format.

        Returns:
            Formatted string for prompt injection, or empty string if no data.
        """
        life_path = self._key_life_path_cache.get(year)
        if not life_path:
            return ""

        parts = []
        parts.append("## 🎯 Key Life Path Anchors (Based on Inferred Path, HIGHEST PRIORITY)")

        # Education status
        edu = life_path.get("education_status", {})
        if edu:
            stage_type = edu.get("stage_type", "")
            grade_label = edu.get("grade_label", "")
            institution = edu.get("institution", "")
            research_dir = edu.get("research_direction")

            edu_line = f"- Education stage: {grade_label}"
            if institution:
                edu_line += f"（{institution}）"
            if research_dir:
                edu_line += f", research direction: {research_dir}"
            parts.append(edu_line)

        # Exams
        exams = life_path.get("exams_this_year", [])
        if exams:
            parts.append("- Exams this year:")
            for exam in exams:
                parts.append(
                    f"  · {exam.get('exam_name', '')}（{exam.get('date_range', '')}）："
                    f"{exam.get('description', '')}"
                )

        # Key events
        events = life_path.get("key_events", [])
        if events:
            parts.append("- Key events:")
            for evt in events:
                transition_tag = " ⚡stage transition" if evt.get("is_stage_transition") else ""
                parts.append(
                    f"  · {evt.get('event_name', '')}（{evt.get('date', '')}）："
                    f"{evt.get('description', '')}{transition_tag}"
                )

        # Career status
        career = life_path.get("career_status", {})
        if career and career.get("is_working"):
            parts.append(
                f"- Career status: {career.get('job_title', '')}, "
                f"{career.get('employer', '')}（{career.get('industry', '')}）"
            )
            if career.get("description"):
                parts.append(f"  {career['description']}")

        # Pathway notes
        notes = life_path.get("pathway_notes", "")
        if notes:
            parts.append(f"- Path notes: {notes}")

        parts.append(
            "⚠️ The above information is definitive fact and must be strictly followed during event generation. "
            "Do not fabricate events inconsistent with the above path."
        )

        return "\n".join(parts)

    def _format_society_enrichment_for_prompt(self, year: int) -> str:
        """Format society enrichment data for prompt injection.

        Uses the unified full-content approach — no truncation, no detail-level grading.
        Data length is controlled at generation time (Phase 0).

        Args:
            year: Calendar year to format.

        Returns:
            Formatted string for prompt injection, or empty string if no data.
        """
        enrichment = self._enrichment_cache.get(year)
        if not enrichment:
            return ""

        parts = []
        parts.append(f"## 🌍 {year} Social and Cultural Context")

        tech = enrichment.get("technology_media", "")
        if tech:
            parts.append(f"### Technology/Media Environment\n{tech}")

        culture = enrichment.get("cultural_atmosphere", "")
        if culture:
            parts.append(f"### Cultural Atmosphere\n{culture}")

        stage_env = enrichment.get("stage_environment", "")
        if stage_env:
            parts.append(f"### Stage Environment\n{stage_env}")

        return "\n\n".join(parts)

    def _merge_multi_year_enrichment(self, start_year: int, end_year: int) -> str:
        """Merge enrichment across multiple years for low-res events spanning >1 year.

        Strategy (token-efficient):
        - 1 year: return that year's full enrichment
        - 2 years: return both years' full enrichment
        - >2 years: first and last year full, middle years abbreviated

        Args:
            start_year: Start year of the time unit.
            end_year: End year of the time unit.

        Returns:
            Merged enrichment string for prompt injection.
        """
        if start_year == end_year:
            # Single year — use standard formatters
            parts = []
            lp = self._format_key_life_path_for_prompt(start_year)
            se = self._format_society_enrichment_for_prompt(start_year)
            if lp:
                parts.append(lp)
            if se:
                parts.append(se)
            return "\n\n".join(parts)

        years = list(range(start_year, end_year + 1))

        if len(years) == 2:
            # Two years — both full
            parts = []
            for yr in years:
                parts.append(f"### Year {yr}")
                lp = self._format_key_life_path_for_prompt(yr)
                se = self._format_society_enrichment_for_prompt(yr)
                if lp:
                    parts.append(lp)
                if se:
                    parts.append(se)
            return "\n\n".join(parts)

        # >2 years — first and last full, middle abbreviated
        parts = []
        for yr in years:
            if yr == years[0] or yr == years[-1]:
                # Full enrichment for first and last year
                parts.append(f"### Year {yr} (Full)")
                lp = self._format_key_life_path_for_prompt(yr)
                se = self._format_society_enrichment_for_prompt(yr)
                if lp:
                    parts.append(lp)
                if se:
                    parts.append(se)
            else:
                # Abbreviated for middle years
                parts.append(f"### Year {yr} (Abbreviated)")
                life_path = self._key_life_path_cache.get(yr, {})
                # Only exams + key_events + grade_label from life path
                edu = life_path.get("education_status", {})
                if edu:
                    parts.append(f"- Grade: {edu.get('grade_label', '')}")
                exams = life_path.get("exams_this_year", [])
                if exams:
                    parts.append(
                        "- Exams: " + ", ".join(e.get("exam_name", "") for e in exams)
                    )
                events = life_path.get("key_events", [])
                if events:
                    parts.append(
                        "- Key events: " + ", ".join(e.get("event_name", "") for e in events)
                    )

        return "\n\n".join(parts)

    # ── Enrichment cache persistence ─────────────────────────────

    def _load_enrichment_cache(self) -> None:
        """Load the year enrichment cache from disk if available.

        Called during __init__ to restore cached enrichments on resume,
        avoiding redundant LLM calls for year-specific context generation.
        """
        if not self._enrichment_cache_path:
            return
        if not os.path.isfile(self._enrichment_cache_path):
            return
        try:
            with open(self._enrichment_cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # JSON keys are strings; convert back to int
            loaded = 0
            for key, value in raw.items():
                try:
                    year = int(key)
                    self._enrichment_cache[year] = value
                    loaded += 1
                except (ValueError, TypeError):
                    logger.warning(f"[TC-Enrichment] Skipping invalid cache key: {key}")
            logger.info(
                f"[TC-Enrichment] Loaded {loaded} cached year enrichments "
                f"from {self._enrichment_cache_path}"
            )
        except Exception as e:
            logger.warning(
                f"[TC-Enrichment] Failed to load enrichment cache "
                f"from {self._enrichment_cache_path}: {e}"
            )

    def _save_enrichment_cache(self) -> None:
        """Persist the year enrichment cache to disk.

        Called after generating new enrichments so that subsequent runs
        (or resume) can skip the LLM calls for already-generated years.
        """
        if not self._enrichment_cache_path:
            return
        if not self._enrichment_cache:
            return
        try:
            # Ensure we save with the new filename (migrate from old name if needed)
            save_path = self._enrichment_cache_path
            if save_path and save_path.endswith("year_enrichment_cache.json"):
                save_path = save_path.replace(
                    "year_enrichment_cache.json",
                    "year_society_enrichment_cache.json",
                )
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            # JSON requires string keys; convert int years to strings
            serializable = {str(year): data for year, data in self._enrichment_cache.items()}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[TC-Enrichment] Saved {len(self._enrichment_cache)} year enrichments "
                f"to {save_path}"
            )
        except Exception as e:
            logger.warning(
                f"[TC-Enrichment] Failed to save enrichment cache "
                f"to {self._enrichment_cache_path}: {e}"
            )

    def _load_key_life_path_cache(self) -> None:
        """Load the key life path cache from disk if available.

        Called during __init__ to restore cached life path data,
        enabling P2 to inject personal path anchors into prompts.
        """
        if not self._key_life_path_cache_path:
            return
        if not os.path.isfile(self._key_life_path_cache_path):
            return
        try:
            with open(self._key_life_path_cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            loaded = 0
            for key, value in raw.items():
                try:
                    year = int(key)
                    self._key_life_path_cache[year] = value
                    loaded += 1
                except (ValueError, TypeError):
                    logger.warning(f"[KeyLifePath] Skipping invalid cache key: {key}")
            logger.info(
                f"[KeyLifePath] Loaded {loaded} cached key life path entries "
                f"from {self._key_life_path_cache_path}"
            )
        except Exception as e:
            logger.warning(
                f"[KeyLifePath] Failed to load key life path cache "
                f"from {self._key_life_path_cache_path}: {e}"
            )

    async def _get_or_generate_year_enrichment(
        self,
        year: int,
        period: Dict[str, Any],
        persona_brief: str = "",
    ) -> Dict[str, Any]:
        """Get or generate year-specific enrichment (technology, culture, stage environment).

        Cached by calendar year. One LLM call per year, shared across all events in that year.

        Args:
            year: Calendar year (e.g. 2003).
            period: Current life period dict (for stage context).
            persona_brief: Target persona brief text.

        Returns:
            Dict with keys: technology_media, cultural_atmosphere, stage_environment.
        """
        # Check cache first
        if year in self._enrichment_cache:
            return self._enrichment_cache[year]

        async with self._enrichment_semaphore:
            # Double-check cache inside semaphore (another coroutine may have filled it)
            if year in self._enrichment_cache:
                return self._enrichment_cache[year]

            # Build context from SocialContextProfile
            social_ctx = self._social_context
            edu_system = social_ctx.get("education_system", [])
            edu_summary = "; ".join(
                f"{s.get('stage_name', '?')} (entry_age={s.get('entry_age', '?')}, "
                f"duration={s.get('duration_years', '?')}y)"
                for s in edu_system
            ) or "N/A"

            career_norms = social_ctx.get("career_path_norms", {})
            if isinstance(career_norms, dict):
                # career_path_norms is a single CareerPathNorm object (dict), not a list
                career_summary = (
                    f"entry_age={career_norms.get('typical_entry_age_min', '?')}-"
                    f"{career_norms.get('typical_entry_age_max', '?')}, "
                    f"prerequisites={career_norms.get('typical_prerequisites', [])}, "
                    f"notes={career_norms.get('notes', '')}"
                )
            elif isinstance(career_norms, list):
                career_summary = "; ".join(
                    f"{n.get('norm_name', '?')}: {n.get('description', '?')}"
                    for n in career_norms
                    if isinstance(n, dict)
                ) or "N/A"
            else:
                career_summary = "N/A"

            institutions = social_ctx.get("social_institutions", [])
            inst_summary = "; ".join(
                f"{i.get('institution_name', '?')}: {i.get('description', '?')}"
                for i in institutions
            ) or "N/A"

            # Compute age in this year
            birth_date_str = getattr(self._tc_computer, 'birth_date', '') if self._tc_computer else ''
            age_in_year = "unknown"
            if birth_date_str:
                try:
                    birth_year = int(birth_date_str.split("-")[0])
                    age_in_year = str(year - birth_year)
                except (ValueError, IndexError):
                    pass

            city = social_ctx.get("city", "") or self.persona.get("growing_up_location", {}).get("city", "unknown")
            country = social_ctx.get("country", "")
            stage_label = period.get("stage_label", "")
            stage_title = period.get("title", "")
            is_edu = period.get("is_education_stage", False)
            is_work = period.get("is_work_stage", False)

            stage_type_hint = ""
            if is_edu and is_work:
                stage_type_hint = "education + work stage"
            elif is_edu:
                stage_type_hint = "education stage"
            elif is_work:
                stage_type_hint = "work stage"
            else:
                stage_type_hint = "life stage"

            system_prompt = (
                "You are a social and cultural expert. Generate dynamic, year-specific context "
                "that supplements the structural social context already available.\n\n"
                "Rules:\n"
                "1. All information must be historically accurate for the specified year\n"
                "2. Do NOT repeat structural information already provided\n"
                "3. Focus on concrete, specific details that can be used in scene descriptions\n"
                "4. Do NOT project later technologies or policies backward\n"
                "5. Respond in English. The output will be used in an English-language simulation."
            )

            user_prompt = (
                f"## Existing structural information (do not regenerate)\n"
                f"- Education system: {edu_summary}\n"
                f"- Career path norms: {career_summary}\n"
                f"- Social institutions: {inst_summary}\n\n"
                f"## Please supplement the following dynamic information for year {year}\n\n"
                f"### Character age in {year}: {age_in_year}\n"
                f"### Character's city: {city}\n"
                f"### Character's country: {country}\n"
                f"### Character's life stage: {stage_title} ({stage_label}) — {stage_type_hint}\n"
                f"### Character brief: {persona_brief if persona_brief else 'N/A'}\n\n"
                f"Generate a JSON with exactly these three keys:\n"
                f"1. **technology_media** (string, max 500 words): "
                f"List 3-5 key technology/communication/media features for {year} in {city or country}. "
                f"Each item 1-2 sentences. Include: device types, social platforms, internet access, "
                f"popular apps, communication methods, etc. "
                f"Focus on what was ACTUALLY prevalent in {city or country} at that time.\n"
                f"2. **cultural_atmosphere** (string, max 400 words): "
                f"List 2-3 key cultural/social trend highlights for {year} in {city or country}. "
                f"Each item 1-2 sentences. Include: popular culture, social events, generational characteristics, etc.\n"
                f"3. **stage_environment** (string, max 600 words): "
            )

            if is_edu:
                user_prompt += (
                    f"Describe 3-5 core scene features of this educational stage in {year} in {city or country}. "
                    f"Each item 1-2 sentences. Include: curriculum/exam system, campus environment, student culture, etc. "
                    f"Focus on perceptible details directly relevant to scene description.\n"
                )
            elif is_work:
                user_prompt += (
                    f"Describe 3-5 core scene features of this professional stage in {year} in {city or country}. "
                    f"Each item 1-2 sentences. Include: industry characteristics, work pace, workplace culture, etc. "
                    f"Focus on perceptible details directly relevant to scene description.\n"
                )
            else:
                user_prompt += (
                    f"Describe 3-5 core scene features of this life stage in {year} in {city or country}. "
                    f"Each item 1-2 sentences. Include: daily living environment, community characteristics, etc.\n"
                )

            user_prompt += (
                f"\n⚠️ Keep each field strictly within the word limits above. Be concise rather than omitting key information."
                f"\n⚠️ Strictly base all content on the real historical situation of {year} in {city or country}. "
                f"Do NOT project later policies or technologies backward. "
                f"Do NOT default to Chinese or American cultural context — use the culture of {city or country}."
                f"\n\nHARD LIMIT: Each field must be under the character limit specified in its description.\n"
                f"technology_media: ≤500 chars\n"
                f"cultural_atmosphere: ≤400 chars\n"
                f"stage_environment: ≤600 chars\n"
                f"If you exceed the limit, truncate at a natural sentence boundary."
            )

            try:
                from pydantic import BaseModel, Field as PydanticField

                class YearEnrichmentOutput(BaseModel):
                    technology_media: str = PydanticField(
                        ...,
                        description="Technology and media environment for this year. "
                        "3-5 key items, each 1-2 sentences. MUST NOT exceed 500 words."
                    )
                    cultural_atmosphere: str = PydanticField(
                        ...,
                        description="Cultural atmosphere and social trends for this year. "
                        "2-3 key items, each 1-2 sentences. MUST NOT exceed 400 words."
                    )
                    stage_environment: str = PydanticField(
                        ...,
                        description="Stage-specific environment details for this year. "
                        "3-5 key items, each 1-2 sentences. MUST NOT exceed 600 words."
                    )

                result: YearEnrichmentOutput = await self.llm.generate_structured(
                    prompt=user_prompt,
                    response_model=YearEnrichmentOutput,
                    system_prompt=system_prompt,
                    task_type="p2_year_enrichment",
                    temperature=0.0,
                )

                enrichment = {
                    "technology_media": result.technology_media,
                    "cultural_atmosphere": result.cultural_atmosphere,
                    "stage_environment": result.stage_environment,
                    "year": year,
                }

                # Save-time validation: warn but never truncate (source-control principle)
                _FIELD_CHAR_LIMITS = {
                    "technology_media": 500,
                    "cultural_atmosphere": 400,
                    "stage_environment": 600,
                }
                for field, limit in _FIELD_CHAR_LIMITS.items():
                    actual_len = len(enrichment[field])
                    if actual_len > limit * 1.2:  # warn only if exceeding 120% of soft limit
                        logger.warning(
                            f"[TC-Enrichment] Year {year}, field '{field}' exceeds soft limit: "
                            f"{actual_len} chars (limit: {limit}). Consider adjusting generation prompt."
                        )

                self._enrichment_cache[year] = enrichment
                logger.info(f"[TC-Enrichment] Generated and cached enrichment for year {year}")
                # Incremental save: persist immediately so progress is not lost if gather is interrupted
                self._save_enrichment_cache()
                return enrichment

            except Exception as e:
                logger.warning(f"[TC-Enrichment] Failed to generate enrichment for year {year}: {e}")
                fallback = {
                    "technology_media": f"Infer based on the actual technology environment of {year}",
                    "cultural_atmosphere": f"Infer based on the actual cultural atmosphere of {year}",
                    "stage_environment": f"Infer based on the actual stage environment of {year}",
                    "year": year,
                }
                self._enrichment_cache[year] = fallback
                return fallback

    async def pregenerate_all_year_enrichments(
        self,
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
    ) -> int:
        """Pre-generate year enrichments for ALL years across ALL periods in parallel.

        This should be called once before the P2 period iteration loop begins.
        It collects all unique years from all life periods (or derived_memory_plan
        segments) and generates their enrichments in a single parallel batch,
        avoiding the per-period serial generation overhead.

        Args:
            life_plan: The complete life plan dict (with life_periods and optionally
                       derived_memory_plan).
            persona_config: The persona configuration dict.

        Returns:
            Number of newly generated enrichments (excludes cache hits).
        """
        # Initialize social context if not already done (needed by enrichment prompts)
        if not self._social_context:
            self._social_context = (
                life_plan.get("global_summary", {}).get("social_context", {})
            )

        # Initialize TC computer if not already done
        if not self._tc_computer:
            birth_date = (
                persona_config.get("derived_birth_date")
                or persona_config.get("birth_date")
                or (persona_config.get("global_summary", {})
                    .get("timeline_anchor", {})
                    .get("derived_birth_date", ""))
            )
            life_periods_list = life_plan.get("life_periods", [])
            if birth_date and self._social_context:
                from lifelong_synth.configs.temporal_context import TemporalContextComputer
                self._tc_computer = TemporalContextComputer(
                    birth_date=birth_date,
                    social_context=self._social_context,
                    life_periods=life_periods_list,
                )

        # Collect all unique years from all periods / segments
        all_years: set = set()
        periods_by_year: Dict[int, Dict[str, Any]] = {}

        life_periods = life_plan.get("life_periods", [])
        derived_segments = life_plan.get("derived_memory_plan", {}).get("segments", [])

        # Build period lookup
        period_lookup = {p["period_id"]: p for p in life_periods}

        if derived_segments:
            for seg in derived_segments:
                start_str = seg.get("start_date", "")
                end_str = seg.get("end_date", "")
                parent_period = period_lookup.get(seg.get("parent_period_id", ""))
                if start_str and end_str and parent_period:
                    try:
                        start_year = int(start_str[:4])
                        end_year = int(end_str[:4])
                        for yr in range(start_year, end_year + 1):
                            all_years.add(yr)
                            if yr not in periods_by_year:
                                periods_by_year[yr] = parent_period
                    except (ValueError, TypeError):
                        pass
        else:
            for period in life_periods:
                dr = period.get("period_date_range", {})
                start_str = dr.get("start_date", "")
                end_str = dr.get("end_date", "")
                if start_str and end_str:
                    try:
                        start_year = int(start_str[:4])
                        end_year = int(end_str[:4])
                        for yr in range(start_year, end_year + 1):
                            all_years.add(yr)
                            if yr not in periods_by_year:
                                periods_by_year[yr] = period
                    except (ValueError, TypeError):
                        pass

        # Filter out already-cached years
        missing_years = sorted(yr for yr in all_years if yr not in self._enrichment_cache)

        if not missing_years:
            logger.info(
                f"[TC-Enrichment-Global] All {len(all_years)} years already cached; "
                f"skipping pre-generation"
            )
            return 0

        persona_brief = persona_config.get("persona_brief_text", "")
        logger.info(
            f"[TC-Enrichment-Global] Pre-generating enrichments for {len(missing_years)} years "
            f"in parallel (total unique years: {len(all_years)}, "
            f"already cached: {len(all_years) - len(missing_years)})"
        )

        await asyncio.gather(*[
            self._get_or_generate_year_enrichment(
                year=yr,
                period=periods_by_year.get(yr, life_periods[0] if life_periods else {}),
                persona_brief=persona_brief,
            )
            for yr in missing_years
        ], return_exceptions=True)

        succeeded = sum(1 for yr in missing_years if yr in self._enrichment_cache)
        failed = len(missing_years) - succeeded
        logger.info(
            f"[TC-Enrichment-Global] Pre-generation complete: "
            f"{succeeded} new enrichments, {failed} failed (fallback used), "
            f"cache size: {len(self._enrichment_cache)}"
        )
        self._save_enrichment_cache()
        return succeeded

    async def pregenerate_all_persona_current_states(
        self,
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
        output_dir: str,
    ) -> int:
        """T17: Pre-generate persona current state cache for all periods.

        Generates period-level occupation/education/values snapshots for P_TARGET,
        so that _build_structured_persona_context() can use accurate period-level
        values instead of the terminal (birth-time) values.

        Args:
            life_plan: The complete life plan dict.
            persona_config: The persona configuration dict.
            output_dir: Output directory for caching.

        Returns:
            Number of newly generated states (0 if loaded from cache).
        """
        import os as _os
        import json as _json

        cache_path = _os.path.join(output_dir, "persona_current_state_cache.json")

        # Resume support: load existing cache
        if _os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._persona_current_state_cache = _json.load(f)
                logger.info(
                    f"[P1d] Loaded persona current state cache from {cache_path} "
                    f"({len(self._persona_current_state_cache)} periods)"
                )
                return 0
            except Exception as e:
                logger.warning(f"[P1d] Failed to load cache from {cache_path}: {e}; regenerating")

        life_periods = life_plan.get("life_periods", [])
        if not life_periods:
            logger.warning("[P1d] No life_periods found; skipping persona current state pre-generation")
            return 0

        persona_name = persona_config.get("persona_name_text", "protagonist")
        persona_brief = persona_config.get("persona_brief_text", "")
        birth_date_str = (
            persona_config.get("derived_birth_date")
            or persona_config.get("birth_date")
            or ""
        )

        # Build life plan summary for context
        life_plan_summary = ""
        for lp in life_periods:
            pid = lp.get("period_id", "")
            title = lp.get("title", "")
            start = lp.get("start_date", "")
            end = lp.get("end_date", "")
            life_plan_summary += f"- {pid} ({start}~{end}): {title}\n"

        from pydantic import BaseModel, Field

        class _PersonaStateOutput(BaseModel):
            target_occupation_group_current: Optional[str] = Field(
                default=None,
                description=(
                    "Occupation category at this period's start. "
                    "Must be one of: student, manager_executive, professional_finance_law_consulting, "
                    "professional_tech_research, professional_health_education, office_admin_support, "
                    "sales_customer_service, service_hospitality_retail, skilled_trades_technical_ops, "
                    "manual_logistics_transport, public_service_military, self_employed_creator, "
                    "retired, unemployed, homemaker. null if not applicable."
                )
            )
            target_education_level_current: Optional[str] = Field(
                default=None,
                description=(
                    "Highest completed education level (ISCED 0-8) at this period's start. "
                    "e.g. '0','1','2','3','4','5','6','7','8'. null if unknown."
                )
            )
            core_values_schwartz_top3: Optional[list] = Field(
                default=None,
                description=(
                    "Top 3 Schwartz core values at this period (may evolve over time). "
                    "e.g. ['achievement', 'benevolence', 'self-direction']. null if unknown."
                )
            )

        async def _generate_one(period: Dict[str, Any]) -> tuple:
            pid = period.get("period_id", "")
            start = period.get("start_date", "")
            title = period.get("title", "")
            tasks_list = period.get("developmental_tasks", [])
            tasks_str = "; ".join(tasks_list[:3]) if tasks_list else ""

            # Compute approximate age at start of this period
            _age_at_period_start = ""
            try:
                from datetime import date as _date_cls
                if birth_date_str and start:
                    _dob_p = _date_cls.fromisoformat(birth_date_str)
                    _start_p = _date_cls.fromisoformat(start)
                    _age_p = _start_p.year - _dob_p.year - (
                        1 if (_start_p.month, _start_p.day) < (_dob_p.month, _dob_p.day) else 0
                    )
                    _age_at_period_start = f"Approximate age at period start: {_age_p} years old\n"
            except Exception:
                pass

            _age_constraint = ""
            if _age_at_period_start:
                try:
                    _age_val = int(_age_at_period_start.split(":")[1].split("years")[0].strip())
                    if _age_val < 3:
                        _age_constraint = (
                            "AGE CONSTRAINT: This character is an infant/toddler (under 3 years old). "
                            "core_values_schwartz_top3 should reflect basic temperament tendencies "
                            "(e.g. security, benevolence, stimulation) — NOT adult values like hedonism, "
                            "achievement, or power. target_occupation_group must be null.\n"
                        )
                    elif _age_val < 6:
                        _age_constraint = (
                            "AGE CONSTRAINT: This character is a young child (3-5 years old). "
                            "core_values_schwartz_top3 should reflect early childhood tendencies "
                            "(e.g. security, benevolence, stimulation, conformity). "
                            "target_occupation_group must be null.\n"
                        )
                except Exception:
                    pass

            prompt = (
                f"Character: {persona_name}\n"
                f"Birth date: {birth_date_str}\n"
                f"{_age_at_period_start}"
                f"Background: {persona_brief[:300]}\n\n"
                f"Life plan overview:\n{life_plan_summary}\n"
                f"Current period: {pid} ({start}) — {title}\n"
                f"Developmental tasks: {tasks_str}\n\n"
                f"{_age_constraint}"
                f"Based on the life plan, infer this character's occupation, education level, "
                f"and core values at the START of period {pid}."
            )
            try:
                result: _PersonaStateOutput = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=_PersonaStateOutput,
                    system_prompt=(
                        "You are inferring a character's life state at a specific period. "
                        "Be accurate based on the life plan context."
                    ),
                    temperature=0.0,
                    task_type="p1d_persona_current_state",
                )
                state = {
                    "target_occupation_group_current": result.target_occupation_group_current,
                    "target_education_level_current": result.target_education_level_current,
                    "self_system_anchor_current": {
                        "core_values_schwartz_top3": result.core_values_schwartz_top3 or []
                    },
                }
                return pid, state
            except Exception as e:
                logger.warning(f"[P1d] Failed to generate state for {pid}: {e}")
                return pid, {}

        logger.info(f"[P1d] Generating persona current states for {len(life_periods)} periods in parallel...")
        results = await asyncio.gather(*[_generate_one(lp) for lp in life_periods])

        for pid, state in results:
            if state:
                # FIX-08: Normalize Schwartz value names: replace hyphens with underscores
                # e.g. "self-direction" → "self_direction"
                anchor = state.get("self_system_anchor_current", {})
                if anchor and isinstance(anchor.get("core_values_schwartz_top3"), list):
                    anchor["core_values_schwartz_top3"] = [
                        v.replace("-", "_") for v in anchor["core_values_schwartz_top3"]
                    ]
                self._persona_current_state_cache[pid] = state

        # FIX-05: Post-process: enforce monotonic non-decreasing education level
        # Education level should never decrease over time (you can't "un-complete" a degree).
        import re as _re
        def _lp_sort_key(pid: str) -> int:
            m = _re.match(r"LP(\d+)", pid)
            return int(m.group(1)) if m else 999

        sorted_pids = sorted(self._persona_current_state_cache.keys(), key=_lp_sort_key)
        max_edu_seen = -1
        for pid in sorted_pids:
            state = self._persona_current_state_cache[pid]
            edu_str = state.get("target_education_level_current")
            if edu_str is not None and str(edu_str).isdigit():
                edu_int = int(edu_str)
                if edu_int < max_edu_seen:
                    logger.warning(
                        f"[P1d] Education level regression detected: {pid} has level '{edu_str}' "
                        f"but previous max was '{max_edu_seen}'. Correcting to '{max_edu_seen}'."
                    )
                    state["target_education_level_current"] = str(max_edu_seen)
                else:
                    max_edu_seen = edu_int

        # Save cache
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                _json.dump(self._persona_current_state_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"[P1d] Saved persona current state cache to {cache_path}")
        except Exception as e:
            logger.warning(f"[P1d] Failed to save cache: {e}")

        return len(self._persona_current_state_cache)

    def _restore_event_counters_from_memory(self) -> None:
        """Restore per-period event counters from existing memory state."""
        pattern = re.compile(r"^(LP\d+)_E(\d+)$")
        for event_id in self.memory.get_all_events().keys():
            match = pattern.match(event_id)
            if not match:
                continue
            period_id = match.group(1)
            sequence = int(match.group(2))
            self._event_counters[period_id] = max(
                self._event_counters.get(period_id, 0),
                sequence,
            )

    # ── Persona pool persistence helper ────────────────────────

    def _track_perf(self, name: str, **tags: Any):
        tracker = self._performance_tracker
        if self._run_id and "run_id" not in tags:
            tags["run_id"] = self._run_id
        if tracker:
            return tracker.track(name, **tags)
        return nullcontext()

    def _save_pool(self) -> None:
        """Persist the participant pool to its JSON file (if a path is configured)."""
        if self._pool_save_path:
            self.pool.save(self._pool_save_path)
        else:
            self.pool.save()

    def _save_memory(self) -> None:
        """Persist memory using the explicit path when available."""
        if self._memory_save_path:
            self.memory.save(self._memory_save_path)
            return
        if getattr(self.memory, "default_file_path", None):
            self.memory.save()
            return
        raise ValueError(
            "No memory save path configured for EventOrganiser. "
            "Provide memory_base_path explicitly or initialize MemoryManager with a default path."
        )

    def _upsert_interaction_history(
        self,
        participant_id: str,
        event_id: str,
        date_str: str,
        summary: str,
        emotional_tone: Optional[str] = None,
        time_period: Optional[str] = None,
    ) -> None:
        """Insert or replace a participant interaction history record by event_id.

        Args:
            date_str: Representative date (YYYY-MM-DD) for recency calculations.
                      For period-spanning events, use the END date of the period.
            time_period: Accurate time period string. Supports:
                         'YYYY-MM-DD', 'YYYY-MM-DD to YYYY-MM-DD', 'YYYY-MM'.
                         Falls back to date_str if None.
        """
        participant = self.pool.get_participant(participant_id)
        if not participant:
            raise KeyError(f"Participant '{participant_id}' not found in pool")

        history = participant.interactions_history_with_the_main_character
        replaced = False
        for idx, record in enumerate(history):
            if record.event_id == event_id:
                history[idx].time_period = time_period or date_str
                history[idx].date = date_str
                history[idx].summary = summary
                history[idx].emotional_tone = emotional_tone
                replaced = True
                break

        if not replaced:
            self.pool.update_interaction_history(
                participant_id=participant_id,
                event_id=event_id,
                date=date_str,
                summary=summary,
                emotional_tone=emotional_tone,
                time_period=time_period,
            )

    def _validate_pool_interactions(self, participant_ids: List[str], event_id: str) -> None:
        """Ensure all listed participants have the given event recorded in interaction history."""
        for pid in participant_ids:
            participant = self.pool.get_participant(pid)
            if not participant:
                raise ValueError(f"Participant '{pid}' not found in pool during validation")
            if not any(rec.event_id == event_id for rec in participant.interactions_history_with_the_main_character):
                raise ValueError(
                    f"Participant '{pid}' missing interaction history for event '{event_id}'"
                )

    # ── Event ID generation ──────────────────────────────────────

    def _next_event_id(self, period_id: str) -> str:
        """Generate the next sequential event ID for a period."""
        count = self._event_counters.get(period_id, 0) + 1
        self._event_counters[period_id] = count
        return f"{period_id}_E{count:03d}"

    # ── Context builders ─────────────────────────────────────────

    def _build_participant_pool_context(self, period_id: str) -> str:
        """
        Build a detailed text block describing ALL available participants
        for participant selection (Step 2). Includes role, relationship,
        current persona, and target persona.
        """
        lines = []
        for p in self.pool.list_all():
            appear = p.appear_period
            if appear and appear != "LP_UNKNOWN":
                try:
                    appear_num = int(appear.replace("LP", ""))
                    period_num = int(period_id.replace("LP", ""))
                    if appear_num > period_num:
                        continue
                except ValueError:
                    pass

            role_str = f"({p.role})" if p.role != "target" else "(target character)"
            current_brief = p.current_persona_brief_text or ""
            target_brief = p.target_persona_brief_text or ""
            lines.append(
                f"- {p.participant_id}: {p.persona_name_text} {role_str}\n"
                f"  Relationship: {p.relationship_towards_the_main_character}\n"
                f"  Current status: {current_brief}\n"
                f"  Target status: {target_brief}"
            )
        return "\n".join(lines) if lines else "No participant information available"

    def _build_participant_brief_context(self, period_id: str) -> str:
        """Build a brief participant list for event framework generation (Step 1)."""
        lines = []
        for p in self.pool.list_all():
            appear = p.appear_period
            if appear and appear != "LP_UNKNOWN":
                try:
                    appear_num = int(appear.replace("LP", ""))
                    period_num = int(period_id.replace("LP", ""))
                    if appear_num > period_num:
                        continue
                except ValueError:
                    pass

            role_str = f"({p.role})" if p.role != "target" else "(target character)"
            lines.append(
                f"- {p.participant_id}: {p.persona_name_text} {role_str} — "
                f"{p.relationship_towards_the_main_character}"
            )
        return "\n".join(lines) if lines else "No participant information available"

    def _build_memory_context(self, current_date=None, period_id="") -> str:
        """Build memory context using the unified retriever API."""
        return self.memory.retriever.build_context(
            mode="general",
            current_date=current_date,
            period_id=period_id,
        )

    # ── Structured persona context builder ───────────────────────

    async def _build_structured_persona_context(
        self, period: Dict[str, Any], unit_start: date
    ) -> str:
        """Build structured, stage-aware persona context for prompt.

        Replaces the flat persona_brief_text with a layered presentation
        that highlights core traits, computed age, and stage relevance.
        """
        name = self.persona.get("persona_name_text", "protagonist")
        brief = self.persona.get("persona_brief_text", "")

        # Compute current age from birth date and unit_start
        birth_date_str = (
            self.persona.get("derived_birth_date")
            or self.persona.get("birth_date")
            or (self.persona.get("global_summary", {})
                .get("timeline_anchor", {})
                .get("derived_birth_date"))
        )
        if not birth_date_str:
            logger.warning(
                "No birth_date found in persona config; "
                "age computation will be skipped for structured persona context"
            )
            age = None
        else:
            birth = date.fromisoformat(birth_date_str)
            age = unit_start.year - birth.year - (
                1 if (unit_start.month, unit_start.day) < (birth.month, birth.day) else 0
            )

        # Extract structured attributes with safe fallbacks
        gender_code = self.persona.get("gender_identity_code", "")
        _GENDER_MAP = {
            "1_male": "male",
            "2_female": "female",
            "3_non_binary": "non-binary",
            "4_other": "other",
            "Z_not_stated": "not stated",
        }
        gender = _GENDER_MAP.get(gender_code, "unknown")

        location = self.persona.get("growing_up_location", {})
        province = location.get("province", "")
        city = location.get("city", "")
        location_str = f"{province} {city}".strip() if province or city else "unknown"

        current_loc = self.persona.get("current_living_location", {})
        current_province = current_loc.get("province", "")
        current_city = current_loc.get("city", "")
        current_location_str = f"{current_province}{current_city}" if current_province or current_city else ""

        period_id = period.get("period_id", "LP?")

        # ── T20: Priority: read from period-level current state cache ──
        current_state = self._persona_current_state_cache.get(period_id, {})
        if current_state.get("self_system_anchor_current"):
            values = current_state["self_system_anchor_current"].get("core_values_schwartz_top3", [])
        else:
            self_system = self.persona.get("self_system_anchor", {})
            values = self_system.get("core_values_schwartz_top3", [])
        attachment = self.persona.get("adult_attachment_rq4cat", "")
        bond = self.persona.get("caregiver_bond_pbi", "")
        title = period.get("title", "")
        tasks = period.get("developmental_tasks", [])
        stage_label = period.get("stage_label", "")

        # ── Infer stage-appropriate location ──
        # Pre-university stages (childhood through senior high school) should
        # default to growing_up_location unless the period title explicitly
        # mentions a different city.  This prevents the LLM from confusing
        # the university city with the hometown.
        pre_university_labels = {
            "early_childhood", "preschool", "kindergarten",
            "primary_school", "junior_middle_school", "senior_middle_school",
            "infancy", "toddler",
        }
        is_pre_university = stage_label in pre_university_labels
        stage_location_hint = ""
        if is_pre_university and location_str and location_str != "unknown":
            stage_location_hint = (
                f"- ⚠️ Current stage location: {location_str} (growing-up area; all stages through high school are here, "
                f"do not confuse with later university/work cities)"
            )
        elif current_location_str and not is_pre_university:
            # For post-education / work stages, hint current living location
            if period.get("is_work_stage", False):
                stage_location_hint = (
                    f"- ⚠️ Current residence/work location: {current_location_str} "
                    f"(protagonist has relocated here; do NOT use growing-up city {location_str} "
                    f"as the scene location for this stage)"
                )

        age_line = (
            f"- Current age: {age} (date of birth: {birth_date_str})"
            if age is not None
            else "- Current age: unknown"
        )
        parts = [
            "## Target Character\n",
            "### Basic Information",
            f"- Name: {name}",
            f"- Gender: {gender}",
            age_line,
            f"- Growing-up location: {location_str}",
        ]
        if stage_location_hint:
            parts.append(stage_location_hint)

        # Core personality traits (only if structured data available)
        trait_lines = []
        if values:
            trait_lines.append(f"- Core values: {', '.join(values)}")
        if attachment:
            trait_lines.append(f"- Attachment style: {attachment}")
        if bond:
            trait_lines.append(f"- Caregiver bond influence: {bond}")
        if trait_lines:
            parts.append("\n### Core Personality Traits")
            parts.extend(trait_lines)

        # Persona brief as a sub-section
        parts.append(f"\n### Character Biography")
        parts.append(brief)

        # Stage relevance
        parts.append(f"\n### Current Stage Context")
        parts.append(f"- Currently in {period_id} ({title})")
        if tasks:
            parts.append(f"- Stage developmental tasks: {', '.join(tasks)}")
        if age is not None:
            parts.append(
                f"- Character age during this stage: {age}. "
                f"Events should match the cognitive level and behavioral traits of this age group."
            )

        # Persona extensions (benchmark-specific attributes)
        extensions_text = format_persona_extensions(
            self.persona.get("persona_extensions") or {}
        )
        if extensions_text:
            parts.append(f"\n{extensions_text}")
            # If specific_attitudes are present, add explicit guidance
            ext = self.persona.get("persona_extensions") or {}
            if ext.get("specific_attitudes"):
                parts.append(
                    "\n> **Attitude Guidance**: The 'Specific Attitudes' listed above are "
                    "confirmed character traits. When generating events for this period, "
                    "naturally reflect these attitudes in the character's reactions, choices, "
                    "and interactions — especially when the event topic overlaps with an attitude target."
                )
            # Identity trait guidance — static fallback only; LLM-generated instruction
            # (identity_salience_instruction) is appended below after signals are fetched.
            if ext.get("identity_traits"):
                parts.append(
                    "\n> **Identity Trait Guidance**: The 'Core Identity Traits' listed above are "
                    "defining characteristics of who this person IS. When generating events for this period, "
                    "consider whether any of these identity traits would naturally surface. "
                    "Do NOT force every event to address identity — only include identity-relevant events "
                    "when they fit naturally within the period's theme and developmental stage."
                )
            # Occupational register guidance (new — §2)
            if ext.get("occupational_register"):
                parts.append(
                    "\n> **Occupational Register Guidance**: The 'Occupational Speech Register' listed above "
                    "describes the character's occupation-specific speech patterns and jargon. "
                    "When generating high-resolution events, ensure the character's dialogue and "
                    "internal monologue naturally use this register — including domain-specific vocabulary, "
                    "professional shorthand, and workplace-specific expressions."
                )

        # Cultural Grounding + Occupation Episodic Depth — delegated to an LLM classifier.
        # See §5.3.3 / Task T29 for the _PersonaContextSignals Pydantic model and the
        # _llm_persona_context_signals() method. The result is cached on self to pay
        # the LLM cost at most once per persona (this method is called 5× per period).
        signals = await self._llm_persona_context_signals()
        if signals.cultural_grounding_needed:
            parts.append(
                f"\n> **Cultural Grounding**: {signals.cultural_grounding_instruction}\n"
                f"The simulation still produces English text, but embed culturally authentic "
                f"details, references, and social dynamics as described above."
            )
        if signals.occupation_depth_needed:
            parts.append(
                f"\n> **Occupation Episodic Depth**: {signals.occupation_depth_instruction}\n"
                f"When generating high-resolution events, ensure at least one SPECIFIC, detailed "
                f"occupation-related episode that demonstrates domain expertise — a specific challenge "
                f"solved, a specific technique used, or a specific interaction that reveals professional "
                f"knowledge. The episode should be concrete enough that the character could recount it in "
                f"detail when asked about their professional experiences. Respond in natural English; "
                f"do not invent facts beyond what the persona brief and period theme support."
            )

        # ── Identity Salience Instruction (LLM-generated, persona-specific) ──
        if signals.identity_salience_needed and signals.identity_salience_instruction:
            parts.append(
                f"\n> **Identity Salience Guidance**: {signals.identity_salience_instruction}\n"
                f"This instruction is persona-specific. Apply it only when the period's theme "
                f"and developmental stage make it natural — do not force identity events into "
                f"every period."
            )

        return "\n".join(parts)

    # ================================================================
    # Step 1a (Batch): Batch-generate LR frameworks for consecutive LR-only units
    # ================================================================

    async def _step1a_batch_generate_all_lr(
        self,
        period: Dict[str, Any],
        time_units: List[Tuple[date, date]],
        unit_labels: List[str],
        density: str,
        detail_budget: int = 999,
        outline_budget: int = 999,
        total_units: int = 0,
        batch_start_index: int = 0,
    ) -> BatchLROutput:
        """Batch-generate LR frameworks for multiple consecutive LR-only time units.

        Uses a single LLM prompt to generate all LR events in the batch,
        reducing N LLM calls → 1. Each time unit in the batch gets its own
        LowResFrameworkOutput.

        Args:
            period: The life stage period dict.
            time_units: List of (unit_start, unit_end) for consecutive units.
            unit_labels: Corresponding labels for each unit (e.g., "month 3/12").
            density: Memory density for this period.
            detail_budget: Remaining detail budget.
            outline_budget: Remaining outline budget.
            total_units: Total number of time units in the period.
            batch_start_index: Starting unit index in the overall period.

        Returns:
            BatchLROutput with one LowResFrameworkOutput per time unit.
        """
        n_units = len(time_units)
        period_id = period.get("period_id", "LP?")
        title = period.get("title", "")
        theme = period.get("dominant_theme", "")
        tasks = period.get("developmental_tasks", [])
        goals = period.get("stage_goals", [])
        pressures = period.get("salient_pressures", [])
        opportunities = period.get("salient_opportunities", [])

        structured_persona_ctx = await self._build_structured_persona_context(period, time_units[0][0])
        memory_ctx = self._build_memory_context()

        # ── Enrichment injection (multi-year for low-res) ──
        evt_start_year = time_units[0][0].year
        evt_end_year = time_units[-1][1].year
        enrichment_block = self._merge_multi_year_enrichment(evt_start_year, evt_end_year)

        # ── Milestone prior injection ──
        milestone_skel = self.get_milestone_for_period(period_id)
        milestone_prior_block = ""
        if milestone_skel:
            ms_name = milestone_skel.get('milestone_name', '')
            ms_date = milestone_skel.get('expected_date_range', '')
            ms_summary = milestone_skel.get('summary_hint', '')
            ms_coherence = milestone_skel.get('low_res_coherence_constraints', '')
            # Check if milestone falls within any unit in this batch
            is_milestone_in_batch = any(
                self.milestone_matches_time_unit(period_id, u_s, u_e)
                for u_s, u_e in time_units
            )
            if is_milestone_in_batch:
                milestone_prior_block = f"""
## 🎯 One Time Unit in This Batch Contains a Pre-Planned Key Milestone
has_key_event MUST be "key_event_detail" for that unit:
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}

The LR summary for that specific unit MUST center on this milestone.
{ms_coherence}
"""
            else:
                milestone_prior_block = f"""
## 📌 Stage Pre-Planned Key Milestone (Reference Information)
This stage has a pre-planned key milestone event (not within this batch):
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}

Your low-res events should reflect the proximity to the milestone accordingly.
{ms_coherence}
"""

        # v8: has_key_event removed from Step 1a output; determined by Step 1b later.
        memory_budget_note = f"""
## Memory Budget for This Period
- Batch size: {n_units} time units
- Time units: {', '.join(unit_labels)}
- Remaining time units after this batch: {total_units - batch_start_index - n_units} / {total_units}
- Detailed events (key_event_detail) budget: {detail_budget}
- Outline events (key_event_outline) budget: {outline_budget}

Note: Key event determination is handled automatically in a later step.
Focus on generating high-quality low-resolution event summaries.
"""

        system_prompt = (
            "You are a life simulation event generator. "
            "Based on the target character's persona, current life stage information, and memory bank, "
            "generate low-resolution event frameworks AND habitual event patterns for MULTIPLE consecutive time units in a single response.\n"
            "[IMPORTANT] Each time unit gets a summary-level event — only 1-2 sentences "
            "describing what happened during that period. Focus on who/what/when/where. "
            "No psychological description, environmental detail, or emotional arc. Keep it concise.\n"
            "Maintain narrative continuity between consecutive time units.\n"
            "Also generate 1-2 habitual/recurring event patterns per time unit in `habitual_events_per_unit`."
        )

        # Build per-unit time blocks
        unit_blocks = []
        for i, (u_start, u_end) in enumerate(time_units):
            unit_idx = batch_start_index + i
            unit_blocks.append(
                f"### Time Unit {i+1}/{n_units}: {unit_labels[i]}\n"
                f"- Time range: {u_start.isoformat()} to {u_end.isoformat()}\n"
                f"- Unit index: {unit_idx}"
            )

        user_prompt = f"""{structured_persona_ctx}

## Current Life Stage
- Stage ID: {period_id}
- Title: {title}
- Dominant theme: {theme}
- Developmental tasks: {json.dumps(tasks, ensure_ascii=False)}
- Stage goals: {json.dumps(goals, ensure_ascii=False)}
- Main pressures: {json.dumps(pressures, ensure_ascii=False)}
- Opportunity windows: {json.dumps(opportunities, ensure_ascii=False)}
{self._get_tc_prompt_for_event(time_units[0][0].isoformat(), period)}
{enrichment_block}
{milestone_prior_block}

## Time Units to Generate ({n_units} consecutive units)
{chr(10).join(unit_blocks)}

## Memory Bank Context
{memory_ctx}

## Task
Generate a **low-resolution event framework** AND **habitual event patterns** for EACH of the {n_units} time units above.
Return {n_units} LowResFrameworkOutput items (one per time unit), each containing:
1. event_type: Event type (e.g., daily_routine, milestone, social, academic, career)
2. event_theme: Core event theme (brief keywords)
3. event_location: Event location
4. value_for_target: Impact on the target character's development (1 sentence)
5. value_for_period: Contribution to stage needs (1 sentence)
6. summary: Main content of this period (1-2 sentences, focus on who/what/when/where)

Also return `habitual_events_per_unit` with EXACTLY {n_units} sub-arrays (one per time unit).
Each sub-array contains 1-2 habitual event seeds (title, frequency, summary).
{memory_budget_note}

Note:
- Each time unit should have its own distinct LR framework
- Maintain narrative continuity between consecutive units
- This is a summary-level event; keep it concise
- ⚠️ event_location must match the character's actual location at the current stage
- ⚠️ habitual_events_per_unit must have EXACTLY {n_units} sub-arrays
"""

        result: BatchLROutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=BatchLROutput,
            system_prompt=system_prompt,
            task_type="step1a_batch_event_framework",
            temperature=0.0,
        )
        return result

    # ================================================================
    # Step 1a Cross-LP Batch: Generate LR frameworks for multiple consecutive
    # LR-only segments from DIFFERENT life periods in a single prompt
    # ================================================================

    async def _step1a_batch_generate_all_lr_multi_period(
        self,
        periods: List[Dict[str, Any]],
        time_units: List[Tuple[date, date]],
        unit_labels: List[str],
    ) -> "BatchLROutput":
        """Batch-generate LR frameworks for multiple consecutive LR-only segments,
        each belonging to a DIFFERENT life period (cross-LP batch).

        Unlike _step1a_batch_generate_all_lr (which handles multiple time units
        within ONE period), this method handles one time unit per period, where
        each period has its own semantic context (title, theme, tasks, goals).

        Uses the STRONGEST model (Tier 3 via task_type='step1a_cross_lp_batch_event_framework')
        to ensure factual accuracy, persona coherence, and narrative continuity.

        Args:
            periods: List of period dicts (one per time unit), in chronological order.
            time_units: List of (unit_start, unit_end) for each period.
            unit_labels: Corresponding labels for each unit.

        Returns:
            BatchLROutput with one LowResFrameworkOutput per period.
        """
        n_units = len(periods)
        assert len(time_units) == n_units
        assert len(unit_labels) == n_units

        # Use the first period's reference date for persona context
        structured_persona_ctx = await self._build_structured_persona_context(
            periods[0], time_units[0][0]
        )
        memory_ctx = self._build_memory_context()

        # Merge enrichment across all years in the batch
        evt_start_year = time_units[0][0].year
        evt_end_year = time_units[-1][1].year
        enrichment_block = self._merge_multi_year_enrichment(evt_start_year, evt_end_year)

        # ── System prompt: emphasise coherence, factual accuracy, format ──
        system_prompt = (
            "You are a life simulation event generator responsible for generating "
            "low-resolution event summaries AND habitual event patterns for MULTIPLE consecutive life stages in a single response.\n\n"
            "## CRITICAL REQUIREMENTS\n"
            "### A. Narrative Coherence\n"
            "- Events across all life stages MUST form a logically coherent life trajectory.\n"
            "- Each stage's event must causally or thematically connect to adjacent stages "
            "(e.g., a decision in Stage 2 should be reflected in Stage 3's circumstances).\n"
            "- Avoid contradictions: if the character moved cities in Stage 1, Stage 2 must "
            "reflect the new location.\n"
            "- The character's emotional arc, relationships, and career path must evolve "
            "consistently across all stages.\n\n"
            "### B. Cultural & Persona Fidelity (No Factual Errors)\n"
            "- Each stage's event MUST be consistent with the cultural background, social norms, "
            "and historical context of that specific year and country.\n"
            "- The character's age, education level, occupation, family status, and location "
            "at each stage must be factually accurate based on the provided life plan.\n"
            "- Do NOT invent institutions, events, or cultural references that did not exist "
            "in the specified year and country.\n"
            "- The character's personality traits, values, and behavioral patterns (as defined "
            "in the persona) must be consistently reflected in each stage's event.\n\n"
            "### C. Output Format\n"
            "- Return EXACTLY {n_units} LowResFrameworkOutput items — one per life stage, "
            "in the same order as the input stages.\n"
            "- Each summary: 1-2 sentences only. Focus on WHO did WHAT, WHERE, and WHEN.\n"
            "- NO psychological introspection, NO environmental description, NO emotional arc.\n"
            "- event_location MUST match the character's actual location at that stage.\n"
            "- summary MUST be in English, plain text, no markdown.\n\n"
            "### D. Habitual Events (General-Event Layer)\n"
            "- For EACH life stage, also generate 1-2 habitual/recurring event patterns.\n"
            "- These represent REPEATED activities (Script Theory, Schank & Abelson 1978) — "
            "things the protagonist does regularly during that stage.\n"
            "- Each habitual event needs: habitual_event_title, habitual_event_frequency "
            "(daily/weekly/monthly/etc.), and habitual_event_summary (1-2 sentences).\n"
            "- Examples: 'Morning school walk | daily', 'Family dinner | daily', "
            "'Piano practice | weekly', 'Semester exam preparation | every semester'\n"
            "- Habitual events MUST be age-appropriate and stage-appropriate.\n"
            "- Return them in the `habitual_events_per_unit` array, one sub-array per stage.\n"
            "- The `habitual_events_per_unit` array MUST have EXACTLY {n_units} elements.\n"
        ).format(n_units=n_units)

        # Build per-period blocks with full semantic context
        period_blocks = []
        for i, (period, (u_start, u_end)) in enumerate(zip(periods, time_units)):
            period_id = period.get("period_id", "LP?")
            title = period.get("title", "")
            theme = period.get("dominant_theme", "")
            tasks = period.get("developmental_tasks", [])
            goals = period.get("stage_goals", [])
            pressures = period.get("salient_pressures", [])
            opportunities = period.get("salient_opportunities", [])
            # Include year-specific enrichment hint per period
            tc_hint = self._get_tc_prompt_for_event(u_start.isoformat(), period)
            period_blocks.append(
                f"### Life Stage {i+1}/{n_units}: {period_id} — {unit_labels[i]}\n"
                f"- Time range: {u_start.isoformat()} to {u_end.isoformat()}\n"
                f"- Title: {title}\n"
                f"- Dominant theme: {theme}\n"
                f"- Developmental tasks: {json.dumps(tasks[:4], ensure_ascii=False)}\n"
                f"- Stage goals: {json.dumps(goals[:3], ensure_ascii=False)}\n"
                f"- Main pressures: {json.dumps(pressures[:3], ensure_ascii=False)}\n"
                f"- Opportunity windows: {json.dumps(opportunities[:2], ensure_ascii=False)}\n"
                + (f"- Cultural/temporal context: {tc_hint.strip()}\n" if tc_hint.strip() else "")
            )

        user_prompt = f"""{structured_persona_ctx}

{enrichment_block}

## Life Stages to Generate ({n_units} consecutive stages, in chronological order)
{chr(10).join(period_blocks)}

## Memory Bank Context (existing events before this batch)
{memory_ctx}

## Task
Generate a **low-resolution event framework** AND **habitual event patterns** for EACH of the {n_units} life stages above.

### Output Requirements (STRICTLY FOLLOW)
Return EXACTLY {n_units} `LowResFrameworkOutput` items in the `lr_frameworks` array, one per stage, in the SAME ORDER as the input stages.

For each item:
| Field | Requirement |
|-------|-------------|
| `event_type` | One of: `daily_routine`, `milestone`, `social`, `academic`, `career`, `family`, `health`, `travel` |
| `event_theme` | 3-6 keywords describing the core theme (English) |
| `event_location` | Exact city/country matching the character's location at this stage |
| `value_for_target` | 1 sentence: how this event impacts the character's development |
| `value_for_period` | 1 sentence: how this event contributes to this stage's goals |
| `summary` | 1-2 sentences: WHO did WHAT WHERE WHEN. Plain English. No markdown. |

### Habitual Events Requirements (STRICTLY FOLLOW)
Also return `habitual_events_per_unit` with EXACTLY {n_units} sub-arrays (one per stage).
Each sub-array contains 2-3 habitual event seeds:
| Field | Requirement |
|-------|-------------|
| `habitual_event_title` | Descriptive title of the recurring activity |
| `habitual_event_frequency` | How often: daily/weekly/monthly/every semester/etc. |
| `habitual_event_summary` | 1-2 sentences describing what this routine involves |

Age-appropriate examples:
- Age 0-5: "Bedtime story with parents | daily", "Playground visit | daily"
- Age 6-12: "Walking to school | daily", "Weekend soccer practice | weekly"
- Age 13-18: "After-school study session | daily", "Part-time job shift | weekly"
- Age 18+: "Morning commute | daily", "Lab meeting | weekly"

### Coherence Checklist (verify before output)
- [ ] Each stage's location is consistent with the character's known whereabouts
- [ ] No institution/event/technology referenced that didn't exist in that year
- [ ] The character's age at each stage matches the time range
- [ ] Events form a causally connected life trajectory (no isolated, unrelated events)
- [ ] Personality traits from the persona are reflected in the event choices
- [ ] habitual_events_per_unit has EXACTLY {n_units} sub-arrays

### Anti-patterns to Avoid
- No "She reflected deeply on her choices" (psychological introspection — forbidden)
- No "The autumn leaves fell gently" (environmental description — forbidden)
- No referencing a university the character hasn't enrolled in yet
- No placing the character in a city they haven't moved to yet
- No inventing cultural events that didn't happen in that year/country
"""

        result: BatchLROutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=BatchLROutput,
            system_prompt=system_prompt,
            task_type="step1a_cross_lp_batch_event_framework",  # → Tier 3 (strongest model)
            temperature=0.0,
        )
        return result

    # ================================================================
    # Step 1a+1b (Merged): Generate LR framework + outlines in single prompt
    # ================================================================

    async def _step1a_generate_lr_with_outlines(
        self,
        period: Dict[str, Any],
        unit_start: date,
        unit_end: date,
        unit_label: str,
        density: str,
        k: int,  # Number of outlines to generate
        detail_budget: int = 999,
        outline_budget: int = 999,
        medium_per_unit: int = 2,
        total_units: int = 0,
        current_unit_index: int = 0,
    ) -> LRWithOutlinesOutput:
        """Generate LR framework + k outlines in a single LLM prompt.

        Merges Step 1a (LR framework) and Step 1b (outline generation) into
        one call, saving 1 LLM round-trip per time unit with outline budget.

        Args:
            period: The life stage period dict.
            unit_start: Start date of the time unit.
            unit_end: End date of the time unit.
            unit_label: Label for this time unit.
            density: Memory density for this period.
            k: Number of key event outlines to generate.
            detail_budget: Remaining detail budget.
            outline_budget: Remaining outline budget.
            total_units: Total number of time units in the period.
            current_unit_index: Index of this unit in the overall period.

        Returns:
            LRWithOutlinesOutput containing both LR framework and outlines.
        """
        period_id = period.get("period_id", "LP?")
        title = period.get("title", "")
        theme = period.get("dominant_theme", "")
        tasks = period.get("developmental_tasks", [])
        goals = period.get("stage_goals", [])
        pressures = period.get("salient_pressures", [])
        opportunities = period.get("salient_opportunities", [])

        structured_persona_ctx = await self._build_structured_persona_context(period, unit_start)
        participant_brief_ctx = self._build_participant_brief_context(period_id)
        memory_ctx = self._build_memory_context()

        # ── Enrichment injection ──
        evt_start_year = unit_start.year
        evt_end_year = unit_end.year
        enrichment_block = self._merge_multi_year_enrichment(evt_start_year, evt_end_year)

        # ── Milestone prior injection ──
        milestone_skel = self.get_milestone_for_period(period_id)
        milestone_prior_block = ""
        if milestone_skel:
            ms_name = milestone_skel.get('milestone_name', '')
            ms_date = milestone_skel.get('expected_date_range', '')
            ms_summary = milestone_skel.get('summary_hint', '')
            is_milestone_unit = self.milestone_matches_time_unit(period_id, unit_start, unit_end)
            if is_milestone_unit:
                milestone_prior_block = f"""
## 🎯 This Time Unit Contains a Pre-Planned Key Milestone
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}
The first outline MUST center on this milestone.
"""
            else:
                milestone_prior_block = f"""
## 📌 Stage Pre-Planned Key Milestone (Reference)
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}
Your events should reflect the proximity to the milestone.
"""

        memory_budget_note = f"""
## Memory Budget
- Time unit: {unit_label}
- Remaining time units: {total_units - current_unit_index} / {total_units}
- Detail budget: {detail_budget}, Outline budget: {outline_budget}
- Outlines to generate: {k}
"""

        system_prompt = (
            "You are the Resolution Arrangement module of MemoryForge, a life simulation framework.\n"
            "Your task is to generate a THREE-LAYER memory plan for a time unit:\n\n"
            "1. **Low-Resolution (L)**: A 1-2 sentence summary of the entire time unit.\n\n"
            f"2. **Medium-Resolution / General Events (G)**: {medium_per_unit} habitual/recurring event patterns "
            "that characterize the protagonist's routine during this period. "
            "These are REPEATED activities (Script Theory, Schank & Abelson 1978) — "
            "things the protagonist does regularly, NOT one-time events.\n"
            "Each must have: title, frequency (daily/weekly/monthly/etc.), and a 1-2 sentence description.\n"
            "Examples: 'Morning lab routine | daily', 'Weekly advisor meeting | weekly', "
            "'Semester exam preparation | every semester'\n\n"
            f"3. **High-Resolution Outlines (E)**: {k} key ONE-TIME event outlines — "
            "the most important milestone/turning-point events within this period. "
            "ALL of these will be expanded into full multi-turn simulations.\n"
            "Each outline should have: theme, summary, start/end dates and times, "
            "motivation, outcome, setting, turning point, value for target, and required participant roles.\n"
            "Outlines must be chronologically ordered and NOT overlap or repeat."
        )

        user_prompt = f"""{structured_persona_ctx}

## Current Life Stage
- Stage ID: {period_id}
- Title: {title}
- Dominant theme: {theme}
- Developmental tasks: {json.dumps(tasks, ensure_ascii=False)}
- Stage goals: {json.dumps(goals, ensure_ascii=False)}
- Main pressures: {json.dumps(pressures, ensure_ascii=False)}
- Opportunity windows: {json.dumps(opportunities, ensure_ascii=False)}
{self._get_tc_prompt_for_event(unit_start.isoformat(), period)}
{enrichment_block}
{milestone_prior_block}

## Current Time Unit
- Time range: {unit_start.isoformat()} to {unit_end.isoformat()} ({unit_label})
- Memory density: {density}

## Available Participants Overview (for reference only)
{participant_brief_ctx}

## Memory Bank Context
{memory_ctx}

## Task
Generate THREE layers:

1. **Low-resolution event framework** — summary of the entire time unit (1-2 sentences)

2. **{medium_per_unit} habitual event seeds** — recurring/routine patterns during this period:
   - `habitual_event_title`: descriptive title (e.g., "Morning lab routine", "Weekly team meeting")
   - `habitual_event_frequency`: how often (e.g., "daily", "weekly", "every semester")
   - `habitual_event_summary`: 1-2 sentence description of the routine

3. **{k} key event outlines** — the most important ONE-TIME events, ordered chronologically
   (ALL will be expanded into full high-resolution simulations)

For each outline provide:
- key_event_theme, key_event_summary
- key_event_start_date, key_event_end_date (ISO format, within the time unit range)
- key_event_start_time, key_event_end_time (e.g., "morning", "afternoon")
- key_event_motivation, key_event_outcome
- key_event_setting, key_event_turning_point
- key_event_value_for_target
- required_participant_roles
{memory_budget_note}

Note:
- Outlines must be chronologically ordered and cover different events
- ⚠️ event_location must match the character's actual location at the current stage
"""

        result: LRWithOutlinesOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=LRWithOutlinesOutput,
            system_prompt=system_prompt,
            task_type="step1a_lr_with_outlines",
            temperature=0.0,
        )
        return result

    # ================================================================
    # Step 1a: Generate Low-Resolution Event Framework (focused)
    # ================================================================

    async def _step1a_generate_low_res_framework(
        self,
        period: Dict[str, Any],
        unit_start: date,
        unit_end: date,
        unit_label: str,
        density: str,
        detail_remaining: int = 999,
        outline_remaining: int = 999,
        detail_budget: int = 999,
        outline_budget: int = 999,
        total_units: int = 0,
        current_unit_index: int = 0,
    ) -> LowResFrameworkOutput:
        """
        Step 1a: Generate the low-resolution event framework.

        Focuses exclusively on the overall narrative for this time unit:
        event type, theme, location, summary, and participant hints.
        Also determines whether a key event exists (three-value + brief reason),
        but does NOT expand key event details.
        """
        period_id = period.get("period_id", "LP?")
        title = period.get("title", "")
        theme = period.get("dominant_theme", "")
        tasks = period.get("developmental_tasks", [])
        goals = period.get("stage_goals", [])
        pressures = period.get("salient_pressures", [])
        opportunities = period.get("salient_opportunities", [])

        structured_persona_ctx = await self._build_structured_persona_context(period, unit_start)
        participant_brief_ctx = self._build_participant_brief_context(period_id)
        memory_ctx = self._build_memory_context()

        # ── Enrichment injection (multi-year for low-res) ──
        evt_start_year = unit_start.year
        evt_end_year = unit_end.year
        enrichment_block = self._merge_multi_year_enrichment(evt_start_year, evt_end_year)

        # ── Milestone prior injection ──
        milestone_skel = self.get_milestone_for_period(period_id)
        milestone_prior_block = ""
        if milestone_skel:
            ms_name = milestone_skel.get('milestone_name', '')
            ms_date = milestone_skel.get('expected_date_range', '')
            ms_summary = milestone_skel.get('summary_hint', '')
            ms_coherence = milestone_skel.get('low_res_coherence_constraints', '')
            is_milestone_unit = self.milestone_matches_time_unit(period_id, unit_start, unit_end)
            if is_milestone_unit:
                milestone_prior_block = f"""
## 🎯 This Time Unit Contains a Pre-Planned Key Milestone
This time unit contains a pre-planned key milestone event. has_key_event MUST be "key_event_detail":
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}

Your low-res summary MUST center on this milestone and remain coherent with the skeleton description.
{ms_coherence}
"""
            else:
                milestone_prior_block = f"""
## 📌 Stage Pre-Planned Key Milestone (Reference Information)
This stage has a pre-planned key milestone event (but not within the current time unit):
- Event name: {ms_name}
- Expected time: {ms_date}
- Event summary hint: {ms_summary}

Your low-res event should:
1. If the current time is before the milestone, serve as lead-up preparation
2. If the current time is after the milestone, reflect its impact
3. has_key_event should NOT be "key_event_detail" (that slot is reserved for the milestone)
{ms_coherence}
"""

        # v8: has_key_event removed from Step 1a output; key_event_instruction no longer needed.
        # Step 1a always initializes has_key_event="no", and Step 1b determines key events.
        memory_budget_note = f"""
## Memory Budget for this Period
- Simulation time unit: {unit_label}
- Remaining time units: {total_units - current_unit_index} / {total_units}
- Detailed events (key_event_detail) budget: remaining {detail_remaining} / max {detail_budget}
- Outline events (key_event_outline) budget: remaining {outline_remaining} / max {outline_budget}

Note: Key event determination is handled automatically in a later step.
Focus on generating a high-quality low-resolution event summary.
"""

        system_prompt = (
            "You are a life simulation event generator. "
            "Based on the target character's persona, current life stage information, and memory bank, "
            "generate a low-resolution event framework for the specified time unit.\n"
            "[IMPORTANT] This is a summary-level event — only 1-2 sentences describing what happened during this period. "
            "Focus on the who/what/when/where elements. No psychological description, environmental detail, or emotional arc. "
            "Keep it concise."
        )

        user_prompt = f"""{structured_persona_ctx}

## Current Life Stage
- Stage ID: {period_id}
- Title: {title}
- Dominant theme: {theme}
- Developmental tasks: {json.dumps(tasks, ensure_ascii=False)}
- Stage goals: {json.dumps(goals, ensure_ascii=False)}
- Main pressures: {json.dumps(pressures, ensure_ascii=False)}
- Opportunity windows: {json.dumps(opportunities, ensure_ascii=False)}
{self._get_tc_prompt_for_event(unit_start.isoformat(), period)}
{enrichment_block}
{milestone_prior_block}
## Current Time Unit
- Time range: {unit_start.isoformat()} to {unit_end.isoformat()} ({unit_label})
- Memory density: {density}

## Available Participants Overview (for reference only, do not select in this step)
{participant_brief_ctx}

## Memory Bank Context
{memory_ctx}

## Task
Generate a **low-resolution event framework** for the above time unit (summary level, 1-2 sentences):
1. event_type: Event type (e.g., daily_routine, milestone, social, academic, career)
2. event_theme: Core event theme (brief keywords)
3. event_location: Event location
4. value_for_target: Impact on the target character's development (1 sentence)
5. value_for_period: Contribution to stage needs (1 sentence)
6. summary: Main content of this period (1-2 sentences, focus on who/what/when/where)
7. required_role_types: List of required participant role types
8. required_relationship_hints: Required participant relationship hints
9. estimated_participant_count: Estimated number of participants
{memory_budget_note}

Note:
- This is a summary-level event; keep it concise, no psychological description or environmental detail
- Maintain narrative continuity, connecting naturally with existing memories
- Do not select specific participants in this step, only provide requirement hints
- ⚠️ event_location must match the character's actual location at the current stage. Pre-high-school stages are typically in the growing-up area (see "Basic Information"); do not confuse with later university/work cities.
"""

        result: LowResFrameworkOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=LowResFrameworkOutput,
            system_prompt=system_prompt,
            task_type="step1a_event_framework",
            temperature=0.0,  # Creative planning (lowered from 0.65 to reduce hallucination)
        )
        return result

    # ================================================================
    # Step 1b: Generate High-Resolution Key Event Outline (focused)
    # ================================================================

    async def _step1b_generate_high_res_outline(
        self,
        low_res: LowResFrameworkOutput,
        period: Dict[str, Any],
        unit_start: date,
        unit_end: date,
        extra_context: str = "",  # v6: preceding outline context for forward dependency
        cross_unit_outline_ctx: str = "",  # [A-05] cross-unit outline context for deduplication
    ) -> HighResOutlineOutput:
        """
        Step 1b: Generate a focused high-resolution key event outline.

        Only called when Step 1a determines has_key_event != 'no'.
        Uses the confirmed low-resolution event framework as context
        to produce precise timing, causal chain, and scene details.
        """
        structured_persona_ctx = await self._build_structured_persona_context(period, unit_start)
        memory_ctx = self._build_memory_context()

        # ── Enrichment injection (single year for high-res outline) ──
        evt_year = unit_start.year
        life_path_block = self._format_key_life_path_for_prompt(evt_year)
        society_block = self._format_society_enrichment_for_prompt(evt_year)

        # ── Milestone skeleton constraints for outline ──
        period_id = period.get("period_id", "LP?")
        milestone_skel = self.get_milestone_for_period(period_id)
        milestone_constraint_block = ""
        if milestone_skel and self.milestone_matches_time_unit(period_id, unit_start, unit_end):
            ms_turning = milestone_skel.get('turning_point_hint', '')
            ms_emotional = milestone_skel.get('emotional_arc_hint', '')
            ms_beats = milestone_skel.get('mandatory_beats', [])
            ms_motivation = milestone_skel.get('motivation_hint', '')
            ms_outcome = milestone_skel.get('outcome_hint', '')
            beats_text = '\n'.join(f'  - {b}' for b in ms_beats) if ms_beats else '  (none)'
            milestone_constraint_block = f"""
## 🎯 Pre-Planned Milestone Skeleton Constraints (MUST FOLLOW)
This event is a pre-planned key milestone. The outline MUST build on the following skeleton:
- Turning point (MUST HAPPEN): {ms_turning}
- Emotional arc (MUST FOLLOW): {ms_emotional}
- Trigger motivation: {ms_motivation}
- Expected outcome: {ms_outcome}
- Mandatory beats to cover:
{beats_text}

Your outline may add details but cannot replace the core plot points above.
"""

        system_prompt = (
            "You are a key event designer for life simulation. "
            "Based on the confirmed low-resolution event framework, design a concrete key event.\n"
            "[Core Principle] The key event you design must satisfy both conditions:\n"
            "  1. Most impactful: During this time period, it produces the most significant change or "
            "driving effect on the target character's life trajectory, values, or important relationships;\n"
            "  2. Most memorable: This event will leave a lasting impression in the target character's memory, "
            "can be clearly recalled years later, and influences subsequent decisions.\n"
            "Key events need precise timing, a clear causal chain, and vivid scene details. "
            "Ensure the key event is consistent with the overall narrative of the low-resolution event framework."
        )

        user_prompt = f"""{structured_persona_ctx}
{self._get_tc_prompt_for_event(unit_start.isoformat(), period)}
{life_path_block}

{society_block}
## Low-Resolution Event Framework (Confirmed)
- Event type: {low_res.event_type}
- Event theme: {low_res.event_theme}
- Event location: {low_res.event_location}
- Event summary: {low_res.summary}
- Time range: {unit_start.isoformat()} to {unit_end.isoformat()}

## Memory Bank Context
{memory_ctx}
{milestone_constraint_block}
## Task
Based on the above low-resolution event framework, design a concrete **high-resolution key event**:

1. key_event_theme: Specific theme of the key event
2. key_event_summary: Detailed description of the key event (3-5 sentences, more specific than the low-res summary)

3. Precise timing (date must be within {unit_start.isoformat()} to {unit_end.isoformat()}):
   - key_event_start_date / key_event_start_time / key_event_precise_start_time
   - key_event_end_date / key_event_end_time / key_event_precise_end_time
   Time period options: early morning (00:00–06:00), morning (06:00–12:00), afternoon (12:00–18:00), night (18:00–24:00)
   Precise time format: HH:MM:SS (24-hour), e.g., "09:30:00"
   Timing requirements:
   * Time must follow realistic logic for the event type (e.g., classes typically 08:00-17:00, social events may be in evening)
   * Duration should match event content (e.g., a class ~40-45 min, a family dinner ~1-2 hrs, an important meeting ~1-3 hrs)
   * Precise time must fall within the corresponding period (e.g., start_time="morning" → precise_start_time between 06:00:00-11:59:59)

4. Causal chain:
   - key_event_motivation: Trigger cause of the event
   - key_event_value_for_target: Specific impact on the target character's development
   - key_event_outcome: Result and subsequent effects of the event

5. Scene details (recommended):
   - key_event_setting: Specific physical environment and atmosphere
   - key_event_turning_point: Key turning point within the event

6. required_participant_roles: List the role types that **must appear** in this key event scene
   (e.g., teacher, parent, friend, classmate, colleague).
   These roles will be passed as hard constraints to the participant selection step.

Note:
- The key event must be consistent with the overall narrative of the low-resolution event framework
- The key event must be the **most impactful and most memorable** event in this time period:
  * Most impactful: produces significant and lasting change in the target character's life direction, core values, or important relationships
  * Most memorable: can be clearly recalled years later and influences subsequent behavior and decisions
  * Priority: life turning points > important relationship changes > skill/cognitive breakthroughs > daily accumulation
  * Avoid: purely routine matters, habitual activities with no emotional or cognitive transformation
- High precision timing is required; carefully consider realistic logic
"""

        # [A-05] Inject cross-unit outline context for deduplication
        if cross_unit_outline_ctx:
            user_prompt += cross_unit_outline_ctx

        # v6: Append preceding outline context for forward dependency
        if extra_context:
            user_prompt += extra_context

        result: HighResOutlineOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=HighResOutlineOutput,
            system_prompt=system_prompt,
            task_type="step1b_high_res_outline",
            temperature=0.0,  # Creative planning
        )
        return result

    # ================================================================
    # Step 1b-batch: Batch Generate Outlines with Forward Dependency (v6)
    # ================================================================

    @staticmethod
    def _build_cross_unit_outline_ctx(
        existing_ctx: str,
        new_outlines: list,  # List of (event_id, HighResOutlineOutput)
    ) -> str:
        """[A-04] Build/update cross-unit outline context for deduplication.

        Accumulates outline summaries from previous time units so that
        subsequent time units can avoid generating duplicate themes.

        Args:
            existing_ctx: Previously accumulated cross-unit context string.
            new_outlines: List of (event_id, outline) tuples from the just-completed unit.

        Returns:
            Updated cross-unit context string.
        """
        if not new_outlines:
            return existing_ctx

        lines = []
        for event_id, outline in new_outlines:
            theme = getattr(outline, 'key_event_theme', '') or ''
            summary = getattr(outline, 'key_event_summary', '') or ''
            start_date = getattr(outline, 'key_event_start_date', '') or ''
            lines.append(f"  - [{start_date}] {theme}: {summary[:100]}")

        if not lines:
            return existing_ctx

        new_block = "\n".join(lines)

        if existing_ctx:
            return existing_ctx + new_block + "\n"
        else:
            return (
                "\n## ⚠️ Already-Generated Outlines from Previous Time Units (MUST NOT REPEAT)\n"
                "The following events have already been planned in earlier time periods. "
                "Your new outline MUST cover a DIFFERENT theme and scenario. "
                "Do NOT generate another version of any of these events:\n"
                + new_block + "\n"
            )

    async def _step1b_batch_generate_outlines(
        self,
        low_res: LowResFrameworkOutput,
        period: Dict[str, Any],
        unit_start: date,
        unit_end: date,
        k: int,  # Number of outlines to generate for this time unit
        cross_unit_outline_ctx: str = "",  # [A-06] cross-unit outline context for deduplication
    ) -> List[HighResOutlineOutput]:
        """v6: Batch generate outlines with forward dependency and coherence.

        Core constraints:
        1. Each outline can only see preceding events (memory_ctx + already generated outlines)
        2. Cannot see subsequent events
        3. Generated one by one, each outline's prompt includes all preceding outlines as context
        """
        if k <= 0:
            return []

        outlines: List[HighResOutlineOutput] = []

        for i in range(k):
            # Build preceding outline context (only includes already generated ones in this batch)
            preceding_outlines_ctx = ""
            if outlines:
                lines = []
                for idx, prev in enumerate(outlines):
                    lines.append(
                        f"  {idx+1}. [{prev.key_event_start_date}] "
                        f"{prev.key_event_theme}: {prev.key_event_summary[:120]}"
                    )
                preceding_outlines_ctx = (
                    "\n## Already Generated Outlines in This Time Unit (Important Constraints)\n"
                    + "\n".join(lines)
                    + "\n\n"
                    + "Constraints:\n"
                    + "1. Your outline must NOT simply restate or paraphrase the above outlines. You must choose one of:\n"
                    + "   - a) A new event of a different type from existing outlines; or\n"
                    + "   - b) An important development of an existing event in a later time period.\n"
                    + "2. The event time for your outline must be strictly later than the above outlines.\n"
                    + "3. Your outline must be an important event within this time period.\n"
                    + "   - Selection criteria: Among all candidate events not yet covered by existing outlines, prioritize the most impactful, most far-reaching, most memory-worthy event.\n"
                    + "   - Avoid events that are essentially the same as existing outlines with no substantial development.\n"
                    + "4. If you choose a continuation of an existing event, the development must match the elapsed time and cannot be an unreasonable leap.\n"
                )

            try:
                outline = await self._step1b_generate_high_res_outline(
                    low_res=low_res,
                    period=period,
                    unit_start=unit_start,
                    unit_end=unit_end,
                    extra_context=preceding_outlines_ctx,
                    cross_unit_outline_ctx=cross_unit_outline_ctx,  # [A-06]
                )
                outlines.append(outline)
            except Exception as e:
                logger.error(
                    f"  _step1b_batch_generate_outlines: failed to generate outline "
                    f"{i+1}/{k}: {e} — this time unit will have fewer events"
                )
                # Continue with remaining outlines
                continue

        # v8: Sort outlines by sub-importance (secondary to primary key event ranking)
        # Outlines from earlier batch entries are de-prioritized as they tend to be
        # less consequential than later ones (temporal progression → importance ↑).
        # This sorting ensures _rank_outlines_for_detail processes them correctly.
        if len(outlines) > 1:
            outlines.sort(key=lambda o: (
                0 if any(kw in (o.key_event_theme or "").lower()
                          for kw in ["graduation", "interview", "promotion",
                                     "defense", "resignation", "entrance exam"]) else 1,
                0 if any(kw in (o.key_event_theme or "").lower()
                          for kw in ["meeting", "dinner", "chat", "overtime",
                                     "discussion", "daily"]) else 1,
            ))

        return outlines

    # ================================================================
    # Rank Outlines for Detail Selection (v6)
    # ================================================================

    async def _rank_outlines_for_detail(
        self,
        outlines: List[tuple],  # List of (event_id, HighResOutlineOutput)
        reference_date: date,
        m: int,  # Number of details to select
        unit_outline_counts: Optional[Dict[int, int]] = None,  # v8: per-unit outline count
        theme_categories: Optional[Dict[str, str]] = None,  # v8: event_id → theme_category
        identity_salience_needed: bool = False,  # v9: True when persona has primary identity traits
        identity_salience_instruction: str = "",  # v9: persona-specific identity instruction
    ) -> List[str]:
        """v9: Rank outlines by importance×recency and select top-m for detail (P3 simulation).

        Enforces per-unit constraint: selected_detail_per_unit ≤ outline_count - 2,
        ensuring each time unit retains at least 2 outline-only events.

        v8 diversity constraint: if theme_categories provided, ensure at least 1 detail
        from each represented category (up to m//2 categories max).

        v9 identity constraint: if identity_salience_needed=True, guarantee at least 1
        selected outline has identity_relevance >= 0.6 (directly identity-related theme).
        Uses the identity_relevance field from _OutlineSalienceScore, populated by the LLM
        when identity_salience_instruction is provided.

        Args:
            unit_outline_counts: Dict mapping unit_index → total outline count for that unit.
                                 Used to enforce: selected_detail_per_unit ≤ outline_count - 2
            theme_categories: Dict mapping event_id → event_theme_category for diversity.
            identity_salience_needed: Whether to enforce the identity-relevance guarantee.
            identity_salience_instruction: Persona-specific instruction passed to the LLM
                                           scorer so it can populate identity_relevance.

        Returns list of selected outline event_ids.
        """
        # v9: Unified scored_with_unit handles both 2-tuple and 3-tuple outlines
        scored_with_unit = []

        # Batch-score all outlines' semantic importance via a single LLM call.
        # Falls back to (0.5, 0.0) per outline if the call fails.
        llm_score_map: Dict[str, Tuple[float, float]] = await self._llm_score_outline_salience(
            outlines,
            identity_salience_instruction=identity_salience_instruction,
        )

        # v9: also track identity_relevance per event_id for the identity guarantee pass
        identity_relevance_map: Dict[str, float] = {}

        for item in outlines:
            if len(item) == 3:
                event_id, outline, unit_idx = item
            else:
                event_id, outline = item
                unit_idx = None

            # Importance scoring — LLM-derived, no keyword lists
            _eid_for_score = (
                item[0] if isinstance(item, tuple) and len(item) >= 1 else None
            )
            score_pair = llm_score_map.get(_eid_for_score, (0.5, 0.0))
            importance = score_pair[0]
            id_relevance = score_pair[1]
            identity_relevance_map[event_id] = id_relevance

            # Recency scoring
            try:
                event_date = date.fromisoformat(outline.key_event_start_date)
                years_ago = (reference_date - event_date).days / 365.25
                recency = max(0.0, 1.0 - years_ago / 5.0)  # Linear decay over 5 years
            except (ValueError, TypeError):
                recency = 0.5

            rank_score = 0.4 * importance + 0.6 * recency
            scored_with_unit.append((event_id, rank_score, unit_idx))

        scored_with_unit.sort(key=lambda x: x[1], reverse=True)

        if unit_outline_counts is None:
            # No per-unit constraint: fall back to simple top-m selection
            selected = [eid for eid, _, _ in scored_with_unit[:m]]
            # v9: identity guarantee — relax if needed (no per-unit constraint here)
            if identity_salience_needed and m > 0:
                selected = self._apply_identity_guarantee(
                    selected=selected,
                    scored_with_unit=scored_with_unit,
                    identity_relevance_map=identity_relevance_map,
                    unit_outline_counts=None,
                    unit_detail_counts={},
                    m=m,
                )
            return selected

        selected: List[str] = []
        unit_detail_counts: Dict[int, int] = {}  # unit_idx → selected detail count
        selected_categories: set = set()  # v8: track covered categories

        # v8 diversity: first pass — ensure at least 1 detail from each category
        if theme_categories and m > 0:
            # Group scored outlines by category
            cat_groups: Dict[str, List[tuple]] = {}
            for event_id, score, unit_idx in scored_with_unit:
                cat = theme_categories.get(event_id, "other")
                if cat not in cat_groups:
                    cat_groups[cat] = []
                cat_groups[cat].append((event_id, score, unit_idx))

            # Pick top-1 from each category (up to m//2 slots for diversity)
            max_diversity_slots = max(1, m // 2)
            categories_by_size = sorted(cat_groups.keys(), key=lambda c: -len(cat_groups[c]))
            diversity_slots_used = 0
            for cat in categories_by_size:
                if diversity_slots_used >= max_diversity_slots:
                    break
                if cat in selected_categories:
                    continue
                # Find best candidate from this category that passes per-unit constraint
                for event_id, score, unit_idx in cat_groups[cat]:
                    if event_id in selected:
                        continue
                    if unit_idx is not None and unit_outline_counts:
                        unit_total_outlines = unit_outline_counts.get(unit_idx, 999)
                        unit_selected = unit_detail_counts.get(unit_idx, 0)
                        max_allowed = max(0, unit_total_outlines - 2)
                        if unit_selected >= max_allowed:
                            continue
                    selected.append(event_id)
                    selected_categories.add(cat)
                    if unit_idx is not None:
                        unit_detail_counts[unit_idx] = unit_detail_counts.get(unit_idx, 0) + 1
                    diversity_slots_used += 1
                    break

        # Second pass: fill remaining slots by score (with per-unit constraint)
        for event_id, score, unit_idx in scored_with_unit:
            if len(selected) >= m:
                break
            if event_id in selected:
                continue
            # v8: per-unit constraint
            if unit_idx is not None and unit_outline_counts:
                unit_total_outlines = unit_outline_counts.get(unit_idx, 999)
                unit_selected = unit_detail_counts.get(unit_idx, 0)
                max_allowed = max(0, unit_total_outlines - 2)
                if unit_selected >= max_allowed:
                    # This unit already has enough details; skip
                    continue
            selected.append(event_id)
            if unit_idx is not None:
                unit_detail_counts[unit_idx] = unit_detail_counts.get(unit_idx, 0) + 1

        # Fallback: if still under budget, relax per-unit constraint and fill remaining slots
        if len(selected) < m:
            for event_id, score, unit_idx in scored_with_unit:
                if event_id not in selected:
                    selected.append(event_id)
                if len(selected) >= m:
                    break

        # v9: identity guarantee — after all other passes, ensure at least 1 identity-relevant
        # outline is selected when identity_salience_needed=True.
        if identity_salience_needed and m > 0:
            selected = self._apply_identity_guarantee(
                selected=selected,
                scored_with_unit=scored_with_unit,
                identity_relevance_map=identity_relevance_map,
                unit_outline_counts=unit_outline_counts,
                unit_detail_counts=unit_detail_counts,
                m=m,
            )

        return selected

    def _apply_identity_guarantee(
        self,
        selected: List[str],
        scored_with_unit: List[tuple],  # (event_id, rank_score, unit_idx)
        identity_relevance_map: Dict[str, float],
        unit_outline_counts: Optional[Dict[int, int]],
        unit_detail_counts: Dict[int, int],
        m: int,
        identity_relevance_threshold: float = 0.6,
    ) -> List[str]:
        """v9: Guarantee at least 1 selected outline has identity_relevance >= threshold.

        If no currently-selected outline meets the threshold, find the highest
        identity_relevance candidate among all outlines and swap it in:
          - If budget allows (len(selected) < m): append it directly.
          - Otherwise: replace the lowest-ranked non-identity outline in selected.

        The per-unit constraint is respected when possible, but relaxed as a last resort
        to ensure the identity guarantee is always honoured.

        Args:
            selected: Current list of selected event_ids (may be mutated in-place).
            scored_with_unit: All outlines sorted by rank_score descending.
            identity_relevance_map: {event_id: identity_relevance} from LLM scoring.
            unit_outline_counts: Per-unit total outline counts (for constraint check).
            unit_detail_counts: Per-unit already-selected detail counts (mutable).
            m: Maximum number of details to select.
            identity_relevance_threshold: Minimum identity_relevance to count as relevant.

        Returns the (possibly modified) selected list.
        """
        # Check if any already-selected outline satisfies the identity guarantee
        already_satisfied = any(
            identity_relevance_map.get(eid, 0.0) >= identity_relevance_threshold
            for eid in selected
        )
        if already_satisfied:
            return selected

        # Find the best identity-relevant candidate not yet selected
        # Sort all outlines by identity_relevance desc, then rank_score desc
        candidates = sorted(
            scored_with_unit,
            key=lambda x: (-identity_relevance_map.get(x[0], 0.0), -x[1]),
        )
        best_identity_eid: Optional[str] = None
        best_identity_unit_idx: Optional[int] = None
        for event_id, rank_score, unit_idx in candidates:
            if identity_relevance_map.get(event_id, 0.0) < identity_relevance_threshold:
                # No more candidates above threshold
                break
            if event_id in selected:
                # Already selected — guarantee already satisfied (shouldn't reach here)
                already_satisfied = True
                break
            best_identity_eid = event_id
            best_identity_unit_idx = unit_idx
            break

        if already_satisfied or best_identity_eid is None:
            # No identity-relevant outline exists at all — nothing to do
            if not already_satisfied:
                logger.info(
                    "[P2-v9] identity_guarantee: no outline with identity_relevance >= "
                    f"{identity_relevance_threshold} found; skipping guarantee."
                )
            return selected

        # Try to add the identity candidate
        if len(selected) < m:
            # Budget available — append directly
            selected.append(best_identity_eid)
            if best_identity_unit_idx is not None:
                unit_detail_counts[best_identity_unit_idx] = (
                    unit_detail_counts.get(best_identity_unit_idx, 0) + 1
                )
            logger.info(
                f"[P2-v9] identity_guarantee: appended identity-relevant outline "
                f"{best_identity_eid} (identity_relevance="
                f"{identity_relevance_map.get(best_identity_eid, 0.0):.2f})"
            )
        else:
            # Budget full — swap out the lowest-ranked non-identity outline
            # Find the selected outline with the lowest rank_score that is NOT identity-relevant
            rank_score_map = {eid: score for eid, score, _ in scored_with_unit}
            swap_target: Optional[str] = None
            swap_target_score = float("inf")
            for eid in selected:
                if identity_relevance_map.get(eid, 0.0) >= identity_relevance_threshold:
                    continue  # Don't swap out another identity-relevant outline
                score = rank_score_map.get(eid, 0.0)
                if score < swap_target_score:
                    swap_target_score = score
                    swap_target = eid
            if swap_target is not None:
                selected.remove(swap_target)
                selected.append(best_identity_eid)
                # Update unit_detail_counts for the swap
                swap_unit_idx = next(
                    (ui for eid, _, ui in scored_with_unit if eid == swap_target), None
                )
                if swap_unit_idx is not None:
                    unit_detail_counts[swap_unit_idx] = max(
                        0, unit_detail_counts.get(swap_unit_idx, 0) - 1
                    )
                if best_identity_unit_idx is not None:
                    unit_detail_counts[best_identity_unit_idx] = (
                        unit_detail_counts.get(best_identity_unit_idx, 0) + 1
                    )
                logger.info(
                    f"[P2-v9] identity_guarantee: swapped {swap_target} "
                    f"(rank={swap_target_score:.3f}) → {best_identity_eid} "
                    f"(identity_relevance="
                    f"{identity_relevance_map.get(best_identity_eid, 0.0):.2f})"
                )
            else:
                logger.info(
                    "[P2-v9] identity_guarantee: all selected outlines are already "
                    "identity-relevant; no swap needed."
                )

        return selected

    async def _step2_select_participants(
        self,
        event_framework: LowResFrameworkOutput,
        period_id: str,
        unit_start: date,
        unit_end: date,
    ) -> ParticipantSelectionOutput:
        """
        Step 2: Browse the participant pool and select suitable participants
        based on role, relationship, current persona, and target persona.
        """
        pool_context = self._build_participant_pool_context(period_id)
        target_name = self.persona.get("persona_name_text", "protagonist")

        system_prompt = (
            "You are a participant matcher for life simulation. "
            "Based on the event framework and participant pool information, select the most suitable participants. "
            "Participants include all characters who will form impressions of the protagonist during the interaction. "
            "Consider: role type, relationship to protagonist, current persona state, target persona state.\n\n"
            # T-B4: Physical presence constraint
            "CRITICAL RULE \u2014 PHYSICAL PRESENCE:\n"
            "Select ONLY participants who will be PHYSICALLY PRESENT in the scene.\n"
            "- For a private family conversation at home: select only the family members in the room.\n"
            "- For a one-on-one meeting: select only the two people meeting.\n"
            "- Do NOT include participants who are merely 'aware of' or 'affected by' the event but not physically present.\n"
            "- WRONG: Including all 20+ participants for a 2-person conversation.\n"
            "- RIGHT: Including only the 2-3 people who are physically in the scene.\n"
            "Typical scene size: 1-5 participants. Only exceed 5 if the event is explicitly a group gathering."
        )

        user_prompt = f"""## Event Framework
- Event type: {event_framework.event_type}
- Event theme: {event_framework.event_theme}
- Event location: {event_framework.event_location}
- Event summary: {event_framework.summary}
- Time range: {unit_start.isoformat()} to {unit_end.isoformat()}
- Required role types: {json.dumps(event_framework.required_role_types, ensure_ascii=False)}
{self._get_tc_prompt_for_event(unit_start.isoformat())}
## Participant Pool- Required relationship hints: {json.dumps(event_framework.required_relationship_hints, ensure_ascii=False)}
- Estimated participant count: {event_framework.estimated_participant_count}

## Target Character
- Name: {target_name} (P_TARGET, must be included in participants)

## Participant Pool (Detailed Information)
{pool_context}

## Task
Select the most suitable participants from the participant pool:
1. P_TARGET (target character) must be included in selected_participants
2. Match based on event's required role types and relationship hints
3. Consider whether each participant's current state is suitable for this event
4. If existing participants are insufficient, set needs_new_participants=true
   and describe what kind of new participants are needed in new_participant_suggestions
5. Provide a selection rationale for each selected participant
"""

        result: ParticipantSelectionOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=ParticipantSelectionOutput,
            system_prompt=system_prompt,
            task_type="step2_participant_selection",
            temperature=0.0,  # Structured decision
        )
        return result

    # ================================================================
    # Step 3: Participant Persona Validation & Update
    # ================================================================

    async def _step3_validate_and_update_persona(
        self,
        participant: Participant,
        event_framework: LowResFrameworkOutput,
        current_date: date,
    ) -> None:
        """
        Step 3: Validate a single participant's persona consistency and
        update if needed. Checks:
          a) Does not deviate from target persona
          b) Accurately reflects current time state
          c) If inconsistent, progressively update current_persona_brief_text
        """
        target_name = self.persona.get("persona_name_text", "protagonist")

        # Calculate participant's current age
        p_age = ""
        if participant.date_of_birth:
            p_age = str(
                current_date.year - participant.date_of_birth.year
                - (1 if (current_date.month, current_date.day) <
                   (participant.date_of_birth.month, participant.date_of_birth.day) else 0)
            )

        system_prompt = (
            "You are a persona consistency checker for life simulation. "
            "Check whether a character's current persona description:\n"
            "1. Does not deviate from the target persona\n"
            "2. Accurately reflects the character's state at the current moment\n"
            "If inconsistencies are found, provide an updated current persona description.\n\n"
            "IMPORTANT — Field semantics:\n"
            "- 'current_persona_brief' is a LIVING description tied to the current simulation date. "
            "The age written here must always match the character's actual age at that date "
            "(computed from date_of_birth), NOT the age mentioned in the target persona.\n"
            "- 'target_persona_brief' is the user-supplied end-state description for directional "
            "reference only. It may contain approximate or characterization ages — do NOT copy "
            "its age literally into the current persona.\n"
            "- Always use the 'Current age' value provided in the prompt as the ground-truth age "
            "when writing any updated persona description."
        )

        user_prompt = f"""## Current Time
- Date: {current_date.isoformat()}
{self._get_tc_prompt_for_event(current_date.isoformat())}
## Character Information
- Name: {participant.persona_name_text}
- Character ID: {participant.participant_id}
- Relationship to {target_name}: {participant.relationship_towards_the_main_character}
- Current age: approximately {p_age}  ← USE THIS EXACT AGE in any updated persona description
- Date of birth: {participant.date_of_birth.year if participant.date_of_birth else '?'}-{participant.date_of_birth.month if participant.date_of_birth else '?'}-{participant.date_of_birth.day if participant.date_of_birth else '?'}

## Persona Descriptions
- Initial persona (at simulation start): {participant.initial_persona_brief_text}
- Current persona: {participant.current_persona_brief_text}
- Target persona (at simulation end, for direction reference only): {participant.target_persona_brief_text}

## Upcoming Event
- Event theme: {event_framework.event_theme}
- Event summary: {event_framework.summary}

## Interaction History (last 5)
{self._format_recent_interactions(participant, limit=5)}

## Check Requirements
1. Does the current persona accurately reflect the character's state at {current_date.isoformat()}?
   - Is the age correct? The age MUST be approximately {p_age} (derived from date_of_birth and current date).
     Do NOT use the age from the target persona description.
   - Is the career/academic status consistent with the timeline?
   - Is the personality development reasonable?
2. Is the current persona maintaining a development direction consistent with the target persona?
3. If inconsistencies are found, provide the updated updated_current_persona_brief_text
   (one paragraph including current age ~{p_age}, career/academic status, personality traits, relationship status with protagonist, etc.)
"""

        try:
            result: PersonaValidationOutput = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=PersonaValidationOutput,
                system_prompt=system_prompt,
                task_type="step3_persona_validation",
                temperature=0.0,  # Factual validation needs max accuracy
            )

            if not result.is_consistent and result.updated_current_persona_brief_text:
                old_brief = participant.current_persona_brief_text[:80]
                participant.current_persona_brief_text = result.updated_current_persona_brief_text
                self._save_pool()
                logger.info(
                    f"  Step 3: Updated persona for {participant.participant_id} "
                    f"({participant.persona_name_text}): "
                    f"issues={result.issues}"
                )
            else:
                logger.debug(
                    f"  Step 3: Persona consistent for {participant.participant_id}"
                )
        except Exception:
            logger.exception(
                f"  Step 3: Persona validation failed for "
                f"{participant.participant_id}; keeping existing persona"
            )

    def _format_recent_interactions(
        self, participant: Participant, limit: int = 5
    ) -> str:
        """Format recent interaction history for a participant.

        Displays time_period (rich, human-readable) instead of date (machine-only).
        """
        history = participant.interactions_history_with_the_main_character
        if not history:
            return "(No interaction history)"
        recent = history[-limit:]
        lines = []
        for rec in recent:
            tone_str = f" (emotion: {rec.emotional_tone})" if rec.emotional_tone else ""
            display_period = rec.time_period or rec.date or "?"
            lines.append(f"- [{display_period}] {rec.summary}{tone_str}")
        return "\n".join(lines)

    # ================================================================
    # Step 4: Participant Sufficiency Assessment
    # ================================================================

    async def _step4_assess_sufficiency(
        self,
        event_framework: LowResFrameworkOutput,
        selected_ids: List[str],
        period_id: str,
    ) -> ParticipantSufficiencyOutput:
        """
        Step 4: Assess whether the current participant count is sufficient
        for the event. If not, provide detailed analysis including:
          - Which specific role types are missing
          - Why additional participants are needed
          - How many new participants to generate
          - Detailed requirements for each new participant
        """
        participant_details = []
        for pid in selected_ids:
            p = self.pool.get_participant(pid)
            if p:
                current_brief = p.current_persona_brief_text or ""
                participant_details.append(
                    f"- {p.participant_id}: {p.persona_name_text} "
                    f"(role={p.role}) — {p.relationship_towards_the_main_character}\n"
                    f"  Current status: {current_brief}"
                )
        participants_text = "\n".join(participant_details)

        # Build a summary of all available roles in the pool for context
        pool_role_summary = {}
        for p in self.pool.list_all():
            role = p.role or "unknown"
            pool_role_summary[role] = pool_role_summary.get(role, 0) + 1
        pool_roles_text = ", ".join(f"{r}: {c}" for r, c in pool_role_summary.items())

        target_name = self.persona.get("persona_name_text", "protagonist")

        system_prompt = (
            "You are a participant assessor for life simulation. "
            "Evaluate whether the currently selected participants are sufficient to support the natural unfolding of the event. "
            "If participants are insufficient, provide detailed analysis and a concrete supplementation plan."
        )

        user_prompt = f"""## Event Framework
- Event type: {event_framework.event_type}
- Event theme: {event_framework.event_theme}
- Event location: {event_framework.event_location}
- Event summary: {event_framework.summary}
- Required role types: {json.dumps(event_framework.required_role_types, ensure_ascii=False)}
- Required relationship hints: {json.dumps(event_framework.required_relationship_hints, ensure_ascii=False)}
- Estimated participant count: {event_framework.estimated_participant_count}

## Target Character
- Name: {target_name}

## Currently Selected Participants ({len(selected_ids)} total)
{participants_text}

## Participant Pool Role Distribution
{pool_roles_text}

## Assessment Task
Evaluate from the following dimensions:

1. **Quantity assessment**: Do the current {len(selected_ids)} participants meet the event's estimated {event_framework.estimated_participant_count} people requirement?

2. **Role coverage assessment**: Are all required role types ({json.dumps(event_framework.required_role_types, ensure_ascii=False)}) covered by participants?
   - Which specific role types are missing? (list in missing_role_types)

3. **Relationship matching assessment**: Are all required relationship hints ({json.dumps(event_framework.required_relationship_hints, ensure_ascii=False)}) matched by participants?

4. **If insufficient**, provide:
   - insufficiency_analysis: Detailed analysis of the insufficiency
   - additional_needed: Specific number of participants to add
   - new_participant_requirements: Detailed requirements for each new participant, including:
     * role_type: Role type (e.g., family, friend, teacher, classmate)
     * relationship_to_main_character: Specific relationship to {target_name}
     * persona_requirements: Detailed persona requirements (age range, personality traits, background, etc.)
     * reason_needed: Why this participant is needed
   - additional_descriptions: Brief text description for each new participant (for generation)

5. **If sufficient**, provide the reasoning
"""

        result: ParticipantSufficiencyOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=ParticipantSufficiencyOutput,
            system_prompt=system_prompt,
            task_type="p2_participant_sufficiency",
            temperature=0.0,  # Structured decision
        )
        return result

    # ================================================================
    # Step 5: New Participant Generation
    # ================================================================

    async def _step5_generate_new_participants(
        self,
        descriptions: List[str],
        appear_period: str,
        event_summary: str,
        requirements: Optional[List[NewParticipantRequirement]] = None,
        current_event_date: Optional[str] = None,
    ) -> List[str]:
        """
        Step 5: Generate new participants one by one following the P1
        standard persona creation flow, then add them to the pool.

        Supports two modes:
          1. Detailed mode (preferred): uses ``requirements`` from Step 4
             with role_type, relationship, persona_requirements, etc.
          2. Fallback mode: uses simple ``descriptions`` strings.

        Each participant is generated individually to ensure quality and
        adherence to the full persona creation specification.

        Args:
            descriptions: Simple text descriptions (fallback).
            appear_period: The period when these participants first appear.
            event_summary: Context about the event triggering creation.
            requirements: Detailed requirements from Step 4 (preferred).

        Returns:
            List of newly created participant IDs.
        """
        target_name = self.persona.get("persona_name_text", "protagonist")
        target_brief = self.persona.get("persona_brief_text", "")

        new_ids = []

        # Determine the generation items: prefer detailed requirements
        gen_items: List[dict] = []
        if requirements:
            for req in requirements:
                gen_items.append({
                    "description": (
                        f"Role type: {req.role_type}, "
                        f"Relationship to {target_name}: {req.relationship_to_main_character}, "
                        f"Persona requirements: {req.persona_requirements}"
                    ),
                    "role_type": req.role_type,
                    "relationship": req.relationship_to_main_character,
                    "persona_requirements": req.persona_requirements,
                    "reason": req.reason_needed,
                })
        else:
            for desc in descriptions:
                gen_items.append({
                    "description": desc,
                    "role_type": "",
                    "relationship": "",
                    "persona_requirements": "",
                    "reason": "",
                })

        for idx, item in enumerate(gen_items):
            logger.info(
                f"  Step 5: Generating participant {idx+1}/{len(gen_items)}: "
                f"{item['description'][:80]}..."
            )

            # Build a comprehensive prompt following P1 standard flow
            system_prompt = (
                "You are a character generator for life simulation, following standard character creation specifications. "
                "You need to generate a complete new character for the life simulation.\n\n"
                "CRITICAL OUTPUT RULES:\n"
                "- Output ACTUAL VALUES for each field, NOT field descriptions or schema metadata.\n"
                "- 'persona_name_text' must be a real name string, e.g. 'John Smith' — NOT a description like 'A new character'.\n"
                "- 'relationship_towards_the_main_character' must be a concrete relationship, e.g. 'college roommate' — NOT a description.\n"
                "- 'persona_brief_text' must be a 3-5 sentence character description — NOT a schema description.\n"
                "- 'date_of_birth_year/month/day' must be integer numbers — NOT strings or descriptions.\n\n"
                "Required fields:\n"
                "1. persona_name_text: Full name matching the character's cultural background and era\n"
                "2. relationship_towards_the_main_character: Specific relationship to the protagonist\n"
                "3. persona_brief_text: Complete character description (age, personality, background, occupation)\n"
                "4. date_of_birth_year / date_of_birth_month / date_of_birth_day: Reasonable birth date as integers\n\n"
                "The generated character should naturally fit into the protagonist's life story."
            )

            # Build detailed context for generation
            requirement_section = ""
            if item["role_type"]:
                requirement_section = f"""
## Detailed Character Requirements
- Role type: {item['role_type']}
- Relationship to protagonist: {item['relationship']}
- Persona requirements: {item['persona_requirements']}
- Reason needed: {item['reason']}
"""
            else:
                requirement_section = f"""
## Character Description
{item['description']}
"""

            # T-A4: Inject already-taken names to prevent LLM from generating duplicates
            taken_names = [
                p.persona_name_text
                for p in self.pool.list_all()
                if p.persona_name_text and p.participant_id != self.pool.TARGET_ID
            ]
            taken_names_section = ""
            if taken_names:
                taken_names_section = (
                    "\n## IMPORTANT: Already-Taken Names\n"
                    "The following names are already used by existing characters. "
                    "You MUST NOT use any of these names for the new character. "
                    "Choose a completely different name:\n"
                    + "\n".join(f"- {n}" for n in taken_names)
                    + "\n"
                )

            # [A-01] Compute protagonist birth year for DOB constraint
            protagonist_birth_year = ""
            protagonist_birth_year_int = None
            if self.simulation_start_date:
                try:
                    from datetime import date as _date
                    sim_start_d = _date.fromisoformat(self.simulation_start_date)
                    protagonist_birth_year_int = sim_start_d.year
                    protagonist_birth_year = str(sim_start_d.year)
                except Exception:
                    pass

            dob_constraint = ""
            if protagonist_birth_year:
                dob_constraint = (
                    f"\n   - **CRITICAL**: The protagonist was born in {protagonist_birth_year}. "
                    f"A 'peer' or 'classmate' character must be born within ±5 years of {protagonist_birth_year} "
                    f"(i.e., between {int(protagonist_birth_year)-5} and {int(protagonist_birth_year)+5}). "
                    f"An 'elder/mentor' character should be born 10-30 years BEFORE {protagonist_birth_year}. "
                    f"Do NOT use birth years from other supporting characters (parents, grandparents) for peer-age characters."
                )

            user_prompt = f"""## Protagonist Information
- Name: {target_name}
- Bio: {target_brief}
- Date of birth: {self.simulation_start_date or 'unknown'} (protagonist born at simulation start)

## Event Context
{event_summary}

## Appearance Stage
{appear_period}
{self._get_tc_prompt_for_event(current_event_date or getattr(self, '_last_event_date_str', '') or self.simulation_start_date or '')}
{requirement_section}
{taken_names_section}
## Generation Requirements
Generate a complete new character following these specifications:

1. **persona_name_text**: Full name, should match the character's cultural background, language environment, and era. MUST be different from all names listed in "Already-Taken Names" above.
2. **relationship_towards_the_main_character**: Specific relationship description to {target_name}
3. **persona_brief_text**: A complete character description (3-5 sentences), must include:
   - Character's age and basic background
   - Personality traits and behavioral patterns
   - Nature of relationship with {target_name} and interaction style
   - Role positioning within the event
4. **date_of_birth_year / date_of_birth_month / date_of_birth_day**: A reasonable date of birth
   - Date of birth MUST be consistent with the age stated in persona_brief_text
   - Consider age relationship to protagonist (e.g., peer, elder){dob_constraint}

Note:
- Character should naturally fit into the protagonist's life story
- Character's background and personality should match the event scene
- Avoid duplication or conflict with existing participants
"""

            try:
                np_info: NewParticipantInfo = await self.llm.generate_structured(
                    prompt=user_prompt,
                    response_model=NewParticipantInfo,
                    system_prompt=system_prompt,
                    task_type="p2_new_participant",
                    temperature=0.0,  # Creative generation
                )

                dob = None
                if np_info.date_of_birth_year and np_info.date_of_birth_month and np_info.date_of_birth_day:
                    # [A-02] Validate DOB against protagonist birth year for peer roles (warning only, no auto-correction)
                    dob_year = np_info.date_of_birth_year
                    if protagonist_birth_year_int is not None:
                        relationship_lower = (np_info.relationship_towards_the_main_character or "").lower()
                        brief_lower = (np_info.persona_brief_text or "").lower()
                        is_peer_role = any(kw in relationship_lower or kw in brief_lower
                                           for kw in ["classmate", "colleague", "friend", "peer",
                                                      "roommate", "teammate", "coworker", "co-worker"])
                        if is_peer_role:
                            peer_min = protagonist_birth_year_int - 10
                            peer_max = protagonist_birth_year_int + 10
                            if not (peer_min <= dob_year <= peer_max):
                                logger.warning(
                                    f"  [A-02] DOB year {dob_year} for peer-role participant "
                                    f"'{np_info.persona_name_text}' is outside expected range "
                                    f"[{peer_min}, {peer_max}] (protagonist born {protagonist_birth_year_int}). "
                                    f"Phase 5 timeline consistency check will handle correction if needed."
                                )
                                # No auto-correction: Phase 5 has full context and should decide
                    dob = DateOfBirth(
                        year=dob_year,
                        month=np_info.date_of_birth_month,
                        day=np_info.date_of_birth_day,
                    )

                enriched_brief = np_info.persona_brief_text
                if event_summary:
                    enriched_brief += (
                        f" (This participant was introduced during the event: "
                        f"{event_summary})"
                    )

                extend = self.pool.llm is not None
                participant = await self.pool.add_participant(
                    name=np_info.persona_name_text,
                    relationship=np_info.relationship_towards_the_main_character,
                    brief=enriched_brief,
                    date_of_birth=dob,
                    appear_period=appear_period,
                    extend_profile=extend,
                    persona_config=self.persona,
                    sim_start_date=self.simulation_start_date,
                    sim_end_date=self.simulation_end_date,
                )
                new_ids.append(participant.participant_id)
                logger.info(
                    f"  Step 5: New participant registered: "
                    f"{participant.participant_id} | {np_info.persona_name_text} "
                    f"(role={item.get('role_type', '?')}, "
                    f"relationship={np_info.relationship_towards_the_main_character})"
                )

                # [A-03] Run Phase 5 timeline consistency check for dynamically added participant
                if self.pool.llm is not None:
                    try:
                        await self.pool._check_timeline_consistency(
                            participant=participant,
                            persona_config=self.persona,
                            life_plan=None,
                            sim_start_date=self.simulation_start_date or "",
                            sim_end_date=self.simulation_end_date or "",
                        )
                        logger.info(
                            f"  [A-03] Phase 5 timeline check completed for "
                            f"{participant.participant_id} | {np_info.persona_name_text}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"  [A-03] Phase 5 timeline check failed for "
                            f"{participant.participant_id}: {e}; skipping"
                        )
            except Exception:
                logger.exception(
                    f"  Step 5: Failed to generate participant {idx+1} "
                    f"from '{item['description'][:60]}'; skipping this candidate"
                )

        # Persist pool after all new participants are added
        if new_ids:
            self._save_pool()
        return new_ids

    def _entry_to_requirement(self, entry) -> "NewParticipantRequirement":
        """Convert NewParticipantRequirementEntry to NewParticipantRequirement.

        T13: Bridges the field name mismatch: persona_sketch → persona_requirements.
        """
        from lifelong_synth.simulation_p2_event_organiser.definition import (
            NewParticipantRequirement,
        )
        return NewParticipantRequirement(
            role_type=entry.role_type,
            relationship_to_main_character=entry.relationship_to_main_character,
            persona_requirements=entry.persona_sketch,  # field name mapping
            reason_needed=entry.reason_needed,
        )

    async def _create_new_participant_from_requirement(
        self,
        requirement: NewParticipantRequirement,
        period_id: str,
        current_date: Any,
    ) -> Optional[str]:
        """Create a new participant from a single NewParticipantRequirement.

        Wraps _step5_generate_new_participants to handle a single requirement.
        Returns the new participant ID or None on failure.
        """
        event_summary = getattr(self, "_last_event_summary", "")
        current_date_str = (
            current_date.isoformat() if hasattr(current_date, 'isoformat')
            else str(current_date) if current_date else None
        )
        try:
            new_ids = await self._step5_generate_new_participants(
                descriptions=[],
                appear_period=period_id,
                event_summary=event_summary,
                requirements=[requirement],
                current_event_date=current_date_str,
            )
            return new_ids[0] if new_ids else None
        except Exception as e:
            logger.warning(
                f"_create_new_participant_from_requirement: failed for "
                f"role_type={requirement.role_type}, "
                f"relationship={requirement.relationship_to_main_character}: {e}; "
                f"returning None (event will proceed without this participant)"
            )
            return None

    # ── Legacy: Register new participants from LowResEventOutput ──

    async def _register_new_participants(
        self,
        new_participants: List[NewParticipantInfo],
        appear_period: str,
        event_summary: str = "",
    ) -> List[str]:
        """Register new participants from LLM output and return their IDs."""
        new_ids = []
        for np_info in new_participants:
            dob = None
            if np_info.date_of_birth_year and np_info.date_of_birth_month and np_info.date_of_birth_day:
                dob = DateOfBirth(
                    year=np_info.date_of_birth_year,
                    month=np_info.date_of_birth_month,
                    day=np_info.date_of_birth_day,
                )

            enriched_brief = np_info.persona_brief_text
            if event_summary:
                enriched_brief += (
                    f" (This participant was introduced during the event: "
                    f"{event_summary})"
                )

            extend = self.pool.llm is not None
            participant = await self.pool.add_participant(
                name=np_info.persona_name_text,
                relationship=np_info.relationship_towards_the_main_character,
                brief=enriched_brief,
                date_of_birth=dob,
                appear_period=appear_period,
                extend_profile=extend,
                persona_config=self.persona,
                sim_start_date=self.simulation_start_date,
                sim_end_date=self.simulation_end_date,
            )
            new_ids.append(participant.participant_id)
            logger.info(
                f"  New participant registered: {participant.participant_id} | "
                f"{np_info.persona_name_text} ({participant.role})"
            )
        # Persist pool after all new participants are added
        if new_ids:
            self._save_pool()
        return new_ids

    # ================================================================
    # Module 1 (Merged): Unified Pool for LR + Outlines (single prompt)
    # ================================================================

    async def _step2_unified_pool_for_lr_and_outlines(
        self,
        lr_with_outlines: LRWithOutlinesOutput,
        period_id: str,
        unit_start: date,
        unit_end: date,
        unit_label: str,
    ) -> LRAndOutlinePoolOutput:
        """Single prompt to select participants for both LR event and all outlines.

        Merges _step2_unified_participant_pool (for LR) and per-outline
        participant selection into one call, saving N LLM calls.

        Args:
            lr_with_outlines: Output from _step1a_generate_lr_with_outlines.
            period_id: Period identifier.
            unit_start: Start date of the time unit.
            unit_end: End date of the time unit.
            unit_label: Label for this time unit.

        Returns:
            LRAndOutlinePoolOutput with LR pool and outline pools.
        """
        target_name = self.persona.get("persona_name_text", "protagonist")
        lr_fw = lr_with_outlines.lr_framework
        outlines = lr_with_outlines.outlines

        # SNAP filtering
        all_themes = {lr_fw.event_theme}
        for ol in outlines:
            all_themes.add(ol.key_event_theme)
        snap = SNAPFilter(self.pool, self.memory)
        filtered_ids = snap.filter_relevant_participants(
            event_theme=" ".join(all_themes),
            event_type=lr_fw.event_type,
            period_id=period_id,
            current_date=unit_start,
            max_candidates=12 + 4 * len(outlines),
        )

        # Build participant context
        participant_details = []
        for pid in filtered_ids:
            p = self.pool.get_participant(pid)
            if p:
                current_brief = p.current_persona_brief_text or ""
                participant_details.append(
                    f"- {p.participant_id}: {p.persona_name_text} "
                    f"(role={p.role}, relationship={p.relationship_towards_the_main_character})\n"
                    f"  Current status: {current_brief}"
                )
        pool_context = "\n".join(participant_details)

        # Build LR event section
        lr_section = f"""#### Low-Resolution Event for This Period
- Event type: {lr_fw.event_type}
- Event theme: {lr_fw.event_theme}
- Event location: {lr_fw.event_location}
- Event summary: {lr_fw.summary}
- Required role types: {json.dumps(lr_fw.required_role_types, ensure_ascii=False)}
- Required relationship hints: {json.dumps(lr_fw.required_relationship_hints, ensure_ascii=False)}
- Estimated participant count: {lr_fw.estimated_participant_count}
"""

        # Build outline sections
        outline_sections = []
        for j, ol in enumerate(outlines):
            outline_sections.append(
                f"#### Key Event Outline {j+1}/{len(outlines)}\n"
                f"- Theme: {ol.key_event_theme}\n"
                f"- Summary: {ol.key_event_summary}\n"
                f"- Time: {ol.key_event_start_date} {ol.key_event_start_time} → "
                f"{ol.key_event_end_date} {ol.key_event_end_time}\n"
                f"- Motivation: {ol.key_event_motivation or 'N/A'}\n"
                f"- Outcome: {ol.key_event_outcome or 'N/A'}\n"
                f"- Setting: {ol.key_event_setting or 'N/A'}\n"
                f"- Turning point: {ol.key_event_turning_point or 'N/A'}\n"
                f"- Required roles: {json.dumps(ol.required_participant_roles, ensure_ascii=False)}\n"
            )

        system_prompt = (
            "You are a participant matcher for life simulation. "
            "Based on the LR event and multiple key event outlines, select participants "
            "for ALL events in one response.\n"
            "1. For the LR event, select participants who cover the full period\n"
            "2. For each outline, select participants who match the specific key event\n"
            "3. Participants can overlap between events\n"
            "4. If insufficient participants, provide new participant requirements"
        )

        user_prompt = f"""## Time Unit: {unit_label} ({unit_start.isoformat()} to {unit_end.isoformat()})

{lr_section}

## Key Event Outlines ({len(outlines)} total)
{chr(10).join(outline_sections)}

## Target Character
- Name: {target_name} (P_TARGET, must be included in participants for each event)

## Candidate Participant Pool
{pool_context}

## Task
For each event (LR + outlines):
1. Select the most suitable participants from candidates (P_TARGET must be included)
2. Evaluate whether selected participants are sufficient
3. If insufficient, describe what kind of new participants are needed

Return:
- lr_pool: UnifiedParticipantPoolOutput for the LR event
- outline_pools: List of OutlinePoolEntry (one per outline, with outline_index)
"""

        result: LRAndOutlinePoolOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=LRAndOutlinePoolOutput,
            system_prompt=system_prompt,
            temperature=0.0,
            task_type="step2_unified_pool_lr_outlines",
        )
        return result

    # ================================================================
    # Module 1 (Batch): Batch Unified Participant Pool for consecutive units
    # ================================================================

    async def _step2_unified_participant_pool_batch(
        self,
        event_frameworks: List[LowResFrameworkOutput],
        high_res_outlines: List[Optional[HighResOutlineOutput]],
        period_id: str,
        time_units: List[Tuple[date, date]],
        unit_labels: List[str],
    ) -> BatchUnifiedParticipantPoolOutput:
        """Batch version: single LLM call to select participants for all time units.

        Replaces N sequential _step2_unified_participant_pool calls with 1 call.
        Each time unit in the batch gets its own UnifiedParticipantPoolOutput.

        Args:
            event_frameworks: LR frameworks for each time unit in the batch.
            high_res_outlines: Optional HR outlines for each unit (None if no key event).
            period_id: Period identifier.
            time_units: List of (start, end) dates for each unit.
            unit_labels: Corresponding labels.

        Returns:
            BatchUnifiedParticipantPoolOutput with one pool output per time unit.
        """
        n_units = len(time_units)
        target_name = self.persona.get("persona_name_text", "protagonist")

        # Use SNAP filtering — pre-filter candidates based on union of themes
        all_themes = set()
        for fw in event_frameworks:
            all_themes.add(fw.event_theme)
        snap = SNAPFilter(self.pool, self.memory)
        filtered_ids = snap.filter_relevant_participants(
            event_theme=" ".join(all_themes),
            event_type=event_frameworks[0].event_type if event_frameworks else "daily_routine",
            period_id=period_id,
            current_date=time_units[0][0],
            max_candidates=12 * n_units,  # Scale max candidates with batch size
        )

        # Build context for filtered participants
        participant_details = []
        for pid in filtered_ids:
            p = self.pool.get_participant(pid)
            if p:
                current_brief = p.current_persona_brief_text or ""
                participant_details.append(
                    f"- {p.participant_id}: {p.persona_name_text} "
                    f"(role={p.role}, relationship={p.relationship_towards_the_main_character})\n"
                    f"  Current status: {current_brief}"
                )
        pool_context = "\n".join(participant_details)

        # Build per-unit event context sections
        unit_event_sections = []
        for i, fw in enumerate(event_frameworks):
            hr_outline = high_res_outlines[i] if i < len(high_res_outlines) else None
            hr_section = ""
            if hr_outline is not None:
                hr_roles = getattr(hr_outline, 'required_participant_roles', []) or []
                hr_roles_text = ""
                if hr_roles:
                    hr_roles_text = (
                        f"\n**HARD CONSTRAINT**: The following role types MUST be covered "
                        f"for the high-res key event in unit {i+1}: {json.dumps(hr_roles, ensure_ascii=False)}. "
                        f"If no existing participant matches a required role, set is_sufficient=false "
                        f"and describe the missing role in new_participant_requirements."
                    )
                hr_section = f"""
#### High-Resolution Key Event for Unit {i+1} (Additional Considerations)
- Theme: {hr_outline.key_event_theme}
- Summary: {hr_outline.key_event_summary}
- Scene setting: {hr_outline.key_event_setting or 'N/A'}
- Turning point: {hr_outline.key_event_turning_point or 'N/A'}
{hr_roles_text}
"""

            unit_event_sections.append(
                f"#### Unit {i+1}/{n_units}: {unit_labels[i]}\n"
                f"- Event type: {fw.event_type}\n"
                f"- Event theme: {fw.event_theme}\n"
                f"- Event location: {fw.event_location}\n"
                f"- Event summary: {fw.summary}\n"
                f"- Time range: {time_units[i][0].isoformat()} to {time_units[i][1].isoformat()}\n"
                f"- Required role types: {json.dumps(fw.required_role_types, ensure_ascii=False)}\n"
                f"- Required relationship hints: {json.dumps(fw.required_relationship_hints, ensure_ascii=False)}\n"
                f"- Estimated participant count: {fw.estimated_participant_count}\n"
                f"{hr_section}"
            )

        system_prompt = (
            "You are a participant matcher for life simulation. "
            "Based on MULTIPLE event frameworks and participant pool information, "
            "select participants for ALL time units in one response.\n"
            "1. For each time unit, select the most suitable participants from the pool\n"
            "2. Evaluate whether the selected participants are sufficient to support the event\n"
            "3. If insufficient, provide requirement descriptions for new participants\n"
            "Complete all evaluations at once, no need to break into steps.\n\n"
            # T-B3: Physical presence constraint
            "CRITICAL RULE \u2014 PHYSICAL PRESENCE:\n"
            "Select ONLY participants who will be PHYSICALLY PRESENT in the scene.\n"
            "- For a private family conversation at home: select only the family members in the room.\n"
            "- For a one-on-one meeting: select only the two people meeting.\n"
            "- Do NOT include participants who are merely 'aware of' or 'affected by' the event but not physically present.\n"
            "- WRONG: Including all 20+ participants for a 2-person conversation.\n"
            "- RIGHT: Including only the 2-3 people who are physically in the scene.\n"
            "Typical scene size: 1-5 participants. Only exceed 5 if the event is explicitly a group gathering."
        )

        user_prompt = f"""## Event Frameworks for {n_units} Time Units
{chr(10).join(unit_event_sections)}

## Target Character
- Name: {target_name} (P_TARGET, must be included in participants for each unit)

## Candidate Participant Pool
{pool_context}

## Task
For each of the {n_units} time units:
1. Select the most suitable participants from candidates (P_TARGET must be included)
2. Evaluate whether selected participants are sufficient to support the natural unfolding of the event
3. If insufficient (is_sufficient=false), describe what kind of new participants are needed

Return one UnifiedParticipantPoolOutput per time unit.
"""

        result: BatchUnifiedParticipantPoolOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=BatchUnifiedParticipantPoolOutput,
            system_prompt=system_prompt,
            temperature=0.0,  # Structured decision
            task_type="step2_batch_participant_selection",
        )
        return result

    # ================================================================
    # Module 1: Unified Participant Pool (replaces Step 2 + Step 4 + Step 5)
    # ================================================================

    async def _step2_unified_participant_pool(
        self,
        event_framework: LowResFrameworkOutput,
        high_res_outline: Optional[HighResOutlineOutput],
        period_id: str,
        unit_start: date,
        unit_end: date,
    ) -> UnifiedParticipantPoolOutput:
        """
        Generate a unified participant pool for both low-res and high-res events.
        Replaces Step 2 + Step 4 + Step 5 with a single LLM call.

        CONSTRAINT: When high_res_outline is provided, the pool MUST cover
        both low-res and high-res participant needs. P3 refine_participants()
        will be constrained to select ONLY from this pool.
        """
        # Use SNAP filtering to pre-filter candidates
        snap = SNAPFilter(self.pool, self.memory)
        filtered_ids = snap.filter_relevant_participants(
            event_theme=event_framework.event_theme,
            event_type=event_framework.event_type,
            period_id=period_id,
            current_date=unit_start,
            max_candidates=12,
        )

        # Build context only for filtered participants
        participant_details = []
        for pid in filtered_ids:
            p = self.pool.get_participant(pid)
            if p:
                current_brief = p.current_persona_brief_text or ""
                participant_details.append(
                    f"- {p.participant_id}: {p.persona_name_text} "
                    f"(role={p.role}, relationship={p.relationship_towards_the_main_character})\n"
                    f"  Current status: {current_brief}"
                )
        pool_context = "\n".join(participant_details)

        target_name = self.persona.get("persona_name_text", "protagonist")

        hr_section = ""
        if high_res_outline is not None:
            hr_roles = getattr(high_res_outline, 'required_participant_roles', []) or []
            hr_roles_text = ""
            if hr_roles:
                hr_roles_text = (
                    f"\n**HARD CONSTRAINT**: The following role types MUST be covered "
                    f"for the high-res key event: {json.dumps(hr_roles, ensure_ascii=False)}. "
                    f"If no existing participant matches a required role, set is_sufficient=false "
                    f"and describe the missing role in new_participant_requirements."
                )
            hr_section = f"""
## High-Resolution Key Event (Additional Considerations)
- Theme: {high_res_outline.key_event_theme}
- Summary: {high_res_outline.key_event_summary}
- Scene setting: {high_res_outline.key_event_setting or 'N/A'}
- Turning point: {high_res_outline.key_event_turning_point or 'N/A'}
Note: Selected participants must satisfy both the low-resolution event and the high-resolution key event requirements.
For participants essential to the high-resolution key event, set is_essential_for_high_res=true.
{hr_roles_text}
"""

        system_prompt = (
            "You are a participant matcher for life simulation. "
            "Based on the event framework and participant pool information, complete the following three tasks:\n"
            "1. Select the most suitable participants from the participant pool\n"
            "2. Evaluate whether the selected participants are sufficient to support the event\n"
            "3. If insufficient, provide requirement descriptions for new participants\n"
            "Complete all evaluations at once, no need to break into steps.\n\n"
            # T-B2: Physical presence constraint
            "CRITICAL RULE \u2014 PHYSICAL PRESENCE:\n"
            "Select ONLY participants who will be PHYSICALLY PRESENT in the scene.\n"
            "- For a private family conversation at home: select only the family members in the room.\n"
            "- For a one-on-one meeting: select only the two people meeting.\n"
            "- Do NOT include participants who are merely 'aware of' or 'affected by' the event but not physically present.\n"
            "- WRONG: Including all 20+ participants for a 2-person conversation.\n"
            "- RIGHT: Including only the 2-3 people who are physically in the scene.\n"
            "Typical scene size: 1-5 participants. Only exceed 5 if the event is explicitly a group gathering."
        )

        user_prompt = f"""## Event Framework
- Event type: {event_framework.event_type}
- Event theme: {event_framework.event_theme}
- Event location: {event_framework.event_location}
- Event summary: {event_framework.summary}
- Time range: {unit_start.isoformat()} to {unit_end.isoformat()}
{self._get_tc_prompt_for_event(unit_start.isoformat())}
{hr_section}
## Target Character
- Name: {target_name} (P_TARGET, must be included in participants)

## Candidate Participant Pool
{pool_context}

## Task
1. Select the most suitable participants from candidates (P_TARGET must be included)
2. Evaluate whether selected participants are sufficient to support the natural unfolding of the event
3. If insufficient (is_sufficient=false), describe what kind of new participants are needed in new_participant_requirements
"""

        result: UnifiedParticipantPoolOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=UnifiedParticipantPoolOutput,
            system_prompt=system_prompt,
            temperature=0.0,  # Structured decision
            task_type="step2_participant_selection",
        )
        return result

    def _snap_fallback_search(
        self,
        requirements: List,
        excluded_ids: set,
    ) -> List[str]:
        """Search the FULL pool (not just SNAP top-k) for participants
        matching the missing role requirements.

        Zero LLM cost — pure keyword/relationship matching.
        Used when unified pool says insufficient, before creating new participants.
        """
        found_ids = []
        all_participants = self.pool.list_all()
        for req in requirements:
            role_keywords = set(req.role_type.lower().split())
            rel_keywords = set(req.relationship_to_main_character.lower().split())
            search_terms = role_keywords | rel_keywords

            best_match = None
            best_score = 0
            for p in all_participants:
                if p.participant_id in excluded_ids:
                    continue
                if p.participant_id == "P_TARGET":
                    continue
                p_text = (
                    f"{p.role} {p.relationship_towards_the_main_character} "
                    f"{p.current_persona_brief_text}"
                ).lower()
                score = sum(1 for term in search_terms if term in p_text)
                if score > best_score:
                    best_score = score
                    best_match = p.participant_id

            if best_match and best_score >= 1:
                found_ids.append(best_match)
                excluded_ids.add(best_match)

        return found_ids

    # ================================================================
    # Module 2: Batch Persona Validation (replaces per-participant Step 3)
    # ================================================================

    async def _step3_batch_validate_personas(
        self,
        participant_ids: List[str],
        event_framework: LowResFrameworkOutput,
        current_date: date,
        high_res_outline: Optional[HighResOutlineOutput] = None,
    ) -> None:
        """Batch validate all participants' personas in a single LLM call.

        MODEL TIER: Tier 1 (gpt-4.1-nano) — structural validation, no creativity needed.

        v4: Always uses LLM validation (rule-based pre-check removed).
        Prompt is generalized to support any language, era, or fictional world.
        """
        if not participant_ids:
            return

        logger.info(
            f"  Step 4 (batch): Validating {len(participant_ids)} participants via LLM"
        )

        # Build batch prompt for LLM validation
        target_name = self.persona.get("persona_name_text", "Target")
        participant_sections = []
        for pid in participant_ids:
            p = self.pool.get_participant(pid)
            if not p:
                continue
            p_age = ""
            if p.date_of_birth:
                p_age = str(
                    current_date.year - p.date_of_birth.year
                    - (1 if (current_date.month, current_date.day) <
                       (p.date_of_birth.month, p.date_of_birth.day) else 0)
                )
            participant_sections.append(
                f"### {p.participant_id}: {p.persona_name_text}\n"
                f"- Relationship to {target_name}: {p.relationship_towards_the_main_character}\n"
                f"- Current age: ~{p_age}\n"
                f"- Initial persona: {p.initial_persona_brief_text}\n"
                f"- Current persona: {p.current_persona_brief_text}\n"
                f"- Target persona: {p.target_persona_brief_text}"
            )

        # Build HR context section if available
        hr_section = ""
        if high_res_outline is not None:
            hr_section = (
                f"\n## Upcoming High-Resolution Key Event\n"
                f"- Theme: {high_res_outline.key_event_theme}\n"
                f"- Summary: {high_res_outline.key_event_summary}\n"
                f"- Setting: {high_res_outline.key_event_setting or 'N/A'}\n"
            )

        system_prompt = (
            "You are a persona consistency validator for a character simulation system. "
            "Check whether each character's current persona description accurately reflects "
            "their state at the given point in time. "
            "If inconsistencies are found, provide an updated persona description "
            "in English. "
            "If the existing persona text contains non-English segments, translate them to English in the updated version."
        )

        user_prompt = f"""## Current Time
- Date: {current_date.isoformat()}
{self._get_tc_prompt_for_event(current_date.isoformat())}
## Upcoming Event
- Theme: {event_framework.event_theme}
- Summary: {event_framework.summary}
{hr_section}
## Characters to Validate
{chr(10).join(participant_sections)}

## Task
For each character, check ALL of the following dimensions:
1. **Age / Life stage**: Does the persona reflect the correct age and life stage at this point in time?
2. **Roles & Status**: Are their roles, titles, occupations, or ranks consistent with the timeline?
   Pay special attention to career progression plausibility — titles and ranks require time to achieve.
   For example, a 35-year-old university teacher should not be described as a "full professor" or
   "second-tier professor" (which typically requires 50+ years of age and decades of experience).
   Similarly, a 30-year-old bank employee should not be a "branch manager".
3. **Relationships**: Have any significant relationship changes occurred that should be reflected?
4. **Circumstances**: Has their living situation, health, location, or other circumstances changed?
5. **Development arc**: Is the current persona progressing naturally from initial_persona toward target_persona?

For each character:
- If ALL dimensions are consistent → set is_consistent=true, leave updated_persona empty
- If ANY dimension is inconsistent → set is_consistent=false, provide updated_persona
  (one paragraph, ≤500 chars, in English)
"""

        try:
            result: BatchPersonaValidationOutput = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=BatchPersonaValidationOutput,
                system_prompt=system_prompt,
                temperature=0.0,  # Factual validation needs max accuracy
                task_type="step4_persona_validation",
            )

            for entry in result.validations:
                if not entry.is_consistent and entry.updated_persona:
                    participant = self.pool.get_participant(entry.participant_id)
                    if participant:
                        participant.current_persona_brief_text = entry.updated_persona
                        self.pool.mark_dirty(entry.participant_id)
                        logger.info(
                            f"  Step 4 (batch): Updated persona for "
                            f"{entry.participant_id}"
                        )
            self._save_pool()
        except Exception:
            logger.exception(
                "  Step 4 (batch): Batch persona validation failed; "
                "keeping existing personas"
            )

    # ================================================================
    # Module 3 (Batch Period): Parallel UPER for multiple events in a period
    # ================================================================

    async def _uper_reflect_batch_period(
        self,
        event_tasks: List[Dict[str, Any]],
    ) -> None:
        """Parallel UPER for multiple events within a single time period.

        Uses asyncio.gather to run per-event UPER calls concurrently,
        reducing wall-clock time from N*T to ~max(T).

        Args:
            event_tasks: List of dicts, each containing:
                - participant_ids: List[str]
                - event_summary: str
                - event_date: date
                - event_id: str
                - high_res_summary: Optional[str]
                - lr_time_period: str
                - hr_time_period: str
                - post_scene_memories: Optional[List[Dict]]
        """
        if not event_tasks:
            return

        async def _single_uper(task: Dict[str, Any]) -> None:
            try:
                await self._uper_reflect_batch(
                    participant_ids=task["participant_ids"],
                    event_summary=task["event_summary"],
                    event_date=task["event_date"],
                    event_id=task["event_id"],
                    high_res_summary=task.get("high_res_summary"),
                    lr_time_period=task.get("lr_time_period", ""),
                    hr_time_period=task.get("hr_time_period", ""),
                    post_scene_memories=task.get("post_scene_memories"),
                    is_habitual=task.get("is_habitual", False),
                )
            except Exception as e:
                logger.error(
                    f"  [M2-UPER] Parallel UPER failed for event "
                    f"{task.get('event_id', '?')}: {e} — participant memories/personas will not be updated for this event"
                )

        results = await asyncio.gather(
            *[_single_uper(task) for task in event_tasks],
            return_exceptions=True,
        )

        n_success = sum(1 for r in results if r is None)
        n_fail = sum(1 for r in results if isinstance(r, Exception))
        logger.info(
            f"[M2-UPER] Period batch UPER complete: "
            f"{n_success} succeeded, {n_fail} failed"
        )

    # ================================================================
    # Module 3: UPER — Unified Post-Event Reflection
    # ================================================================

    async def _uper_reflect_batch(
        self,
        participant_ids: List[str],
        event_summary: str,
        event_date: date,
        event_id: str,
        high_res_summary: Optional[str] = None,
        lr_time_period: str = "",
        hr_time_period: str = "",
        post_scene_memories: Optional[List[Dict]] = None,
        is_habitual: bool = False,
    ) -> None:
        """
        Unified Post-Event Reflection.

        - P_TARGET: single LLM call generates first-person memory + persona update.
          post_scene_memories for P_TARGET (if provided) are included as context.
        - Side-characters: event_summary written directly to interaction_history (no LLM).
          post_scene_memories for side-chars are ignored.

        MODEL TIER: Tier 2 (only for P_TARGET)
        """
        target_name = self.persona.get("persona_name_text", "protagonist")

        # ── Side-characters: write event summary directly (no LLM) ──
        side_char_ids = [pid for pid in participant_ids if pid != "P_TARGET"]
        for pid in side_char_ids:
            try:
                self._upsert_interaction_history(
                    participant_id=pid,
                    event_id=event_id,
                    date_str=event_date.isoformat(),
                    summary=event_summary,
                    time_period=hr_time_period or lr_time_period or event_date.isoformat(),
                )
            except KeyError:
                logger.warning(f"  UPER: side-char '{pid}' not found; skipping interaction history")

        # ── P_TARGET: LLM generates first-person memory + persona update ──
        if "P_TARGET" not in participant_ids:
            return

        p_target = self.pool.get_participant("P_TARGET")
        if not p_target:
            return

        p_age = ""
        if p_target.date_of_birth:
            p_age = str(
                event_date.year - p_target.date_of_birth.year
                - (1 if (event_date.month, event_date.day) <
                   (p_target.date_of_birth.month, p_target.date_of_birth.day) else 0)
            )
        recent_history = self._format_recent_interactions(p_target, limit=3)

        # Build event context
        event_context = f"## Event Summary\n{event_summary}"
        if high_res_summary:
            event_context += f"\n\n## High-Resolution Key Event (Most Important)\n{high_res_summary}"
        if lr_time_period:
            event_context += f"\n\nLow-resolution time range: {lr_time_period}"
        if hr_time_period:
            event_context += f"\nHigh-resolution time range: {hr_time_period}"

        # Include only P_TARGET's post-scene memory as context (not side-chars)
        scene_memory_section = ""
        if post_scene_memories:
            target_mem = next(
                (m for m in post_scene_memories if m.get("participant_id") == "P_TARGET"),
                None,
            )
            if target_mem and target_mem.get("summary"):
                scene_memory_section = (
                    f"\n\n## Your Post-Scene Memory (Reference)\n"
                    f"{target_mem['summary']}"
                )

        # Build memory instruction based on event type
        if is_habitual:
            memory_instruction = (
                "1. memory: First-person autobiographical memory of this HABITUAL/RECURRING activity (1-2 sentences). "
                "Describe what this routine feels like from the inside — the sensory details, "
                "the emotional texture of repetition. Use habitual-aspect language "
                "(e.g., 'Every morning I would...', 'I remember the routine of...').\n"
            )
        else:
            memory_instruction = (
                "1. memory: First-person autobiographical memory of the event (1-2 sentences). "
                "This becomes the General-Event Memory (g_j) in the protagonist's memory base.\n"
            )

        system_prompt = (
            f"You are generating a post-event reflection for {target_name} in a life simulation.\n"
            "Generate two items:\n"
            f"{memory_instruction}"
            "2. updated_persona: Updated current persona description after the event\n\n"
            "CRITICAL MEMORY RULES:\n"
            "- memory MUST be written in first person — always start with 'I'\n"
            "- NEVER use the character's name or third-person pronouns (he/she/they) as subject\n"
            "- ✓ Correct: 'I finally got my diploma today; what a relief.'\n"
            "- ✗ Wrong: 'Tyler Johnson graduates from high school.' (third person)\n\n"
            "CRITICAL ID RULES: participant_id MUST be 'P_TARGET'."
        )

        user_prompt = f"""{event_context}
{scene_memory_section}

## Protagonist: {p_target.participant_id}: {p_target.persona_name_text}
- **participant_id (use this exact string)**: P_TARGET
- Current age: approximately {p_age}
- Current persona: {p_target.current_persona_brief_text}
- Target persona: {p_target.target_persona_brief_text}
- Recent interactions: {recent_history}

## Task
Generate for P_TARGET:
1. **memory**: First-person autobiographical memory (1-2 sentences). Start with 'I'.
2. **updated_persona**: Updated current persona description reflecting the event's impact.
"""

        try:
            result: BatchReflectionOutput = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=BatchReflectionOutput,
                system_prompt=system_prompt,
                temperature=0.0,
                task_type="uper_reflection",
            )

            for reflection in result.reflections:
                pid = reflection.participant_id
                # Normalise: accept name as fallback
                if pid not in self.pool._participants:
                    if p_target.persona_name_text.lower() in pid.lower() or pid.lower() in p_target.persona_name_text.lower():
                        pid = "P_TARGET"

                try:
                    self._upsert_interaction_history(
                        participant_id=pid,
                        event_id=event_id,
                        date_str=event_date.isoformat(),
                        summary=reflection.memory,
                        time_period=hr_time_period or lr_time_period or event_date.isoformat(),
                    )
                except KeyError:
                    logger.error(f"  UPER: P_TARGET '{pid}' not found after resolution; skipping")

                participant = self.pool.get_participant(pid)
                if participant and reflection.updated_persona:
                    participant.current_persona_brief_text = reflection.updated_persona
                    self.pool.mark_dirty(pid)

                # Write first-person memory to the EventRecord in memory_base
                if pid == "P_TARGET" and reflection.memory:
                    event_record = self.memory.get_event(event_id)
                    if event_record:
                        event_record.first_person_memory = reflection.memory
                        logger.debug(
                            f"  UPER: Written first_person_memory for {event_id}: "
                            f"{reflection.memory[:50]}..."
                        )

            self._save_pool()
            self._save_memory()
            logger.info(f"  UPER: P_TARGET reflected for event on {event_date.isoformat()}")

        except Exception:
            logger.exception("  UPER: P_TARGET reflection failed; writing event_summary as fallback")
            try:
                self._upsert_interaction_history(
                    participant_id="P_TARGET",
                    event_id=event_id,
                    date_str=event_date.isoformat(),
                    summary=event_summary,
                    time_period=hr_time_period or lr_time_period or event_date.isoformat(),
                )
            except KeyError:
                pass
            # Fallback: write a simple first_person_memory for habitual events
            if is_habitual:
                event_record = self.memory.get_event(event_id)
                if event_record and not event_record.first_person_memory:
                    title_part = event_summary.split("[")[0].strip() if "[" in event_summary else event_summary.split(":")[0].strip()
                    event_record.first_person_memory = f"I remember the routine of {title_part.lower()}."
                    self._save_memory()
            self._save_pool()

    # ================================================================
    # Post-Event: Interaction History Generation
    # ================================================================

    async def _generate_interaction_histories(
        self,
        event_id: str,
        event_summary: str,
        participant_ids: List[str],
        event_date: str,
    ) -> List[InteractionHistoryEntry]:
        """Generate first-person interaction history entries for all participants."""
        target_name = self.persona.get("persona_name_text", "protagonist")

        participant_details = []
        for pid in participant_ids:
            p = self.pool.get_participant(pid)
            if p:
                participant_details.append(
                    f"- {p.participant_id}: {p.persona_name_text} "
                    f"({p.relationship_towards_the_main_character})"
                )
        participants_text = "\n".join(participant_details)

        system_prompt = (
            "You are a memory generator for life simulation. "
            "Generate a FIRST-PERSON perspective interaction record summary for each participant about this event. "
            "Each summary MUST start with 'I' and describe the event from that participant's own perspective. "
            "NEVER use third-person pronouns (he/she/they) or the character's name as the subject. "
            "Each summary should be concise (1-2 sentences), colloquial, and personal."
        )

        user_prompt = f"""## Event Information
- Event date: {event_date}
- Event summary: {event_summary}
- Target character: {target_name} (P_TARGET)

## Participants
{participants_text}

## Task
For each participant, generate one FIRST-PERSON interaction memory summary.

Rules:
- MUST start with "I" — this is a personal memory, not a biography
- Use colloquial, spoken language
- ✓ Correct: "I finally got my diploma today; what a relief to be done with school."
- ✗ Wrong: "Tyler Johnson graduates from high school." (third person)
- ✗ Wrong: "He/She attended the graduation ceremony." (third person pronoun)

Examples:
- P_TARGET: "Today I discussed my future academic plans with my father; he suggested I consider a PhD."
- P_001: "Today I talked with my son about his future and suggested he consider continuing his studies."

Generate entries for all participants.
"""

        result: InteractionHistoryBatch = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=InteractionHistoryBatch,
            system_prompt=system_prompt,
            task_type="interaction_history",
            temperature=0.0,  # Summary/recap
        )
        return result.entries

    # ================================================================
    # Post-Event: Persona Dynamic Update
    # ================================================================

    async def _post_event_update_personas(
        self,
        participant_ids: List[str],
        event_summary: str,
        event_date: date,
    ) -> None:
        """
        After each event, update all participants' current_persona_brief_text
        to reflect the event's impact on their character development.

        Update principles:
          1) Maintain consistency with target persona
          2) Accurately reflect current time state evolution
        """
        target_name = self.persona.get("persona_name_text", "protagonist")

        for pid in participant_ids:
            participant = self.pool.get_participant(pid)
            if not participant:
                continue

            # Calculate current age
            p_age = ""
            if participant.date_of_birth:
                p_age = str(
                    event_date.year - participant.date_of_birth.year
                    - (1 if (event_date.month, event_date.day) <
                       (participant.date_of_birth.month, participant.date_of_birth.day) else 0)
                )

            system_prompt = (
                "You are a character profile updater for life simulation. "
                "Update the character's current persona description based on the event that just occurred. "
                "The updated description should:\n"
                "1. Maintain consistent development direction toward the target persona\n"
                "2. Accurately reflect the character's state evolution at the current moment\n"
                "3. Incorporate the event's impact on the character\n\n"
                "IMPORTANT — Field semantics:\n"
                "- 'current_persona_brief' is a LIVING description tied to the current simulation date. "
                "The age written here must always match the character's actual age at that date "
                "(computed from date_of_birth), NOT the age mentioned in the target persona.\n"
                "- 'target_persona_brief' is the user-supplied end-state description for directional "
                "reference only. It may contain approximate or characterization ages — do NOT copy "
                "its age literally into the updated current persona.\n"
                "- Always use the 'Current age' value provided in the prompt as the ground-truth age."
            )

            user_prompt = f"""## Current Time
- Date: {event_date.isoformat()}
{self._get_tc_prompt_for_event(event_date.isoformat())}
## Character Information
- Name: {participant.persona_name_text} ({pid})
- Relationship to {target_name}: {participant.relationship_towards_the_main_character}
- Current age: approximately {p_age}  ← USE THIS EXACT AGE in the updated persona description

## Persona Descriptions
- Initial persona: {participant.initial_persona_brief_text}
- Current persona (before update): {participant.current_persona_brief_text}
- Target persona (for direction reference only): {participant.target_persona_brief_text}

## Event That Just Occurred
{event_summary}

## Recent Interaction History
{self._format_recent_interactions(participant, limit=5)}

## Task
Generate the updated updated_current_persona_brief_text:
1. One paragraph containing current age (MUST be ~{p_age}, do NOT use the target persona's age),
   career/academic status, personality traits, relationship status with protagonist
2. Reflect the event's impact on the character (if any)
3. Maintain consistent development toward the target persona
4. If the event has little impact on this character, minor adjustments or no change are acceptable
"""

            try:
                result: PostEventPersonaUpdateOutput = await self.llm.generate_structured(
                    prompt=user_prompt,
                    response_model=PostEventPersonaUpdateOutput,
                    system_prompt=system_prompt,
                    task_type="persona_update",
                    temperature=0.0,  # Summary/recap
                )
                participant.current_persona_brief_text = result.updated_current_persona_brief_text
                logger.debug(
                    f"  Post-event persona updated: {pid} ({participant.persona_name_text})"
                )
            except Exception:
                logger.exception(
                    f"  Post-event persona update failed for {pid}; keeping existing persona"
                )

        # Persist pool after all participants' personas are updated
        self._save_pool()
        logger.info(f"  Persona pool persisted after post-event updates")

    # ================================================================
    # Core: Process a single time unit (Multi-Step Pipeline)
    # ================================================================

    async def _process_time_unit(
        self,
        period: Dict[str, Any],
        unit_start: date,
        unit_end: date,
        unit_label: str,
        density: str,
        detail_remaining: int = 999,
        outline_remaining: int = 999,
        detail_budget: int = 999,
        outline_budget: int = 999,
        total_units: int = 0,
        current_unit_index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single time unit using the multi-step pipeline:
          Step 1: Generate event framework
          Step 2: Select participants
          Step 3: Validate & update personas
          Step 4: Assess participant sufficiency
          Step 5: Generate new participants if needed
          Post:   Interaction histories, memory update, persona updates

        Returns:
            A dict with the high-res event data if one was generated,
            otherwise None.
        """
        period_id = period.get("period_id", "LP?")
        event_id = self._next_event_id(period_id)

        # Track current event date for TC injection in sub-steps (e.g., Step 5)
        self._last_event_date_str = unit_start.isoformat()

        with self._track_perf(
            "event.time_unit",
            period_id=period_id,
            unit_label=unit_label,
            density=density,
            event_id=event_id,
            resolution_level="low",
        ):
            logger.info(
                f"Processing {unit_label}: {unit_start} ~ {unit_end} "
                f"(density={density}, event_id={event_id})"
            )

            logger.info(f"  Step 1a: Generating low-resolution event framework...")
            event_framework = await self._step1a_generate_low_res_framework(
                period=period,
                unit_start=unit_start,
                unit_end=unit_end,
                unit_label=unit_label,
                density=density,
                detail_remaining=detail_remaining,
                outline_remaining=outline_remaining,
                detail_budget=detail_budget,
                outline_budget=outline_budget,
                total_units=total_units,
                current_unit_index=current_unit_index,
            )

            # ── v8: has_key_event is always "no" after Step 1a (field in model with default) ──
            # Key event determination is now done by Step 1b at segment level.
            event_framework.has_key_event = "no"
            logger.info(
                f"  Step 1a complete: type={event_framework.event_type}, "
                f"theme={event_framework.event_theme}, "
                f"key_event={event_framework.has_key_event}"
            )
            high_res_outline: Optional[HighResOutlineOutput] = None
            milestone_active = False  # v6: milestone handling moved to segment level
            milestone_skel = None

            # ── Step 3: Select participants from pool + create if missing ──
            logger.info(f"  Step 3: Selecting participants from pool...")

            pool_result = await self._step2_unified_participant_pool(
                event_framework=event_framework,
                high_res_outline=high_res_outline,
                period_id=period_id,
                unit_start=unit_start,
                unit_end=unit_end,
            )

            # Extract selected IDs from unified pool result
            selected_ids = [c.participant_id for c in pool_result.selected_participants]
            if "P_TARGET" not in selected_ids:
                selected_ids.insert(0, "P_TARGET")
            selected_ids = [
                pid for pid in selected_ids
                if self.pool.get_participant(pid) is not None
            ]
            if "P_TARGET" not in selected_ids:
                selected_ids.insert(0, "P_TARGET")

            logger.info(
                f"  Step 3 (select) complete: selected {len(selected_ids)} participants: "
                f"{selected_ids}, sufficient={pool_result.is_sufficient}"
            )

            # Conditional: Create new participants if unified pool says insufficient
            new_ids = []
            if not pool_result.is_sufficient and pool_result.new_participant_requirements:
                # SNAP fallback: search full pool before creating new participants
                fallback_ids = self._snap_fallback_search(
                    requirements=pool_result.new_participant_requirements,
                    excluded_ids=set(selected_ids),
                )
                if fallback_ids:
                    logger.info(
                        f"  Step 3 (SNAP fallback): Found {len(fallback_ids)} "
                        f"matching participants in pool: {fallback_ids}"
                    )
                    selected_ids.extend(fallback_ids)
                    # Re-evaluate: reduce requirements by number of fallback matches
                    remaining_requirements = pool_result.new_participant_requirements[len(fallback_ids):]
                else:
                    remaining_requirements = list(pool_result.new_participant_requirements)

                if remaining_requirements:
                    logger.info(
                        f"  Step 3 (create): Generating "
                        f"{len(remaining_requirements)} new participants..."
                    )
                    descriptions = [
                        f"Role type: {req.role_type}, "
                        f"Relationship: {req.relationship_to_main_character}, "
                        f"Reason: {req.reason_needed}"
                        for req in remaining_requirements
                    ]
                    new_ids = await self._step5_generate_new_participants(
                        descriptions=descriptions,
                        appear_period=period_id,
                        event_summary=event_framework.summary,
                        current_event_date=unit_start.isoformat(),
                    )
                    logger.info(f"  Step 3 (create) complete: new participants: {new_ids}")
            else:
                logger.info(f"  Participants sufficient, skipping creation")

            all_participant_ids = list(dict.fromkeys(selected_ids + new_ids))
            if "P_TARGET" not in all_participant_ids:
                all_participant_ids.insert(0, "P_TARGET")

            # ── Step 4: Validate all selected participants' personas ──
            logger.info(
                f"  Step 4: Validating {len(all_participant_ids)} participants' personas..."
            )
            await self._step3_batch_validate_personas(
                participant_ids=all_participant_ids,
                event_framework=event_framework,
                high_res_outline=high_res_outline,
                current_date=unit_start,
            )

            high_res_result = None
            pending_hr_event = None
            lr_event_id = event_id
            lr_time_period = _format_time_period(
                unit_start, "early morning",
                unit_end, "night",
            )

            lr_event_data = {
                "event_id": lr_event_id,
                "resolution_level": "low",
                "time_period": lr_time_period,
                "summary": event_framework.summary,
                "initial_summary": event_framework.summary,
                "refined_summary": "",
                "summary_stage": (
                    "initial"
                    if event_framework.has_key_event == "key_event_detail"
                    else "final_same_as_initial"
                ),
                "participants": all_participant_ids,
                "languages_in_use": ["eng"],
                "interaction_details": [],
                "period_id": period_id,
                "event_type": getattr(event_framework, 'event_type', ''),
                "linked_event_id": "",
                "source_event_id": lr_event_id,
                "storage_status": (
                    EVENT_STORAGE_STATUS_GENERAL_EVENT_READY
                    if event_framework.has_key_event == "key_event_detail"
                    else EVENT_STORAGE_STATUS_FINAL
                ),
                "run_id": self._run_id,
                "version": 1,
                "use_llm_summary": False,
            }

            pending_lr_event = dict(lr_event_data)
            await self.memory.upsert_low_res_event(
                event_id=lr_event_data["event_id"],
                time_period=lr_event_data["time_period"],
                summary=lr_event_data["summary"],
                participants=lr_event_data.get("participants"),
                languages_in_use=["eng"],
                interaction_details=lr_event_data.get("interaction_details"),
                period_id=lr_event_data.get("period_id"),
                event_type=lr_event_data.get("event_type", ""),
                linked_event_id=lr_event_data.get("linked_event_id", ""),
                source_event_id=lr_event_data.get("source_event_id", lr_event_id),
                storage_status=lr_event_data.get("storage_status", EVENT_STORAGE_STATUS_FINAL),
                run_id=lr_event_data.get("run_id", self._run_id),
                version=lr_event_data.get("version"),
                use_llm_summary=lr_event_data.get("use_llm_summary", False),
                initial_summary=lr_event_data.get("initial_summary", lr_event_data["summary"]),
                refined_summary=lr_event_data.get("refined_summary", ""),
                summary_stage=lr_event_data.get("summary_stage", "initial"),
            )
            self._save_memory()
            if event_framework.has_key_event == "key_event_detail":
                logger.info(
                    f"  Low-res event {lr_event_id} stored in memory_base "
                    f"and queued for P3 refinement"
                )
            elif event_framework.has_key_event == "key_event_general":
                logger.info(
                    f"  Low-res event {lr_event_id} stored in memory_base "
                    f"(outline path, no P3 refinement)"
                )

            lr_event_date = unit_start.isoformat()

            # Module 3: Use UPER for combined interaction history + persona update
            # For LR-only events, UPER handles everything in one call
            if high_res_outline is None:
                await self._uper_reflect_batch(
                    participant_ids=all_participant_ids,
                    event_summary=event_framework.summary,
                    event_date=unit_start,
                    event_id=lr_event_id,
                    lr_time_period=_format_time_period(
                        unit_start, "early morning", unit_end, "night",
                    ),
                )
            else:
                # When HR event exists, generate LR interaction histories
                # but skip persona update (will be done after HR event via UPER)
                lr_histories = await self._generate_interaction_histories(
                    event_id=lr_event_id,
                    event_summary=event_framework.summary,
                    participant_ids=all_participant_ids,
                    event_date=lr_event_date,
                )
                for entry in lr_histories:
                    try:
                        self.pool.update_interaction_history(
                            participant_id=entry.participant_id,
                            event_id=lr_event_id,
                            date=lr_event_date,
                            summary=entry.summary,
                        )
                    except KeyError:
                        logger.warning(
                            f"  Participant {entry.participant_id} not found; "
                            f"skipping interaction history update"
                        )
                self._save_pool()

            logger.info(f"  Low-res summary event {lr_event_id} stored")

            # ── Outline-only path: write outline event + UPER, skip P3 ──
            if high_res_outline is not None and event_framework.has_key_event == "key_event_general":
                logger.info(
                    f"  Outline-only path: finalizing outline event "
                    f"(parent_low_res={lr_event_id})..."
                )
                # Mark LR event as final (no P3 refinement will follow)
                if pending_lr_event is not None:
                    pending_lr_event["summary_stage"] = "final_same_as_initial"
                    pending_lr_event["storage_status"] = EVENT_STORAGE_STATUS_FINAL

                await self._finalize_outline_only(
                    event_framework=event_framework,
                    high_res_outline=high_res_outline,
                    selected_ids=all_participant_ids,
                    period_id=period_id,
                    event_id=lr_event_id,
                    unit_start=unit_start,
                    unit_end=unit_end,
                )

                self._save_memory()
                logger.info(f"  Memory base persisted after outline event {event_id}")

                # Return bundle WITHOUT high_res_event (no P3 simulation needed)
                return {
                    "high_res_event": None,
                    "pending_lr_event": pending_lr_event,
                    "pending_hr_event": None,
                    "runtime_meta": None,
                    "key_event_type": event_framework.has_key_event,
                }

            # ── Detail path: create HR event for P3 simulation ──
            if high_res_outline is not None and event_framework.has_key_event == "key_event_detail":
                hr_event_id = self._next_event_id(period_id)
                logger.info(
                    f"  Storing high-res outline {hr_event_id} "
                    f"(parent_low_res={lr_event_id})..."
                )

                key_participants = all_participant_ids

                hr_start_date = high_res_outline.key_event_start_date or unit_start.isoformat()
                hr_start_seg = high_res_outline.key_event_start_time or "morning"
                hr_end_date = high_res_outline.key_event_end_date or unit_end.isoformat()
                hr_end_seg = high_res_outline.key_event_end_time or "afternoon"

                _SEGMENT_DEFAULTS = {
                    "early morning": ("05:30:00", "06:30:00"),
                    "morning":       ("09:00:00", "11:00:00"),
                    "afternoon":     ("14:00:00", "16:00:00"),
                    "night":         ("19:00:00", "21:00:00"),
                }
                default_start, _ = _SEGMENT_DEFAULTS.get(hr_start_seg, ("09:00:00", "11:00:00"))
                _, default_end = _SEGMENT_DEFAULTS.get(hr_end_seg, ("14:00:00", "16:00:00"))

                hr_precise_start = high_res_outline.key_event_precise_start_time or default_start
                hr_precise_end = high_res_outline.key_event_precise_end_time or default_end

                _TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
                if not _TIME_RE.match(hr_precise_start):
                    logger.warning(
                        f"  Invalid precise_start_time '{hr_precise_start}', "
                        f"falling back to default '{default_start}'"
                    )
                    hr_precise_start = default_start
                if not _TIME_RE.match(hr_precise_end):
                    logger.warning(
                        f"  Invalid precise_end_time '{hr_precise_end}', "
                        f"falling back to default '{default_end}'"
                    )
                    hr_precise_end = default_end

                hr_time_period = (
                    f"from {hr_start_date} {hr_precise_start} "
                    f"to {hr_end_date} {hr_precise_end}"
                )

                logger.info(
                    f"  High-res precise time: {hr_start_date} {hr_precise_start} → "
                    f"{hr_end_date} {hr_precise_end} (segment: {hr_start_seg}→{hr_end_seg})"
                )

                pending_hr_event = {
                    "event_id": hr_event_id,
                    "resolution_level": "high",
                    "time_period": hr_time_period,
                    "summary": high_res_outline.key_event_summary,
                    "initial_summary": high_res_outline.key_event_summary,
                    "refined_summary": "",
                    "summary_stage": "initial",
                    "participants": key_participants,
                    "languages_in_use": ["eng"],
                    "interaction_details": [],
                    "period_id": period_id,
                    "event_type": getattr(event_framework, 'event_type', 'milestone'),
                    "linked_event_id": lr_event_id,
                    "source_event_id": hr_event_id,
                    "storage_status": EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
                    "run_id": self._run_id,
                    "version": 1,
                    "use_llm_summary": False,
                    "linked_low_res_summary": event_framework.summary,
                }
                await self.memory.upsert_high_res_outline_event(
                    event_id=hr_event_id,
                    time_period=hr_time_period,
                    summary=high_res_outline.key_event_summary,
                    participants=key_participants,
                    languages_in_use=["eng"],
                    interaction_details=[],
                    period_id=period_id,
                    event_type=getattr(event_framework, 'event_type', 'milestone'),
                    linked_event_id=lr_event_id,
                    source_event_id=hr_event_id,
                    storage_status=EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
                    run_id=self._run_id,
                    version=1,
                    use_llm_summary=False,
                    initial_summary=high_res_outline.key_event_summary,
                    refined_summary="",
                    summary_stage="initial",
                )
                self._save_memory()
                logger.info(
                    f"  High-res outline {hr_event_id} stored in memory_base "
                    f"and queued for P3 refinement"
                )

                hr_event_date = hr_start_date
                # v4: HR event UPER is deferred to finalize_time_unit()
                # after P3 simulation completes. Only record LR interaction
                # history here (no persona update).
                lr_histories = await self._generate_interaction_histories(
                    event_id=lr_event_id,
                    event_summary=event_framework.summary,
                    participant_ids=key_participants,
                    event_date=lr_event_date,
                )
                for entry in lr_histories:
                    try:
                        self.pool.update_interaction_history(
                            participant_id=entry.participant_id,
                            event_id=lr_event_id,
                            date=lr_event_date,
                            summary=entry.summary,
                        )
                    except KeyError:
                        logger.warning(
                            f"  Participant {entry.participant_id} not found; "
                            f"skipping LR interaction history update for HR event"
                        )
                self._save_pool()

                high_res_result = {
                    "event_id": hr_event_id,
                    "parent_low_res_event_id": lr_event_id,
                    "linked_event_id": lr_event_id,
                    "source_event_id": hr_event_id,
                    "run_id": self._run_id,
                    "period_id": period_id,
                    "resolution_level": "high",
                    "value_for_target": high_res_outline.key_event_value_for_target,
                    "value_for_period": event_framework.value_for_period,
                    "summary": high_res_outline.key_event_summary,
                    "low_res_context": event_framework.summary,
                    "motivation": high_res_outline.key_event_motivation,
                    "duration": {
                        "start_date": hr_start_date,
                        "start_time": hr_start_seg,
                        "precise_start_time": hr_precise_start,
                        "end_date": hr_end_date,
                        "end_time": hr_end_seg,
                        "precise_end_time": hr_precise_end,
                    },
                    "outcome": high_res_outline.key_event_outcome,
                    "participants": key_participants,
                    "original_low_res_participants": key_participants,
                    "languages_in_use": ["eng"],
                    "setting": high_res_outline.key_event_setting,
                    "turning_point": high_res_outline.key_event_turning_point,
                    "linked_low_res_summary": event_framework.summary,
                    # Simulation mode flags (read by high_res_event_simulator.py)
                    "use_parallel_protagonist_mode": True,  # Fastest: skeleton + parallel fill
                    "use_m3_script_mode": True,             # Fallback: M3 one-shot mode
                }                # Attach milestone skeleton for P3 mandatory beats
                if milestone_active and milestone_skel:
                    high_res_result["milestone_skeleton"] = milestone_skel

                if pending_lr_event is not None:
                    pending_lr_event["linked_event_id"] = hr_event_id
                logger.info(f"  High-res outline {hr_event_id} saved successfully")

                # v4: UPER deferred to finalize_time_unit() after P3 simulation

            self._save_memory()
            logger.info(f"  Memory base persisted after event {event_id}")

            runtime_meta = None
            if high_res_outline is not None and event_framework.has_key_event == "key_event_detail":
                runtime_meta = {
                    "original_low_res_participants": key_participants,
                }

            return {
                "high_res_event": high_res_result,
                "pending_lr_event": pending_lr_event,
                "pending_hr_event": pending_hr_event if (high_res_outline is not None and event_framework.has_key_event == "key_event_detail") else None,
                "runtime_meta": runtime_meta,
                "key_event_type": event_framework.has_key_event,
                # v6: additional fields for segment-level outline generation
                "lr_summary": event_framework.summary,
                "unit_start": unit_start,
                "unit_end": unit_end,
                "all_participant_ids": all_participant_ids,
                "event_framework": event_framework,
            }

    async def _finalize_outline_only(
        self,
        event_framework,
        high_res_outline,
        selected_ids: List[str],
        period_id: str,
        event_id: str,
        unit_start: date,
        unit_end: date,
    ) -> None:
        """Finalize an outline-only key event (no P3 simulation).

        This method handles the outline path which:
        1. Writes the event to memory_base with resolution_level="medium"
        2. Calls UPER to update participant personas

        NOTE: Participant selection (Step 2-5) must be completed BEFORE
        calling this method. The selected_ids parameter should already
        contain all participants including any newly created ones.
        """
        # ── Step A: Write outline event to memory ──
        time_period_str = _format_time_period(
            start=date.fromisoformat(high_res_outline.key_event_start_date),
            start_seg=high_res_outline.key_event_start_time,
            end=date.fromisoformat(high_res_outline.key_event_end_date),
            end_seg=high_res_outline.key_event_end_time,
        )

        outline_event_id = f"{event_id}_outline"
        self.memory.add_event(
            event_id=outline_event_id,
            period_id=period_id,
            resolution_level="medium",
            time_period=time_period_str,
            summary=(
                f"{high_res_outline.key_event_summary} "
                f"[Motivation: {high_res_outline.key_event_motivation}] "
                f"[Outcome: {high_res_outline.key_event_outcome}]"
            ),
            participants=selected_ids,
            languages_in_use=["eng"],
        )

        logger.info(
            f"  Outline-only event finalized: {outline_event_id} "
            f"(participants={selected_ids})"
        )

        # ── Step B: UPER — Update participant personas ──
        # Even though we skip P3 simulation, the outline event still
        # affects participant relationships and states. We must call
        # the same UPER (Unified Post-Event Reflection) as the detail path.
        try:
            await self._uper_reflect_batch(
                participant_ids=selected_ids,
                event_summary=high_res_outline.key_event_summary,
                event_date=unit_start,
                event_id=outline_event_id,
                lr_time_period=time_period_str,
            )
            logger.info(f"  UPER completed for outline event {outline_event_id}")
        except Exception as e:
            logger.error(
                f"  UPER failed for outline event {outline_event_id}: {e}. "
                f"Participant personas will not be updated for this event."
            )

    async def finalize_time_unit(
        self,
        bundle: Dict[str, Any],
        simulated_result: Optional[Dict[str, Any]] = None,
        final_participants: Optional[List[str]] = None,
        mark_high_res_failed: bool = False,
        is_last_hr_for_lr: bool = True,  # kept for API compat; no longer used
    ) -> None:
        """Finalize a prepared time unit: write memory records and status flags.

        UPER (interaction histories) is now handled externally by _run_period_uper()
        after all detail simulations in the period complete. This method only:
          - upserts LR event record
          - upserts HR outline / detail records
          - marks storage status flags
          - does NOT call _uper_reflect_batch or update low-res refined_summary
        """
        if not bundle:
            return

        pending_lr = bundle.get("pending_lr_event")
        pending_hr = bundle.get("pending_hr_event")
        participant_ids = list(dict.fromkeys(final_participants or []))

        if pending_hr:
            self.memory.mark_event_storage_status(
                pending_hr["event_id"],
                EVENT_STORAGE_STATUS_SIMULATING if simulated_result is not None else EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
            )

        if pending_lr:
            lr_payload = dict(pending_lr)
            if participant_ids:
                lr_payload["participants"] = participant_ids
            linked_hr_id = (pending_hr or {}).get("event_id", "") or lr_payload.get("linked_event_id", "")
            if not lr_payload.get("refined_summary"):
                lr_payload["refined_summary"] = lr_payload.get(
                    "initial_summary", lr_payload.get("summary", "")
                )
            await self.memory.upsert_low_res_event(
                event_id=lr_payload["event_id"],
                time_period=lr_payload["time_period"],
                summary=lr_payload["summary"],
                participants=lr_payload.get("participants"),
                languages_in_use=["eng"],
                interaction_details=lr_payload.get("interaction_details"),
                period_id=lr_payload.get("period_id"),
                event_type=lr_payload.get("event_type", ""),
                linked_event_id=linked_hr_id,
                source_event_id=lr_payload.get("source_event_id", lr_payload["event_id"]),
                storage_status=(
                    EVENT_STORAGE_STATUS_FINAL
                    if not pending_hr or simulated_result is not None
                    else EVENT_STORAGE_STATUS_GENERAL_EVENT_READY
                ),
                run_id=lr_payload.get("run_id", self._run_id),
                version=lr_payload.get("version"),
                use_llm_summary=lr_payload.get("use_llm_summary", False),
                initial_summary=lr_payload.get("initial_summary", lr_payload.get("summary", "")),
                refined_summary=lr_payload.get("refined_summary", ""),
                summary_stage=lr_payload.get("summary_stage", "initial"),
            )
            logger.info(f"  Time-unit LR event finalized: {lr_payload['event_id']}")
        else:
            lr_payload = {}

        if pending_hr:
            hr_payload = dict(pending_hr)
            if participant_ids:
                hr_payload["participants"] = participant_ids
            linked_lr_id = (pending_lr or {}).get("event_id", "") or hr_payload.get("linked_event_id", "")

            if simulated_result is not None:
                # Detail simulation succeeded → write FINAL directly, skip outline record.
                # upsert_high_res_event_from_scene_result will upsert with the same
                # event_id, overwriting any pre-existing outline record if present.
                final_hr_record = await self.memory.upsert_high_res_event_from_scene_result(
                    event_id=hr_payload["event_id"],
                    scene_result=simulated_result,
                    period_id=hr_payload.get("period_id"),
                    event_type=hr_payload.get("event_type", ""),
                    linked_event_id=linked_lr_id,
                    run_id=hr_payload.get("run_id", self._run_id),
                    source_event_id=hr_payload.get("source_event_id", hr_payload["event_id"]),
                )
                self.memory.extract_legacy_memories_from_scene_result(
                    simulated_result,
                    hr_payload["event_id"],
                )
                self.memory.mark_event_storage_status(
                    hr_payload["event_id"],
                    EVENT_STORAGE_STATUS_FINAL,
                )
                logger.info(f"  Final HR scene result finalized: {hr_payload['event_id']}")

            elif not mark_high_res_failed:
                # No simulation result and not explicitly failed → outline-only event
                # (e.g. outline budget event that won't become a detail). Write outline.
                await self.memory.upsert_high_res_outline_event(
                    event_id=hr_payload["event_id"],
                    time_period=hr_payload["time_period"],
                    summary=hr_payload["summary"],
                    participants=hr_payload.get("participants"),
                    languages_in_use=["eng"],
                    interaction_details=hr_payload.get("interaction_details"),
                    period_id=hr_payload.get("period_id"),
                    event_type=hr_payload.get("event_type", ""),
                    linked_event_id=linked_lr_id,
                    source_event_id=hr_payload.get("source_event_id", hr_payload["event_id"]),
                    storage_status=EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
                    run_id=hr_payload.get("run_id", self._run_id),
                    version=hr_payload.get("version"),
                    use_llm_summary=hr_payload.get("use_llm_summary", False),
                    initial_summary=hr_payload.get("initial_summary", hr_payload.get("summary", "")),
                    refined_summary=hr_payload.get("refined_summary", ""),
                    summary_stage=hr_payload.get("summary_stage", "initial"),
                )
                logger.info(f"  HR outline-only record written: {hr_payload['event_id']}")

            else:
                # Detail simulation failed → do not write any HR record to memory.
                # The LR event (already written above) remains as the sole record
                # for this time unit.
                logger.warning(
                    f"  HR detail simulation failed for {hr_payload['event_id']}; "
                    f"skipping HR record write (LR event retained)."
                )

        self._save_memory()
        self._save_pool()

    async def flush_deferred_bundle(
        self,
        bundle: Dict[str, Any],
        simulated_result: Optional[Dict[str, Any]] = None,
        final_participants: Optional[List[str]] = None,
        mark_high_res_failed: bool = False,
    ) -> None:
        """Backward-compatible wrapper around finalize_time_unit()."""
        await self.finalize_time_unit(
            bundle,
            simulated_result=simulated_result,
            final_participants=final_participants,
            mark_high_res_failed=mark_high_res_failed,
        )

    async def run_high_res_for_time_unit(
        self,
        bundle: Dict[str, Any],
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
        max_turns: int,
        hr_events_dir: Optional[str] = None,
        skip_refinement: bool = False,
        _refinement_only: bool = False,
    ) -> Dict[str, Any]:
        """Run P3 immediately for a prepared time unit and return the final scene result.

        skip_refinement=True: skip refine_participants(), use bundle["pre_selected_participants"].
            Used when refinement was already run sequentially before parallel dispatch.
        _refinement_only=True: run only the refinement step, then return immediately.
            Used by the sequential pre-refinement loop (Phase 3a).
        """
        hr_event = bundle.get("high_res_event")
        if not hr_event:
            return {"sim_result": None, "final_participants": []}

        event_id = hr_event.get("event_id", "")
        period_id = hr_event.get("period_id", "")

        with self._track_perf(
            f"event.P3.total.{event_id}",
            event_id=event_id,
            period_id=period_id,
            resolution_level="high",
        ):
            simulator = HighResEventSimulator(
                llm_client=self.llm,
                persona_pool=self.pool.to_dict(),
                memory_base=self.memory.to_dict(),
                life_plan=life_plan,
                persona_config=persona_config,
                high_res_event=hr_event,
                retriever=self.memory.retriever,
                performance_tracker=self._performance_tracker,
                run_id=self._run_id,
                event_id=event_id,
                period_id=period_id,
                pre_selected_participants=bundle.get("pre_selected_participants"),
            )

            if skip_refinement:
                # Refinement was already run sequentially before parallel dispatch;
                # pre_selected_participants in bundle already holds the refined list.
                final_ids = list(dict.fromkeys(bundle.get("pre_selected_participants") or []))
                logger.info(f"[P3-inline] {event_id}: Skipping refinement (pre-refined); participants={final_ids}")
            else:
                logger.info(f"[P3-inline] {event_id}: Starting participant refinement...")
                refinement = await simulator.refine_participants()
                refined_ids = list(refinement.final_participants)

                for update in refinement.step4.persona_updates:
                    if not update.needs_update or not update.updated_brief:
                        continue
                    p = self.pool.get_participant(update.participant_id)
                    if not p:
                        logger.warning(
                            f"[P3-inline] Participant not found for persona update: {update.participant_id}"
                        )
                        continue
                    p.current_persona_brief_text = update.updated_brief
                    logger.info(
                        f"[P3-inline] Persona updated: {update.participant_id} ({update.update_reason})"
                    )
                self._save_pool()

                newly_created_ids = []
                if refinement.new_participant_suggestions:
                    logger.info(
                        f"[P3-inline] Creating {len(refinement.new_participant_suggestions)} new participants..."
                    )
                    for suggestion in refinement.new_participant_suggestions:
                        new_p = await self.pool.add_participant(
                            name=f"new_role_{suggestion.role_type}",
                            relationship=suggestion.relationship_to_main_character,
                            brief=suggestion.persona_requirements,
                            role=suggestion.role_type,
                            appear_period=hr_event.get("period_id") or period_id,
                            extend_profile=True,
                            persona_config=persona_config,
                            sim_end_date=self.simulation_end_date,
                        )
                        if new_p:
                            newly_created_ids.append(new_p.participant_id)
                            logger.info(
                                f"[P3-inline] New participant created: {new_p.participant_id} - {new_p.persona_name_text} ({suggestion.role_type})"
                            )
                    self._save_pool()

                final_ids = list(dict.fromkeys(refined_ids + newly_created_ids))

            hr_event["participants"] = final_ids
            bundle["high_res_event"] = hr_event
            if bundle.get("pending_hr_event"):
                bundle["pending_hr_event"]["participants"] = final_ids
            if bundle.get("pending_lr_event"):
                bundle["pending_lr_event"]["participants"] = final_ids
            logger.info(f"[P3-inline] Final participants for {event_id}: {final_ids}")

            # Pre-refinement-only mode: store result back and return immediately.
            # The caller (Phase 3a sequential loop) will then dispatch the actual
            # simulation in parallel with skip_refinement=True.
            if _refinement_only:
                bundle["pre_selected_participants"] = final_ids
                return {"sim_result": None, "final_participants": final_ids}

            simulator = HighResEventSimulator(
                llm_client=self.llm,
                persona_pool=self.pool.to_dict(),
                memory_base=self.memory.to_dict(),
                life_plan=life_plan,
                persona_config=persona_config,
                high_res_event=hr_event,
                retriever=self.memory.retriever,
                performance_tracker=self._performance_tracker,
                run_id=self._run_id,
                event_id=event_id,
                period_id=period_id,
                pre_selected_participants=final_ids,
            )

            with self._track_perf(
                "event.P3.run_simulation",
                event_id=event_id,
                period_id=period_id,
                resolution_level="high",
            ):
                sim_result = await simulator.run_simulation(max_turns=max_turns)

            if hr_events_dir:
                os.makedirs(hr_events_dir, exist_ok=True)
                sim_path = os.path.join(hr_events_dir, f"{event_id}_simulation.json")
                with open(sim_path, "w", encoding="utf-8") as f:
                    json.dump(sim_result, f, ensure_ascii=False, indent=2)
                logger.info(f"[P3-inline] Simulation saved to {sim_path}")

            return {
                "sim_result": sim_result,
                "final_participants": final_ids,
            }

    async def _build_low_res_refined_summary(
        self,
        low_res_payload: Dict[str, Any],
        high_res_record,
        simulated_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a low-resolution refined summary via LLM call.

        Replaces the old concatenation + hard truncation approach.
        Length is controlled implicitly via prompt (<=280 chars), no post-processing truncation.
        """
        low_res_initial = (low_res_payload or {}).get("initial_summary") or (low_res_payload or {}).get("summary", "")
        high_res_summary = self.memory.get_preferred_summary(high_res_record)
        event_context = (simulated_result or {}).get("event_context", {}) or {}
        outcome = event_context.get("outcome", "") or (simulated_result or {}).get("outcome", "") or getattr(high_res_record, "summary_diff_note", "")
        post_memories = (simulated_result or {}).get("post_scene_memories", []) or []

        memory_fragments = []
        scored_memories = [(mem, score_memory_fragment(mem or {})) for mem in post_memories]
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        for mem, _score in scored_memories:
            summary = (mem or {}).get("summary", "").strip()
            if summary:
                memory_fragments.append(summary)

        # If no extra info beyond initial, just return initial
        has_extra = high_res_summary or outcome or memory_fragments
        if not has_extra:
            return low_res_initial

        # Build context for LLM
        context_parts = []
        if low_res_initial:
            context_parts.append(f"Low-resolution event overview: {low_res_initial.strip()}")
        if high_res_summary and high_res_summary.strip() != low_res_initial.strip():
            context_parts.append(f"High-resolution event summary: {high_res_summary.strip()}")
        if outcome:
            context_parts.append(f"Event outcome: {outcome.strip()}")
        if memory_fragments:
            context_parts.append(f"Participant memories: {'; '.join(memory_fragments)}")

        context_text = "\n".join(context_parts)

        system_prompt = (
            "You are an event summary generator for a character simulation system. "
            "Merge a low-resolution time period overview with high-resolution event details "
            "into a coherent summary. "
            "You MUST respond in English, regardless of the language of the input overview."
        )

        user_prompt = (
            f"Generate a summary that blends the overall overview with key event details.\n\n"
            f"{context_text}\n\n"
            f"Requirements:\n"
            f"- Use the low-resolution overview as the main body (~70% of content), "
            f"supplemented by high-resolution event highlights (~30%)\n"
            f"- Maintain the completeness of the macro narrative; "
            f"high-resolution details serve only as highlights\n"
            f"- Maintain narrative completeness — no half-sentences or incomplete descriptions\n"
            f"- 2-4 sentences, ≤280 chars\n"
            f"- Use a colloquial narrative style matching the original overview's language\n"
            f"- Output the summary text directly, no prefixes or labels"
        )

        from pydantic import BaseModel, Field

        class _LRSummaryOutput(BaseModel):
            summary: str = Field(
                ...,
                description=(
                    "Merged event summary. 2-4 sentences, ≤280 chars. "
                    "Plain text only — no markdown, no labels, no prefixes."
                )
            )

        MAX_SUMMARY_RETRIES = 1000
        for attempt in range(MAX_SUMMARY_RETRIES):
            try:
                result: _LRSummaryOutput = await self.llm.generate_structured(
                    prompt=user_prompt,
                    response_model=_LRSummaryOutput,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.0,
                    task_type="lr_refined_summary",
                )
                if result.summary.strip():
                    return result.summary.strip()
            except Exception as e:
                logger.warning(
                    f"[Summary] LR attempt {attempt + 1}/{MAX_SUMMARY_RETRIES} failed: {e}"
                )
                if attempt < MAX_SUMMARY_RETRIES - 1:
                    import asyncio as _asyncio
                    await _asyncio.sleep(1.0 * (attempt + 1))
                    continue

        logger.warning("[Summary] LR all retries exhausted; falling back to initial summary")
        # Fallback: return initial summary (no truncation)
        return low_res_initial or high_res_summary or ""

    def validate_time_unit_commit(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the current time unit has been fully committed before moving on."""
        pending_lr = bundle.get("pending_lr_event") or {}
        pending_hr = bundle.get("pending_hr_event") or {}
        low_res_event_id = pending_lr.get("event_id", "")
        high_res_event_id = pending_hr.get("event_id", "") or None
        if not low_res_event_id:
            raise ValueError("Missing low-resolution event_id in time-unit bundle")

        report = self.memory.build_time_unit_commit_report(
            low_res_event_id,
            high_res_event_id,
        )
        # NOTE: Do NOT validate HR interaction history here.
        # UPER is now deferred to _run_period_uper() after all detail simulations
        # complete (parallel detail mode). Interaction history is not written yet
        # at this point, so validating it would always fail. The UPER step is
        # responsible for ensuring histories are written — validate there instead.
        if False and high_res_event_id:
            self._validate_pool_interactions(
                pending_hr.get("participants") or [],
                high_res_event_id,
            )

        low_res_record = self.memory.get_event(low_res_event_id)
        high_res_record = self.memory.get_event(high_res_event_id) if high_res_event_id else None
        if low_res_record is None:
            raise ValueError(f"Low-resolution event '{low_res_event_id}' missing after commit")
        if not low_res_record.initial_summary:
            raise ValueError(f"Low-resolution event '{low_res_event_id}' is missing initial_summary")

        if high_res_record is not None:
            if not high_res_record.initial_summary:
                raise ValueError(f"High-resolution event '{high_res_event_id}' is missing initial_summary")
            if high_res_record.storage_status == EVENT_STORAGE_STATUS_FINAL:
                if not high_res_record.refined_summary:
                    raise ValueError(f"Final high-resolution event '{high_res_event_id}' is missing refined_summary")
                # NOTE: Do NOT check low_res_record.refined_summary or summary_stage='refined' here.
                # low-res no longer gets a LLM-refined summary (requirement 2: no back-update to low-res).
                # The low-res summary stays at its initial value; that is the intended behavior.
        else:
            pass  # No HR event — LR-only, nothing further to validate.

        return report

    # ================================================================
    # Period-End: Life Summary & Period Summary Rewriting
    # ================================================================

    def _build_period_events_text(
        self,
        period_id: str,
        pending_events: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a merged textual view of committed and pending period events."""
        lines: List[str] = []
        seen_event_ids = set()
        period_events = self.memory.get_events_for_period(period_id)
        if period_events:
            for eid, rec in period_events.items():
                seen_event_ids.add(eid)
                lines.append(
                    f"- [{eid}] ({rec.resolution_level}) {rec.time_period}: {rec.summary}"
                )

        if pending_events:
            for idx, event in enumerate(pending_events, start=1):
                if not event:
                    continue
                event_id = event.get("event_id", f"pending_{idx}")
                if event_id in seen_event_ids:
                    continue
                resolution_level = event.get("resolution_level", "pending")
                time_period = event.get("time_period", "")
                summary = event.get("summary", "")
                lines.append(
                    f"- [{event_id}] ({resolution_level}, pending) {time_period}: {summary}"
                )

        return "\n".join(lines) if lines else "(No events recorded for this stage yet)"

    async def _generate_and_update_life_summary(
        self,
        period: Dict[str, Any],
        pending_events: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a current_life_summary after completing a life period
        and update the memory base.
        """
        period_id = period.get("period_id", "LP?")
        target_name = self.persona.get("persona_name_text", "protagonist")
        target_brief = self.persona.get("persona_brief_text", "")

        memory_ctx = self._build_memory_context()
        events_text = self._build_period_events_text(period_id, pending_events=pending_events)

        system_prompt = (
            "You are a memory summarizer for life simulation. "
            "Based on the target character's persona and existing memory bank information, "
            "generate a concise current life summary. "
            "The summary should be in the third person, covering the target character's current life state, "
            "major achievements, key relationships, and current situation up to this point."
        )

        end_date = period.get('period_date_range', {}).get('end_date', '')

        # Compute protagonist's exact age at the end of this period
        computed_age_line = ""
        try:
            from datetime import date as _date
            birth_date_str_p4 = self.persona.get("date_of_birth", "")
            if birth_date_str_p4 and end_date:
                _dob = _date.fromisoformat(birth_date_str_p4)
                _end = _date.fromisoformat(end_date)
                _age = _end.year - _dob.year - (
                    1 if (_end.month, _end.day) < (_dob.month, _dob.day) else 0
                )
                computed_age_line = f"- Protagonist's exact age at end of this stage: {_age} years old"
        except Exception:
            pass

        user_prompt = f"""## Target Character
- Name: {target_name}
- Bio: {target_brief}
{computed_age_line}

## Just-Completed Life Stage
- Stage ID: {period_id}
- Title: {period.get('title', '')}
- Dominant theme: {period.get('dominant_theme', '')}
- Time range: {period.get('period_date_range', {}).get('start_date', '')} to {end_date}
{self._get_tc_prompt_for_event(end_date, period) if end_date else ''}
## Stage Events
{events_text}

## Existing Memory Context
{memory_ctx}

## Task
Generate a current life summary (current_life_summary):
1. Describe the target character's life state up to this point in the third person
2. Cover major achievements, key relationships, personality development, and current situation
3. Length: 3-6 sentences
4. Connect naturally with existing memories, reflecting stage-based growth
"""

        result: CurrentLifeSummaryOutput = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=CurrentLifeSummaryOutput,
            system_prompt=system_prompt,
            task_type="life_summary",
            temperature=0.0,  # Summary/recap
        )

        self.memory.update_current_life_summary(result.current_life_summary)
        self._save_memory()
        logger.info(
            f"  current_life_summary updated after period {period_id}: "
            f"{result.current_life_summary[:80]}..."
        )
        return result.current_life_summary

    async def _rewrite_period_summary(
        self,
        period: Dict[str, Any],
        pending_events: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Rewrite the period_summary in memory_base based on:
          - The original period plan (developmental tasks, goals, themes)
          - The actual simulated events that occurred

        This integrates planned goals with actual execution outcomes.
        """
        period_id = period.get("period_id", "LP?")
        title = period.get("title", "")
        theme = period.get("dominant_theme", "")
        tasks = period.get("developmental_tasks", [])
        goals = period.get("stage_goals", [])

        events_text = self._build_period_events_text(period_id, pending_events=pending_events)

        system_prompt = (
            "You are a stage summary writer for life simulation. "
            "Based on the original stage plan and the actual events that occurred, rewrite the stage summary. "
            "The summary should integrate planned goals with actual execution outcomes."
        )

        user_prompt = f"""## Original Stage Plan
- Stage ID: {period_id}
- Title: {title}
- Dominant theme: {theme}
- Developmental tasks: {json.dumps(tasks, ensure_ascii=False)}
- Stage goals: {json.dumps(goals, ensure_ascii=False)}

## Actual Events That Occurred
{events_text}

## Task
Generate TWO versions of the period summary:

1. **period_summary** (third-person, 2-4 sentences):
   - Integrate the original planned goals with the actual events that occurred
   - Reflect the comparison between plan and actual execution
   - Describe in the third person

2. **period_summary_first_person** (first-person retrospective, 3-5 sentences):
   - Written as if the protagonist is recalling this era of their life
   - MUST use 'I' as subject throughout
   - Tone: reflective, personal, like someone looking back on a chapter of their life
   - Example style: "I remember those years as a time of intense growth. I was juggling..."
"""

        try:
            result: PeriodSummaryOutput = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=PeriodSummaryOutput,
                system_prompt=system_prompt,
                task_type="period_summary",
                temperature=0.0,  # Summary/recap
            )

            # Update the period summary in memory base
            self.memory.update_period_summary(period_id, result.period_summary)
            # Write first-person retrospective summary (paper's Low-Res Simulator output ℓ_i)
            period_record = self.memory.get_life_period(period_id)
            if period_record and result.period_summary_first_person:
                period_record.period_summary_first_person = result.period_summary_first_person
            with self._track_perf("event.summary_rewrite", period_id=period_id, summary_type="period_summary"):
                self._save_memory()
            logger.info(
                f"  Period summary rewritten for {period_id}: "
                f"{result.period_summary[:80]}..."
            )
            return result.period_summary
        except Exception:
            logger.error(
                f"  [BUG] Failed to rewrite period summary for {period_id}; leaving existing summary unchanged",
                exc_info=True,
            )
            return ""

    # ── Core: Batch-process multiple consecutive LR-only segments ─────────

    async def process_lr_only_batch_segments(
        self,
        segments: List[Dict[str, Any]],
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
        hr_events_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-process multiple consecutive LR-only segments in a single LLM call.

        All segments must have max_detail_events=0 and max_outline_events=0.
        Each segment corresponds to one LP (time_unit=module), so each segment
        produces exactly one LR event.

        Workflow:
          Step 0: Build (period, segment, time_unit) tuples
          Step 1: Single LLM call → generate ALL LR frameworks at once
                  (uses _step1a_batch_generate_all_lr_multi_period with Tier-3 model)
          Step 2: Commit all LR events to memory (sequential — memory writes must be ordered)
          Step 3: Parallel UPER: generate first-person summaries for P_TARGET for ALL events
                  simultaneously (asyncio.gather)
          Step 4: Parallel period summary rewrite for all LPs (asyncio.gather)

        Args:
            segments: List of segment dicts from derived_memory_plan, all LR-only.
            life_plan: Full life plan dict.
            persona_config: Persona configuration dict.
            hr_events_dir: Directory for high-res event files (unused for LR-only).

        Returns:
            List of result dicts (one per segment), each with the same shape as
            process_period_atomic() return value.
        """
        if not segments:
            return []

        # ── Step 0: Build (period, segment, time_unit) tuples ──
        period_lookup = {p["period_id"]: p for p in life_plan.get("life_periods", [])}
        batch_items = []
        for seg in segments:
            parent_period_id = seg["parent_period_id"]
            parent_period = period_lookup.get(parent_period_id)
            if not parent_period:
                logger.warning(
                    f"[P2-CrossLP-Batch] Segment {seg.get('segment_id')} references "
                    f"unknown parent_period_id={parent_period_id}; skipping"
                )
                continue
            start_str = seg["start_date"]
            end_str = seg["end_date"]
            batch_items.append({
                "seg": seg,
                "period": parent_period,
                "start": date.fromisoformat(start_str),
                "end": date.fromisoformat(end_str),
            })

        if not batch_items:
            return []

        logger.info(
            f"[P2-CrossLP-Batch] Batch-generating LR events for "
            f"{len(batch_items)} consecutive LR-only segments: "
            f"{[item['seg']['segment_id'] for item in batch_items]}"
        )

        # ── Step 1: Single LLM call → ALL LR frameworks ──
        time_units_list = [(item["start"], item["end"]) for item in batch_items]
        unit_labels = [
            f"module {item['seg']['segment_id']}"
            for item in batch_items
        ]
        periods_list = [item["period"] for item in batch_items]

        try:
            batch_lr_result = await self._step1a_batch_generate_all_lr_multi_period(
                periods=periods_list,
                time_units=time_units_list,
                unit_labels=unit_labels,
            )
            lr_frameworks = batch_lr_result.lr_frameworks
            # Pad or trim to match batch size
            while len(lr_frameworks) < len(batch_items):
                lr_frameworks.append(LowResFrameworkOutput(
                    event_type="daily_routine",
                    event_theme="daily life",
                    event_location="",
                    value_for_target="",
                    value_for_period="",
                    summary="Uneventful period.",
                ))
            lr_frameworks = lr_frameworks[:len(batch_items)]
        except Exception:
            logger.exception(
                "[P2-CrossLP-Batch] Batch LR generation failed; "
                "falling back to sequential process_period_atomic"
            )
            # Fallback: process each segment individually
            results = []
            for item in batch_items:
                try:
                    r = await self.process_period_atomic(
                        period=item["period"],
                        life_plan=life_plan,
                        persona_config=persona_config,
                        max_turns=0,
                        skip_p3=True,
                        hr_events_dir=hr_events_dir,
                        segment_override=item["seg"],
                        is_last_segment_of_period=True,
                    )
                    results.append(r)
                except Exception:
                    logger.exception(
                        f"[P2-CrossLP-Batch] Fallback failed for "
                        f"{item['seg']['segment_id']}"
                    )
                    results.append({
                        "period_id": item["period"].get("period_id", "?"),
                        "time_unit_count": 0,
                        "high_res_count": 0,
                        "committed_event_ids": [],
                        "failed_units": ["fallback_failed"],
                    })
            return results

        # ── Step 2: Commit all LR events to memory (sequential — memory writes must be ordered) ──
        results = []
        uper_tasks = []  # Collect for parallel UPER in Step 3

        for i, item in enumerate(batch_items):
            seg = item["seg"]
            period = item["period"]
            period_id = period.get("period_id", "LP?")
            segment_id = seg.get("segment_id", "?")
            event_fw = lr_frameworks[i]
            u_start, u_end = item["start"], item["end"]

            # Register canonical period in memory
            canonical_time_period_str = f"{seg['start_date']} to {seg['end_date']}"
            existing = self.memory.get_life_period(period_id)
            if not existing:
                title = period.get("title", "")
                theme = period.get("dominant_theme", "")
                self.memory.add_life_period(
                    period_id=period_id,
                    time_period=canonical_time_period_str,
                    period_summary=f"{title}. {theme}" if theme else title,
                )

            low_res_event_id = self._next_event_id(period_id)
            time_period_str = _format_time_period(
                start=u_start, start_seg="morning",
                end=u_end, end_seg="afternoon",
            )
            await self.memory.upsert_low_res_event(
                event_id=low_res_event_id,
                period_id=period_id,
                time_period=time_period_str,
                summary=event_fw.summary,
            )
            self._save_memory()

            # Collect UPER task for P_TARGET (first-person summary)
            uper_tasks.append({
                "participant_ids": ["P_TARGET"],
                "event_summary": event_fw.summary,
                "event_date": u_start,
                "event_id": low_res_event_id,
                "lr_time_period": time_period_str,
                "hr_time_period": "",
                "high_res_summary": None,
                "post_scene_memories": None,
            })

            # ── Write Medium (habitual) events from batch LR output ──
            hab_events_for_unit = []
            if hasattr(batch_lr_result, 'habitual_events_per_unit') and batch_lr_result.habitual_events_per_unit:
                if i < len(batch_lr_result.habitual_events_per_unit):
                    hab_events_for_unit = batch_lr_result.habitual_events_per_unit[i] or []

            hab_event_ids = []
            for hab_event in hab_events_for_unit:
                hab_event_id = self._next_event_id(period_id)
                hab_time_period_str = f"from {u_start.isoformat()} to {u_end.isoformat()}"
                self.memory.add_event(
                    event_id=hab_event_id,
                    period_id=period_id,
                    resolution_level="medium",
                    time_period=hab_time_period_str,
                    summary=f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]",
                    participants=["P_TARGET"],
                    languages_in_use=["eng"],
                )
                hab_record = self.memory.get_event(hab_event_id)
                if hab_record:
                    hab_record.general_event_title = hab_event.habitual_event_title
                    hab_record.general_event_frequency = hab_event.habitual_event_frequency
                    hab_record.storage_status = "final"
                hab_event_ids.append(hab_event_id)

                # Add UPER task for habitual event (first-person memory generation)
                uper_tasks.append({
                    "participant_ids": ["P_TARGET"],
                    "event_summary": f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]: {hab_event.habitual_event_summary}",
                    "event_date": u_start,
                    "event_id": hab_event_id,
                    "lr_time_period": hab_time_period_str,
                    "hr_time_period": "",
                    "high_res_summary": None,
                    "post_scene_memories": None,
                    "is_habitual": True,
                })

            if hab_events_for_unit:
                self._save_memory()
                logger.info(
                    f"[P2-CrossLP-Batch] Written {len(hab_events_for_unit)} habitual events "
                    f"for segment {segment_id}"
                )

            logger.info(
                f"[P2-CrossLP-Batch] Committed LR event {low_res_event_id} "
                f"for segment {segment_id}"
            )
            results.append({
                "period_id": period_id,
                "segment_id": segment_id,
                "time_unit_count": 1,
                "high_res_count": 0,
                "simulated_high_res_count": 0,
                "committed_event_ids": [low_res_event_id] + hab_event_ids,
                "failed_units": [],
            })

        # ── Step 3: Parallel UPER — generate first-person summaries for P_TARGET ──
        # All LR events are committed; now generate first-person memories in parallel.
        if uper_tasks:
            logger.info(
                f"[P2-CrossLP-Batch] Running parallel UPER for {len(uper_tasks)} LR events..."
            )
            try:
                await self._uper_reflect_batch_period(event_tasks=uper_tasks)
                logger.info(
                    f"[P2-CrossLP-Batch] Parallel UPER complete for {len(uper_tasks)} events"
                )
            except Exception:
                logger.warning(
                    "[P2-CrossLP-Batch] Parallel UPER failed; continuing without first-person summaries"
                )

        # ── Step 4: Parallel period summary rewrite for all LPs ──
        logger.info(
            f"[P2-CrossLP-Batch] Running parallel period summary rewrite for "
            f"{len(batch_items)} LPs..."
        )
        period_summary_coroutines = [
            self._rewrite_period_summary(period=item["period"])
            for item in batch_items
        ]
        period_summary_results = await asyncio.gather(
            *period_summary_coroutines, return_exceptions=True
        )
        for i, res in enumerate(period_summary_results):
            if isinstance(res, Exception):
                logger.warning(
                    f"[P2-CrossLP-Batch] Period summary rewrite failed for "
                    f"{batch_items[i]['period'].get('period_id', '?')}: {res}"
                )

        return results

    async def _run_period_uper(
        self,
        lr_bundles: List[tuple],
        all_outlines: List[tuple],
        detail_event_ids: List[str],
        detail_results: List,
        p3_tasks: List[tuple],
    ) -> None:
        """Unified period UPER: run after ALL detail simulations complete.

        For each event in the period, dispatches _uper_reflect_batch for P_TARGET
        in parallel (asyncio.gather). Side-chars get event summary written directly.

        Events included:
          - All LR events (from lr_bundles)
          - Outline events NOT expanded into a detail (all_outlines minus detail_event_ids)
          - All detail events (from p3_tasks + detail_results)

        Outline events that WERE expanded into a detail are excluded (detail replaces them).
        """
        uper_coros = []

        # ── 1. LR events ────────────────────────────────────────────────────
        for _unit_idx, lr_bundle in lr_bundles:
            pending_lr = lr_bundle.get("pending_lr_event") or {}
            lr_event_id = pending_lr.get("event_id", "")
            lr_summary = pending_lr.get("summary", "") or lr_bundle.get("lr_summary", "")
            lr_participants = pending_lr.get("participants") or lr_bundle.get("all_participant_ids", ["P_TARGET"])
            lr_time_period = pending_lr.get("time_period", "")
            lr_date_str = lr_time_period.split(" ")[1] if " " in lr_time_period else ""
            if not lr_event_id or not lr_summary or not lr_date_str:
                continue
            try:
                lr_event_date = date.fromisoformat(lr_date_str)
            except (ValueError, TypeError):
                continue

            uper_coros.append(self._uper_reflect_batch(
                participant_ids=list(lr_participants),
                event_summary=lr_summary,
                event_date=lr_event_date,
                event_id=lr_event_id,
                lr_time_period=lr_time_period,
            ))

        # ── 2. Outline-only events (not expanded into detail) ────────────────
        expanded_outline_eids = {eid for eid, _, _ in p3_tasks}  # outlines that became detail
        for ol_eid, outline, _unit_idx, lr_bundle, _ in all_outlines:
            if ol_eid in detail_event_ids:
                continue  # this outline was expanded; skip (detail handles it)
            ol_summary = outline.key_event_summary if hasattr(outline, "key_event_summary") else str(outline)
            all_participant_ids = lr_bundle.get("all_participant_ids", ["P_TARGET"])
            # Approximate date from outline start
            ol_date_str = getattr(outline, "key_event_start_date", "") or ""
            if not ol_date_str:
                pending_lr = lr_bundle.get("pending_lr_event") or {}
                tp = pending_lr.get("time_period", "")
                ol_date_str = tp.split(" ")[1] if " " in tp else ""
            if not ol_date_str:
                continue
            try:
                ol_date = date.fromisoformat(ol_date_str)
            except (ValueError, TypeError):
                continue

            uper_coros.append(self._uper_reflect_batch(
                participant_ids=list(all_participant_ids),
                event_summary=ol_summary,
                event_date=ol_date,
                event_id=ol_eid,
            ))

        # ── 3. Detail events ─────────────────────────────────────────────────
        # Map p3_task eid → (p3_bundle, all_participant_ids)
        p3_map = {eid: (bundle, pids) for eid, bundle, pids in p3_tasks}
        # Map detail eid → sim_result from detail_results
        detail_sim_map: Dict[str, Any] = {}
        for i, res in enumerate(detail_results):
            if isinstance(res, Exception):
                continue
            if isinstance(res, tuple) and len(res) >= 2:
                d_eid, sim_result, *_ = res
                if d_eid:
                    detail_sim_map[d_eid] = sim_result

        for eid in detail_event_ids:
            if eid not in p3_map:
                continue
            p3_bundle, all_pids = p3_map[eid]
            sim_result = detail_sim_map.get(eid)
            hr_payload = p3_bundle.get("pending_hr_event") or {}
            hr_event_id = hr_payload.get("event_id", "")
            if not hr_event_id:
                continue

            # Use the detail simulation's preferred summary
            if sim_result:
                hr_summary = (
                    (sim_result.get("refined_summary") or "")
                    or (sim_result.get("summary") or "")
                    or hr_payload.get("summary", "")
                )
                # P_TARGET post_scene_memory (for context only)
                post_memories = sim_result.get("post_scene_memories", []) or []
                scene_date_str = (
                    (sim_result.get("event_context", {}) or {})
                    .get("duration", {})
                    .get("start_date", "")
                    or hr_payload["time_period"].split(" ")[1]
                )
            else:
                hr_summary = hr_payload.get("summary", "")
                post_memories = []
                scene_date_str = hr_payload["time_period"].split(" ")[1] if " " in hr_payload.get("time_period", "") else ""

            if not scene_date_str:
                continue
            try:
                scene_date = date.fromisoformat(scene_date_str)
            except (ValueError, TypeError):
                continue

            pending_lr = p3_bundle.get("pending_lr_event") or {}
            lr_time_period = pending_lr.get("time_period", "")

            uper_coros.append(self._uper_reflect_batch(
                participant_ids=list(all_pids),
                event_summary=hr_summary,
                event_date=scene_date,
                event_id=hr_event_id,
                high_res_summary=hr_summary if sim_result else None,
                lr_time_period=lr_time_period,
                hr_time_period=hr_payload.get("time_period", ""),
                post_scene_memories=post_memories,
            ))

        # ── Run all UPER coroutines in parallel ──────────────────────────────
        if not uper_coros:
            return
        logger.info(f"[MemorySystem::Write] Running {len(uper_coros)} UPER tasks in parallel...")
        results = await asyncio.gather(*uper_coros, return_exceptions=True)
        n_ok = sum(1 for r in results if not isinstance(r, Exception))
        n_err = sum(1 for r in results if isinstance(r, Exception))
        logger.info(f"[MemorySystem::Write] Period UPER complete: {n_ok} OK, {n_err} failed")

    # ── Core: Process a full life period ─────────────────────────

    async def process_period_atomic(
        self,
        period: Dict[str, Any],
        life_plan: Dict[str, Any],
        persona_config: Dict[str, Any],
        max_turns: int,
        skip_p3: bool = False,
        hr_events_dir: Optional[str] = None,
        stop_after_first_high_res: bool = False,
        segment_override: Optional[Dict[str, Any]] = None,
        is_last_segment_of_period: bool = True,
    ) -> Dict[str, Any]:
        """Process a full life period (or a segment thereof) with time-unit level prepare/simulate/finalize/validate.

        Args:
            segment_override: If provided, use this segment's date range and
                density (time_unit, max_detail_events, max_outline_events)
                instead of recomputing from AM Model. The ``period`` arg
                still supplies all semantic info (developmental_tasks, etc.).
            is_last_segment_of_period: When True (default), period summary
                and life summary are rewritten after processing. Set to
                False for intermediate segments of a split period so that
                summaries are only written once after the final segment.
        """
        period_id = period.get("period_id", "LP?")

        # ── Determine date range and density ──
        if segment_override:
            # Use pre-computed segment from derived_memory_plan
            segment_id = segment_override.get("segment_id", "")
            start_str = segment_override["start_date"]
            end_str = segment_override["end_date"]
            unit = segment_override["time_unit"]
            detail_budget = segment_override["max_detail_events"]
            outline_budget = segment_override["max_outline_events"]
            medium_per_unit = segment_override.get("max_medium_events_per_unit", 2)

            # High-res gate: enforce recency rule even for pre-computed segments
            # Only zero out detail_budget; preserve outline_budget for general-event generation
            # (guards against stale derived_memory_plan generated before this rule was added)
            _global_ref_str = (
                life_plan.get("reference_date")
                or persona_config.get("reference_date")
                or end_str
            )
            try:
                _seg_end = date.fromisoformat(end_str)
                _ref_d = date.fromisoformat(_global_ref_str)
                _boundary_5yr = _ref_d - timedelta(days=5 * 365)
                if _seg_end <= _boundary_5yr:
                    detail_budget = 0
                    # outline_budget preserved — general events are generated for all non-amnesia periods
            except (ValueError, TypeError):
                pass

            logger.info(
                f"[P2] Using segment_override {segment_id} for period {period_id}: "
                f"date={start_str}~{end_str}, unit={unit}, "
                f"detail_budget={detail_budget}, outline_budget={outline_budget}"
            )
        else:
            # Fallback: compute density from AM Model
            dr = period.get("period_date_range", {})
            start_str = dr.get("start_date", "")
            end_str = dr.get("end_date", "")

        if not start_str or not end_str:
            logger.warning(f"Period {period_id} has no date range; skipping")
            return {
                "period_id": period_id,
                "time_unit_count": 0,
                "high_res_count": 0,
                "committed_event_ids": [],
                "failed_units": ["missing_date_range"],
            }

        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)

        if not segment_override:
            # ── Compute density from AM Model (original path) ──
            from lifelong_synth.configs.temporal_context import (
                AutobiographicalMemoryModel,
                compute_period_density,
            )

            birth_date_for_am = (
                persona_config.get("derived_birth_date")
                or persona_config.get("birth_date")
                or (persona_config.get("global_summary", {})
                    .get("timeline_anchor", {})
                    .get("derived_birth_date", ""))
            )
            country_for_am = life_plan.get("global_summary", {}).get("social_context", {}).get("country", "")

            # Use global reference_date (current date) for recency calculation,
            # NOT period end_str. This ensures years_ago reflects true temporal
            # distance from the present, consistent with life_period_planner.py.
            global_reference_date = (
                life_plan.get("reference_date")
                or persona_config.get("reference_date")
                or end_str  # fallback to period end if not available
            )
            if birth_date_for_am and start_str and end_str:
                am_model_new = AutobiographicalMemoryModel(
                    birth_date=birth_date_for_am,
                    reference_date=global_reference_date,
                    country=country_for_am,
                )
                density_dict = compute_period_density(period, am_model_new, birth_date_for_am)
            else:
                density_dict = {"time_unit": "year", "max_detail_events": 1, "max_outline_events": 2, "max_medium_events_per_unit": 2}

            unit = density_dict["time_unit"]
            detail_budget = density_dict["max_detail_events"]
            outline_budget = density_dict["max_outline_events"]
            medium_per_unit = density_dict.get("max_medium_events_per_unit", 2)

        density = "none"  # Keep variable for backward compat with _step1a prompt

        # Register canonical period in memory (use canonical date range, not segment)
        canonical_dr = period.get("period_date_range", {})
        canonical_time_period_str = (
            f"{canonical_dr.get('start_date', '')} to {canonical_dr.get('end_date', '')}"
        )
        existing = self.memory.get_life_period(period_id)
        if not existing:
            title = period.get("title", "")
            theme = period.get("dominant_theme", "")
            self.memory.add_life_period(
                period_id=period_id,
                time_period=canonical_time_period_str,
                period_summary=f"{title}. {theme}" if theme else title,
            )

        time_units = _iter_time_units(start, end, unit)
        logger.info(
            f"Period {period_id}: {len(time_units)} time units "
            f"(unit={unit}, detail_budget={detail_budget}, outline_budget={outline_budget}, mode=atomic)"
        )

        # ── TC Engine: Initialize TemporalContextComputer for this period ──
        from lifelong_synth.configs.temporal_context import (
            TemporalContextComputer,
            AutobiographicalMemoryDistributionModel,
        )

        # ── v6: Budget tracking (used at segment level, not unit level) ──
        # detail_remaining/outline_remaining are kept for backward compat with _process_time_unit signature
        detail_remaining = detail_budget
        outline_remaining = outline_budget

        # Bridge SocialContextProfile from life_plan (P1 output)
        social_context = life_plan.get("global_summary", {}).get("social_context", {})
        birth_date = (
            persona_config.get("derived_birth_date")
            or persona_config.get("birth_date")
            or (persona_config.get("global_summary", {})
                .get("timeline_anchor", {})
                .get("derived_birth_date", ""))
        )
        life_periods_list = life_plan.get("life_periods", [])

        if birth_date and social_context:
            self._tc_computer = TemporalContextComputer(
                birth_date=birth_date,
                social_context=social_context,
                life_periods=life_periods_list,
            )
            self._social_context = social_context
            logger.info(f"[TC] TemporalContextComputer initialized for period {period_id}")

        # ── AM Distribution: Compute weight for P3 telescoping (no cap in P2) ──
        am_weight = None
        if birth_date and start_str and end_str:
            country = social_context.get("country", "")
            am_model = AutobiographicalMemoryDistributionModel(
                birth_date=birth_date,
                reference_date=end_str,
                country=country,
            )
            am_weight = am_model.compute_period_weight(start_str, end_str)
            logger.info(
                f"[AM] Period {period_id}: weight={am_weight:.2f} "
                f"(used for P3 telescoping only, no high-res cap in P2)"
            )

        # ── TC Enrichment: Pre-generate year enrichments for this period (parallel) ──
        persona_brief = persona_config.get("persona_brief_text", "")
        if start_str and end_str:
            try:
                start_year = int(start_str[:4])
                end_year = int(end_str[:4])
                missing_years = [
                    yr for yr in range(start_year, end_year + 1)
                    if yr not in self._enrichment_cache
                ]
                if missing_years:
                    await asyncio.gather(*[
                        self._get_or_generate_year_enrichment(
                            year=yr,
                            period=period,
                            persona_brief=persona_brief,
                        )
                        for yr in missing_years
                    ], return_exceptions=True)
                logger.info(
                    f"[TC-Enrichment] Pre-generated enrichments for years "
                    f"{start_year}-{end_year} (parallel, {len(missing_years)} new, "
                    f"cache size: {len(self._enrichment_cache)})"
                )
                if missing_years:
                    self._save_enrichment_cache()
            except (ValueError, TypeError) as e:
                logger.warning(f"[TC-Enrichment] Failed to pre-generate enrichments: {e}")

        committed_event_ids: List[str] = []
        failed_units: List[str] = []
        high_res_count = 0
        simulated_high_res_count = 0

        # v9: Compute high-res quotas per time unit (outline = high-res, no separate budget)
        # All outlines will be simulated in P3, so quota = detail_budget distributed across units
        n_units = len(time_units)
        unit_outline_quotas = [0] * n_units
        if n_units > 0 and detail_budget > 0:
            # Evenly distributed, later units get more (recency effect)
            base_quota = detail_budget // n_units
            remainder = detail_budget % n_units
            for i in range(n_units):
                unit_outline_quotas[i] = base_quota
            # Recency bias: remainder goes to last units
            for i in range(remainder):
                unit_outline_quotas[-(i + 1)] += 1

            logger.info(
                f"[P2-v9] High-res quotas (all outlines → P3): "
                f"quotas={unit_outline_quotas}"
            )

        # ★ M1 block removed — budget is fully controlled by AM-weight algorithm
        # in _compile_memory_segments_from_plan (v3). No secondary cap needed.

        # ═══ Phase 1: Generate LR events for all time units ═══
        logger.info(f"[MultiResSimulator::LowRes] Generating LR events for {n_units} time units...")
        all_outlines = []  # M4+M5: Initialize early so merged generation can add outlines

        # ★ M2-d: Batch grouping — identify consecutive LR-only segments
        # LR-only = units with outline_quota == 0 (no outline detail needed)
        # These can be batch-generated with single LLM calls
        MAX_BATCH_SIZE = 6  # Cap batch size to avoid overly long prompts

        # Build groups: each group is either a batch of LR-only units or a single outline unit
        unit_groups: List[Dict[str, Any]] = []  # List of {"type": "batch"|"single", "indices": [...]}
        i = 0
        while i < n_units:
            if unit_outline_quotas[i] == 0:
                # Start a batch of consecutive LR-only units
                batch_indices = [i]
                i += 1
                while i < n_units and unit_outline_quotas[i] == 0 and len(batch_indices) < MAX_BATCH_SIZE:
                    batch_indices.append(i)
                    i += 1
                unit_groups.append({"type": "batch", "indices": batch_indices})
            else:
                # Single unit with outline budget
                unit_groups.append({"type": "single", "indices": [i]})
                i += 1

        lr_bundles = []  # List of (unit_index, bundle)
        uper_tasks = []  # M2-c: collect UPER tasks for parallel execution

        for group in unit_groups:
            group_type = group["type"]
            group_indices = group["indices"]

            if group_type == "batch" and len(group_indices) >= 1:
                # ★ M2-a: Batch LR generation for consecutive LR-only units
                # (Also applies to single LR-only unit — no outline needed)
                batch_time_units = [time_units[idx] for idx in group_indices]
                batch_labels = [f"{unit} {idx+1}/{n_units}" for idx in group_indices]
                logger.info(
                    f"[P2-M2] Batch LR generation for {len(group_indices)} consecutive units: "
                    f"{batch_labels}"
                )
                try:
                    batch_lr_result = await self._step1a_batch_generate_all_lr(
                        period=period,
                        time_units=batch_time_units,
                        unit_labels=batch_labels,
                        density=density,
                        detail_budget=detail_budget,
                        outline_budget=outline_budget,
                        total_units=n_units,
                        batch_start_index=group_indices[0],
                    )

                    # ★ M2-b: Batch participant pool for these LR units
                    lr_frameworks = batch_lr_result.lr_frameworks
                    # Pad or trim to match batch size
                    if len(lr_frameworks) < len(group_indices):
                        logger.warning(
                            f"[P2-M2] Batch LR returned {len(lr_frameworks)} frameworks "
                            f"for {len(group_indices)} units; padding with defaults"
                        )
                        while len(lr_frameworks) < len(group_indices):
                            lr_frameworks.append(LowResFrameworkOutput(
                                event_type="daily_routine",
                                event_theme="daily life",
                                event_location="",
                                value_for_target="",
                                value_for_period="",
                                summary="Uneventful period.",
                            ))
                    elif len(lr_frameworks) > len(group_indices):
                        lr_frameworks = lr_frameworks[:len(group_indices)]

                    # ★ LR-only: no participant pool, no UPER — just commit directly
                    for j, idx in enumerate(group_indices):
                        u_start, u_end = time_units[idx]
                        unit_label = batch_labels[j]
                        event_fw = lr_frameworks[j]

                        # Commit LR event to memory (resolution_level + time_period + summary only)
                        low_res_event_id = self._next_event_id(period_id)
                        time_period_str = _format_time_period(
                            start=u_start, start_seg="morning",
                            end=u_end, end_seg="afternoon",
                        )

                        await self.memory.upsert_low_res_event(
                            event_id=low_res_event_id,
                            period_id=period_id,
                            time_period=time_period_str,
                            summary=event_fw.summary,
                        )
                        self._save_memory()
                        committed_event_ids.append(low_res_event_id)

                        # Update protagonist interaction history via UPER (first-person).
                        # Side-characters are NOT written for LR-only events (no participant pool).
                        uper_tasks.append({
                            "participant_ids": ["P_TARGET"],
                            "event_summary": event_fw.summary,
                            "event_date": u_start,
                            "event_id": low_res_event_id,
                            "lr_time_period": time_period_str,
                        })

                        # ── Write Medium (habitual) events from batch LR output ──
                        hab_events_for_unit = []
                        if hasattr(batch_lr_result, 'habitual_events_per_unit') and batch_lr_result.habitual_events_per_unit:
                            if j < len(batch_lr_result.habitual_events_per_unit):
                                hab_events_for_unit = batch_lr_result.habitual_events_per_unit[j] or []

                        for hab_event in hab_events_for_unit:
                            hab_event_id = self._next_event_id(period_id)
                            hab_time_period_str = f"from {u_start.isoformat()} to {u_end.isoformat()}"
                            self.memory.add_event(
                                event_id=hab_event_id,
                                period_id=period_id,
                                resolution_level="medium",
                                time_period=hab_time_period_str,
                                summary=f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]",
                                participants=["P_TARGET"],
                                languages_in_use=["eng"],
                            )
                            hab_record = self.memory.get_event(hab_event_id)
                            if hab_record:
                                hab_record.general_event_title = hab_event.habitual_event_title
                                hab_record.general_event_frequency = hab_event.habitual_event_frequency
                                hab_record.storage_status = "final"
                            committed_event_ids.append(hab_event_id)
                            uper_tasks.append({
                                "participant_ids": ["P_TARGET"],
                                "event_summary": f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]: {hab_event.habitual_event_summary}",
                                "event_date": u_start,
                                "event_id": hab_event_id,
                                "lr_time_period": hab_time_period_str,
                                "is_habitual": True,
                            })
                        if hab_events_for_unit:
                            self._save_memory()
                            logger.info(
                                f"[P2-M2] Written {len(hab_events_for_unit)} habitual events "
                                f"for unit {unit_label}"
                            )

                        bundle = {
                            "pending_lr_event": {
                                "event_id": low_res_event_id,
                                "resolution_level": "low",
                                "time_period": time_period_str,
                                "summary": event_fw.summary,
                                "initial_summary": event_fw.summary,
                                "period_id": period_id,
                                "run_id": self._run_id,
                            },
                            "event_framework": event_fw,
                            "lr_summary": event_fw.summary,
                        }
                        lr_bundles.append((idx, bundle))

                    logger.info(
                        f"[P2-M2] Batch LR: {len(group_indices)} units processed"
                    )
                except Exception:
                    logger.exception(
                        f"[P2-M2] Batch LR failed for units {group_indices}; "
                        f"falling back to sequential processing"
                    )
                    # Fallback to sequential processing
                    for idx in group_indices:
                        u_start, u_end = time_units[idx]
                        unit_label = f"{unit} {idx+1}/{n_units}"
                        try:
                            bundle = await self._process_time_unit(
                                period=period,
                                unit_start=u_start,
                                unit_end=u_end,
                                unit_label=unit_label,
                                density=density,
                                detail_remaining=0,
                                outline_remaining=0,
                                detail_budget=detail_budget,
                                outline_budget=outline_budget,
                                total_units=n_units,
                                current_unit_index=idx,
                            )
                            if bundle:
                                lr_bundles.append((idx, bundle))
                                pending_lr = bundle.get("pending_lr_event") or {}
                                low_res_event_id = pending_lr.get("event_id", "")
                                if low_res_event_id:
                                    committed_event_ids.append(low_res_event_id)
                            else:
                                failed_units.append(unit_label)
                        except Exception:
                            logger.exception(
                                f"[MultiResSimulator::LowRes] ❌ Failed during {period_id} {unit_label}"
                            )
                            failed_units.append(unit_label)
            else:
                # ★ M4+M5-d: Single unit with outline budget — use merged LR+Outline generation
                idx = group_indices[0]
                u_start, u_end = time_units[idx]
                unit_label = f"{unit} {idx+1}/{n_units}"
                k = unit_outline_quotas[idx]  # Number of outlines for this unit
                logger.info(
                    f"[P2-M4+5] Phase 1: Merged LR+Outline for {period_id} {unit_label} "
                    f"(k={k} outlines)..."
                )
                try:
                    # M4+M5-b: Generate LR + outlines in single prompt
                    lr_with_ol = await self._step1a_generate_lr_with_outlines(
                        period=period,
                        unit_start=u_start,
                        unit_end=u_end,
                        unit_label=unit_label,
                        density=density,
                        k=k,
                        detail_budget=detail_budget,
                        outline_budget=outline_budget,
                        medium_per_unit=medium_per_unit,
                        total_units=n_units,
                        current_unit_index=idx,
                    )
                    event_fw = lr_with_ol.lr_framework
                    generated_outlines = lr_with_ol.outlines

                    # M4+M5-c: Unified pool for LR + outlines in single prompt
                    lr_and_pool = await self._step2_unified_pool_for_lr_and_outlines(
                        lr_with_outlines=lr_with_ol,
                        period_id=period_id,
                        unit_start=u_start,
                        unit_end=u_end,
                        unit_label=unit_label,
                    )
                    lr_pool = lr_and_pool.lr_pool
                    outline_pools = lr_and_pool.outline_pools

                    # Process LR event
                    selected_ids = [c.participant_id for c in lr_pool.selected_participants]
                    if "P_TARGET" not in selected_ids:
                        selected_ids.insert(0, "P_TARGET")
                    selected_ids = [
                        pid for pid in selected_ids
                        if self.pool.get_participant(pid) is not None
                    ]

                    if not lr_pool.is_sufficient and lr_pool.new_participant_requirements:
                        for req in lr_pool.new_participant_requirements[:2]:
                            new_pid = await self._create_new_participant_from_requirement(
                                requirement=self._entry_to_requirement(req),
                                period_id=period_id,
                                current_date=u_start,
                            )
                            if new_pid:
                                selected_ids.append(new_pid)
                            else:
                                logger.warning(
                                    f"  [M4+5] Could not create new participant for LR requirement: "
                                    f"role_type={req.role_type}, relationship={req.relationship_to_main_character}; "
                                    f"event will proceed with available participants"
                                )

                    # Commit LR event to memory
                    low_res_event_id = self._next_event_id(period_id)
                    time_period_str = _format_time_period(
                        start=u_start, start_seg="morning",
                        end=u_end, end_seg="afternoon",
                    )

                    await self.memory.upsert_low_res_event(
                        event_id=low_res_event_id,
                        period_id=period_id,
                        time_period=time_period_str,
                        summary=event_fw.summary,
                        participants=selected_ids,
                        languages_in_use=["eng"],
                        )
                    self._save_memory()
                    committed_event_ids.append(low_res_event_id)

                    bundle = {
                        "pending_lr_event": {
                            "event_id": low_res_event_id,
                            "resolution_level": "low",
                            "time_period": time_period_str,
                            "summary": event_fw.summary,
                            "initial_summary": event_fw.summary,
                            "participants": selected_ids,
                            "period_id": period_id,
                            "run_id": self._run_id,
                        },
                        "event_framework": event_fw,
                        "lr_summary": event_fw.summary,
                        "all_participant_ids": selected_ids,
                    }
                    lr_bundles.append((idx, bundle))

                    # Collect UPER task for LR event
                    uper_tasks.append({
                        "participant_ids": selected_ids,
                        "event_summary": event_fw.summary,
                        "event_date": u_start,
                        "event_id": low_res_event_id,
                        "lr_time_period": time_period_str,
                    })

                    # Process outlines (skip Phase 2 for this unit)
                    for j, outline_in_lr in enumerate(generated_outlines):
                        ol_event_id = self._next_event_id(period_id)
                        # Find the outline pool for this outline
                        ol_pool = None
                        for op in outline_pools:
                            if op.outline_index == j:
                                ol_pool = op
                                break
                        if ol_pool is None and outline_pools:
                            ol_pool = outline_pools[min(j, len(outline_pools) - 1)]

                        ol_selected_ids = []
                        if ol_pool:
                            ol_selected_ids = [
                                c.participant_id for c in ol_pool.selected_participants
                            ]
                        if "P_TARGET" not in ol_selected_ids:
                            ol_selected_ids.insert(0, "P_TARGET")
                        ol_selected_ids = [
                            pid for pid in ol_selected_ids
                            if self.pool.get_participant(pid) is not None
                        ]

                        # Create new participants if needed for outline
                        if ol_pool and not ol_pool.is_sufficient and ol_pool.new_participant_requirements:
                            for req in ol_pool.new_participant_requirements[:2]:
                                new_pid = await self._create_new_participant_from_requirement(
                                    requirement=self._entry_to_requirement(req),
                                    period_id=period_id,
                                    current_date=u_start,
                                )
                                if new_pid:
                                    ol_selected_ids.append(new_pid)
                                else:
                                    logger.warning(
                                        f"  [M4+5] Could not create new participant for outline requirement: "
                                        f"role_type={req.role_type}, relationship={req.relationship_to_main_character}; "
                                        f"HL event will proceed with available participants"
                                    )

                        # Convert OutlineInLR to HighResOutlineOutput for compatibility
                        outline_compat = HighResOutlineOutput(
                            key_event_theme=outline_in_lr.key_event_theme,
                            key_event_summary=outline_in_lr.key_event_summary,
                            key_event_start_date=outline_in_lr.key_event_start_date,
                            key_event_end_date=outline_in_lr.key_event_end_date,
                            key_event_start_time=outline_in_lr.key_event_start_time,
                            key_event_end_time=outline_in_lr.key_event_end_time,
                            key_event_precise_start_time=getattr(outline_in_lr, "key_event_precise_start_time", None) or "09:00:00",
                            key_event_precise_end_time=getattr(outline_in_lr, "key_event_precise_end_time", None) or "18:00:00",
                            key_event_motivation=outline_in_lr.key_event_motivation,
                            key_event_outcome=outline_in_lr.key_event_outcome,
                            key_event_setting=outline_in_lr.key_event_setting,
                            key_event_turning_point=outline_in_lr.key_event_turning_point,
                            key_event_value_for_target=outline_in_lr.key_event_value_for_target,
                            required_participant_roles=outline_in_lr.required_participant_roles,
                        )

                        # Commit outline event
                        try:
                            ol_time_period_str = _format_time_period(
                                start=date.fromisoformat(outline_in_lr.key_event_start_date),
                                start_seg=outline_in_lr.key_event_start_time,
                                end=date.fromisoformat(outline_in_lr.key_event_end_date),
                                end_seg=outline_in_lr.key_event_end_time,
                            )
                        except (ValueError, TypeError):
                            ol_time_period_str = time_period_str

                        await self.memory.upsert_high_res_outline_event(
                            event_id=ol_event_id,
                            period_id=period_id,
                            time_period=ol_time_period_str,
                            summary=outline_in_lr.key_event_summary,
                            participants=ol_selected_ids,
                            languages_in_use=["eng"],
                            interaction_details=[],
                            event_type=getattr(event_fw, 'event_type', 'milestone'),
                            linked_event_id=low_res_event_id,
                            source_event_id=ol_event_id,
                            storage_status=EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
                            run_id=self._run_id,
                            version=1,
                            use_llm_summary=False,
                            initial_summary=outline_in_lr.key_event_summary,
                            refined_summary="",
                            summary_stage="initial",
                        )
                        self._save_memory()
                        committed_event_ids.append(ol_event_id)

                        # Store for Phase 3 (high-res simulation)
                        # ★ M4+M5-e: Pass ol_selected_ids as pre_selected_participants
                        # so P3 refine_participants skips Step 1/2
                        all_outlines.append((ol_event_id, outline_compat, idx, bundle, ol_selected_ids))

                    # ── Write Medium (habitual) events from LLM output ──
                    habitual_events = getattr(lr_with_ol, 'habitual_events', []) or []
                    for hab_event in habitual_events:
                        hab_event_id = self._next_event_id(period_id)
                        hab_time_period_str = f"from {u_start.isoformat()} to {u_end.isoformat()}"
                        self.memory.add_event(
                            event_id=hab_event_id,
                            period_id=period_id,
                            resolution_level="medium",
                            time_period=hab_time_period_str,
                            summary=f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]",
                            participants=["P_TARGET"],
                            languages_in_use=["eng"],
                        )
                        # Set the title and frequency fields
                        hab_record = self.memory.get_event(hab_event_id)
                        if hab_record:
                            hab_record.general_event_title = hab_event.habitual_event_title
                            hab_record.general_event_frequency = hab_event.habitual_event_frequency
                            hab_record.storage_status = "final"
                        committed_event_ids.append(hab_event_id)
                        # Add UPER task for habitual event
                        uper_tasks.append({
                            "participant_ids": ["P_TARGET"],
                            "event_summary": f"{hab_event.habitual_event_title} [{hab_event.habitual_event_frequency}]: {hab_event.habitual_event_summary}",
                            "event_date": u_start,
                            "event_id": hab_event_id,
                            "lr_time_period": hab_time_period_str,
                            "is_habitual": True,
                        })
                    self._save_memory()

                    logger.info(
                        f"[P2-M4+5] Merged LR+Outline for {unit_label}: "
                        f"1 LR + {len(generated_outlines)} outlines + {len(habitual_events)} habitual events"
                    )

                except Exception:
                    logger.exception(
                        f"[P2-M4+5] Merged LR+Outline failed for {period_id} {unit_label}; "
                        f"falling back to sequential processing"
                    )
                    # Fallback to sequential processing
                    try:
                        fb_bundle = await self._process_time_unit(
                            period=period,
                            unit_start=u_start,
                            unit_end=u_end,
                            unit_label=unit_label,
                            density=density,
                            detail_remaining=0,
                            outline_remaining=0,
                            detail_budget=detail_budget,
                            outline_budget=outline_budget,
                            total_units=n_units,
                            current_unit_index=idx,
                        )
                        if fb_bundle:
                            lr_bundles.append((idx, fb_bundle))
                            pending_lr = fb_bundle.get("pending_lr_event") or {}
                            low_res_event_id = pending_lr.get("event_id", "")
                            if low_res_event_id:
                                committed_event_ids.append(low_res_event_id)
                        else:
                            failed_units.append(unit_label)
                    except Exception:
                        logger.exception(
                            f"[MultiResSimulator::LowRes] ❌ Failed during {period_id} {unit_label}"
                        )
                        failed_units.append(unit_label)

        logger.info(
            f"[P2-v6] Phase 1 complete: {len(lr_bundles)} LR events generated, "
            f"{len(failed_units)} failed"
        )

        # ═══ Phase 2: Batch generate outlines per time unit (forward dependency) ═══
        logger.info(f"[MultiResSimulator::MediumRes] Generating outlines (budget={outline_budget})...")
        # M4+M5-d: all_outlines already initialized in Phase 1; check which units
        # already have outlines from merged generation
        units_with_merged_outlines = {idx for _, _, idx, _, _ in all_outlines}

        # [A-07] Maintain cross-unit outline context for deduplication
        cross_unit_outline_ctx = ""

        for i, (u_start, u_end) in enumerate(time_units):
            k = unit_outline_quotas[i]
            if k <= 0:
                continue

            # M4+M5-d: Skip units already processed by merged LR+Outline generation
            if i in units_with_merged_outlines:
                # [A-07] Still collect outlines from merged units for cross-unit ctx
                merged_outlines_for_unit = [
                    (eid, ol) for eid, ol, unit_idx, _, _ in all_outlines if unit_idx == i
                ]
                if merged_outlines_for_unit:
                    cross_unit_outline_ctx = self._build_cross_unit_outline_ctx(
                        cross_unit_outline_ctx, merged_outlines_for_unit
                    )
                logger.info(
                    f"[P2-M4+5] Phase 2: Skipping unit {i+1}/{n_units} "
                    f"(outlines already generated in Phase 1 merged call)"
                )
                continue

            # Find the LR bundle for this unit
            lr_bundle = next((b for idx, b in lr_bundles if idx == i), None)
            if not lr_bundle:
                continue

            lr_summary = lr_bundle.get("lr_summary", "")
            event_fw = lr_bundle.get("event_framework")

            # Build a LowResFrameworkOutput for the batch outline generator
            temp_lr = LowResFrameworkOutput(
                event_type=getattr(event_fw, 'event_type', '') if event_fw else '',
                event_theme=getattr(event_fw, 'event_theme', '') if event_fw else '',
                value_for_target=getattr(event_fw, 'value_for_target', '') if event_fw else '',
                value_for_period=getattr(event_fw, 'value_for_period', '') if event_fw else '',
                summary=lr_summary,
            )

            try:
                outlines = await self._step1b_batch_generate_outlines(
                    low_res=temp_lr,
                    period=period,
                    unit_start=u_start,
                    unit_end=u_end,
                    k=k,
                    cross_unit_outline_ctx=cross_unit_outline_ctx,  # [A-07]
                )

                for outline in outlines:
                    ol_event_id = self._next_event_id(period_id)
                    all_outlines.append((ol_event_id, outline, i, lr_bundle, None))

                    # Write outline event to memory_base (OUTLINE_READY status)
                    time_period_str = _format_time_period(
                        start=date.fromisoformat(outline.key_event_start_date),
                        start_seg=outline.key_event_start_time,
                        end=date.fromisoformat(outline.key_event_end_date),
                        end_seg=outline.key_event_end_time,
                    )
                    all_participant_ids = lr_bundle.get("all_participant_ids", ["P_TARGET"])
                    self.memory.add_event(
                        event_id=ol_event_id,
                        period_id=period_id,
                        resolution_level="medium",
                        time_period=time_period_str,
                        summary=(
                            f"{outline.key_event_summary} "
                            f"[Motivation: {outline.key_event_motivation}] "
                            f"[Outcome: {outline.key_event_outcome}]"
                        ),
                        participants=all_participant_ids,
                        languages_in_use=["eng"],
                    )
                    committed_event_ids.append(ol_event_id)

                # [A-07] Update cross-unit outline context after each unit
                new_outlines_for_ctx = [
                    (eid, ol)
                    for eid, ol, unit_idx, _, _ in all_outlines
                    if unit_idx == i
                ]
                cross_unit_outline_ctx = self._build_cross_unit_outline_ctx(
                    cross_unit_outline_ctx, new_outlines_for_ctx
                )

                self._save_memory()
                logger.info(
                    f"[MultiResSimulator::MediumRes] Generated {len(outlines)} outlines for unit {i+1}/{n_units}"
                )
            except Exception:
                logger.exception(
                    f"[MultiResSimulator::MediumRes] ❌ Failed to generate outlines for unit {i+1}/{n_units}"
                )

        logger.info(
            f"[P2-v6] Phase 2 complete: {len(all_outlines)} outlines generated"
        )

        # ═══ Phase 3: ALL outlines go to P3 simulation (no ranking needed) ═══
        if detail_budget > 0 and all_outlines and not skip_p3:
            logger.info(
                f"[MultiResSimulator::HighRes] Simulating all {len(all_outlines)} outlines in P3..."
            )
            # All outlines are selected for detail simulation
            detail_event_ids = [eid for eid, _, _, _, _ in all_outlines]

            logger.info(
                f"[MultiResSimulator::HighRes] Selected {len(detail_event_ids)} events for detail: "
                f"{detail_event_ids}"
            )

            # ── Phase 3: Parallel detail simulations ──────────────────────────
            # Build p3_bundles for all selected outlines, then run in parallel.
            p3_tasks = []  # list of (eid, p3_bundle, all_participant_ids) for parallel dispatch

            for eid, outline, unit_idx, lr_bundle, outline_pre_selected in all_outlines:

                high_res_count += 1
                u_start, u_end = time_units[unit_idx]
                event_fw = lr_bundle.get("event_framework")
                all_participant_ids = lr_bundle.get("all_participant_ids", ["P_TARGET"])

                # Reuse the outline's event_id for the detail record so that
                # the detail overwrites the outline in memory (same key).
                # This prevents duplicate outline+detail records for the same event.
                hr_event_id = eid
                lr_event_id = (lr_bundle.get("pending_lr_event") or {}).get("event_id", "")

                hr_start_date = outline.key_event_start_date or u_start.isoformat()
                hr_start_seg = outline.key_event_start_time or "morning"
                hr_end_date = outline.key_event_end_date or u_end.isoformat()
                hr_end_seg = outline.key_event_end_time or "afternoon"

                _SEGMENT_DEFAULTS = {
                    "early morning": ("05:30:00", "06:30:00"),
                    "morning":       ("09:00:00", "11:00:00"),
                    "afternoon":     ("14:00:00", "16:00:00"),
                    "night":         ("19:00:00", "21:00:00"),
                }
                default_start, _ = _SEGMENT_DEFAULTS.get(hr_start_seg, ("09:00:00", "11:00:00"))
                _, default_end = _SEGMENT_DEFAULTS.get(hr_end_seg, ("14:00:00", "16:00:00"))

                hr_precise_start = outline.key_event_precise_start_time or default_start
                hr_precise_end = outline.key_event_precise_end_time or default_end

                _TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
                if not _TIME_RE.match(hr_precise_start):
                    hr_precise_start = default_start
                if not _TIME_RE.match(hr_precise_end):
                    hr_precise_end = default_end

                hr_time_period = (
                    f"from {hr_start_date} {hr_precise_start} "
                    f"to {hr_end_date} {hr_precise_end}"
                )

                pending_hr_event = {
                    "event_id": hr_event_id,
                    "resolution_level": "high",
                    "time_period": hr_time_period,
                    "summary": outline.key_event_summary,
                    "initial_summary": outline.key_event_summary,
                    "refined_summary": "",
                    "summary_stage": "initial",
                    "participants": all_participant_ids,
                    "languages_in_use": ["eng"],
                    "interaction_details": [],
                    "period_id": period_id,
                    "event_type": getattr(event_fw, "event_type", "milestone") if event_fw else "milestone",
                    "linked_event_id": lr_event_id,
                    "source_event_id": hr_event_id,
                    "storage_status": EVENT_STORAGE_STATUS_GENERAL_EVENT_READY,
                    "run_id": self._run_id,
                    "version": 1,
                    "use_llm_summary": False,
                    "linked_low_res_summary": lr_bundle.get("lr_summary", ""),
                }

                # Do NOT write outline to memory here — finalize_time_unit will write
                # the FINAL detail record directly once simulation completes.
                # This prevents orphaned outline_ready records when detail fails or
                # the budget is exhausted before this event is simulated.
                committed_event_ids.append(hr_event_id)

                high_res_result = {
                    "event_id": hr_event_id,
                    "parent_low_res_event_id": lr_event_id,
                    "linked_event_id": lr_event_id,
                    "source_event_id": hr_event_id,
                    "run_id": self._run_id,
                    "period_id": period_id,
                    "resolution_level": "high",
                    "value_for_target": outline.key_event_value_for_target,
                    "value_for_period": getattr(event_fw, "value_for_period", "") if event_fw else "",
                    "summary": outline.key_event_summary,
                    "low_res_context": lr_bundle.get("lr_summary", ""),
                    "motivation": outline.key_event_motivation,
                    "duration": {
                        "start_date": hr_start_date,
                        "start_time": hr_start_seg,
                        "precise_start_time": hr_precise_start,
                        "end_date": hr_end_date,
                        "end_time": hr_end_seg,
                        "precise_end_time": hr_precise_end,
                    },
                    "outcome": outline.key_event_outcome,
                    "participants": all_participant_ids,
                    "original_low_res_participants": all_participant_ids,
                    "languages_in_use": ["eng"],
                    "setting": outline.key_event_setting,
                    "turning_point": outline.key_event_turning_point,
                    "linked_low_res_summary": lr_bundle.get("lr_summary", ""),
                    "use_parallel_protagonist_mode": True,
                    "use_m3_script_mode": True,
                }

                if am_weight is not None:
                    high_res_result["am_weight"] = am_weight
                try:
                    evt_year = int(hr_start_date[:4])
                    if evt_year in self._enrichment_cache:
                        high_res_result["year_enrichment"] = self._enrichment_cache[evt_year]
                    if evt_year in self._key_life_path_cache:
                        high_res_result["key_life_path"] = self._key_life_path_cache[evt_year]
                except (ValueError, TypeError, IndexError):
                    pass

                p3_bundle = {
                    "high_res_event": high_res_result,
                    "pending_lr_event": lr_bundle.get("pending_lr_event"),
                    "pending_hr_event": pending_hr_event,
                    "runtime_meta": {"original_low_res_participants": all_participant_ids},
                    "key_event_type": "key_event_detail",
                    "pre_selected_participants": outline_pre_selected,
                }
                p3_tasks.append((eid, p3_bundle, all_participant_ids))

            # ── Run all detail simulations in parallel ────────────────────────
            if p3_tasks:
                logger.info(
                    f"[MultiResSimulator::HighRes] Running {len(p3_tasks)} detail simulations in parallel..."
                )

                # ── Step 3a: Sequential pre-refinement ─────────────────────────
                # Run refine_participants() for each task sequentially BEFORE the
                # parallel simulation dispatch. This eliminates last-write-wins
                # races on self.pool when multiple events share participants.
                # After this loop, each bundle's pre_selected_participants holds
                # the definitive refined list; run_high_res_for_time_unit is
                # then called with skip_refinement=True.
                logger.info(f"[MultiResSimulator::HighRes] Sequential pre-refinement for {len(p3_tasks)} tasks...")
                for _eid, _p3_bundle, _all_pids in p3_tasks:
                    _hr_event_id = _p3_bundle["pending_hr_event"]["event_id"]
                    try:
                        await self.run_high_res_for_time_unit(
                            bundle=_p3_bundle,
                            life_plan=life_plan,
                            persona_config=persona_config,
                            max_turns=0,  # refinement only — simulation skipped when max_turns=0
                            hr_events_dir=None,
                            skip_refinement=False,  # actually run refinement here
                            _refinement_only=True,  # new flag: stop after refinement, no simulation
                        )
                    except Exception:
                        logger.exception(
                            f"[MultiResSimulator::HighRes] ❌ Pre-refinement failed for {_hr_event_id}; "
                            f"will use original participants"
                        )

                # ── Step 3b: Parallel simulations (refinement already done) ───

                async def _run_one_detail(eid_bundle_pids):
                    eid, p3_bundle, all_pids = eid_bundle_pids
                    hr_event_id = p3_bundle["pending_hr_event"]["event_id"]
                    try:
                        simulated_high_res_count_delta = 1
                        run_result = await self.run_high_res_for_time_unit(
                            bundle=p3_bundle,
                            life_plan=life_plan,
                            persona_config=persona_config,
                            max_turns=max_turns,
                            hr_events_dir=hr_events_dir,
                            skip_refinement=True,
                        )
                        sim_result = run_result.get("sim_result")
                        final_participants = run_result.get("final_participants") or all_pids
                        await self.finalize_time_unit(
                            p3_bundle,
                            simulated_result=sim_result,
                            final_participants=final_participants,
                            mark_high_res_failed=False,
                        )
                        self.validate_time_unit_commit(p3_bundle)
                        logger.info(f"[MultiResSimulator::HighRes] ✅ Detail event {hr_event_id} committed")
                        return (eid, sim_result, final_participants)
                    except Exception:
                        logger.exception(f"[MultiResSimulator::HighRes] ❌ Failed P3 simulation for {hr_event_id}")
                        try:
                            await self.finalize_time_unit(
                                p3_bundle,
                                simulated_result=None,
                                final_participants=all_pids,
                                mark_high_res_failed=True,
                            )
                        except Exception:
                            logger.exception(f"[MultiResSimulator::HighRes] Failed to mark {hr_event_id} as failed")
                        return (eid, None, all_pids)

                detail_results = await asyncio.gather(
                    *[_run_one_detail(t) for t in p3_tasks],
                    return_exceptions=True,
                )
                for r in detail_results:
                    if not isinstance(r, Exception):
                        simulated_high_res_count += 1

                # ── Unified UPER: all events in period after all details complete ──
                logger.info("[MemorySystem::EndOfPeriod] Running end-of-period memory write (UPER)...")
                try:
                    await self._run_period_uper(
                        lr_bundles=lr_bundles,
                        all_outlines=all_outlines,
                        detail_event_ids=detail_event_ids,
                        detail_results=detail_results,
                        p3_tasks=p3_tasks,
                    )
                except Exception:
                    logger.exception("[P2-v6] Unified UPER failed; continuing")

            if stop_after_first_high_res and simulated_high_res_count > 0:
                logger.info("[P2] stop-after-first-high-res triggered; stopping")

        else:
            # No detail simulations — run UPER now for LR + outline-only events
            if lr_bundles or all_outlines:
                logger.info("[MemorySystem::Write] No details; running UPER for LR/outline events...")
                try:
                    await self._run_period_uper(
                        lr_bundles=lr_bundles,
                        all_outlines=all_outlines,
                        detail_event_ids=[],
                        detail_results=[],
                        p3_tasks=[],
                    )
                except Exception:
                    logger.exception("[MemorySystem::Write] UPER for LR/outline events failed; continuing")

        # ── Dispatch UPER for habitual (medium-res) events ──────────────────
        # uper_tasks contains habitual events added during batch LR generation.
        # These are NOT included in _run_period_uper (which only handles LR/outline/detail).
        habitual_uper_tasks = [t for t in uper_tasks if t.get("is_habitual")]
        if habitual_uper_tasks:
            logger.info(
                f"[MemorySystem::Write] Running UPER for {len(habitual_uper_tasks)} habitual events..."
            )
            try:
                await self._uper_reflect_batch_period(event_tasks=habitual_uper_tasks)
                logger.info(
                    f"[MemorySystem::Write] Habitual UPER complete for {len(habitual_uper_tasks)} events"
                )
            except Exception:
                logger.exception(
                    "[MemorySystem::Write] Habitual UPER failed; habitual events will have empty first_person_memory"
                )

        if is_last_segment_of_period:
            await self._rewrite_period_summary(period)
            await self._generate_and_update_life_summary(period)
        else:
            logger.info(
                f"[P2] Skipping period/life summary for {period_id} "
                f"(not last segment; segment={segment_override.get('segment_id', '?') if segment_override else 'N/A'})"
            )

        return {
            "period_id": period_id,
            "time_unit_count": len(time_units),
            "high_res_count": high_res_count,
            "simulated_high_res_count": simulated_high_res_count,
            "committed_event_ids": committed_event_ids,
            "failed_units": failed_units,
        }

    async def process_period(
        self,
        period: Dict[str, Any],
        stop_after_first_high_res: bool = False,
    ) -> List[Dict[str, Any]]:
        """DEPRECATED: Backward-compatible wrapper; prefer process_period_atomic().

        This method still reads default_memory_density from the period dict
        for backward compatibility with old life plans. New code should use
        process_period_atomic() which computes density from AM Model.
        """
        period_id = period.get("period_id", "LP?")
        density = period.get("default_memory_density", "none")
        dr = period.get("period_date_range", {})
        start_str = dr.get("start_date", "")
        end_str = dr.get("end_date", "")
        if not start_str or not end_str:
            return []

        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        unit = DENSITY_TO_UNIT.get(density, "month")
        time_units = _iter_time_units(start, end, unit)
        bundles: List[Dict[str, Any]] = []
        for i, (u_start, u_end) in enumerate(time_units):
            unit_label = f"{unit} {i+1}/{len(time_units)}"
            bundle = await self._process_time_unit(
                period=period,
                unit_start=u_start,
                unit_end=u_end,
                unit_label=unit_label,
                density=density,
            )
            if bundle and bundle.get("high_res_event"):
                bundles.append(bundle)
                if stop_after_first_high_res:
                    break
        return bundles

    # ── Core: Process all periods ────────────────────────────────

    async def process_all_periods(
        self,
        life_periods: List[Dict[str, Any]],
        stop_after_first_high_res: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible wrapper that preserves the old stop-after-first API shape."""
        for period in life_periods:
            period_result = await self.process_period(
                period=period,
                stop_after_first_high_res=stop_after_first_high_res,
            )
            if period_result and stop_after_first_high_res:
                return period_result[0]

        return None


# ================================================================
# Test / Demo Entry Point
# ================================================================

async def _demo_main() -> None:
    """
    Test function: read all input files, generate events using the
    multi-step pipeline, and stop at the first high-resolution event,
    saving it to JSON.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File paths
    memory_base_path = os.path.join(
        base_dir, "simulation_p4_memory_organiser", "memory_base_test.json"
    )
    participant_pool_path = os.path.join(
        base_dir, "simulation_p1_initialisation", "persona_pool.json"
    )
    persona_path = os.path.join(
        base_dir, "simulation_p0_persona_settings", "test_sample.json"
    )
    plan_path = os.path.join(
        base_dir, "simulation_p1_initialisation", "test_plan.json"
    )
    output_path = os.path.join(
        base_dir, "simulation_p2_event_organiser", "high_resolution_event_test.json"
    )

    # ── 1. Load input files ──────────────────────────────────────
    logger.info("Loading input files...")

    with open(persona_path, "r", encoding="utf-8") as f:
        persona_config = json.load(f)
    logger.info(f"  Persona loaded: {persona_config.get('persona_name_text', '?')}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    life_periods = plan.get("life_periods", [])
    logger.info(f"  Plan loaded: {len(life_periods)} life periods")

    # ── 2. Initialize LLM client ─────────────────────────────────
    llm_client = AsyncLLMClient(
        default_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_base=os.environ.get("OPENAI_API_BASE", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Load participant pool
    with open(participant_pool_path, "r", encoding="utf-8") as f:
        pool_data = json.load(f)

    pool = ParticipantPoolManager(llm_client=llm_client, save_path=participant_pool_path)
    participants_list = pool_data.get("participants", [])
    if isinstance(participants_list, list):
        for pdata in participants_list:
            p = Participant(**pdata)
            pool._participants[p.participant_id] = p
            if p.participant_id.startswith("P_") and p.participant_id != "P_TARGET":
                try:
                    num = int(p.participant_id.split("_")[1])
                    if num >= pool._next_id_counter:
                        pool._next_id_counter = num + 1
                except ValueError:
                    pass
    elif isinstance(participants_list, dict):
        for pid, pdata in participants_list.items():
            p = Participant(**pdata)
            pool._participants[pid] = p
    logger.info(f"  Participant pool loaded: {pool.count()} participants")

    # Load memory base
    memory = MemoryManager(memory_base_path=memory_base_path)
    logger.info(
        f"  Memory base loaded: {memory.period_count} periods, "
        f"{memory.event_count} events"
    )

    # ── 3. Create event organiser and run ─────────────────────────
    organiser = EventOrganiser(
        llm_client=llm_client,
        memory_manager=memory,
        participant_pool=pool,
        target_persona=persona_config,
        participant_pool_path=participant_pool_path,
    )

    logger.info("Starting multi-step event generation (will stop at first high-res event)...")
    first_hr_event = await organiser.process_all_periods(
        life_periods=life_periods,
        stop_after_first_high_res=True,
    )

    # ── 4. Save outputs ──────────────────────────────────────────
    if first_hr_event:
        logger.info(f"First high-res event generated: {first_hr_event['event_id']}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(first_hr_event, f, ensure_ascii=False, indent=2)
        logger.info(f"High-res event saved to: {output_path}")
    else:
        logger.warning("No high-resolution event was generated across all periods")

    # Save updated memory base
    memory.save(memory_base_path)
    logger.info(f"Memory base saved to: {memory_base_path}")

    # Final save of participant pool (also saved in real-time during simulation)
    pool.save(participant_pool_path)
    logger.info(f"Participant pool saved to: {participant_pool_path}")

    # Print summary
    logger.info(
        f"\nFinal state: "
        f"{memory.period_count} periods, "
        f"{memory.event_count} events, "
        f"{pool.count()} participants"
    )


if __name__ == "__main__":
    asyncio.run(_demo_main())
