"""
Participant Pool Manager — Initialization & Lifecycle Management
================================================================
Manages the pool of participants (target persona + supporting characters)
for the lifelong simulation system.

This module belongs to the P1 Initialisation layer and is responsible for:
  - Target persona registration and temporal briefs generation:
    Step 1: Register target persona from persona config
            Birth date resolved from life_plan.derived_birth_date (preferred)
            or persona_config.birth_date (fallback)
    Step 1.5: Build structured "current status + future goal" context for target
              (no LLM call — purely factual description based on config & life_plan)
    Step 1.6: Extend target profile with simulation-start attributes via LLM
              (current_living_location, occupation, education, etc.)
  - Six-phase LLM-driven supporting character initialization:
    Phase 1 (Step 2): Basic identity (name, DOB, gender, role, relationship, brief)
                       Only characters at least 10 years old at target's birth are generated
    Phase 2 (Step 3): Three temporal persona briefs (initial / current / target)
    Phase 3 (Step 4): Full profile extension (location, occupation, education, etc.)
    Phase 5 (Step 5): Timeline consistency check & auto-correction
    Phase 6 (Step 6): Generate target's initial_persona_brief_text for the first time
                       Uses supporting characters' initial states for full coherence
  - Maintaining participant profiles aligned with PersonaInputSchema
  - Tracking interaction history between participants and the main character
  - Dynamically updating current_persona_brief_text based on accumulated interactions
  - Serialization / deserialization for downstream consumption and persistence

Field naming follows `definiton.py` conventions (PersonaInputSchema).
"""

import os
import re
import json
import logging
import uuid
import asyncio
from contextlib import nullcontext
from typing import Awaitable, Callable, Dict, List, Optional, Any

from lifelong_synth.performance_tracker import PerformanceTracker
from llm.client import AsyncLLMClient
from lifelong_synth.simulation_p1_initialisation.definition import (
    # Participant Pool data models
    DateOfBirth,
    LocationInfo,
    SelfSystemAnchor,
    InteractionRecord,
    Participant,
    # LLM structured output models for multi-phase initialization
    Phase1BasicInfo,
    Phase1ParticipantList,
    Phase2TemporalBriefs,
    Phase3FullProfile,
    Phase5ConsistencyResult,
    Phase6TargetInitialBrief,
)

logger = logging.getLogger(__name__)


# ================================================================
# Participant Pool Manager
# ================================================================

class ParticipantPoolManager:
    """
    Manages the full lifecycle of participants in a lifelong simulation.

    Responsibilities:
      1. **Register target persona** from persona config
      2. **Build target status context** (no LLM call):
         - Structured "current status + future goal" description
         - Used as context for supporting character generation
         - initial_persona_brief_text deferred to Phase 6
      2b. **Extend target profile** with simulation-start attributes:
         - current_living_location, target_occupation_group,
           target_education_level, self_system_anchor
         - All inferred for the simulation start time, not end time
      3. **Initialize supporting characters** via five-phase LLM calls:
         - Phase 1: Generate basic identity (name, DOB, gender, role, relationship, brief)
         - Phase 2: Generate three temporal persona briefs (initial / current / target)
         - Phase 3: Extend each participant with full profile fields (at simulation start)
         - Phase 4: Generate initial memories in interactions_history (first-person)
         - Phase 5: Timeline consistency check & auto-correction
      4. **Add participants** on-the-fly during simulation
      5. **Update interaction history** after each simulated event
      6. **Update persona text** dynamically based on accumulated interactions
      7. **Serialize / deserialize** the pool for persistence

    The target persona is always registered first with id ``P_TARGET``.
    Supporting characters receive auto-generated ids ``P_001``, ``P_002``, …
    """

    TARGET_ID = "P_TARGET"

    def __init__(
        self,
        llm_client: Optional[AsyncLLMClient] = None,
        save_path: Optional[str] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        institution_naming_policy: str = "real",
    ):
        """
        Args:
            llm_client: Async LLM client for participant generation.
                        Can be ``None`` if only loading from serialized data.
            save_path: Optional file path for auto-persisting the pool to JSON.
                       When set, calling ``save()`` writes the pool to this path.
            institution_naming_policy: Institution naming strategy — 'real' (default),
                        'fictional', or 'anonymized'.
        """
        self.llm = llm_client
        self._participants: Dict[str, Participant] = {}
        self._next_id_counter: int = 1
        self._simulation_start_date: str = ""
        self._simulation_end_date: str = ""
        self._target_status_context: str = ""
        self._save_path: Optional[str] = save_path
        self.institution_naming_policy = institution_naming_policy
        self._performance_tracker = performance_tracker
        # Module 12: Incremental pool serialization — dirty tracking
        self._dirty_participants: set = set()

    # ── Institution Naming Policy ───────────────────────────────

    def _get_institution_naming_prompt(self) -> str:
        """Return institution naming rules based on the configured policy."""
        if self.institution_naming_policy == "fictional":
            return (
                "**Institution naming rules**:\n"
                "- Use fictional names for schools, companies, hospitals, and other institutions (do not use real institution names)\n"
                "- Fictional names should conform to local naming conventions, sounding plausible but not matching real institutions\n"
                "- For workplaces, if the persona already specifies a real company name, replace it with a fictional equivalent"
            )
        elif self.institution_naming_policy == "anonymized":
            return (
                "**Institution naming rules**:\n"
                "- Use anonymized institution names (e.g., 'a top-ranked university', 'a major tech company', 'a large general hospital')\n"
                "- Do not reveal any identifiable real institution information\n"
                "- For workplaces, if the persona already specifies a real company name, anonymize it"
            )
        else:  # "real" (default)
            return (
                "**Institution naming rules**:\n"
                "- Use real names for schools, companies, hospitals, and other institutions\n"
                "- If the persona settings mention a specific city and institution type (e.g., 'a 211 university in Nanning'), "
                "use the corresponding real institution in that city (e.g., 'Guangxi University')\n"
                "- If the specific institution cannot be determined, use the most well-known institution of that type in the region\n"
                "- For workplaces, if the persona already specifies a real company name, use it directly"
            )

    # ── Module 12: Dirty Tracking ────────────────────────────────

    def mark_dirty(self, participant_id: str) -> None:
        """Mark a participant as modified for incremental serialization."""
        self._dirty_participants.add(participant_id)

    def save_incremental(self, path: Optional[str] = None) -> None:
        """Save only modified participants (incremental update).

        Falls back to full save if the file doesn't exist yet.
        """
        target_path = path or self._save_path
        if not target_path:
            return
        if not self._dirty_participants:
            logger.debug("save_incremental: no dirty participants, skipping")
            return

        if not os.path.exists(target_path):
            # First save — do a full save
            self.save(target_path)
            self._dirty_participants.clear()
            return

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.save(target_path)
            self._dirty_participants.clear()
            return

        # Build index of existing participants
        participants_list = data.get("participants", [])
        existing_map = {}
        for i, p in enumerate(participants_list):
            pid = p.get("participant_id", "")
            if pid:
                existing_map[pid] = i

        # Update only dirty participants
        for pid in self._dirty_participants:
            p = self._participants.get(pid)
            if p is None:
                continue
            p_data = p.model_dump(mode="json", exclude_none=False)
            if pid in existing_map:
                participants_list[existing_map[pid]] = p_data
            else:
                participants_list.append(p_data)

        data["participants"] = participants_list
        data["total_count"] = len(participants_list)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(
            f"save_incremental: updated {len(self._dirty_participants)} participants"
        )
        self._dirty_participants.clear()

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """
        Persist the participant pool to a JSON file.

        Args:
            path: File path to write to. If None, uses the ``save_path``
                  provided at construction time. If neither is set, this
                  method is a no-op (with a debug log).
        """
        target_path = path or self._save_path
        if not target_path:
            logger.debug("ParticipantPoolManager.save() called but no save_path configured; skipping.")
            return

        def _write() -> None:
            pool_data = {
                "simulation_start_date": self._simulation_start_date,
                "simulation_end_date": self._simulation_end_date,
                "total_count": self.count(),
                "participants": [
                    p.model_dump(mode="json", exclude_none=False)
                    for p in self.list_all()
                ],
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(pool_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Participant pool persisted to {target_path} ({self.count()} participants)")

        tracker = self._performance_tracker
        if tracker:
            with tracker.track("participant.save_pool", file_path=target_path, participant_count=self.count()):
                _write()
            return
        _write()

    # ── Private helpers ──────────────────────────────────────────

    def _generate_participant_id(self) -> str:
        """Generate the next auto-incremented participant ID."""
        pid = f"P_{self._next_id_counter:03d}"
        self._next_id_counter += 1
        return pid

    def _track_perf(self, name: str, **tags: Any):
        tracker = self._performance_tracker
        if tracker:
            return tracker.track(name, **tags)
        return nullcontext()

    async def _run_participant_phase_parallel(
        self,
        phase_name: str,
        participants: List[Participant],
        worker: Callable[[Participant], Awaitable[None]],
        max_concurrency: int,
    ) -> List[str]:
        if not participants:
            logger.info(f"{phase_name}: no participants to process")
            return []

        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        failed_ids: List[str] = []

        async def _run_single(participant: Participant) -> None:
            async with semaphore:
                try:
                    with self._track_perf(
                        "participant.task",
                        phase=phase_name,
                        participant_id=participant.participant_id,
                        participant_name=participant.persona_name_text,
                    ):
                        await worker(participant)
                except Exception as e:
                    failed_ids.append(participant.participant_id)
                    logger.error(
                        f"{phase_name} failed for {participant.participant_id} | "
                        f"{participant.persona_name_text}: {e}"
                    )

        with self._track_perf(
            "participant.phase",
            phase=phase_name,
            participant_count=len(participants),
            max_concurrency=max(1, max_concurrency),
        ):
            await asyncio.gather(*[_run_single(participant) for participant in participants])

        success_count = len(participants) - len(failed_ids)
        logger.info(
            f"{phase_name} complete: success={success_count}, failed={len(failed_ids)}"
        )
        if failed_ids:
            logger.info(f"{phase_name} failed participants: {failed_ids}")
        return failed_ids

    def _infer_role_from_relationship(self, relationship_text: str) -> str:
        """
        Heuristically infer a role tag from the relationship description.
        Falls back to 'acquaintance' if no keyword matches.
        """
        text = relationship_text.lower()
        family_keywords = [
            "parent", "father", "mother",
            "brother", "sister", "sibling",
            "grandparent", "grandfather", "grandmother",
            "uncle", "aunt",
            "spouse", "wife", "husband", "partner",
        ]
        friend_keywords = ["friend", "buddy", "pal", "bestie", "mate"]
        colleague_keywords = ["colleague", "coworker", "classmate", "schoolmate"]
        mentor_keywords = ["mentor", "teacher", "professor", "advisor", "instructor"]
        romantic_keywords = ["girlfriend", "boyfriend", "partner", "significant other", "lover"]

        for kw in family_keywords:
            if kw in text:
                return "family"
        for kw in romantic_keywords:
            if kw in text:
                return "romantic_partner"
        for kw in mentor_keywords:
            if kw in text:
                return "mentor"
        for kw in colleague_keywords:
            if kw in text:
                return "colleague"
        for kw in friend_keywords:
            if kw in text:
                return "friend"
        return "acquaintance"

    # ── 1. Participant Initialization ────────────────────────────

    def register_target_persona(
        self,
        persona_config: Dict[str, Any],
        life_plan: Optional[Dict[str, Any]] = None,
    ) -> Participant:
        """
        Register the main character (target persona) from the persona config.

        This is always called first before any supporting characters are created.

        The target's date_of_birth is resolved in the following priority order:
          1. ``life_plan.global_summary.timeline_anchor.derived_birth_date``
          2. ``persona_config["birth_date"]``
          3. ``None`` (if neither source provides a birth date)

        Args:
            persona_config: The full persona constraint sheet (test_sample.json).
            life_plan: The generated life-period plan (optional but recommended).

        Returns:
            The registered target Participant.
        """
        # ── Resolve birth date: prefer life_plan derived_birth_date ──
        birth_date_str: Optional[str] = None
        if life_plan:
            timeline_anchor = life_plan.get("global_summary", {}).get("timeline_anchor", {})
            birth_date_str = timeline_anchor.get("derived_birth_date") or None
            # Also cache simulation time range
            self._simulation_start_date = timeline_anchor.get("simulation_start_date", "")
            self._simulation_end_date = timeline_anchor.get("simulation_end_date", "")
        if not birth_date_str:
            birth_date_str = persona_config.get("birth_date") or None

        target_dob: Optional[DateOfBirth] = None
        if birth_date_str:
            try:
                parts = birth_date_str.split("-")
                target_dob = DateOfBirth(
                    year=int(parts[0]),
                    month=int(parts[1]),
                    day=int(parts[2]),
                )
            except (IndexError, ValueError) as e:
                logger.error(f"Failed to parse birth_date '{birth_date_str}': {e} — target DOB will be None, all timeline calculations will be incorrect")

        # Build location objects
        growing_up_loc = None
        if "growing_up_location" in persona_config:
            growing_up_loc = LocationInfo(**persona_config["growing_up_location"])

        current_loc = None
        if "current_living_location" in persona_config:
            current_loc = LocationInfo(**persona_config["current_living_location"])

        # Build self-system anchor
        anchor = None
        if "self_system_anchor" in persona_config:
            anchor = SelfSystemAnchor(**persona_config["self_system_anchor"])

        # NOTE: persona_config (test_sample.json) contains the TARGET END-STATE
        # information.  Fields that are time-invariant (childhood_*, growing_up_*,
        # gender, language, caregiver_bond, etc.) can be used directly.
        # Fields that may differ between simulation start and end (e.g.
        # current_living_location, target_occupation_group, target_education_level,
        # adult_attachment_rq4cat) are stored here as the TARGET end-state values.
        # The actual simulation-start values will be generated later by
        # _extend_target_profile() called from generate_target_temporal_briefs().
        target = Participant(
            participant_id=self.TARGET_ID,
            role="target",
            date_of_birth=target_dob,
            relationship_towards_the_main_character="protagonist (target persona)",
            persona_name_text=persona_config.get("persona_name_text", "unnamed"),
            persona_brief_text=persona_config.get("persona_brief_text", ""),
            initial_persona_brief_text="",  # to be generated by generate_target_temporal_briefs
            current_persona_brief_text="",  # to be generated by generate_target_temporal_briefs
            target_persona_brief_text=persona_config.get("persona_brief_text", ""),
            appear_period="LP1",
            # Time-invariant fields — same at simulation start and end
            growing_up_location=growing_up_loc,
            childhood_primary_residential_context=persona_config.get("childhood_primary_residential_context"),
            primary_language="en",  # Hardcoded for English-only simulation
            working_language=["en"],  # Hardcoded for English-only simulation
            childhood_living_arrangement=persona_config.get("childhood_living_arrangement"),  # [SENSITIVE] from user input only
            gender_identity_code=persona_config.get("gender_identity_code"),
            caregiver_bond_pbi=persona_config.get("caregiver_bond_pbi"),  # [SENSITIVE] from user input only
            childhood_adversity_aceiq_13=persona_config.get("childhood_adversity_aceiq_13"),  # [SENSITIVE] from user input only
            # Time-variant fields — set to None initially; will be populated
            # by _extend_target_profile() with simulation-start values.
            # The target end-state values are preserved in persona_config.
            current_living_location=None,
            target_occupation_group=None,
            target_education_level=None,
            self_system_anchor=None,
            interactions_history_with_the_main_character=[],
            persona_extensions=persona_config.get("persona_extensions"),
        )

        self._participants[self.TARGET_ID] = target
        logger.info(f"Registered target persona: {target.persona_name_text} ({self.TARGET_ID})")
        return target

    async def generate_target_temporal_briefs(
        self,
        persona_config: Dict[str, Any],
        life_plan: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Participant:
        """
        Build structured "current status + future goal" context for the target
        and extend the target profile with simulation-start attributes.

        This step should be executed **before** supporting character generation.
        Instead of generating an LLM-based initial_persona_brief_text (which
        would be inaccurate without supporting character info), it builds a
        factual, structured context string (``_target_status_context``) that
        downstream phases (Phase 1/2/4) can use as context.

        The actual ``initial_persona_brief_text`` will be generated later in
        Phase 6, after all supporting characters are created, ensuring full
        coherence.

        Produces:
          - ``_target_status_context``: structured factual description stored
            on the pool instance, used as context for supporting char generation.
          - ``target_persona_brief_text``: kept identical to the input
            ``persona_brief_text`` (no generation needed).

        Args:
            persona_config: The full persona constraint sheet (test_sample.json).
            life_plan: The generated life-period plan.
            model: Optional LLM model override.

        Returns:
            The updated target Participant.

        Raises:
            RuntimeError: If LLM client is not available.
            KeyError: If target persona has not been registered yet.
        """
        if self.TARGET_ID not in self._participants:
            raise KeyError(
                f"Target persona '{self.TARGET_ID}' not registered. "
                f"Call register_target_persona() first."
            )
        if self.llm is None:
            raise RuntimeError("LLM client is required for target temporal briefs generation")

        target = self._participants[self.TARGET_ID]
        target_name = persona_config.get("persona_name_text", "protagonist")
        target_brief = persona_config.get("persona_brief_text", "")

        # Compute simulation start and end dates — prefer explicit
        # timeline_anchor from life_plan.global_summary, fall back to
        # first/last period date range boundaries.
        all_periods = life_plan.get("life_periods", [])
        timeline_anchor = life_plan.get("global_summary", {}).get("timeline_anchor", {})
        sim_start_date = timeline_anchor.get("simulation_start_date", "")
        sim_end_date = timeline_anchor.get("simulation_end_date", "")
        if not sim_start_date or not sim_end_date:
            if all_periods:
                first_dr = all_periods[0].get("period_date_range", {})
                last_dr = all_periods[-1].get("period_date_range", {})
                sim_start_date = sim_start_date or first_dr.get("start_date", "")
                sim_end_date = sim_end_date or last_dr.get("end_date", "")
        logger.info(
            f"Target temporal briefs: using simulation time range "
            f"{sim_start_date} ~ {sim_end_date} (source: timeline_anchor)"
        )

        # Build period timeline for context
        period_summaries = []
        for p in all_periods:
            dr = p.get("period_date_range", {})
            period_summaries.append(
                f"- {p.get('period_id', '?')}: {p.get('title', '?')} "
                f"({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')})"
            )
        periods_text = "\n".join(period_summaries) if period_summaries else "No life stage information provided"

        # Compute age at simulation start
        age_at_start = ""
        dob_str = ""
        if target.date_of_birth and sim_start_date:
            dob_str = f"{target.date_of_birth.year}-{target.date_of_birth.month:02d}-{target.date_of_birth.day:02d}"
            try:
                from datetime import date as _date
                start_d = _date.fromisoformat(sim_start_date)
                age_at_start = str(start_d.year - target.date_of_birth.year)
            except Exception:
                pass

        # Build the first period's context for richer description
        first_period = all_periods[0] if all_periods else {}
        first_period_title = first_period.get("title", "")
        first_period_theme = first_period.get("dominant_theme", "")
        first_period_tasks = first_period.get("developmental_tasks", [])

        # Check if simulation starts at birth
        sim_starts_at_birth_target = (sim_start_date == dob_str) if sim_start_date and dob_str else False

        # ── Build structured "current status + future goal" context ──
        # Instead of generating an LLM-based initial_persona_brief_text
        # (which would be inaccurate without supporting character info),
        # we build a factual, structured context string that downstream
        # phases can use.  The actual initial_persona_brief_text will be
        # generated in Phase 6 after all supporting characters are created.

        if sim_starts_at_birth_target:
            age_desc = "newborn (age 0)"
        elif age_at_start:
            age_desc = f"approximately {age_at_start} years old"
        else:
            age_desc = "unknown"

        growing_up_loc = persona_config.get("growing_up_location", {})
        loc_desc = f"{growing_up_loc.get('province', '')}{growing_up_loc.get('city', '')}"

        status_context = (
            f"[Protagonist's Current State (at simulation start {sim_start_date})]\n"
            f"- Name: {target_name}\n"
            f"- Date of birth: {dob_str or 'unknown'}\n"
            f"- Age: {age_desc}\n"
            f"- Location: {loc_desc or 'unknown'}\n"
            f"- Current life stage: {first_period_title} ({first_period_theme})\n"
        )
        if sim_starts_at_birth_target:
            status_context += (
                "- Development state: newborn, no autonomous behavioral capacity\n"
                "- Note: protagonist is a newborn at this point, cannot walk, talk, or do anything\n"
            )

        goal_context = (
            f"\n[Protagonist's Future Goals (at simulation end {sim_end_date})]\n"
            f"- Target persona brief: {target_brief}\n"
            f"- Target education level: {persona_config.get('target_education_level', 'unknown')}\n"
            f"- Target occupation group: {persona_config.get('target_occupation_group', 'unknown')}\n"
        )

        timeline_context = (
            f"\n[Life Stage Timeline]\n{periods_text}\n"
        )

        # Store the structured context on the pool instance for downstream use
        self._target_status_context = status_context + goal_context + timeline_context

        # Do NOT set initial_persona_brief_text here — it will be generated
        # in Phase 6 with full supporting character context for coherence.
        # Leave it empty so downstream code knows it's not yet available.
        target.initial_persona_brief_text = ""
        target.current_persona_brief_text = ""
        # target_persona_brief_text stays as persona_brief_text (already set)

        logger.info(
            f"  Target status context built (no LLM call): {target.participant_id} | "
            f"{target.persona_name_text}"
        )
        logger.info(f"    age_at_start: {age_desc}, location: {loc_desc}")
        logger.info(
            f"    initial_persona_brief_text deferred to Phase 6 "
            f"(after supporting characters are created)"
        )

        # ── Extend target profile with simulation-start attributes ──
        # Fields like current_living_location, target_occupation_group,
        # target_education_level etc. may differ between simulation start
        # and end.  Use LLM to infer the correct values at simulation start.
        await self._extend_target_profile(
            persona_config=persona_config,
            life_plan=life_plan,
            sim_start_date=sim_start_date,
            sim_end_date=sim_end_date,
            model=model,
        )

        return target

    async def regenerate_target_initial_brief(
        self,
        persona_config: Dict[str, Any],
        life_plan: Dict[str, Any],
        supporting_participants: List[Participant],
        model: Optional[str] = None,
    ) -> Participant:
        """
        Generate the target's initial_persona_brief_text for the FIRST time,
        AFTER supporting characters have been created, so that the target's
        description is fully coherent with the supporting characters' initial
        states.

        This is the first time initial_persona_brief_text is generated — Step
        1.5 intentionally deferred this to avoid inaccurate descriptions
        (e.g. writing "professor" when the father is still a "lecturer" at
        simulation start).

        This method should be called after Phase 5 (timeline consistency check)
        of supporting character initialization, so that supporting characters'
        initial_persona_brief_text fields are already populated and validated.

        Args:
            persona_config: The full persona constraint sheet.
            life_plan: The generated life-period plan.
            supporting_participants: List of supporting characters with
                initial_persona_brief_text already populated.
            model: Optional LLM model override.

        Returns:
            The updated target Participant.
        """
        if self.llm is None:
            logger.warning("LLM client not available; skipping target initial brief generation")
            return self._participants[self.TARGET_ID]

        target = self._participants.get(self.TARGET_ID)
        if not target:
            raise KeyError("Target persona not registered")

        # Compute simulation start date
        timeline_anchor = life_plan.get("global_summary", {}).get("timeline_anchor", {})
        sim_start_date = timeline_anchor.get("simulation_start_date", self._simulation_start_date)

        # Compute age at simulation start
        age_at_start = ""
        dob_str = ""
        if target.date_of_birth and sim_start_date:
            dob_str = f"{target.date_of_birth.year}-{target.date_of_birth.month:02d}-{target.date_of_birth.day:02d}"
            try:
                from datetime import date as _date
                start_d = _date.fromisoformat(sim_start_date)
                age_at_start = str(start_d.year - target.date_of_birth.year)
            except Exception:
                pass

        # Build supporting characters context
        supporting_info_lines = []
        for sp in supporting_participants:
            sp_initial = sp.initial_persona_brief_text or sp.persona_brief_text
            supporting_info_lines.append(
                f"- {sp.persona_name_text}（{sp.relationship_towards_the_main_character}）: "
                f"{sp_initial}"
            )
        supporting_info = "\n".join(supporting_info_lines)

        first_period = life_plan.get("life_periods", [{}])[0] if life_plan.get("life_periods") else {}

        # Check if simulation starts at birth
        sim_starts_at_birth = (sim_start_date == dob_str) if sim_start_date and dob_str else False

        # Adjust age description for newborn
        age_description = f"approximately {age_at_start} years old" if age_at_start and age_at_start != "0" else "newborn"

        # Determine language instruction (hardcoded to English for English-only simulation)
        lang_instruction = "English"

        system_prompt = (
            "You are a character background description generator for a life simulation."
            "Based on the protagonist's basic information and the initial state of surrounding characters, "
            "generate the protagonist's identity description at the simulation start point (initial_persona_brief_text).\n\n"
            "**Core requirement**: Only describe the protagonist's identity state at the moment the simulation starts. "
            "Any referenced character information (e.g., parents' occupations) must be consistent with the supporting character data."
        )

        user_prompt = f"""## Protagonist Basic Information
- Name: {persona_config.get('persona_name_text', 'Protagonist')}
- Date of birth: {dob_str or 'unknown'}
- Gender: {persona_config.get('gender_identity_code', 'unknown')}
- Residence: {json.dumps(persona_config.get('growing_up_location', {}), ensure_ascii=False)}

## Simulation Start Point
- Date: {sim_start_date}
- Protagonist status: {age_description}
- Current life stage: {first_period.get('title', '')}
{f'- Note: The protagonist has just been born; is a newborn' if sim_starts_at_birth else ''}

## Supporting Characters' Status at Simulation Start
{supporting_info}

## Task
Generate the protagonist's identity description at {sim_start_date} (initial_persona_brief_text).

Requirements:
{f'- The protagonist has just been born; focus on: the family environment at birth, parents\' occupations and status at the time' if sim_starts_at_birth else f'- Include: age ({age_description}), education stage, family environment, living situation'}
- Referenced character information must be consistent with the supporting character data above (e.g., occupations, ages)
- Length: 3-5 sentences, in {lang_instruction}
"""

        logger.info("Phase 6: Generating target initial brief with supporting character context (first time)...")

        try:
            result: Phase6TargetInitialBrief = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Phase6TargetInitialBrief,
                system_prompt=system_prompt,
                task_type="p1b_target_brief",
                temperature=0.0,
            )

            target.initial_persona_brief_text = result.initial_persona_brief_text
            target.current_persona_brief_text = result.initial_persona_brief_text

            logger.info(
                f"  Target initial brief generated with supporting character context.\n"
                f"    RESULT: {target.initial_persona_brief_text[:120]}..."
            )

        except Exception as e:
            logger.warning(
                f"  Target initial brief generation failed: {e}. "
                f"Falling back to structured status context."
            )
            # Use the structured status context as a reasonable fallback
            # instead of persona_brief_text (which describes the END state).
            fallback_text = getattr(self, '_target_status_context', '') or ''
            if fallback_text:
                target.initial_persona_brief_text = fallback_text
                target.current_persona_brief_text = fallback_text
            else:
                # Last resort: leave empty rather than using end-state text
                logger.error(
                    "  No structured status context available for fallback. "
                    "initial_persona_brief_text will remain empty."
                )

        return target

    async def initialize_supporting_characters(
        self,
        persona_config: Dict[str, Any],
        life_plan: Dict[str, Any],
        model: Optional[str] = None,
        max_concurrency: int = 3,
    ) -> List[Participant]:
        """
        Six-phase LLM-driven initialization of supporting characters.

        **Phase 1** — Generate basic identity for each supporting character:
          - date_of_birth, relationship, name, gender, role, brief description
          - Only characters that already exist at the target's birth AND are
            at least 10 years old at that time are generated
          - Post-generation validation: filter out any character born after the target

        **Phase 2** — Generate three temporal persona briefs for each character:
          - initial_persona_brief_text (at simulation start)
          - current_persona_brief_text (initially same as initial)
          - target_persona_brief_text (at simulation end)

        **Phase 3** — Extend each character with full profile fields:
          - location, occupation, education, psychological attributes, etc.

        **Phase 4** — Generate initial interaction memories:
          - First-person memories for each character's interactions_history
          - Covers meaningful interactions before simulation start
          - For supporting characters: memory window starts from mother's
            pregnancy (~9 months before target's birth) to simulation start
          - For the target: if simulation starts at birth, no memories are
            generated (a newborn has no memories)
          - Validates memory dates: must be within the valid time window

        **Phase 5** — Timeline consistency check & auto-correction:
          - Verify target_persona_brief_text matches simulation end time
          - Verify initial_persona_brief_text matches simulation start time
          - Check overall timeline coherence (DOB → start → end)
          - Auto-correct any identified inconsistencies

        **Phase 6** — Generate target's initial_persona_brief_text (first time):
          - Generate the target's initial brief for the first time using
            supporting characters' initial states as context, ensuring full
            coherence (e.g. if father is a lecturer at sim start, target's
            brief will correctly say "lecturer" instead of "professor")

        Note:
            ``generate_target_temporal_briefs()`` should be called before this
            method so that the target's structured status context is available.
        Args:
            persona_config: The full persona constraint sheet.
            life_plan: The generated life-period plan (from DevelopmentAwareLifePeriodPlanner).
            model: Optional LLM model override.

        Returns:
            List of newly created Participant objects.
        """
        if self.llm is None:
            raise RuntimeError("LLM client is required for participant initialization")

        target_name = persona_config.get("persona_name_text", "Protagonist")
        target_brief = persona_config.get("persona_brief_text", "")

        # Retrieve target's structured status context (if available)
        target_participant = self._participants.get(self.TARGET_ID)
        target_status_context = getattr(self, '_target_status_context', '')

        # Summarize life periods for context
        period_summaries = []
        for p in life_plan.get("life_periods", []):
            dr = p.get("period_date_range", {})
            period_summaries.append(
                f"- {p.get('period_id', '?')}: {p.get('title', '?')} "
                f"({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')}), "
                f"theme: {p.get('dominant_theme', '?')}"
            )
        periods_text = "\n".join(period_summaries)

        # Compute simulation start and end dates from life_plan (needed by all phases)
        # Prefer explicit timeline_anchor from global_summary; fall back to
        # first/last period date range boundaries.
        all_periods = life_plan.get("life_periods", [])
        timeline_anchor = life_plan.get("global_summary", {}).get("timeline_anchor", {})
        sim_start_date = timeline_anchor.get("simulation_start_date", "")
        sim_end_date = timeline_anchor.get("simulation_end_date", "")
        if not sim_start_date or not sim_end_date:
            if all_periods:
                first_dr = all_periods[0].get("period_date_range", {})
                last_dr = all_periods[-1].get("period_date_range", {})
                sim_start_date = sim_start_date or first_dr.get("start_date", "")
                sim_end_date = sim_end_date or last_dr.get("end_date", "")
        logger.info(
            f"Supporting characters init: using simulation time range "
            f"{sim_start_date} ~ {sim_end_date} (source: timeline_anchor)"
        )

        # Compute target's birth date string (needed by Phase 1 for existence validation)
        target_birth_date = ""
        target_participant_for_dob = self._participants.get(self.TARGET_ID)
        if target_participant_for_dob and target_participant_for_dob.date_of_birth:
            dob = target_participant_for_dob.date_of_birth
            target_birth_date = f"{dob.year}-{dob.month:02d}-{dob.day:02d}"

        # ── Phase 1: Basic identity generation ───────────────────
        # Extract social_context from life_plan for temporal reasoning
        social_context_data = life_plan.get("global_summary", {}).get("social_context", {})
        social_context_text_for_p1 = ""
        if social_context_data:
            sc_lines = []
            for edu_stage in social_context_data.get("education_system", []):
                sc_lines.append(
                    f"  - {edu_stage.get('stage_name', '?')}: "
                    f"entry age ~{edu_stage.get('typical_entry_age', '?')}, "
                    f"duration ~{edu_stage.get('typical_duration_years', '?')} years"
                )
            cpn = social_context_data.get("career_path_norms", {})
            if cpn:
                sc_lines.append(
                    f"  - Career entry: age {cpn.get('typical_entry_age_min', '?')}-"
                    f"{cpn.get('typical_entry_age_max', '?')}"
                )
            if sc_lines:
                social_context_text_for_p1 = "\n".join(sc_lines)

        phase1_system_prompt = (
            "You are a life simulation character designer."
            "Based on the protagonist's persona information and life-stage plan, "
            "generate a set of supporting characters who already existed at the time of the protagonist's birth.\n\n"
            "**Age constraint**: All supporting characters must be at least 10 years old at the protagonist's birth "
            "(i.e., their date of birth must be at least 10 years before the protagonist's). "
            "This means you can only generate characters who were teenagers or adults when the protagonist was born, "
            "including parents, grandparents, older relatives, parents' friends/colleagues, etc.\n"
            "**Maximum age constraint**: No supporting character should be older than 95 years at the simulation end date. "
            "If a character (e.g., a great-grandparent) would be older than 95 at simulation end, either omit them "
            "or replace them with a younger relative of the same generation (e.g., a grandparent instead of a great-grandparent).\n"
            "Do NOT generate classmates, colleagues, spouses, or children — these appear later in the protagonist's life.\n"
            "Each character should be realistic and culturally consistent with the protagonist's upbringing.\n\n"
            "**Crucial — terminal state vs. initial state distinction**:\n"
            "The protagonist's persona_brief_text describes their state at the **end** of the simulation (i.e., the target terminal state), "
            "and any supporting character titles, ranks, or positions mentioned therein are also **terminal-state** descriptions.\n"
            "When generating a supporting character's persona_brief_text, you must describe that character's **complete career trajectory overview**, "
            "not just the terminal state. Specifically:\n"
            "- If the protagonist's terminal description mentions parents as 'professors' or 'full professors', that is their title at the end of the simulation, "
            "not necessarily their title when the protagonist was born.\n"
            "- A supporting character's persona_brief_text should include their complete career trajectory overview, "
            "e.g., 'started as a lecturer and gradually advanced to professor through years of research', rather than simply 'currently a professor'.\n"
            "- For all supporting characters, infer their reasonable occupational/academic state at the protagonist's birth "
            "based on their birth decade, educational background, and industry.\n\n" +
            self._get_institution_naming_prompt()
        )

        # Build target status context section for Phase 1 prompt
        target_context_section = ""
        if target_status_context:
            target_context_section = f"\n## Protagonist's Current Status and Future Goals\n{target_status_context}\n"

        phase1_user_prompt = f"""Protagonist Information:
- Name: {target_name}
- Date of birth: {target_birth_date or 'unknown'}
- Target persona summary: {target_brief}
{target_context_section}- Upbringing location: {json.dumps(persona_config.get('growing_up_location', {}), ensure_ascii=False)}
- Current residence: {json.dumps(persona_config.get('current_living_location', {}), ensure_ascii=False)}
- Education level: {persona_config.get('target_education_level', 'unknown')}
- Occupation group: {persona_config.get('target_occupation_group', 'unknown')}
- Gender: {persona_config.get('gender_identity_code', 'unknown')}

Life-stage plan:
{periods_text}

Based on the above information, generate 5-10 supporting characters who already existed at the time of the protagonist's birth.

**Core constraint — only generate characters who already existed at the protagonist's birth and were at least 10 years old**:
- All supporting characters must be born at least 10 years before the protagonist's birth date ({target_birth_date}), i.e., birth date ≤ {target_birth_date} minus 10 years
- This means all supporting characters must be at least teenagers or adults when the protagonist is born
- Allowed character types: parents, grandparents (paternal and maternal), older relatives (uncles, aunts, etc.), parents' friends/colleagues,
  older neighbors, etc.
- **Strictly exclude** the following types of characters (they do not yet exist or have no relationship with the protagonist at birth):
  × Classmates (appear after school enrollment)
  × Colleagues (appear after starting work)
  × Spouses/romantic partners (appear in adulthood)
  × Children (appear after childbirth)
  × Friends the protagonist meets while growing up
  × Cousins of similar age to the protagonist (under 10 at protagonist's birth)

For each character, provide:
1. A realistic date of birth (year, month, day) — must be strictly at least 10 years before the protagonist's birth date {target_birth_date}, AND the character must be no older than 95 years at the simulation end date ({sim_end_date})
2. Relationship to the protagonist (described in natural language, e.g., "mother", "maternal grandfather", "father's colleague", "neighbor")
3. A culturally appropriate full name
4. Gender identity code (1_male / 2_female / 3_non_binary / 4_other / Z_not_stated)
5. Role type (family / friend / colleague / mentor / acquaintance / stranger / romantic_partner)
6. A brief persona overview (one paragraph covering personality, background, and role in the protagonist's life)
   **Important**: When the overview mentions occupations/titles, it must describe the complete career trajectory overview,
   not just the terminal title. For example, if someone ultimately becomes a professor, write "started as a lecturer and
   gradually advanced to professor through years of research", rather than simply "currently a professor".
   Note: The supporting character titles in the protagonist's target persona summary reflect the **end-of-simulation** terminal state,
   not the state at the simulation start.
{f"""
## Social Background Reference (for inferring reasonable career trajectories of supporting characters)
{social_context_text_for_p1}
""" if social_context_text_for_p1 else ""}
Ensure diversity:
- Cover core family members (parents must be included) and extended family members (grandparents, aunts/uncles, etc.)
- May include figures from the parents' social circle who existed at the protagonist's birth (e.g., neighbors, parents' colleagues)
- Reasonable age range (all characters must be at least 10 years old at the protagonist's birth)
"""

        logger.info("Phase 1: Generating basic identity for supporting characters...")
        phase1_result: Phase1ParticipantList = await self.llm.generate_structured(
            prompt=phase1_user_prompt,
            response_model=Phase1ParticipantList,
            system_prompt=phase1_system_prompt,
            task_type="p1b_basic_identity",
            temperature=0.0,
        )

        new_participants: List[Participant] = []

        for basic_info in phase1_result.participants:
            pid = self._generate_participant_id()
            role = basic_info.role or self._infer_role_from_relationship(
                basic_info.relationship_towards_the_main_character
            )

            participant = Participant(
                participant_id=pid,
                role=role,
                date_of_birth=basic_info.date_of_birth,
                relationship_towards_the_main_character=basic_info.relationship_towards_the_main_character,
                persona_name_text=basic_info.persona_name_text,
                persona_brief_text=basic_info.persona_brief_text,
                gender_identity_code=basic_info.gender_identity_code,
                interactions_history_with_the_main_character=[],
                appear_period="LP0",  # T22: Phase 1 characters exist at protagonist's birth
            )
            self._participants[pid] = participant
            new_participants.append(participant)
            logger.info(
                f"  Phase 1 registered: {pid} | {participant.persona_name_text} "
                f"({role}) — {basic_info.relationship_towards_the_main_character}"
            )

        # ── Phase 1.5: Birth-existence & minimum-age validation ────
        # Verify that every supporting character was born at least 10 years
        # before the target.  When simulation starts at birth, we need
        # characters who are old enough to meaningfully interact.
        MIN_AGE_AT_TARGET_BIRTH = 10
        if target_birth_date:
            from datetime import date as _date
            try:
                target_dob_date = _date.fromisoformat(target_birth_date)
            except ValueError:
                target_dob_date = None

            if target_dob_date:
                # Cutoff: character must be born on or before this date
                min_birth_cutoff = target_dob_date.replace(
                    year=target_dob_date.year - MIN_AGE_AT_TARGET_BIRTH
                )
                validated_participants: List[Participant] = []
                for p in new_participants:
                    if p.date_of_birth:
                        try:
                            p_dob = _date(
                                p.date_of_birth.year,
                                p.date_of_birth.month,
                                p.date_of_birth.day,
                            )
                            if p_dob > min_birth_cutoff:
                                age_at_birth = target_dob_date.year - p_dob.year
                                logger.warning(
                                    f"  Phase 1.5 REMOVED: {p.participant_id} | "
                                    f"{p.persona_name_text} (born {p_dob}, "
                                    f"age {age_at_birth} at target birth) — "
                                    f"must be at least {MIN_AGE_AT_TARGET_BIRTH} "
                                    f"years old at target birth ({target_birth_date})"
                                )
                                # Remove from pool
                                self._participants.pop(p.participant_id, None)
                                continue
                        except (ValueError, TypeError):
                            pass  # keep if date is unparseable

                        # ── v4: Maximum age check (95 years at sim end) ──
                        if p.date_of_birth:
                            try:
                                p_dob_chk = _date(
                                    p.date_of_birth.year,
                                    p.date_of_birth.month,
                                    p.date_of_birth.day,
                                )
                                MAX_AGE_AT_SIM_END = 95
                                sim_end_dt = _date.fromisoformat(sim_end_date) if sim_end_date else _date.today()
                                age_at_sim_end = sim_end_dt.year - p_dob_chk.year
                                if age_at_sim_end > MAX_AGE_AT_SIM_END:
                                    logger.warning(
                                        f"  Phase 1.5 AGE WARNING: {p.participant_id} | "
                                        f"{p.persona_name_text} (born {p_dob_chk}, "
                                        f"age {age_at_sim_end} at sim end {sim_end_date}) — "
                                        f"exceeds max age {MAX_AGE_AT_SIM_END}. "
                                        f"Consider replacing with a younger relative."
                                    )
                                    # Non-fatal: log warning but do not remove
                            except (ValueError, TypeError):
                                pass
                    validated_participants.append(p)

                removed_count = len(new_participants) - len(validated_participants)
                if removed_count > 0:
                    logger.info(
                        f"  Phase 1.5: Removed {removed_count} character(s) "
                        f"not at least {MIN_AGE_AT_TARGET_BIRTH} years old "
                        f"at target's birth date ({target_birth_date})"
                    )
                new_participants = validated_participants

        # ── Phase 1.6: Key Relationship Audit ────────────────────
        logger.info("Phase 1.6: Auditing key relationship completeness...")
        audit_added = await self._step1_5_audit_key_relationships(
            target=target_participant,
            persona_config=persona_config,
            life_plan=life_plan,
            sim_start_date=sim_start_date,
            sim_end_date=sim_end_date,
        )
        if audit_added:
            new_participants.extend(audit_added)
            logger.info(f"  Phase 1.6: Added {len(audit_added)} new participant(s) to pipeline.")

        # ── Phase 2: Three temporal persona briefs ───────────
        logger.info("Phase 2: Generating temporal persona briefs (initial / current / target)...")
        phase2_failed = await self._run_participant_phase_parallel(
            phase_name="Phase 2",
            participants=new_participants,
            worker=lambda participant: self._generate_temporal_briefs(
                participant=participant,
                persona_config=persona_config,
                life_plan=life_plan,
                sim_start_date=sim_start_date,
                sim_end_date=sim_end_date,
                model=model,
            ),
            max_concurrency=max_concurrency,
        )
        self.save()

        # ── Phase 3: Full profile extension ──────────────────────
        logger.info("Phase 3: Extending participants with full profile fields (at simulation start)...")
        phase3_failed = await self._run_participant_phase_parallel(
            phase_name="Phase 3",
            participants=new_participants,
            worker=lambda participant: self._extend_participant_profile(
                participant=participant,
                persona_config=persona_config,
                life_plan=life_plan,
                sim_start_date=sim_start_date,
                sim_end_date=sim_end_date,
                model=model,
            ),
            max_concurrency=max_concurrency,
        )
        self.save()

        # ── Phase 5: Timeline consistency check & auto-correction ─
        logger.info("Phase 5: Checking timeline consistency & auto-correcting...")
        # Only check supporting characters here — the target's initial brief
        # has not been generated yet (deferred to Phase 6), so checking it
        # would be meaningless.
        for participant in new_participants:
            await self._check_timeline_consistency(
                participant=participant,
                persona_config=persona_config,
                life_plan=life_plan,
                sim_start_date=sim_start_date,
                sim_end_date=sim_end_date,
                model=model,
            )

        # ── Phase 6: Generate target initial brief with supporting context ─
        # Now that all supporting characters have their initial_persona_brief_text
        # populated, generate the target's initial brief for the FIRST time.
        # Step 1.5 intentionally deferred this to avoid inaccurate descriptions.
        logger.info("Phase 6: Generating target initial brief with supporting character context...")
        if target_participant:
            await self.regenerate_target_initial_brief(
                persona_config=persona_config,
                life_plan=life_plan,
                supporting_participants=new_participants,
                model=model,
            )

        logger.info(
            f"Participant pool initialized: {len(self._participants)} total "
            f"({len(new_participants)} supporting + 1 target)"
        )
        return new_participants

    async def _check_timeline_consistency(
        self,
        participant: Participant,
        persona_config: Dict[str, Any],
        life_plan: Optional[Dict[str, Any]] = None,
        sim_start_date: str = "",
        sim_end_date: str = "",
        model: Optional[str] = None,
    ) -> None:
        """
        Phase-5: Check timeline consistency for a single participant and auto-correct.

        Performs three checks:
          1. target_persona_brief_text vs simulation end time consistency
          2. initial_persona_brief_text vs simulation start time consistency
          3. Overall timeline coherence (DOB → start → end)

        If inconsistencies are found, the LLM generates corrected versions
        and the participant is updated in-place.

        Args:
            participant: The participant to check.
            persona_config: The full persona constraint sheet.
            life_plan: The generated life-period plan.
            sim_start_date: Simulation start date string (YYYY-MM-DD).
            sim_end_date: Simulation end date string (YYYY-MM-DD).
            model: Optional LLM model override.
        """
        if self.llm is None:
            logger.warning("LLM client not available; skipping timeline consistency check")
            return

        # Compute participant's age at simulation start and end
        dob_str = ""
        age_at_start = ""
        age_at_end = ""
        if participant.date_of_birth:
            dob_str = (
                f"{participant.date_of_birth.year}-"
                f"{participant.date_of_birth.month:02d}-"
                f"{participant.date_of_birth.day:02d}"
            )
            try:
                from datetime import date as _date
                if sim_start_date:
                    start_d = _date.fromisoformat(sim_start_date)
                    age_at_start = str(start_d.year - participant.date_of_birth.year)
                if sim_end_date:
                    end_d = _date.fromisoformat(sim_end_date)
                    age_at_end = str(end_d.year - participant.date_of_birth.year)
            except Exception:
                pass

        # Build period timeline for context
        period_summaries = []
        if life_plan is not None:
            for p in life_plan.get("life_periods", []):
                dr = p.get("period_date_range", {})
                period_summaries.append(
                    f"- {p.get('period_id', '?')}: {p.get('title', '?')} "
                    f"({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')})"
                )
        periods_text = "\n".join(period_summaries) if period_summaries else "No life-stage information provided"

        is_target = participant.participant_id == self.TARGET_ID

        # Compute target's birth date and age info for coherence checking
        target_participant = self._participants.get(self.TARGET_ID)
        target_birth_date_str_p5 = ""
        if target_participant and target_participant.date_of_birth:
            target_birth_date_str_p5 = (
                f"{target_participant.date_of_birth.year}-"
                f"{target_participant.date_of_birth.month:02d}-"
                f"{target_participant.date_of_birth.day:02d}"
            )
        sim_starts_at_birth_p5 = (
            sim_start_date == target_birth_date_str_p5
            if sim_start_date and target_birth_date_str_p5 else False
        )

        newborn_check_note = ""
        if sim_starts_at_birth_p5 and not is_target:
            newborn_check_note = (
                "\n4. Consistency between memory content and the protagonist's age"
                " (the simulation starts at the protagonist's birth; within the memory time window the protagonist is either unborn or a newborn — "
                "descriptions of the protagonist walking/talking/playing etc. are age-inappropriate and must not appear)\n"
            )

        system_prompt = (
            "You are a timeline consistency checker for a life simulation.\n"
            "Your task is to verify whether a character's background descriptions are consistent with the timeline information, "
            "and to provide corrected descriptions when inconsistencies are found.\n\n"
            "Check the following aspects:\n"
            "1. Whether target_persona_brief_text is consistent with the simulation end point "
            "(are age, occupational status, etc. temporally reasonable?)\n"
            "2. Whether initial_persona_brief_text is consistent with the simulation start point "
            "(are age, education stage, etc. consistent with the date of birth?)\n"
            "3. Whether the complete timeline from date of birth to simulation start to simulation end is coherent "
            "(do the state changes across stages follow natural development patterns?)\n"
            "4. **Temporal reasonableness of titles/ranks** (critically important):\n"
            "   - Does the title/rank in initial_persona_brief_text match the character's age and experience at simulation start?\n"
            "   - Title promotion requires time accumulation; terminal-state titles cannot be directly used for the initial state.\n"
            "   - For example: a 33-year-old university teacher cannot be a 'full professor' (typically requires 50+ years of age),\n"
            "     a 30-year-old bank employee cannot be a 'branch manager', a 25-year-old doctor cannot be a 'chief physician'.\n"
            "   - If a title/rank does not match the age, it must be corrected to a reasonable title for that age.\n"
            + newborn_check_note +
            "\nIf everything is consistent, set has_issues=false and return an empty issues_found.\n"
            "If inconsistencies are found, set has_issues=true, list the specific issues, "
            "and provide corrected description text."
        )

        role_label = "Protagonist" if is_target else "Supporting character"

        user_prompt = f"""## {role_label} Information
- Name: {participant.persona_name_text}
- Date of birth: {dob_str or 'unknown'}
- Role type: {participant.role}
- Relationship to protagonist: {participant.relationship_towards_the_main_character}
- Persona overview: {participant.persona_brief_text}

## Simulation Time Range
- Start date: {sim_start_date}
- End date: {sim_end_date}
- Character's age at simulation start: approximately {age_at_start} years old
- Character's age at simulation end: approximately {age_at_end} years old

## Life-stage Timeline
{periods_text}

## Current Background Descriptions (to be checked)

### initial_persona_brief_text (background at simulation start {sim_start_date}):
{participant.initial_persona_brief_text}

### target_persona_brief_text (expected background at simulation end {sim_end_date}):
{participant.target_persona_brief_text}

### persona_brief_text (overall persona overview):
{participant.persona_brief_text}

## Protagonist Information (for consistency comparison)
- Protagonist's date of birth: {target_birth_date_str_p5 or 'unknown'}
- Does simulation start at protagonist's birth?: {'Yes (protagonist is a newborn at simulation start)' if sim_starts_at_birth_p5 else 'No'}

## Checking Task

Please check the following three aspects one by one:

**Check 1: Consistency between target description and simulation end time**
- Is the age mentioned in target_persona_brief_text consistent with the age calculated from date of birth ({dob_str}) and simulation end date ({sim_end_date}) (approximately {age_at_end} years old)?
- Are occupational status, educational background, etc. temporally reasonable?
- Are there any unreasonable temporal jumps or logical contradictions?

**Check 2: Consistency between initial description and simulation start time**
- Is the age mentioned in initial_persona_brief_text consistent with the age calculated from date of birth ({dob_str}) and simulation start date ({sim_start_date}) (approximately {age_at_start} years old)?
- Do education stage, work status, etc. match that age?
- **Temporal reasonableness of titles/ranks**: Does the title/rank in initial_persona_brief_text match the character's age and experience at simulation start?
  Title promotion requires time accumulation. For example: from lecturer to associate professor typically takes 5-8 years, from associate professor to professor 8-12 years,
  full professor typically requires 50+ years of age. Corporate managers from entry-level to executive typically take 10-20 years.
  If a title does not match the age (e.g., a 33-year-old 'full professor', a 30-year-old 'branch manager'), it must be corrected to a reasonable title.
- Is the transition from initial state to target state temporally feasible?

**Check 3: Overall timeline coherence**
- Is the complete timeline from date of birth to simulation start to simulation end coherent?
- Do the state changes across stages follow natural development patterns?
- Is the information in persona_brief_text consistent with the timeline?

If any inconsistencies are found, please provide corrected description text. When correcting, preserve the original style and information density; only correct timeline-related inconsistencies.
"""

        try:
            result: Phase5ConsistencyResult = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Phase5ConsistencyResult,
                system_prompt=system_prompt,
                task_type="step3_persona_validation",
                temperature=0.0,
            )

            if result.has_issues:
                logger.info(
                    f"  Phase 5 issues found for {participant.participant_id} | "
                    f"{participant.persona_name_text}:"
                )
                for issue in result.issues_found:
                    logger.info(f"    - {issue}")

                # Apply corrections
                corrections_applied = []
                if result.corrected_initial_persona_brief_text:
                    participant.initial_persona_brief_text = result.corrected_initial_persona_brief_text
                    participant.current_persona_brief_text = result.corrected_initial_persona_brief_text
                    corrections_applied.append("initial_persona_brief_text")

                if result.corrected_target_persona_brief_text:
                    participant.target_persona_brief_text = result.corrected_target_persona_brief_text
                    corrections_applied.append("target_persona_brief_text")

                if result.corrected_persona_brief_text:
                    participant.persona_brief_text = result.corrected_persona_brief_text
                    corrections_applied.append("persona_brief_text")

                if corrections_applied:
                    logger.info(
                        f"    Auto-corrected fields: {', '.join(corrections_applied)}"
                    )
                else:
                    logger.info(
                        f"    Issues identified but no corrections provided by LLM."
                    )
            else:
                logger.info(
                    f"  Phase 5 consistency OK: {participant.participant_id} | "
                    f"{participant.persona_name_text}"
                )

        except Exception as e:
            logger.error(
                f"  Phase 5 consistency check failed for {participant.participant_id}: {e}. "
                f"Skipping consistency check for this participant."
            )

    # _generate_initial_memories removed: interactions_history_with_the_main_character
    # is now always initialized empty; no pre-simulation memories are generated.

    async def _generate_temporal_briefs(
        self,
        participant: Participant,
        persona_config: Dict[str, Any],
        life_plan: Optional[Dict[str, Any]] = None,
        sim_start_date: str = "",
        sim_end_date: str = "",
        model: Optional[str] = None,
    ) -> None:
        """
        Phase-2: Generate three temporal persona briefs for a single participant.

        Produces:
          - initial_persona_brief_text: background at simulation start
          - current_persona_brief_text: initially same as initial
          - target_persona_brief_text: background at simulation end

        Updates the participant object in-place.
        """
        if self.llm is None:
            logger.warning("LLM client not available; skipping temporal briefs generation")
            # Fallback: copy persona_brief_text to all three fields
            participant.initial_persona_brief_text = participant.persona_brief_text
            participant.current_persona_brief_text = participant.persona_brief_text
            participant.target_persona_brief_text = participant.persona_brief_text
            return

        target_name = persona_config.get("persona_name_text", "Protagonist")
        target_brief = persona_config.get("persona_brief_text", "")

        # Retrieve target's structured status context for richer context
        target_participant = self._participants.get(self.TARGET_ID)
        target_status_context = getattr(self, '_target_status_context', '')

        # Build period timeline for context
        period_summaries = []
        if life_plan is not None:
            for p in life_plan.get("life_periods", []):
                dr = p.get("period_date_range", {})
                period_summaries.append(
                    f"- {p.get('period_id', '?')}: {p.get('title', '?')} "
                    f"({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')})"
                )
        periods_text = "\n".join(period_summaries) if period_summaries else "No life-stage information provided"

        # Compute participant's age at simulation start and end
        dob_str = ""
        age_at_start = ""
        age_at_end = ""
        if participant.date_of_birth:
            dob_str = f"{participant.date_of_birth.year}-{participant.date_of_birth.month:02d}-{participant.date_of_birth.day:02d}"
            if sim_start_date:
                try:
                    from datetime import date as _date
                    start_d = _date.fromisoformat(sim_start_date)
                    age_at_start = str(start_d.year - participant.date_of_birth.year)
                except Exception:
                    pass
            if sim_end_date:
                try:
                    from datetime import date as _date
                    end_d = _date.fromisoformat(sim_end_date)
                    age_at_end = str(end_d.year - participant.date_of_birth.year)
                except Exception:
                    pass

        # Determine if simulation starts at birth
        target_birth_date_str = ""
        if target_participant and target_participant.date_of_birth:
            target_birth_date_str = (
                f"{target_participant.date_of_birth.year}-"
                f"{target_participant.date_of_birth.month:02d}-"
                f"{target_participant.date_of_birth.day:02d}"
            )
        sim_starts_at_birth_p2 = (
            sim_start_date == target_birth_date_str
            if sim_start_date and target_birth_date_str else False
        )

        newborn_constraint = ""
        if sim_starts_at_birth_p2:
            newborn_constraint = (
                "\n\n**Critically important constraint**: The simulation starts on the day the protagonist is born, "
                "so at the simulation start point (initial_persona_brief_text), "
                "the protagonist is a newborn (0 years old), unable to walk, talk, or do anything.\n"
                "When describing the relationship with the protagonist in the supporting character's initial_persona_brief_text, "
                "this fact must be reflected. For example:\n"
                "- Father: 'filled with anticipation and love for the newborn son' (not 'accompanying the 3-year-old to study')\n"
                "- Grandmother: 'looking forward to caring for the newborn grandson' (not 'often reading picture books with the grandson')\n"
                "- Neighbor: 'noticing the new little life next door' (not 'helping look after the young child')"
            )

        # Determine language instruction (hardcoded to English for English-only simulation)
        lang_instruction = "English"

        # Build social context section for Phase 2 temporal reasoning
        social_context_data = (life_plan or {}).get("global_summary", {}).get("social_context", {})
        social_context_section_p2 = ""
        if social_context_data:
            sc_p2_lines = ["\n**Social Background Reference (for inferring reasonable occupational/academic states of supporting characters at different time points)**:"]
            for edu_stage in social_context_data.get("education_system", []):
                notes = edu_stage.get('notes', '')
                sc_p2_lines.append(
                    f"  - {edu_stage.get('stage_name', '?')}: "
                    f"entry age ~{edu_stage.get('typical_entry_age', '?')}, "
                    f"duration ~{edu_stage.get('typical_duration_years', '?')} years"
                    + (f" ({notes})" if notes else "")
                )
            cpn = social_context_data.get("career_path_norms", {})
            if cpn:
                sc_p2_lines.append(
                    f"  - Career entry: age {cpn.get('typical_entry_age_min', '?')}-"
                    f"{cpn.get('typical_entry_age_max', '?')}"
                )
                if cpn.get('notes'):
                    sc_p2_lines.append(f"    Notes: {cpn['notes']}")
            for adj in social_context_data.get("persona_specific_adjustments", []):
                sc_p2_lines.append(
                    f"  - [{adj.get('adjustment_type', '?')}] {adj.get('description', '')}"
                )
            social_context_section_p2 = "\n".join(sc_p2_lines)

        phase2_system_prompt = (
            "You are a character background description generator for a life simulation."
            "Based on the supporting character's basic information, the protagonist's background, and the life-stage timeline, "
            "generate two time-dimensioned background descriptions for this supporting character:\n"
            "1. initial_persona_brief_text: the character's background at the simulation start point\n"
            "2. target_persona_brief_text: the character's expected background at the simulation end point\n\n"
            "Each description should include: age at the time, occupational/academic status, educational background, "
            "personality traits, relationship state with the protagonist, etc. "
            "Descriptions should be realistic, culturally appropriate, and reflect changes brought by the passage of time.\n\n"
            "**Critically important — temporal reasonableness of titles/ranks**:\n"
            "A supporting character's persona_brief_text (overall overview) may contain their terminal title (e.g., 'professor', 'director', etc.), "
            "but the initial_persona_brief_text must reflect the character's true occupational state at the simulation **start point**.\n"
            "You must infer the supporting character's reasonable title/rank at simulation start based on the following factors:\n"
            "- The supporting character's date of birth and age at simulation start\n"
            "- The supporting character's educational background and career field\n"
            "- Career promotion norms in that country/region in that era\n"
            "- The typical number of years from entry to reaching the terminal title\n\n"
            "For example: if a person is a 'full professor' at the end of the simulation (2026), but at simulation start (1998) "
            "they are only 33 years old and recently completed their doctorate, then the most reasonable title in 1998 is 'lecturer' or 'associate professor', "
            "not 'full professor'. Title promotion requires time accumulation.\n"
            "Similarly: if a person's terminal state is 'bank president', but they are only 30 at simulation start, "
            "then the initial state should be 'bank clerk' or 'account manager', etc.\n\n"
            "**Critically important — role-context consistency**:\n"
            "The character's role description MUST be consistent with the protagonist's life stage "
            "at the time of introduction (appear_period). For example:\n"
            "- If the character is introduced when the protagonist is in secondary school (Years 9-13), "
            "describe them as a 'secondary school teacher', NOT 'intermediate school teacher'.\n"
            "- If the character is introduced when the protagonist is in primary school, "
            "describe them as a 'primary school teacher'.\n"
            "- If the character is introduced when the protagonist is at university, "
            "describe them as a 'university lecturer/professor' or 'industry professional'.\n"
            "Always cross-check the character's role description against the protagonist's "
            "life-stage timeline provided in the prompt.\n"
            + newborn_constraint
        )

        # Build target status context section for Phase 2 prompt
        target_status_section = ""
        if target_status_context:
            target_status_section = f"\n## Protagonist's Current Status and Future Goals\n{target_status_context}\n"

        # Compute target's age at simulation start for Phase 2 prompt
        target_age_at_start_desc = ""
        if sim_starts_at_birth_p2:
            target_age_at_start_desc = "newborn (0 years old)"
        elif target_participant and target_participant.date_of_birth and sim_start_date:
            try:
                from datetime import date as _date
                _start = _date.fromisoformat(sim_start_date)
                _tage = _start.year - target_participant.date_of_birth.year
                target_age_at_start_desc = f"approximately {_tage} years old"
            except Exception:
                target_age_at_start_desc = "unknown"

        phase2_user_prompt = f"""## Protagonist Information
- Name: {target_name}
- Target persona summary: {target_brief}
- Protagonist's date of birth: {target_birth_date_str or 'unknown'}
- Protagonist's status at simulation start: {target_age_at_start_desc}
{target_status_section}
## Simulation Time Range
- Start date: {sim_start_date}
- End date: {sim_end_date}

## Protagonist's Life-stage Timeline
{periods_text}

## Supporting Character Information (to generate background for)
- Name: {participant.persona_name_text}
- Date of birth: {dob_str or 'unknown'}
- Age at simulation start: approximately {age_at_start} years old (if available)
- Age at simulation end: approximately {age_at_end} years old (if available)
- Gender: {participant.gender_identity_code or 'unknown'}
- Role type: {participant.role}
- Relationship to protagonist: {participant.relationship_towards_the_main_character}
- Overall persona overview: {participant.persona_brief_text}
{social_context_section_p2}

## Task
Generate the following two background descriptions:

1. **initial_persona_brief_text** (background at simulation start {sim_start_date}):
   - Describe the character's state at the simulation start point
   - Include: age at the time, occupational/academic status, educational background, personality traits, relationship state with the protagonist
{f'   - **The protagonist is a newborn (0 years old) at this point; descriptions of the relationship with the protagonist must reflect this fact**' if sim_starts_at_birth_p2 else ''}
   - Length: 2-4 sentences

2. **target_persona_brief_text** (expected background at simulation end {sim_end_date}):
   - Describe the character's expected state at the simulation end point
   - Include: expected age, occupational status, educational background, personality development, relationship evolution with the protagonist
   - Length: 2-4 sentences

Notes:
- Both descriptions should reflect natural changes brought by the passage of time (aging, career development, relationship evolution, etc.)
- Maintain consistency with the protagonist's life-stage timeline
- Write in {lang_instruction}
"""

        try:
            briefs: Phase2TemporalBriefs = await self.llm.generate_structured(
                prompt=phase2_user_prompt,
                response_model=Phase2TemporalBriefs,
                system_prompt=phase2_system_prompt,
                task_type="p1b_temporal_briefs",
                temperature=0.0,
            )

            participant.initial_persona_brief_text = briefs.initial_persona_brief_text
            participant.current_persona_brief_text = briefs.initial_persona_brief_text  # initially same
            participant.target_persona_brief_text = briefs.target_persona_brief_text

            logger.info(
                f"  Phase 2 temporal briefs generated: {participant.participant_id} | "
                f"{participant.persona_name_text}"
            )

        except Exception as e:
            logger.error(
                f"  Phase 2 temporal briefs failed for {participant.participant_id}: {e}. "
                f"Falling back to persona_brief_text for all three fields."
            )
            participant.initial_persona_brief_text = participant.persona_brief_text
            participant.current_persona_brief_text = participant.persona_brief_text
            participant.target_persona_brief_text = participant.persona_brief_text

    async def _extend_participant_profile(
        self,
        participant: Participant,
        persona_config: Dict[str, Any],
        life_plan: Optional[Dict[str, Any]] = None,
        sim_start_date: str = "",
        sim_end_date: str = "",
        model: Optional[str] = None,
    ) -> None:
        """
        Phase-3: Call LLM to fill in extended profile fields for a single participant.

        All generated attributes reflect the participant's state at **simulation
        start time**, not the simulation end time.  Time-invariant attributes
        (childhood_*, growing_up_*, gender, language) are the same at both
        points; time-variant attributes (current_living_location,
        target_occupation_group, target_education_level, etc.) are inferred
        for the simulation start date.

        Args:
            participant: The participant to extend.
            persona_config: The full persona constraint sheet.
            life_plan: The generated life-period plan.
            sim_start_date: Simulation start date string (YYYY-MM-DD).
            sim_end_date: Simulation end date string (YYYY-MM-DD).
            model: Optional LLM model override.

        Updates the participant object in-place.
        """
        if self.llm is None:
            logger.warning("LLM client not available; skipping profile extension")
            return

        phase3_system_prompt = (
            "You are a character profile generator for a life simulation."
            "Based on the supporting character's basic information, infer their gender identity code."
            "Output only the gender_identity_code field."
        )

        phase3_user_prompt = f"""## Supporting Character Information
- Name: {participant.persona_name_text}
- Relationship to protagonist: {participant.relationship_towards_the_main_character}
- Persona overview: {participant.persona_brief_text}

## Task
Based on the character's name and background, infer:
- gender_identity_code: "1_male" | "2_female" | "3_non_binary" | "4_other" | "Z_not_stated"
"""

        try:
            profile: Phase3FullProfile = await self.llm.generate_structured(
                prompt=phase3_user_prompt,
                response_model=Phase3FullProfile,
                system_prompt=phase3_system_prompt,
                task_type="p1b_full_profile",
                temperature=0.0,
            )

            # Only apply gender_identity_code (the only field now in Phase3FullProfile)
            if profile.gender_identity_code is not None:
                participant.gender_identity_code = profile.gender_identity_code

            logger.info(f"  Phase 3 extended: {participant.participant_id} | {participant.persona_name_text}")

        except Exception as e:
            logger.error(
                f"  Phase 3 extension failed for {participant.participant_id}: {e}. "
                f"Participant will retain Phase-1/2 fields only."
            )

    async def _extend_target_profile(
        self,
        persona_config: Dict[str, Any],
        life_plan: Dict[str, Any],
        sim_start_date: str = "",
        sim_end_date: str = "",
        model: Optional[str] = None,
    ) -> None:
        """
        Extend the target persona's profile with simulation-start attributes.

        The persona_config (test_sample.json) contains the TARGET END-STATE
        information.  This method uses LLM to infer what the time-variant
        attributes looked like at the simulation start time.

        Time-invariant fields (childhood_*, growing_up_*, gender, language,
        caregiver_bond) are already set in register_target_persona().

        Time-variant fields that need inference:
          - current_living_location (at simulation start)
          - target_occupation_group (at simulation start)
          - target_education_level (at simulation start)
          - self_system_anchor (at simulation start)

        Note: adult_attachment_rq4cat is set directly from persona_config (not LLM-inferred).

        Updates the target participant object in-place.
        """
        if self.llm is None:
            logger.warning("LLM client not available; falling back to end-state values for target profile")
            target = self._participants.get(self.TARGET_ID)
            if target:
                # Fallback: use end-state values directly
                if "current_living_location" in persona_config:
                    target.current_living_location = LocationInfo(**persona_config["current_living_location"])
                target.target_occupation_group = persona_config.get("target_occupation_group")
                target.target_education_level = persona_config.get("target_education_level")
                if "self_system_anchor" in persona_config:
                    target.self_system_anchor = SelfSystemAnchor(**persona_config["self_system_anchor"])
                target.adult_attachment_rq4cat = persona_config.get("adult_attachment_rq4cat")
            return

        target = self._participants.get(self.TARGET_ID)
        if not target:
            logger.error("Target persona not registered; skipping profile extension — this indicates a call-order bug")
            return

        target_name = persona_config.get("persona_name_text", "Protagonist")

        # Compute age at simulation start
        age_at_start = ""
        dob_str = ""
        if target.date_of_birth and sim_start_date:
            dob_str = f"{target.date_of_birth.year}-{target.date_of_birth.month:02d}-{target.date_of_birth.day:02d}"
            try:
                from datetime import date as _date
                start_d = _date.fromisoformat(sim_start_date)
                age_at_start = str(start_d.year - target.date_of_birth.year)
            except Exception:
                pass

        # Build period timeline for context
        period_summaries = []
        if life_plan is not None:
            for p in life_plan.get("life_periods", []):
                dr = p.get("period_date_range", {})
                period_summaries.append(
                    f"- {p.get('period_id', '?')}: {p.get('title', '?')} "
                    f"({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')})"
                )
        periods_text = "\n".join(period_summaries) if period_summaries else "No life-stage information provided"

        system_prompt = (
            "You are a character profile generator for a life simulation."
            "Your task is to infer the protagonist's attribute states at the simulation start point, "
            "based on the protagonist's target terminal persona information and life-stage timeline.\n\n"
            "**Important**: All generated attributes must reflect the protagonist's state at the simulation start point, "
            "not the target state at the simulation end.\n"
            "For example, if the simulation starts at birth, the protagonist is still an infant at simulation start, "
            "with no occupation (should be null or not applicable) and education level 0 (pre-primary).\n"
            "All field values must be selected from the provided allowed value sets."
        )

        user_prompt = f"""## Time Reference
- Simulation start date: {sim_start_date}
- Simulation end date: {sim_end_date}
- Protagonist's age at simulation start: approximately {age_at_start} years old
- **All attributes must reflect the protagonist's state at the simulation start point ({sim_start_date})**

## Protagonist Target Terminal Information (state at simulation end, for reference only)
- Name: {target_name}
- Target persona summary: {persona_config.get('persona_brief_text', '')}
- Date of birth: {dob_str or 'unknown'}
- Gender: {persona_config.get('gender_identity_code', 'unknown')}
- Upbringing location: {json.dumps(persona_config.get('growing_up_location', {}), ensure_ascii=False)}
- Target residence (at simulation end): {json.dumps(persona_config.get('current_living_location', {}), ensure_ascii=False)}
- Target occupation group (at simulation end): {persona_config.get('target_occupation_group', 'unknown')}
- Target education level (at simulation end): {persona_config.get('target_education_level', 'unknown')}
- Target values (at simulation end): {json.dumps(persona_config.get('self_system_anchor', {}), ensure_ascii=False)}
- Target adult attachment type (at simulation end): {persona_config.get('adult_attachment_rq4cat') or 'not specified'}

## Protagonist's Current Status and Future Goals
{getattr(self, '_target_status_context', 'not generated')}

## Protagonist's Life-stage Timeline
{periods_text}

## Task
Based on the protagonist's persona brief and background, infer:

1. gender_identity_code: Select one from ["1_male", "2_female", "3_non_binary", "4_other", "Z_not_stated"]
   - Note: All other simulation-start attributes (location, occupation, education, values) have been
     pre-determined in the persona config and do not need to be inferred here.

Notes:
- Only output gender_identity_code. Do not infer any other attributes.
"""

        try:
            profile: Phase3FullProfile = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=Phase3FullProfile,
                system_prompt=system_prompt,
                task_type="p1b_target_profile",
                temperature=0.0,
            )

            # Apply sim-start attributes from persona_config (pre-computed in P0.5)
            # location
            if persona_config.get("sim_start_living_location"):
                target.current_living_location = LocationInfo(**persona_config["sim_start_living_location"])
            elif target.growing_up_location is not None:
                target.current_living_location = target.growing_up_location
            elif persona_config.get("current_living_location"):
                target.current_living_location = LocationInfo(**persona_config["current_living_location"])

            # occupation
            if persona_config.get("sim_start_occupation_group") is not None:
                target.target_occupation_group = persona_config["sim_start_occupation_group"]
            # else: leave as None (infant/toddler at simulation start)

            # education level
            if persona_config.get("sim_start_education_level") is not None:
                target.target_education_level = persona_config["sim_start_education_level"]
            else:
                target.target_education_level = "0"  # default: pre-primary

            # values
            if persona_config.get("sim_start_self_system_anchor"):
                target.self_system_anchor = SelfSystemAnchor(**persona_config["sim_start_self_system_anchor"])
            elif persona_config.get("self_system_anchor"):
                # Fallback: use end-state values if sim_start not available
                target.self_system_anchor = SelfSystemAnchor(**persona_config["self_system_anchor"])

            # adult_attachment_rq4cat: set from user-provided config, not from LLM inference
            if persona_config.get("adult_attachment_rq4cat") is not None:
                target.adult_attachment_rq4cat = persona_config["adult_attachment_rq4cat"]

            logger.info(
                f"  Target profile extended with simulation-start attributes: "
                f"occupation={target.target_occupation_group}, "
                f"education={target.target_education_level}, "
                f"location={target.current_living_location}"
            )

        except Exception as e:
            logger.error(
                f"  Target profile extension failed: {e}. "
                f"Falling back to sim_start_* values from persona_config."
            )
            # Fallback: read sim_start_* from persona_config (same as normal path)
            if persona_config.get("sim_start_living_location"):
                target.current_living_location = LocationInfo(**persona_config["sim_start_living_location"])
            elif "current_living_location" in persona_config:
                target.current_living_location = LocationInfo(**persona_config["current_living_location"])
            target.target_occupation_group = persona_config.get("sim_start_occupation_group")
            target.target_education_level = persona_config.get("sim_start_education_level") or "0"
            if persona_config.get("sim_start_self_system_anchor"):
                target.self_system_anchor = SelfSystemAnchor(**persona_config["sim_start_self_system_anchor"])
            elif persona_config.get("self_system_anchor"):
                target.self_system_anchor = SelfSystemAnchor(**persona_config["self_system_anchor"])
            # adult_attachment_rq4cat: set from user-provided config only
            if persona_config.get("adult_attachment_rq4cat") is not None:
                target.adult_attachment_rq4cat = persona_config["adult_attachment_rq4cat"]

    # ── 2. Add Participant ───────────────────────────────────────
    def _resolve_unique_name(self, name: str) -> str:
        """Ensure the given name is unique within the participant pool.

        If a collision is detected, appends a suffix to disambiguate.
        Returns the (possibly modified) unique name.
        """
        existing_names = {
            p.persona_name_text.strip().lower()
            for p in self._participants.values()
            if p.persona_name_text
        }
        original_name = name.strip()
        if original_name.lower() not in existing_names:
            return original_name

        suffix_candidates = ["Jr.", "II", "R.", "M.", "A.", "B.", "C."]
        for suffix in suffix_candidates:
            candidate = f"{original_name} {suffix}"
            if candidate.lower() not in existing_names:
                logger.warning(
                    f"[ParticipantPool] Name collision: '{original_name}' already exists. "
                    f"Renamed to '{candidate}' to avoid ambiguity."
                )
                return candidate

        for i in range(2, 20):
            candidate = f"{original_name} ({i})"
            if candidate.lower() not in existing_names:
                logger.warning(
                    f"[ParticipantPool] Name collision: '{original_name}' → '{candidate}'"
                )
                return candidate

        return original_name  # Last resort: keep original

    async def add_participant(
        self,
        name: str,
        relationship: str,
        brief: str,
        date_of_birth: Optional[DateOfBirth] = None,
        role: Optional[str] = None,
        appear_period: str = "LP_UNKNOWN",
        extend_profile: bool = True,
        persona_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        sim_start_date: Optional[str] = None,
        sim_end_date: Optional[str] = None,
    ) -> Participant:
        """
        Add a new participant to the pool with given basic information.

        Optionally extends the profile via Phase-3 LLM call if ``extend_profile``
        is True and an LLM client is available.

        For dynamically added participants (e.g. during event simulation),
        the three temporal brief fields are all initialised to the same ``brief``
        value. They can be refined later via ``_generate_temporal_briefs``.

        Args:
            name: Participant's display name.
            relationship: Natural-language relationship description.
            brief: One-paragraph persona description.
            date_of_birth: Optional structured DOB.
            role: Optional explicit role tag; auto-inferred from relationship if None.
            appear_period: period_id when this participant starts to appear (e.g., LP3).
            extend_profile: Whether to call LLM for Phase-3 profile extension.
            persona_config: Main character's persona config (needed for Phase-3).
            model: Optional LLM model override.
            sim_start_date: Simulation start date (YYYY-MM-DD) for Phase-3 profile extension.
            sim_end_date: Simulation end date (YYYY-MM-DD) for target brief generation.

        Returns:
            The newly created Participant.
        """
        pid = self._generate_participant_id()
        # T-A2: Ensure name uniqueness before creating participant
        name = self._resolve_unique_name(name)
        inferred_role = role or self._infer_role_from_relationship(relationship)

        participant = Participant(
            participant_id=pid,
            role=inferred_role,
            date_of_birth=date_of_birth,
            relationship_towards_the_main_character=relationship,
            persona_name_text=name,
            persona_brief_text=brief,
            initial_persona_brief_text=brief,
            current_persona_brief_text=brief,
            target_persona_brief_text=brief,
            appear_period=appear_period,
            interactions_history_with_the_main_character=[],
        )
        self._participants[pid] = participant
        logger.info(f"Added participant: {pid} | {name} ({inferred_role})")

        if extend_profile and self.llm is not None and persona_config is not None:
            try:
                # Use provided sim_start_date; fall back to the pool's cached value
                resolved_sim_start = sim_start_date or self._simulation_start_date or ""
                resolved_sim_end = sim_end_date or self._simulation_end_date or ""
                await self._extend_participant_profile(
                    participant=participant,
                    persona_config=persona_config,
                    life_plan=None,
                    sim_start_date=resolved_sim_start,
                    sim_end_date=resolved_sim_end,
                    model=model,
                )
            except Exception as e:
                logger.warning(
                    f"add_participant: profile extension failed for {pid} ({name}): {e}; "
                    f"participant added with basic profile only"
                )

        # Generate target_persona_brief_text if sim_end_date is provided
        # and the participant has a date_of_birth (needed for age calculation)
        if sim_end_date and date_of_birth and self.llm is not None:
            try:
                from datetime import date as _date
                end_d = _date.fromisoformat(sim_end_date)
                age_at_end = end_d.year - date_of_birth.year
                # Simple LLM call to generate target brief
                target_prompt = (
                    f"Based on the following character description at their introduction time:\n"
                    f"{brief}\n\n"
                    f"Generate a 2-3 sentence description of this character at the simulation "
                    f"end date ({sim_end_date}), when they will be approximately {age_at_end} "
                    f"years old. Reflect natural changes: aging, career/academic progression, "
                    f"relationship evolution with the protagonist. Keep the same personality "
                    f"traits but update age, role, and status appropriately."
                )
                from pydantic import BaseModel as _BaseModel, Field as _Field
                class _TargetBriefOutput(_BaseModel):
                    target_persona_brief_text: str = _Field(
                        ..., description="Character description at simulation end date"
                    )
                result = await self.llm.generate_structured(
                    prompt=target_prompt,
                    response_model=_TargetBriefOutput,
                    system_prompt="You are a character background description generator.",
                    task_type="p2_target_brief_snap",
                    temperature=0.0,
                )
                participant.target_persona_brief_text = result.target_persona_brief_text
                logger.info(f"  Generated target_persona_brief_text for {pid} (age at end: {age_at_end})")
            except Exception as e:
                logger.warning(f"  Failed to generate target_persona_brief_text for {pid}: {e}")

        return participant

    # ── 3. Interaction History Update ────────────────────────────

    def update_interaction_history(
        self,
        participant_id: str,
        event_id: str,
        date: str,
        summary: str,
        emotional_tone: Optional[str] = None,
        time_period: Optional[str] = None,
    ) -> None:
        """
        Append a new interaction record to a participant's history.

        Args:
            participant_id: ID of the participant to update.
            event_id: ID of the event where the interaction occurred.
            date: Representative date string (YYYY-MM-DD) for recency calculations.
                  For period-spanning events, use the END date of the period.
            summary: Brief summary of the interaction.
            emotional_tone: Optional emotional tone descriptor.
            time_period: Accurate time period string. If None, falls back to date.
                         Supports: 'YYYY-MM-DD', 'YYYY-MM-DD to YYYY-MM-DD', 'YYYY-MM'.

        Raises:
            KeyError: If participant_id is not found in the pool.
        """
        if participant_id not in self._participants:
            raise KeyError(f"Participant '{participant_id}' not found in pool")

        record = InteractionRecord(
            event_id=event_id,
            time_period=time_period or date,
            date=date,
            summary=summary,
            emotional_tone=emotional_tone,
        )
        self._participants[participant_id].interactions_history_with_the_main_character.append(record)
        self.mark_dirty(participant_id)
        logger.debug(
            f"Updated interaction history for {participant_id}: "
            f"event={event_id}, date={date}"
        )

    # ── 4. Persona Text Update ───────────────────────────────────

    async def update_persona_brief(
        self,
        participant_id: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Dynamically update a participant's current_persona_brief_text based on their
        accumulated interaction history with the main character.

        Uses LLM to synthesize a new brief that reflects the evolved relationship.
        Only updates ``current_persona_brief_text``; ``initial_persona_brief_text``
        and ``target_persona_brief_text`` remain unchanged.

        Args:
            participant_id: ID of the participant to update.
            model: Optional LLM model override.

        Returns:
            The updated current_persona_brief_text.

        Raises:
            KeyError: If participant_id is not found.
            RuntimeError: If LLM client is not available.
        """
        if participant_id not in self._participants:
            raise KeyError(f"Participant '{participant_id}' not found in pool")
        if self.llm is None:
            raise RuntimeError("LLM client is required for persona brief update")

        participant = self._participants[participant_id]
        history = participant.interactions_history_with_the_main_character

        if not history:
            logger.info(f"No interaction history for {participant_id}; brief unchanged.")
            return participant.current_persona_brief_text

        # Build interaction summary for LLM
        history_lines = []
        for rec in history:
            tone_str = f" (emotional tone: {rec.emotional_tone})" if rec.emotional_tone else ""
            history_lines.append(f"- [{rec.date}] {rec.summary}{tone_str}")
        history_text = "\n".join(history_lines)

        target = self._participants.get(self.TARGET_ID)
        target_name = target.persona_name_text if target else "protagonist"

        # Determine language instruction (hardcoded to English for English-only simulation)
        lang_instruction = "English"

        prompt = f"""Based on the following character and their interaction history with {target_name},
write an updated character background description reflecting the character's current real-time state.
The description should be concise but capture key personality traits, current career/academic status, relationship dynamics, and any significant changes.

Character information:
- Name: {participant.persona_name_text}
- Role type: {participant.role}
- Relationship with protagonist: {participant.relationship_towards_the_main_character}
- Initial background: {participant.initial_persona_brief_text}
- Current background: {participant.current_persona_brief_text}
- Target background: {participant.target_persona_brief_text}

Interaction history:
{history_text}

Write the updated current character background description (one paragraph, in {lang_instruction}, including current age, career/academic status, personality traits, etc.):"""

        system_prompt = (
            "You are a character profile writer for a life simulation."
            "Update the character's current persona brief based on accumulated interaction history."
            "The description should reflect the character's real-time state at the current point in time."
        )

        try:
            from pydantic import BaseModel as _BM, Field as _F

            class _PersonaBriefOutput(_BM):
                updated_persona_brief: str = _F(
                    ...,
                    description=(
                        "Updated current character background description. "
                        "One paragraph in English. Include current age, career/academic status, "
                        "personality traits, and relationship dynamics. "
                        "Plain text only — no markdown, no labels, no prefixes."
                    )
                )

            result: _PersonaBriefOutput = await self.llm.generate_structured(
                prompt=prompt,
                response_model=_PersonaBriefOutput,
                system_prompt=system_prompt,
                task_type="persona_update",
                temperature=0.0,
            )
            new_brief = result.updated_persona_brief.strip()
            participant.current_persona_brief_text = new_brief
            self.mark_dirty(participant_id)
            logger.info(f"Updated current_persona_brief_text for {participant_id}")
            return participant.current_persona_brief_text
        except Exception as e:
            logger.warning(f"Failed to update persona brief for {participant_id}: {e}")
            return participant.current_persona_brief_text

    # ── Query & Retrieval ────────────────────────────────────────

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        """Retrieve a participant by ID, or None if not found."""
        return self._participants.get(participant_id)

    def get_target(self) -> Optional[Participant]:
        """Retrieve the target persona participant."""
        return self._participants.get(self.TARGET_ID)

    def list_all(self) -> List[Participant]:
        """Return all registered participants (target + supporting)."""
        return list(self._participants.values())

    def list_supporting(self) -> List[Participant]:
        """Return only supporting characters (excluding target)."""
        return [p for p in self._participants.values() if p.participant_id != self.TARGET_ID]

    def get_participants_by_role(self, role: str) -> List[Participant]:
        """Return all participants with a given role tag."""
        return [p for p in self._participants.values() if p.role == role]

    def count(self) -> int:
        """Return total number of participants in the pool."""
        return len(self._participants)

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the entire pool to a JSON-compatible dict.

        Returns:
            Dict with 'participants' mapping pid → participant data,
            plus simulation_start_date and simulation_end_date.
        """
        return {
            "simulation_start_date": self._simulation_start_date,
            "simulation_end_date": self._simulation_end_date,
            "total_count": len(self._participants),
            "participants": {
                pid: p.model_dump(mode="json", exclude_none=True)
                for pid, p in self._participants.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the pool to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_client: Optional[AsyncLLMClient] = None,
    ) -> "ParticipantPoolManager":
        """
        Deserialize a pool from a dict (inverse of ``to_dict``).

        Args:
            data: Dict with 'participants' key.
            llm_client: Optional LLM client for future operations.

        Returns:
            A reconstructed ParticipantPoolManager.
        """
        manager = cls(llm_client=llm_client)
        # Restore simulation time range if present
        manager._simulation_start_date = data.get("simulation_start_date", "")
        manager._simulation_end_date = data.get("simulation_end_date", "")
        for pid, pdata in data.get("participants", {}).items():
            participant = Participant(**pdata)
            manager._participants[pid] = participant
            # Update counter to avoid ID collisions
            if pid.startswith("P_") and pid != cls.TARGET_ID:
                try:
                    num = int(pid.split("_")[1])
                    if num >= manager._next_id_counter:
                        manager._next_id_counter = num + 1
                except ValueError:
                    pass
        logger.info(f"Loaded participant pool from dict: {len(manager._participants)} participants")
        return manager

    # ── Display ──────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a human-readable summary of the participant pool."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Participant Pool Summary — {len(self._participants)} participants")
        logger.info(f"{'='*60}")
        for pid, p in self._participants.items():
            dob_str = (
                f"{p.date_of_birth.year}-{p.date_of_birth.month:02d}-{p.date_of_birth.day:02d}"
                if p.date_of_birth else "N/A"
            )
            logger.info(f"\n  [{pid}] {p.persona_name_text}")
            logger.info(f"    Role: {p.role}")
            logger.info(f"    DOB:  {dob_str}")
            logger.info(f"    Appear period: {p.appear_period}")
            logger.info(f"    Relationship: {p.relationship_towards_the_main_character}")
            logger.info(f"    Brief (overall): {p.persona_brief_text[:80]}{'...' if len(p.persona_brief_text) > 80 else ''}")
            if p.initial_persona_brief_text:
                logger.info(f"    Brief (initial): {p.initial_persona_brief_text[:80]}{'...' if len(p.initial_persona_brief_text) > 80 else ''}")
            if p.current_persona_brief_text:
                logger.info(f"    Brief (current): {p.current_persona_brief_text[:80]}{'...' if len(p.current_persona_brief_text) > 80 else ''}")
            if p.target_persona_brief_text:
                logger.info(f"    Brief (target):  {p.target_persona_brief_text[:80]}{'...' if len(p.target_persona_brief_text) > 80 else ''}")
            logger.info(f"    Interactions: {len(p.interactions_history_with_the_main_character)} records")
        logger.info(f"{'='*60}\n")

    async def _step1_5_audit_key_relationships(
        self,
        target: "Participant",
        persona_config: Dict[str, Any],
        life_plan: Dict[str, Any],
        sim_start_date: str,
        sim_end_date: str,
    ) -> list:
        """
        Phase 1.6: Audit and supplement key relationship characters.

        Checks the protagonist's life plan for expected relationship milestones
        (marriage, children, long-term colleagues, etc.) and generates missing
        key figures who are NOT covered by Phase 1 (birth-time characters).
        """
        existing_summary = "\n".join(
            f"- {p.persona_name_text}: {p.relationship_towards_the_main_character} ({p.role})"
            for p in self._participants.values()
            if p.persona_name_text
        )

        life_plan_summary = "\n".join(
            f"- {period.get('period_id', '?')} ({period.get('stage_label', '?')}): {period.get('title', '')} — {period.get('dominant_theme', '')}"
            for period in (life_plan.get("life_periods", []) if isinstance(life_plan, dict) else [])
        )

        target_name = target.persona_name_text or persona_config.get("persona_name_text", "Unknown")
        target_brief = target.persona_brief_text or persona_config.get("persona_brief_text", "")

        system_prompt = (
            "You are a life simulation relationship auditor. "
            "Given a protagonist's persona description and life-stage plan, identify "
            "key relationship figures that should exist but may be missing.\n\n"
            "Focus on figures implied by the life plan:\n"
            "- Spouse/partner (if the plan mentions marriage, domestic partnership, or widowhood)\n"
            "- Children (if the plan mentions parenthood)\n"
            "- Long-term colleagues/mentors (if the plan mentions a significant career)\n"
            "- Close friends (if the plan mentions social milestones)\n\n"
            "For each missing figure, provide:\n"
            "1. relationship_to_main_character (e.g., 'wife', 'adult son', 'business partner')\n"
            "2. role_type (family / friend / colleague / mentor / romantic_partner)\n"
            "3. When this relationship likely began (life period, e.g., LP5)\n"
            "4. A brief description of this person\n"
            "5. date_of_birth: an object with keys year (int), month (int 1-12), day (int 1-31) — infer a realistic DOB consistent with the character's described age and role\n\n"
            "Output as a JSON array. If no key figures are missing, output an empty array []."
        )

        user_prompt = (
            f"## Protagonist\n"
            f"- Name: {target_name}\n"
            f"- Description: {target_brief}\n\n"
            f"## Life plan\n{life_plan_summary}\n\n"
            f"## Already Generated Characters\n{existing_summary or 'None yet'}\n\n"
            f"## Task\n"
            f"Identify up to 5 additional key relationship figures that the protagonist "
            f"would realistically have, based on the life plan. These are characters who "
            f"enter the protagonist's life AFTER birth (unlike Phase 1 characters).\n\n"
            f"Important constraints:\n"
            f"- Each character must have a clear relationship to the protagonist\n"
            f"- Prioritize: spouse > children > long-term colleagues > close friends\n"
            f"- Characters should be culturally and temporally consistent\n"
            f"- Output as JSON array of objects with keys: name, relationship_to_main_character, "
            f"role_type, life_period_entered, brief_description, date_of_birth\n"
            f"- date_of_birth must be a JSON object: {{\"year\": <int>, \"month\": <int>, \"day\": <int>}}"
        )

        if self.llm is None:
            logger.warning("Phase 1.6: LLM client not available. Skipping relationship audit.")
            return []

        from pydantic import BaseModel as _BaseModel, Field as _Field

        class _RelationshipAuditOutput(_BaseModel):
            characters: list = _Field(
                ...,
                description="JSON array of relationship figures with keys: name, relationship_to_main_character, role_type, life_period_entered, brief_description, date_of_birth",
            )

        try:
            result: _RelationshipAuditOutput = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=_RelationshipAuditOutput,
                system_prompt=system_prompt,
                task_type="p1a_social_context",
                temperature=0.0,
            )
            new_chars = result.characters if isinstance(result.characters, list) else []
        except Exception as e:
            logger.error(f"Phase 1.6: LLM call failed: {e}. Skipping relationship audit — key relationship figures (spouse, children, etc.) will be missing.")
            return []

        added = []
        for char_data in new_chars[:5]:
            try:
                new_participant = self._create_participant_from_audit(char_data=char_data)
                if new_participant:
                    self._participants[new_participant.participant_id] = new_participant
                    added.append(new_participant)
                    logger.info(
                        f"  Phase 1.6 added: {char_data.get('name', '?')} "
                        f"({char_data.get('relationship_to_main_character', '?')})"
                    )
            except Exception as e:
                logger.error(f"Phase 1.6: Failed to create participant: {e}")

        return added

    def _create_participant_from_audit(self, char_data: dict, current_period_id: str = "") -> "Optional[Participant]":
        """Create a Participant object from Phase 1.6 audit data.

        T23: appear_period is set from char_data['life_period_entered'] if available,
        otherwise falls back to current_period_id (the period being processed).
        T-P06: date_of_birth is extracted from char_data['date_of_birth'] (year/month/day keys).
        """
        try:
            pid = self._generate_participant_id()
            # T-A3: Ensure name uniqueness before creating participant
            raw_name = char_data.get("name", "Unknown")
            char_data = dict(char_data)  # avoid mutating caller's dict
            char_data["name"] = self._resolve_unique_name(raw_name)
            # T23: Use life_period_entered from LLM output, fallback to current_period_id
            appear = char_data.get("life_period_entered", "")
            if not appear or appear == "LP_UNKNOWN":
                appear = current_period_id if current_period_id else "LP_UNKNOWN"
            # T-P06: Extract date_of_birth from LLM output
            dob: Optional[DateOfBirth] = None
            dob_raw = char_data.get("date_of_birth")
            if isinstance(dob_raw, dict):
                try:
                    dob = DateOfBirth(
                        year=int(dob_raw["year"]),
                        month=int(dob_raw["month"]),
                        day=int(dob_raw["day"]),
                    )
                except (KeyError, ValueError, TypeError) as dob_err:
                    logger.warning(
                        f"Phase 1.6: Could not parse date_of_birth for "
                        f"{char_data.get('name', '?')}: {dob_err}"
                    )
            participant = Participant(
                participant_id=pid,
                persona_name_text=char_data.get("name", "Unknown"),
                relationship_towards_the_main_character=char_data.get(
                    "relationship_to_main_character", "acquaintance"
                ),
                role=char_data.get("role_type", "friend"),
                persona_brief_text=char_data.get("brief_description", ""),
                date_of_birth=dob,
                appear_period=appear,
                interactions_history_with_the_main_character=[],
            )
            return participant
        except Exception as e:
            logger.error(f"Phase 1.6: Failed to create participant from audit data: {e}")
            return None


# ================================================================
# Backward Compatibility — ParticipantPool alias
# ================================================================

# The test_pipeline.py imports ParticipantPool; provide an alias
# that wraps the new manager with the old interface.

class ParticipantPool(ParticipantPoolManager):
    """
    Backward-compatible wrapper around ParticipantPoolManager.

    Provides the ``initialize_from_persona`` method expected by
    ``test_pipeline.py`` while delegating to the new manager internals.
    """

    def __init__(self, llm_client: Optional[AsyncLLMClient] = None):
        super().__init__(llm_client=llm_client)

    def initialize_from_persona(
        self,
        persona_config: Dict[str, Any],
        life_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synchronous initialization of the target persona (Phase 0).

        Creates the target participant and bootstraps minimal family members
        based on childhood_living_arrangement. For full LLM-driven supporting
        character generation, call ``initialize_supporting_characters`` afterwards.

        Args:
            persona_config: The full persona constraint sheet.
            life_plan: Optional life-period plan (used to resolve birth date).

        Returns:
            The participant_id of the target persona.
        """
        target = self.register_target_persona(persona_config, life_plan=life_plan)

        # Bootstrap minimal family members based on living arrangement
        arrangement = persona_config.get("childhood_living_arrangement", "")
        target_name = persona_config.get("persona_name_text", "Unnamed")

        # Skip family bootstrap if childhood_living_arrangement is not provided
        # (sensitive field, not auto-inferred). Family members will be created
        # later during initialize_supporting_characters() if needed.
        if not arrangement:
            logger.info(
                "childhood_living_arrangement not provided — skipping family bootstrap. "
                "Family members will be created during supporting character generation."
            )
        elif arrangement in ("two_married_parents", "two_cohabiting_parents"):
            father = Participant(
                participant_id=self._generate_participant_id(),
                role="family",
                relationship_towards_the_main_character="father",
                persona_name_text=f"{target_name}'s father",
                persona_brief_text="Father figure, important family member",
                initial_persona_brief_text="Father figure, important family member",
                current_persona_brief_text="Father figure, important family member",
                target_persona_brief_text="Father figure, important family member",
                appear_period="LP1",
                interactions_history_with_the_main_character=[],
            )
            mother = Participant(
                participant_id=self._generate_participant_id(),
                role="family",
                relationship_towards_the_main_character="mother",
                persona_name_text=f"{target_name}'s mother",
                persona_brief_text="Mother figure, important family member",
                initial_persona_brief_text="Mother figure, important family member",
                current_persona_brief_text="Mother figure, important family member",
                target_persona_brief_text="Mother figure, important family member",                appear_period="LP1",
                interactions_history_with_the_main_character=[],
            )
            self._participants[father.participant_id] = father
            self._participants[mother.participant_id] = mother
        elif arrangement == "single_parent":
            parent = Participant(
                participant_id=self._generate_participant_id(),
                role="family",
                relationship_towards_the_main_character="parent",
                persona_name_text=f"{target_name}'s parent",
                persona_brief_text="Single parent, solely responsible for raising the child",
                initial_persona_brief_text="Single parent, solely responsible for raising the child",
                current_persona_brief_text="Single parent, solely responsible for raising the child",
                target_persona_brief_text="Single parent, solely responsible for raising the child",
                appear_period="LP1",
                interactions_history_with_the_main_character=[],
            )
            self._participants[parent.participant_id] = parent

        logger.info(
            f"ParticipantPool initialized: target={self.TARGET_ID}, "
            f"total={len(self._participants)} participants"
        )
        return self.TARGET_ID


async def _demo_main() -> None:
    """Demonstrate participant pool initialization flow and export results to persona_pool.json."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    persona_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_p0_persona_settings", "test_sample.json")
    life_plan_path = os.path.join(base_dir, "test_plan.json")
    output_path = os.path.join(base_dir, "persona_pool.json")

    if not os.path.exists(persona_path):
        raise FileNotFoundError(f"persona sample not found: {persona_path}")
    if not os.path.exists(life_plan_path):
        raise FileNotFoundError(f"life plan file not found: {life_plan_path}")

    with open(persona_path, "r", encoding="utf-8") as f:
        persona_config = json.load(f)
    with open(life_plan_path, "r", encoding="utf-8") as f:
        life_plan = json.load(f)

    llm_client = AsyncLLMClient(
        default_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_base=os.environ.get("OPENAI_API_BASE", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    pool = ParticipantPoolManager(llm_client=llm_client)
    target = pool.register_target_persona(persona_config, life_plan=life_plan)
    logger.info(f"Target persona registered: {target.persona_name_text} ({target.participant_id}), DOB={target.date_of_birth}")

    # Generate temporal briefs for the target persona first
    target = await pool.generate_target_temporal_briefs(
        persona_config=persona_config,
        life_plan=life_plan,
    )
    logger.info(f"Protagonist status context constructed (initial brief will be generated in Phase 6): {target.persona_name_text}")

    supporting = await pool.initialize_supporting_characters(
        persona_config=persona_config,
        life_plan=life_plan,
    )
    logger.info(f"Supporting characters initialized: {len(supporting)}")

    pool.print_summary()

    # Explicitly output all fields (including appear_period and simulation dates)
    pool_data = {
        "simulation_start_date": pool._simulation_start_date,
        "simulation_end_date": pool._simulation_end_date,
        "total_count": pool.count(),
        "participants": [
            p.model_dump(mode="json", exclude_none=False)
            for p in pool.list_all()
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pool_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Participant pool saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(_demo_main())

