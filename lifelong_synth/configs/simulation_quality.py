"""
Simulation Quality Engine (SQE) — Turn-level quality control for P3 simulation.

Provides:
  - SimulationPlan: Immutable simulation blueprint with mathematical guarantees
  - SimulationPlanBuilder: Builds SimulationPlan from TelescopingMode
  - TurnDirective: Per-turn navigation instruction for the director
  - TurnDirectiveComputer: Computes TurnDirective from simulation state
  - RepetitionGuard: Character-level Jaccard similarity detection + forced skip
  - BeatTracker: Keyword-overlap beat completion detection

Solves: Q1 (repetition), Q2 (scene incompletion), Q3 (summary disconnect)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from lifelong_synth.configs.temporal_context import (
    PHASE_ORDER,
    AdaptiveTelescopingPolicy,
    TelescopingMode,
)

logger = logging.getLogger(__name__)


# ================================================================
# SimulationPlan
# ================================================================

@dataclass
class SimulationPlan:
    """Immutable simulation blueprint with mathematical guarantees.

    Invariants:
      - max_beats <= simulation_turns // 3
      - all phase allocations >= 1
      - reserved_turns >= 2
    """
    total_turns: int
    simulation_turns: int           # total_turns - reserved_turns
    reserved_turns: int = 2         # System-reserved for resolution + closing
    max_beats: int = 5
    phase_allocation: Dict[str, int] = field(default_factory=dict)
    phase_boundaries: Dict[str, int] = field(default_factory=dict)
    telescoping_mode: str = "standard"

    @property
    def is_valid(self) -> bool:
        return (
            self.max_beats <= self.simulation_turns // 3
            and all(v >= 1 for v in self.phase_allocation.values() if v > 0)
            and self.reserved_turns >= 2
        )

    def get_phase_for_turn(self, turn_index: int) -> str:
        """Determine the narrative phase for a given turn index.

        System-level rule — cannot be overridden by LLM.
        """
        for phase in PHASE_ORDER:
            boundary = self.phase_boundaries.get(phase, self.total_turns)
            if turn_index < boundary:
                return phase
        return "closing"


class SimulationPlanBuilder:
    """Build a SimulationPlan from a TelescopingMode."""

    def __init__(self, telescoping: Optional[AdaptiveTelescopingPolicy] = None):
        self._telescoping = telescoping or AdaptiveTelescopingPolicy()

    def build(self, am_weight: float) -> SimulationPlan:
        """Build a SimulationPlan based on AM weight."""
        mode = self._telescoping.select_mode(am_weight)
        total_turns = (mode.turn_range[0] + mode.turn_range[1]) // 2
        reserved = 2
        simulation_turns = total_turns - reserved

        # Ensure max_beats respects the mathematical guarantee
        max_beats = min(mode.max_beats, simulation_turns // 3)

        phase_allocation = self._telescoping.compute_phase_allocation(
            total_turns, reserved
        )
        phase_boundaries = self._telescoping.compute_phase_boundaries(
            phase_allocation
        )

        plan = SimulationPlan(
            total_turns=total_turns,
            simulation_turns=simulation_turns,
            reserved_turns=reserved,
            max_beats=max_beats,
            phase_allocation=phase_allocation,
            phase_boundaries=phase_boundaries,
            telescoping_mode=mode.name,
        )

        if not plan.is_valid:
            logger.warning(
                f"[SQE] SimulationPlan validation failed: "
                f"beats={max_beats}, sim_turns={simulation_turns}. "
                f"Adjusting max_beats."
            )
            plan.max_beats = simulation_turns // 3

        return plan

    def build_from_turns(self, total_turns: int, max_beats: int = 5) -> SimulationPlan:
        """Build a SimulationPlan from explicit turn count (for backward compat)."""
        reserved = 2
        simulation_turns = total_turns - reserved
        max_beats = min(max_beats, simulation_turns // 3)

        phase_allocation = self._telescoping.compute_phase_allocation(
            total_turns, reserved
        )
        phase_boundaries = self._telescoping.compute_phase_boundaries(
            phase_allocation
        )

        return SimulationPlan(
            total_turns=total_turns,
            simulation_turns=simulation_turns,
            reserved_turns=reserved,
            max_beats=max_beats,
            phase_allocation=phase_allocation,
            phase_boundaries=phase_boundaries,
            telescoping_mode="custom",
        )


# ================================================================
# TurnDirective
# ================================================================

@dataclass
class TurnDirective:
    """Per-turn navigation instruction injected into the director prompt."""
    turn_number: int
    total_turns: int
    progress_pct: float
    current_phase: str
    turns_until_next_phase: int
    current_beat_index: int
    current_beat_description: str
    remaining_beats: int
    pacing_instruction: str         # "Normal" / "Accelerate" / "Wind down" / "⚠️ Final stage"
    recent_summary: str             # Summary of last 3 turns
    forbidden_patterns: List[str]   # Detected repetitive patterns to avoid

    def format_for_prompt(self, language: str = "en") -> str:
        """Format as a prompt-injectable text block."""
        return (
            f"## 🎬 Scene Navigation (system-generated, follow strictly)\n"
            f"- Turn: {self.turn_number}/{self.total_turns}, "
            f"Progress: {self.progress_pct:.0%}\n"
            f"- Phase: {self.current_phase}, "
            f"Turns until next phase: {self.turns_until_next_phase}\n"
            f"- Current beat: #{self.current_beat_index + 1} — "
            f"{self.current_beat_description}, "
            f"Remaining beats: {self.remaining_beats}\n"
            f"- Pacing: {self.pacing_instruction}\n"
            f"- Recent: {self.recent_summary}\n"
            + (f"- ⛔ Avoid repeating: {'; '.join(self.forbidden_patterns)}\n"
               if self.forbidden_patterns else "")
        )


class TurnDirectiveComputer:
    """Compute TurnDirective from current simulation state."""

    def __init__(self, plan: SimulationPlan, beats: List[str]):
        self.plan = plan
        self.beats = beats

    def compute(
        self,
        turn_index: int,
        current_beat_index: int,
        completed_beats: List[int],
        recent_turns: List[Dict[str, Any]],
        forbidden_patterns: List[str],
    ) -> TurnDirective:
        """Compute the TurnDirective for the current turn."""
        total = self.plan.total_turns
        progress = turn_index / max(total, 1)
        current_phase = self.plan.get_phase_for_turn(turn_index)

        # Turns until next phase
        boundary = self.plan.phase_boundaries.get(current_phase, total)
        turns_until_next = max(0, boundary - turn_index)

        # Current beat description
        beat_desc = (
            self.beats[current_beat_index]
            if current_beat_index < len(self.beats)
            else "(All beats completed)"
        )

        remaining_beats = len(self.beats) - len(completed_beats)

        # Pacing instruction
        if progress >= 0.85:
            pacing = "⚠️ Final stage: must wrap up immediately"
        elif progress >= 0.7:
            pacing = "Wind down: begin transitioning to resolution"
        elif remaining_beats > 0 and turns_until_next <= remaining_beats:
            pacing = "Accelerate: more beats remaining than turns left, advance one beat per turn"
        else:
            pacing = "Normal progression"

        # Recent summary (last 3 turns) — full content for accurate context
        recent_texts = []
        for t in recent_turns[-3:]:
            name = t.get("speaker_name", "?")
            content = t.get("content", "")
            recent_texts.append(f"{name}: {content}")
        recent_summary = " → ".join(recent_texts) if recent_texts else "(Scene just started)"

        return TurnDirective(
            turn_number=turn_index + 1,
            total_turns=total,
            progress_pct=progress,
            current_phase=current_phase,
            turns_until_next_phase=turns_until_next,
            current_beat_index=current_beat_index,
            current_beat_description=beat_desc,
            remaining_beats=remaining_beats,
            pacing_instruction=pacing,
            recent_summary=recent_summary,
            forbidden_patterns=forbidden_patterns,
        )


# ================================================================
# Repetition Guard
# ================================================================

# CJK Unicode ranges for language detection
_CJK_RANGES = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u3040-\u309f"
    r"\u30a0-\u30ff\uac00-\ud7af]"
)


def _is_cjk_dominant(text: str) -> bool:
    """Check if text is predominantly CJK characters."""
    if not text:
        return False
    cjk_count = len(_CJK_RANGES.findall(text))
    return cjk_count / max(len(text), 1) > 0.3


def _extract_ngrams(text: str, n: int) -> Set[str]:
    """Extract character-level n-grams from text."""
    text = text.strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / max(union, 1)


class RepetitionGuard:
    """Detect and prevent repetitive content in simulation turns.

    Uses character-level Jaccard similarity with language-adaptive n-gram size:
      - CJK text: 4-gram (Chinese/Japanese/Korean characters are denser)
      - Latin text: 8-gram

    When consecutive repetitions reach `max_consecutive`, triggers forced beat skip.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        max_consecutive: int = 3,
    ):
        self.threshold = similarity_threshold
        self.max_consecutive = max_consecutive
        self._consecutive_count: int = 0
        self._detected_patterns: List[str] = []

    def check(self, new_content: str, recent_contents: List[str]) -> bool:
        """Check if new_content is repetitive compared to recent turns.

        Returns:
            True if repetition detected, False otherwise.
        """
        if not recent_contents:
            self._consecutive_count = 0
            return False

        is_cjk = _is_cjk_dominant(new_content)
        n = 4 if is_cjk else 8
        new_ngrams = _extract_ngrams(new_content, n)

        is_repetitive = False
        for prev in recent_contents[-3:]:
            prev_ngrams = _extract_ngrams(prev, n)
            sim = _jaccard_similarity(new_ngrams, prev_ngrams)
            if sim >= self.threshold:
                is_repetitive = True
                # Extract the repeated pattern (longest common substring approximation)
                common = new_ngrams & prev_ngrams
                if common:
                    pattern = max(common, key=len)
                    if pattern not in self._detected_patterns:
                        self._detected_patterns.append(pattern)
                        # Keep only last 5 patterns
                        self._detected_patterns = self._detected_patterns[-5:]
                break

        if is_repetitive:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        return is_repetitive

    @property
    def should_force_skip(self) -> bool:
        """Whether to force-skip to the next beat."""
        return self._consecutive_count >= self.max_consecutive

    @property
    def detected_patterns(self) -> List[str]:
        """Return detected repetitive patterns for injection into TurnDirective."""
        return list(self._detected_patterns)

    def reset(self) -> None:
        """Reset state."""
        self._consecutive_count = 0
        self._detected_patterns.clear()


# ================================================================
# Beat Tracker
# ================================================================

class BeatTracker:
    """Track narrative beat completion using LLM semantic judgment + rule-based fallback.

    After each turn, checks if the turn content covers the current beat.
    Uses LLM if available for semantic understanding, otherwise falls back
    to keyword overlap.
    """

    def __init__(
        self,
        beats: List[str],
        llm_client=None,           # v8: LLM client for semantic beat detection
overlap_threshold: float = 0.2,  # v8: lowered from 0.3 for CJK beat matching
    ):
        self.beats = beats
        self.threshold = overlap_threshold
        self._llm = llm_client
        self.completed: List[int] = []
        self.current_index: int = 0

    def check_and_advance(self, turn_content: str) -> bool:
        """Synchronous check — rule-based fallback only (no LLM)."""
        if self.current_index >= len(self.beats):
            return False

        beat_text = self.beats[self.current_index]
        if self._check_overlap(beat_text, turn_content):
            self.completed.append(self.current_index)
            self.current_index += 1
            return True
        return False

    async def check_and_advance_async(
        self,
        turn_content: str,
        recent_turns: List[str],   # Last 3 turns for context
    ) -> bool:
        """Check if current beat is covered. Uses LLM if available, else falls back to overlap.

        Args:
            turn_content: Content of the current turn
            recent_turns: Content of the last 3 turns (for cumulative context)

        Returns:
            True if beat was completed and index advanced.
        """
        if self.current_index >= len(self.beats):
            return False

        beat_text = self.beats[self.current_index]

        # LLM path
        if self._llm:
            try:
                context = "\n".join(f"- {t}" for t in recent_turns[-3:] + [turn_content])
                prompt = (
                    f"Current scene beat objective: {beat_text}\n\n"
                    f"Recent dialogue turns:\n{context}\n\n"
                    f"Judge: Has the above dialogue covered the current beat objective?\n"
                    f"Criteria: Has the core event or emotional turning point of the beat objective "
                    f"occurred in the dialogue (either explicitly described or implicitly expressed)?\n"
                    f"Answer only YES or NO, with no explanation."
                )
                from pydantic import BaseModel as _BM, Field as _F

                class _BeatJudgment(_BM):
                    completed: bool = _F(
                        ...,
                        description=(
                            "True if the core event or emotional turning point of the beat objective "
                            "has occurred in the dialogue (explicitly or implicitly). False otherwise."
                        )
                    )

                result: _BeatJudgment = await self._llm.generate_structured(
                    prompt=prompt,
                    response_model=_BeatJudgment,
                    max_tokens=20,
                    temperature=0.0,
                    task_type="beat_completion_check",
                )
                if result.completed:
                    self.completed.append(self.current_index)
                    self.current_index += 1
                    return True
                return False
            except Exception:
                # Fallback to rule-based on LLM failure
                pass

        # Rule-based fallback
        return self._check_overlap(beat_text, turn_content)

    def _check_overlap(self, beat_text: str, turn_content: str) -> bool:
        """Rule-based overlap check (fallback)."""
        if _is_cjk_dominant(beat_text):
            beat_chars = set(beat_text)
            content_chars = set(turn_content)
            overlap = len(beat_chars & content_chars) / max(len(beat_chars), 1)
        else:
            beat_words = set(beat_text.lower().split())
            content_words = set(turn_content.lower().split())
            overlap = len(beat_words & content_words) / max(len(beat_words), 1)

        if overlap >= self.threshold:
            self.completed.append(self.current_index)
            self.current_index += 1
            return True
        return False

    def force_advance(self) -> None:
        """Force-advance to the next beat (used by RepetitionGuard)."""
        if self.current_index < len(self.beats):
            if self.current_index not in self.completed:
                self.completed.append(self.current_index)
            self.current_index += 1
            logger.info(
                f"[BeatTracker] Force-advanced to beat {self.current_index}"
            )

    @property
    def all_completed(self) -> bool:
        return self.current_index >= len(self.beats)

    @property
    def remaining_count(self) -> int:
        return max(0, len(self.beats) - len(self.completed))
