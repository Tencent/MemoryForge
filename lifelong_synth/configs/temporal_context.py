"""
Temporal Context Engine — Rule-based temporal context computation.

Provides:
  - compute_exact_age(): Precise age calculation considering birthday (fixes F2)
  - TemporalContext: Dataclass for event temporal context snapshot
  - TemporalContextComputer: Rule-based engine computing grade/semester/stage (fixes F1+F3+F5)
  - AutobiographicalMemoryDistributionModel: Literature-driven high-res frequency allocation (fixes Q1)
  - AdaptiveTelescopingPolicy: Adaptive simulation depth based on AM weight (fixes Q2)

References:
  - Rubin, Wetzler & Nebes (1986). Autobiographical memory across the lifespan.
  - Rubin & Schulkind (1997). Distribution of important autobiographical memories.
  - Conway & Pleydell-Pearce (2000). The construction of autobiographical memories.
  - Pillemer & White (1989). Childhood amnesia and memory development.
  - Wang & Conway (2004). The stories we keep: Autobiographical memory in American and Chinese.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# ================================================================
# F2 Fix: Precise Age Calculation
# ================================================================

def compute_exact_age(birth_date_str: str, event_date_str: str) -> int:
    """Compute exact age at event date, considering whether birthday has passed.

    Args:
        birth_date_str: Birth date in "YYYY-MM-DD" format.
        event_date_str: Event date in "YYYY-MM-DD" format.

    Returns:
        Exact integer age. Returns 0 if event is before birth.
    """
    birth = date.fromisoformat(birth_date_str)
    event = date.fromisoformat(event_date_str)
    age = event.year - birth.year
    if (event.month, event.day) < (birth.month, birth.day):
        age -= 1  # Birthday hasn't occurred yet this year
    return max(age, 0)


def compute_exact_age_safe(
    birth_date_str: Optional[str],
    event_date_str: str,
    fallback_birth_year: Optional[int] = None,
) -> Tuple[int, bool]:
    """Safe wrapper that handles missing/partial birth dates.

    Returns:
        (age, is_approximate): Tuple of age and whether it's approximate.
    """
    if birth_date_str:
        try:
            return (compute_exact_age(birth_date_str, event_date_str), False)
        except (ValueError, TypeError):
            pass

    if fallback_birth_year:
        try:
            event = date.fromisoformat(event_date_str)
            return (event.year - fallback_birth_year, True)
        except (ValueError, TypeError):
            pass

    return (0, True)


# ================================================================
# Temporal Context Dataclass
# ================================================================

@dataclass
class TemporalContext:
    """Universal temporal context snapshot for any event at any life stage."""

    # ── Core temporal facts (always available) ──
    exact_age: int
    age_approximate: bool = False
    event_date: str = ""                    # "YYYY-MM-DD"
    day_of_week: str = ""                   # "Monday", etc.
    season: str = ""                        # "spring" / "summer" / "autumn" / "winter"

    # ── Life stage context (always available) ──
    period_id: str = ""
    stage_label: str = ""                   # "junior_secondary"
    stage_title: str = ""                   # "Key Junior Secondary School"
    years_into_stage: float = 0.0
    stage_progress: str = "mid"             # "early" / "mid" / "late"

    # ── Education context (only when is_education_stage=True) ──
    is_education_stage: bool = False
    academic_year: Optional[str] = None     # "2010-2011"
    grade_label: Optional[str] = None       # "Grade 8" / "Year 9"
    grade_number: Optional[int] = None      # 1-based within this education stage
    semester: Optional[str] = None          # "Semester 1" / "Fall"
    semester_phase: Optional[str] = None    # "start" / "mid-term" / "end-of-term" / "vacation"
    formatted_education_label: Optional[str] = None  # e.g. "Year 8 Semester 1 mid-term"

    # ── Work context (only when is_work_stage=True) ──
    is_work_stage: bool = False
    months_into_job: Optional[int] = None
    job_phase: Optional[str] = None         # "onboarding" / "ramping_up" / "established"

    # ── Transition context ──
    is_transition_stage: bool = False
    transition_from: Optional[str] = None
    transition_to: Optional[str] = None

    def format_for_prompt(self, language: str = "en") -> str:
        """Format as a prompt-injectable text block."""
        lines = []
        lines.append("## Temporal Context (system-computed, immutable)")

        approx = " (approximate)" if self.age_approximate else ""
        lines.append(f"- Event date: {self.event_date} ({self.day_of_week})")
        lines.append(f"- Protagonist exact age: {self.exact_age}{approx}")
        lines.append(f"- Life stage: {self.stage_title} ({self.stage_progress})")

        if self.is_education_stage and self.formatted_education_label:
            lines.append(f"- Education stage: {self.formatted_education_label}")
            if self.academic_year:
                lines.append(f"- Academic year: {self.academic_year}")

        if self.is_work_stage and self.months_into_job is not None:
            lines.append(f"- Months into job: {self.months_into_job} ({self.job_phase})")

        lines.append("")
        lines.append("\u26a0\ufe0f Your generated content MUST be consistent with the above temporal context. "
                      "Do NOT recalculate age, grade, or semester \u2014 use the data provided above directly.")

        return "\n".join(lines)


# ================================================================
# Semester System Definitions (Cross-Cultural)
# ================================================================

# (semester_name, start_month) tuples
SEMESTER_SYSTEMS: Dict[str, List[Tuple[str, int]]] = {
    # China: two semesters (Sep, Feb)
    "china": [("Semester 1", 9), ("Semester 2", 2)],
    # Japan: three terms (Apr, Sep, Jan)
    "japan": [("Term 1", 4), ("Term 2", 9), ("Term 3", 1)],
    # UK: three terms (Sep, Jan, Apr)
    "uk": [("Autumn", 9), ("Spring", 1), ("Summer", 4)],
    # US: two semesters (Sep, Jan)
    "us": [("Fall", 9), ("Spring", 1)],
    # Australia: two semesters (Feb, Jul)
    "australia": [("Semester 1", 2), ("Semester 2", 7)],
}

# Default fallback: two semesters (Sep, Feb)
DEFAULT_SEMESTER_SYSTEM = [("Semester 1", 9), ("Semester 2", 2)]

# Country name normalization
COUNTRY_ALIASES: Dict[str, str] = {
    "china": "china", "cn": "china", "156": "china",
    "japan": "japan", "jp": "japan", "392": "japan",
    "uk": "uk", "gb": "uk", "united kingdom": "uk", "826": "uk",
    "us": "usa", "usa": "us", "united states": "us", "840": "us",
    "australia": "australia", "au": "australia", "036": "australia",
}

def _normalize_country(country: str) -> str:
    """Normalize country string to a canonical key."""
    return COUNTRY_ALIASES.get(country.lower().strip(), "")


def _get_semester_system(country: str) -> List[Tuple[str, int]]:
    """Get semester system for a country, with fallback."""
    normalized = _normalize_country(country)
    return SEMESTER_SYSTEMS.get(normalized, DEFAULT_SEMESTER_SYSTEM)


# ================================================================
# Education Stage Detection Patterns
# ================================================================

EDUCATION_STAGE_PATTERNS = [
    "school", "primary", "secondary", "undergraduate", "bachelor",
    "master", "doctoral", "phd", "college", "university",
    "kindergarten", "preschool", "high_school", "middle_school",
    "kosen", "gymnasium", "lycée", "polytechnic", "academy",
    "education", "学", "幼儿园", "小学", "初中", "高中", "大学", "研究生", "博士",
]

WORK_STAGE_PATTERNS = [
    "career", "work", "job", "employment", "professional",
    "intern", "position", "职业", "工作", "实习",
    # "entry" removed: ambiguous in education contexts (e.g., "school_entry")
]

TRANSITION_STAGE_PATTERNS = [
    "transition", "gap", "break", "sabbatical", "between",
    "过渡", "间隔", "待业",
]


def _is_education_stage(stage_label: str, dynamic_patterns: Optional[List[str]] = None) -> bool:
    """Detect if a stage label represents an education stage.

    DEPRECATED: This function is a fallback for old life plans that don't have
    is_education_stage field. New plans should use period["is_education_stage"] directly.
    """
    label_lower = stage_label.lower()
    all_patterns = EDUCATION_STAGE_PATTERNS + (dynamic_patterns or [])
    return any(p in label_lower for p in all_patterns)


def _is_work_stage(stage_label: str) -> bool:
    """Detect if a stage label represents a work stage.

    DEPRECATED: Fallback for old life plans. New plans use period["is_work_stage"].
    """
    label_lower = stage_label.lower()
    return any(p in label_lower for p in WORK_STAGE_PATTERNS)


def _is_transition_stage(stage_label: str) -> bool:
    """Detect if a stage label represents a transition stage.

    DEPRECATED: Fallback for old life plans. New plans use period["is_transition_stage"].
    """
    label_lower = stage_label.lower()
    return any(p in label_lower for p in TRANSITION_STAGE_PATTERNS)


# ================================================================
# TemporalContextComputer
# ================================================================

class TemporalContextComputer:
    """Rule-based engine that computes TemporalContext for any event.

    Reads parameters from SocialContextProfile (P1 output) to compute
    precise grade, semester, and stage information. Zero LLM calls.

    Args:
        birth_date: Target persona's birth date "YYYY-MM-DD".
        social_context: SocialContextProfile dict from life_plan.global_summary.social_context.
        life_periods: List of life period dicts from life_plan.life_periods.
    """

    def __init__(
        self,
        birth_date: str,
        social_context: Dict[str, Any],
        life_periods: List[Dict[str, Any]],
    ):
        self.birth_date = birth_date
        self.social_context = social_context
        self.life_periods = life_periods
        self.country = social_context.get("country", "")
        self.semester_system = _get_semester_system(self.country)

        # Extract dynamic education stage names from SocialContextProfile
        self._dynamic_edu_patterns: List[str] = []
        for stage in social_context.get("education_system", []):
            stage_name = stage.get("stage_name", "")
            if stage_name:
                self._dynamic_edu_patterns.append(stage_name.lower())

        # Build enrollment timeline from education_system
        self._enrollment_timeline: List[Dict[str, Any]] = []
        self._build_enrollment_timeline()

    def _build_enrollment_timeline(self) -> None:
        """Build precise enrollment timeline from SocialContextProfile.education_system.

        Each entry: {stage_name, entry_age, duration_years, entry_month, grade_labels}
        """
        for stage in self.social_context.get("education_system", []):
            entry = {
                "stage_name": stage.get("stage_name", ""),
                "entry_age": stage.get("entry_age", 0),
                "duration_years": stage.get("duration_years", 0),
                "entry_month": stage.get("entry_month", 9),  # Default: September
                "grade_labels": stage.get("grade_labels", []),
            }
            self._enrollment_timeline.append(entry)

    def _find_current_period(self, event_date_str: str) -> Optional[Dict[str, Any]]:
        """Find the life period that contains the event date."""
        event_d = date.fromisoformat(event_date_str)
        for period in self.life_periods:
            dr = period.get("period_date_range", {})
            start_str = dr.get("start_date", "")
            end_str = dr.get("end_date", "")
            if not start_str or not end_str:
                continue
            try:
                start = date.fromisoformat(start_str)
                end = date.fromisoformat(end_str)
                if start <= event_d <= end:
                    return period
            except ValueError:
                continue
        return None

    def _compute_season(self, event_d: date) -> str:
        """Compute season from date."""
        month = event_d.month
        if month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        elif month in (9, 10, 11):
            return "autumn"
        else:
            return "winter"

    def _compute_stage_progress(self, event_d: date, period: Dict[str, Any]) -> Tuple[float, str]:
        """Compute years into stage and progress label."""
        dr = period.get("period_date_range", {})
        start_str = dr.get("start_date", "")
        end_str = dr.get("end_date", "")
        if not start_str or not end_str:
            return (0.0, "mid")
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        total_days = max((end - start).days, 1)
        elapsed_days = max((event_d - start).days, 0)
        ratio = elapsed_days / total_days
        years_into = elapsed_days / 365.25

        if ratio < 0.25:
            progress = "early"
        elif ratio < 0.75:
            progress = "mid"
        else:
            progress = "late"

        return (round(years_into, 2), progress)

    def _compute_education_context(
        self, event_d: date, age: int, stage_label: str
    ) -> Dict[str, Any]:
        """Compute grade, semester, semester_phase for education stages.

        Returns empty result dict for non-education stages (career, retirement, etc.)
        to avoid generating meaningless semester/grade labels.
        """
        result: Dict[str, Any] = {
            "academic_year": None,
            "grade_label": None,
            "grade_number": None,
            "semester": None,
            "semester_phase": None,
            "formatted_education_label": None,
        }

        # Early exit for non-education stage labels
        # These labels indicate career/transition/retirement stages that have
        # no meaningful grade or semester information.
        _NON_EDUCATION_LABELS = {
            "gap_transition", "early_career", "mid_career", "senior_career",
            "career_entry", "career_growth", "career_peak", "pre_retirement",
            "retirement", "post_retirement", "work_stage", "employment",
        }
        stage_label_lower = stage_label.lower()
        if any(label in stage_label_lower for label in _NON_EDUCATION_LABELS):
            return result

        # Find matching education stage in enrollment timeline
        matched_stage = None
        for stage in self._enrollment_timeline:
            sn = stage["stage_name"].lower()
            if sn in stage_label.lower() or stage_label.lower() in sn:
                matched_stage = stage
                break

        if not matched_stage:
            # Fallback: try to match by age range
            for stage in self._enrollment_timeline:
                entry_age = stage["entry_age"]
                duration = stage["duration_years"]
                if entry_age <= age < entry_age + duration:
                    matched_stage = stage
                    break

        if not matched_stage:
            return result

        # Compute grade number (1-based within this education stage)
        entry_age = matched_stage["entry_age"]
        entry_month = matched_stage["entry_month"]
        birth = date.fromisoformat(self.birth_date)

        # Compute enrollment year
        enrollment_year = birth.year + entry_age
        if birth.month > entry_month:
            enrollment_year += 1  # Born after entry month → enroll next year

        # Compute grade number based on academic years elapsed
        academic_years_elapsed = event_d.year - enrollment_year
        if event_d.month < entry_month:
            academic_years_elapsed -= 1  # Before new academic year starts
        grade_number = max(1, academic_years_elapsed + 1)

        # Cap at stage duration
        duration = matched_stage["duration_years"]
        grade_number = min(grade_number, duration)

        result["grade_number"] = grade_number

        # Academic year
        if event_d.month >= entry_month:
            result["academic_year"] = f"{event_d.year}-{event_d.year + 1}"
        else:
            result["academic_year"] = f"{event_d.year - 1}-{event_d.year}"

        # Grade label from predefined labels or auto-generate
        grade_labels = matched_stage.get("grade_labels", [])
        if grade_labels and grade_number <= len(grade_labels):
            result["grade_label"] = grade_labels[grade_number - 1]
        else:
            # Auto-generate: "Year {n}" or stage-specific
            result["grade_label"] = f"Year {grade_number}"

        # Semester detection
        month = event_d.month
        current_semester = None
        for i, (sem_name, sem_start) in enumerate(self.semester_system):
            next_start = (
                self.semester_system[(i + 1) % len(self.semester_system)][1]
            )
            # Handle wrap-around (e.g., semester starting in Sep, next in Feb)
            if sem_start <= next_start:
                if sem_start <= month < next_start:
                    current_semester = sem_name
                    break
            else:
                if month >= sem_start or month < next_start:
                    current_semester = sem_name
                    break

        if current_semester is None:
            current_semester = self.semester_system[0][0]
        result["semester"] = current_semester

        # Semester phase detection
        # Find current semester's start month
        sem_start_month = None
        for sem_name, sm in self.semester_system:
            if sem_name == current_semester:
                sem_start_month = sm
                break

        if sem_start_month is not None:
            months_into_sem = (month - sem_start_month) % 12
            if months_into_sem <= 1:
                result["semester_phase"] = "start"
            elif months_into_sem <= 2:
                result["semester_phase"] = "mid-term"
            else:
                result["semester_phase"] = "end-of-term"

        # Check if in vacation
        vacation_months = {7, 8}  # Summer vacation (universal)
        if _normalize_country(self.country) == "china":
            vacation_months.update({1, 2})  # Winter vacation
        if month in vacation_months:
            result["semester_phase"] = "vacation"

        # Formatted education label
        grade = result["grade_label"] or f"Year {grade_number}"
        sem = result["semester"] or ""
        phase = result["semester_phase"] or ""
        parts = [p for p in [grade, sem, phase] if p]
        result["formatted_education_label"] = " ".join(parts)

        return result

    def _compute_work_context(self, event_d: date, period: Dict[str, Any]) -> Dict[str, Any]:
        """Compute work-stage context."""
        dr = period.get("period_date_range", {})
        start_str = dr.get("start_date", "")
        if not start_str:
            return {"months_into_job": None, "job_phase": None}

        start = date.fromisoformat(start_str)
        months = (event_d.year - start.year) * 12 + (event_d.month - start.month)
        months = max(0, months)

        if months <= 3:
            phase = "onboarding"
        elif months <= 12:
            phase = "ramping_up"
        else:
            phase = "established"

        return {"months_into_job": months, "job_phase": phase}

    def compute(self, event_date_str: str, period_override: Optional[Dict[str, Any]] = None) -> TemporalContext:
        """Compute full TemporalContext for an event.

        Args:
            event_date_str: Event date in "YYYY-MM-DD" format.
            period_override: Optional period dict to use instead of auto-detecting.
                If the period dict contains is_education_stage / is_work_stage /
                is_transition_stage fields (P1 LLM generated), those are used
                directly. Otherwise falls back to rule-based pattern matching.

        Returns:
            TemporalContext with all applicable fields populated.
        """
        event_d = date.fromisoformat(event_date_str)
        age, approx = compute_exact_age_safe(self.birth_date, event_date_str)

        period = period_override or self._find_current_period(event_date_str)
        if not period:
            return TemporalContext(
                exact_age=age,
                age_approximate=approx,
                event_date=event_date_str,
            )

        stage_label = period.get("stage_label", period.get("title", ""))
        stage_title = period.get("title", "")
        years_into, progress = self._compute_stage_progress(event_d, period)

        # Priority 1: Read from period dict (P1 LLM generated, most accurate)
        if "is_education_stage" in period:
            is_edu = period.get("is_education_stage", False)
            is_work = period.get("is_work_stage", False)
            is_transition = period.get("is_transition_stage", False)
        else:
            # Priority 2: Fallback to rule matching (backward compat with old plans)
            is_edu = _is_education_stage(stage_label, self._dynamic_edu_patterns)
            is_work = _is_work_stage(stage_label)
            is_transition = _is_transition_stage(stage_label)

        tc = TemporalContext(
            exact_age=age,
            age_approximate=approx,
            event_date=event_date_str,
            day_of_week=event_d.strftime("%A"),
            season=self._compute_season(event_d),
            period_id=period.get("period_id", ""),
            stage_label=stage_label,
            stage_title=stage_title,
            years_into_stage=years_into,
            stage_progress=progress,
            is_education_stage=is_edu,
            is_work_stage=is_work,
            is_transition_stage=is_transition,
        )

        if is_edu:
            edu_ctx = self._compute_education_context(event_d, age, stage_label)
            tc.academic_year = edu_ctx["academic_year"]
            tc.grade_label = edu_ctx["grade_label"]
            tc.grade_number = edu_ctx["grade_number"]
            tc.semester = edu_ctx["semester"]
            tc.semester_phase = edu_ctx["semester_phase"]
            tc.formatted_education_label = edu_ctx["formatted_education_label"]

        if is_work:
            work_ctx = self._compute_work_context(event_d, period)
            tc.months_into_job = work_ctx["months_into_job"]
            tc.job_phase = work_ctx["job_phase"]

        return tc


# ================================================================
# Autobiographical Memory Distribution Model
# ================================================================

@dataclass
class AMModelCulturalPreset:
    """Cultural preset for the AM distribution model.

    References:
      - Wang & Conway (2004): East Asian bump center ~22
      - Rubin & Schulkind (1997): Western bump center ~18
    """
    name: str
    childhood_amnesia_thresholds: Tuple[float, float, float]  # (near_zero, sparse, partial)
    bump_center: float
    bump_sigma: float = 8.0

    @staticmethod
    def from_country(country: str) -> "AMModelCulturalPreset":
        """Auto-select preset from country string."""
        normalized = _normalize_country(country)
        if normalized in ("china", "japan"):
            return AMModelCulturalPreset(
                name="east_asian",
                childhood_amnesia_thresholds=(3.0, 5.5, 7.5),
                bump_center=22.0,
                bump_sigma=8.0,
            )
        elif normalized in ("us", "uk", "australia"):
            return AMModelCulturalPreset(
                name="western",
                childhood_amnesia_thresholds=(2.5, 4.5, 6.5),
                bump_center=18.0,
                bump_sigma=8.0,
            )
        else:
            return AMModelCulturalPreset(
                name="default",
                childhood_amnesia_thresholds=(3.0, 5.0, 7.0),
                bump_center=20.0,
                bump_sigma=8.0,
            )


# ================================================================
# Per-Age Memory Profile (v2 — replaces period-level density)
# ================================================================

@dataclass
class PerAgeMemoryProfile:
    """Memory characteristics for a single year of age.

    Computed by AutobiographicalMemoryModel based on the three-component model.
    Used to determine simulation granularity and event budget per age.
    """
    age: int
    temporal_resolution: str       # "none" | "year" | "season" | "month"
    max_detail_per_year: int       # max high-res with details events
    max_outline_per_year: int      # max high-res with outline events
    am_weight: float = 0.0        # raw AM weight for telescoping policy


class AutobiographicalMemoryModel:
    """
    Literature-grounded per-age memory profile generator.

    Replaces the old period-level AutobiographicalMemoryDistributionModel
    with a per-age model that computes:
      1. Temporal resolution: none / year / month
      2. Max detail events per year (episodic-rich)
      3. Max outline events per year (general-event)
      4. AM weight (for P3 telescoping policy)

    Three-Component Model:
      1. Childhood Amnesia  — Pillemer & White (1989); Usher & Neisser (1993)
      2. Reminiscence Bump  — Rubin & Schulkind (1997); Wang & Conway (2004)
      3. Recency Effect     — Rubin & Wenzel (1996)

    Hierarchy Model:
      Conway & Pleydell-Pearce (2000); Levine et al. (2002)
    """

    def __init__(self, birth_date: str, reference_date: str, country: str = ""):
        self.birth_date = birth_date
        self.reference_date = reference_date
        self.preset = AMModelCulturalPreset.from_country(country)
        self._ref_date = date.fromisoformat(reference_date)
        self._birth = date.fromisoformat(birth_date)
        self._current_age = (self._ref_date - self._birth).days / 365.25

    def compute_age_profile(self, age: int) -> PerAgeMemoryProfile:
        """Compute memory profile for a given age."""
        years_ago = self._current_age - age - 0.5  # midpoint of that year

        temporal_resolution = self._compute_temporal_resolution(age, years_ago)
        am_weight = self._compute_am_weight(age, years_ago)
        max_detail = self._compute_max_detail(age, years_ago, am_weight)
        max_outline = self._compute_max_outline(age, years_ago, am_weight)

        return PerAgeMemoryProfile(
            age=age,
            temporal_resolution=temporal_resolution,
            max_detail_per_year=max_detail,
            max_outline_per_year=max_outline,
            am_weight=am_weight,
        )

    def _compute_temporal_resolution(self, age: int, years_ago: float) -> str:
        """Determine temporal granularity = min(encoding_ceiling, decay_ceiling).

        Encoding ceiling (age-based):
          - age < t1: none
          - t1 <= age < 13: year
          - age >= 13: month

        Decay ceiling (years_ago-based):
          - <=0.25y (3 months): month
          - <=2y:               season
          - >2y:                year

        Final = min(encoding, decay) using order none < year < season < month.
        """
        t1, _, _ = self.preset.childhood_amnesia_thresholds
        RESOLUTION_RANK = {"none": 0, "year": 1, "month": 2}

        # Step 1: Encoding ceiling
        if age < t1:
            encoding = "none"
        elif age < 13:
            encoding = "year"
        else:
            encoding = "month"

        # Step 2: Decay ceiling
        # v8: decay ceiling capped at "year" — season resolution removed.
        # Rationale: season granularity produces too many low-density time units
        # in recent years; unified "year" step produces 1-2 events/year, more natural.
        decay = "year"

        # Step 3: min(encoding, decay)
        if RESOLUTION_RANK[encoding] <= RESOLUTION_RANK[decay]:
            return encoding
        else:
            return decay

    def _compute_am_weight(self, age: int, years_ago: float) -> float:
        """Three-component AM weight (same formula as the old model, per-age version).

        Formula: raw = ca * (0.4 * bump + 0.6 * recency)
        Range: [0.0, 1.0]
        """
        t1, t2, t3 = self.preset.childhood_amnesia_thresholds

        # Childhood amnesia factor
        if age < t1:
            ca = 0.0
        elif age < t2:
            ca = 0.10
        elif age < t3:
            ca = 0.30
        else:
            ca = 1.0

        # Reminiscence bump (Gaussian)
        bump = math.exp(
            -0.5 * ((age - self.preset.bump_center) / self.preset.bump_sigma) ** 2
        )

        # Recency (power-law)
        if years_ago <= 0:
            recency = 1.0
        elif years_ago <= 1:
            recency = 0.95
        else:
            recency = 1.0 / (1.0 + years_ago) ** 0.5

        raw = ca * (0.4 * bump + 0.6 * recency)
        return min(max(raw, 0.0), 1.0)

    def _compute_max_detail(self, age: int, years_ago: float, w: float) -> int:
        """Max episodic-rich (high-res with details) events for this age."""
        if age < 5:
            return 0
        if age < 7:
            return 1 if w >= 0.05 else 0
        if age < 10:
            return 1 if w >= 0.08 else 0
        if age < 13:
            return 1

        is_bump = 13 <= age <= 25
        if is_bump:
            if years_ago <= 2:
                return 3
            elif years_ago <= 5:
                return 2
            elif years_ago <= 10:
                return 1
            else:
                return 1
        else:
            if years_ago <= 1:
                return 3
            elif years_ago <= 3:
                return 2
            elif years_ago <= 8:
                return 1
            else:
                return 1 if w >= 0.10 else 0

    def _compute_max_outline(self, age: int, years_ago: float, w: float) -> int:
        """Max general-event (high-res with outline) events for this age."""
        if age < 3:
            return 0
        if age < 5:
            return 1 if w >= 0.05 else 0
        if age < 7:
            return 1
        if age < 10:
            return 2 if w >= 0.10 else 1
        if age < 13:
            return 2

        is_bump = 13 <= age <= 25
        if is_bump:
            if years_ago <= 2:
                return 4
            elif years_ago <= 5:
                return 3
            elif years_ago <= 10:
                return 2
            else:
                return 1
        else:
            if years_ago <= 1:
                return 4
            elif years_ago <= 3:
                return 3
            elif years_ago <= 8:
                return 2
            else:
                return 1


def compute_period_density(
    period: dict,
    am_model: AutobiographicalMemoryModel,
    birth_date: str,
    is_most_recent: bool = False,
) -> dict:
    """
    Compute density dict for a life period based on per-age AM profiles.

    This replaces the old LLM-generated default_memory_density field.
    Called in P2 at the start of each period processing.

    Returns:
        {
            "time_unit": "year" | "month" | "period",
            "max_detail_events": int,
            "max_outline_events": int,
        }

    time_unit values:
      - "year":  step by year, each time unit ~365 days
      - "month": step by month, each time unit ~30 days
      - "period": fallback when period duration is too short (<30d or <365d)
                  for month/year stepping; entire period as one time unit

    Note: PerAgeMemoryProfile.temporal_resolution has "none"/"year"/"season"/"month".
    "period" is produced by this function during the clamp stage, not by AM Model.
    "none" is promoted to "year" in this function (still generates minimal events).
    """
    birth = date.fromisoformat(birth_date)
    start = date.fromisoformat(period["period_date_range"]["start_date"])
    end = date.fromisoformat(period["period_date_range"]["end_date"])
    period_days = (end - start).days

    start_age = max(0, int((start - birth).days / 365.25))
    end_age = int((end - birth).days / 365.25)

    profiles = []
    for age in range(start_age, end_age + 1):
        profiles.append(am_model.compute_age_profile(age))

    if not profiles:
        return {"time_unit": "year", "max_detail_events": 0, "max_outline_events": 0}

    # Time unit: use the FINEST resolution among all ages in this period
    resolution_order = {"none": 0, "year": 1, "season": 2, "month": 3}
    finest_profile = max(profiles, key=lambda p: resolution_order[p.temporal_resolution])
    time_unit = finest_profile.temporal_resolution

    # If time_unit is "none", treat as "year" (will generate minimal events)
    if time_unit == "none":
        time_unit = "year"

    # If time_unit granularity exceeds period duration, clamp
    # v8: season resolution removed; only year/period clamp needed
    if time_unit == "month":
        # Backward compat: if old data still has month, treat as year
        time_unit = "year"
    if time_unit == "season":
        # v8: season removed, treat as year
        time_unit = "year"
    if time_unit == "year" and period_days < 365:
        time_unit = "period"

    total_max_detail = sum(p.max_detail_per_year for p in profiles)
    total_max_outline = sum(p.max_outline_per_year for p in profiles)

    # ── v3: Literature-driven per-period cap ──
    # Based on autobiographical memory research:
    # - Singer & Salovey (1993): ~10-15 self-defining memories per lifetime
    # - McAdams (2001): ~10-15 nuclear episodes in life story
    # - Berntsen & Rubin (2004): ~7-10 major turning points
    # Target: ~8-12 detail events across entire lifespan.
    period_years = max(0.25, period_days / 365.25)

    # ── High-res gate: extended to 20yr window for outline events ──
    # Rationale: high-res detail events only for the recent 5yr window,
    # but outline events are extended to the 20yr window so older career
    # milestones and formative events can appear as general (outline) events.
    # 20yr covers the full career-formative period for mid-50s personas
    # (e.g. LP7 1995-2009 for a 55yr-old ends ~16yr ago, within 20yr window).
    ref = date.fromisoformat(am_model.reference_date)
    boundary_5yr = ref - timedelta(days=5 * 365)
    boundary_20yr = ref - timedelta(days=20 * 365)
    if end <= boundary_20yr:
        # Period ended more than 20 years ago → no high-res at all
        return {
            "time_unit": time_unit,
            "max_detail_events": 0,
            "max_outline_events": 0,
        }

    # v7: Proportional detail_cap — split period by recency windows and weight
    # Rationale: a period spanning both recent and remote years should get
    # detail proportional to how much of it falls in the recent window.
    # Handles cross-boundary periods naturally via overlap calculation.
    #
    # Time windows (from reference_date backwards):
    #   [R-1yr, R]     → weight 3  (dense behavioral sampling)
    #   [R-3yr, R-1yr] → weight 2  (important events)
    #   [0,     R-3yr] → weight 0  (remote: outline sufficient)
    boundary_1yr = ref - timedelta(days=365)
    boundary_3yr = ref - timedelta(days=3 * 365)

    period_days_f = max(1.0, float(period_days))

    def _overlap_days(seg_start: date, seg_end: date, win_start: date, win_end: date) -> float:
        """Return number of days that [seg_start, seg_end] overlaps with [win_start, win_end]."""
        lo = max(seg_start, win_start)
        hi = min(seg_end, win_end)
        return max(0.0, (hi - lo).days)

    # Window 1: last 1 year → weight 3
    overlap_w1 = _overlap_days(start, end, boundary_1yr, ref)
    # Window 2: last 1-3 years → weight 2
    overlap_w2 = _overlap_days(start, end, boundary_3yr, boundary_1yr)
    # Window 3: older than 3 years → weight 0 (no contribution)

    weighted_detail = (overlap_w1 * 3.0 + overlap_w2 * 2.0) / period_days_f
    detail_cap = int(round(weighted_detail))

    # Guarantee: the most-recent period always gets at least 1 detail event
    # so the agent's latest life chapter is never left without episodic memory.
    if is_most_recent and detail_cap == 0:
        detail_cap = 1

    # outline_cap: based on period length, but must be >= detail_cap
    # (Conway & Pleydell-Pearce, 2000: general events are prerequisites for
    # episodic memories — you cannot select more details than outlines)
    # High-res outline gate: only allow outlines within the 5-year window.
    # For periods that partially overlap the 5-year window, scale outline_cap
    # by the fraction of the period that falls within the window.
    # Outline cap: combine 5yr window (existing) + 5-20yr window (new, outline-only)
    overlap_5yr = _overlap_days(start, end, boundary_5yr, ref)
    fraction_in_5yr_window = overlap_5yr / period_days_f
    outline_cap_by_length = max(1, min(round(period_years * 0.5) + 1, 5))
    outline_cap_in_5yr = max(0, round(outline_cap_by_length * fraction_in_5yr_window))

    # Additional outline budget for 5-20yr window (outline-only, no detail)
    # v4: Increased coefficient from 0.3 to 0.5, and raised max from 2 to 3.
    # Rationale: identity-critical events (coming-out, career pivots, religious conflicts)
    # often occur in the 5-20yr window and should be eligible for high-res simulation.
    # The outline budget increase does NOT affect detail_cap (episodic memory density).
    overlap_5_20yr = _overlap_days(start, end, boundary_20yr, boundary_5yr)
    fraction_in_5_20yr_window = overlap_5_20yr / period_days_f
    # Cap at 1 outline event per year in the 5-20yr window, max 3 total (was 2, v4)
    outline_cap_in_5_20yr = min(3, max(0, round(fraction_in_5_20yr_window * period_years * 0.5)))

    outline_cap = max(outline_cap_in_5yr + outline_cap_in_5_20yr, detail_cap)

    total_max_detail = min(total_max_detail, detail_cap)
    total_max_outline = min(total_max_outline, outline_cap)

    # Invariant: outline >= detail (episodic memories nest inside general events)
    total_max_outline = max(total_max_outline, total_max_detail)

    return {
        "time_unit": time_unit,
        "max_detail_events": total_max_detail,
        "max_outline_events": total_max_outline,
        # New unified keys: outline = high-res (no separate budget)
        "max_high_res_events": total_max_detail,
        "max_medium_events_per_unit": 3,
    }


class AutobiographicalMemoryDistributionModel:
    """DEPRECATED: Use AutobiographicalMemoryModel instead.

    This class is kept for backward compatibility only.
    New code should use AutobiographicalMemoryModel which provides
    per-age profiles instead of period-level weights.

    Literature-driven high-res event frequency allocation model.

    Based on the Three-Component Model of autobiographical memory:
      1. Childhood Amnesia (Pillemer & White, 1989)
      2. Reminiscence Bump (Rubin & Schulkind, 1997)
      3. Recency Effect (Rubin & Wenzel, 1996)
    """

    def __init__(
        self,
        birth_date: str,
        reference_date: str,
        country: str = "",
        preset: Optional[AMModelCulturalPreset] = None,
    ):
        self.birth_date = birth_date
        self.reference_date = reference_date
        self.preset = preset or AMModelCulturalPreset.from_country(country)

    def compute_period_weight(
        self,
        period_start_date: str,
        period_end_date: str,
    ) -> float:
        """Compute relative weight for a life period based on the three-component model.

        Returns a weight in [0.05, 1.0].
        """
        birth = date.fromisoformat(self.birth_date)
        ref = date.fromisoformat(self.reference_date)
        start = date.fromisoformat(period_start_date)
        end = date.fromisoformat(period_end_date)
        midpoint = start + (end - start) / 2

        age_at_mid = (midpoint - birth).days / 365.25
        years_ago = (ref - midpoint).days / 365.25

        # Component 1: Childhood Amnesia
        t1, t2, t3 = self.preset.childhood_amnesia_thresholds
        if age_at_mid < t1:
            childhood_factor = 0.05
        elif age_at_mid < t2:
            childhood_factor = 0.15
        elif age_at_mid < t3:
            childhood_factor = 0.35
        else:
            childhood_factor = 1.0

        # Component 2: Reminiscence Bump (Gaussian)
        bump_factor = math.exp(
            -0.5 * ((age_at_mid - self.preset.bump_center) / self.preset.bump_sigma) ** 2
        )

        # Component 3: Recency Effect (power-law)
        if years_ago <= 0:
            recency_factor = 1.0
        else:
            recency_factor = 1.0 / (1.0 + years_ago) ** 0.5

        # Combine: childhood amnesia as hard ceiling
        raw_weight = childhood_factor * (0.4 * bump_factor + 0.6 * recency_factor)
        return min(max(raw_weight, 0.05), 1.0)

    def compute_period_high_res_cap(
        self,
        period_start_date: str,
        period_end_date: str,
    ) -> int:
        """Compute maximum high-res events for a period.

        Base rate: 2 high-res events per year, scaled by AM weight.
        """
        start = date.fromisoformat(period_start_date)
        end = date.fromisoformat(period_end_date)
        duration_years = (end - start).days / 365.25
        weight = self.compute_period_weight(period_start_date, period_end_date)
        return max(1, round(duration_years * 2.0 * weight))


# ================================================================
# Adaptive Telescoping Policy
# ================================================================

@dataclass
class TelescopingMode:
    """Configuration for a single telescoping mode."""
    name: str                    # "deep" / "standard" / "telescoped"
    turn_range: Tuple[int, int]  # (min_turns, max_turns)
    max_beats: int               # Maximum number of narrative beats
    description: str = ""


# All modes maintain the full 5-phase narrative arc
TELESCOPING_MODES: Dict[str, TelescopingMode] = {
    "deep": TelescopingMode(
        name="deep",
        turn_range=(16, 20),
        max_beats=5,
        description="Full simulation for recent/bump events — extended for beat coverage (v8)",
    ),
    "standard": TelescopingMode(
        name="standard",
        turn_range=(8, 10),
        max_beats=4,
        description="Standard simulation for mid-range events (v2 optimised)",
    ),
    "telescoped": TelescopingMode(
        name="telescoped",
        turn_range=(5, 7),
        max_beats=3,
        description="Compressed simulation for distant/childhood events (v2 optimised)",
    ),
}

# Narrative phases — using "turning_point" instead of "climax" (v6 neutralization)
PHASE_ORDER = ["opening", "development", "turning_point", "resolution", "closing"]


class AdaptiveTelescopingPolicy:
    """Select simulation depth based on AM weight.

    All modes maintain the full 5-phase narrative arc:
    opening → development → turning_point → resolution → closing
    """

    def select_mode(self, am_weight: float) -> TelescopingMode:
        """Select telescoping mode based on AM weight."""
        if am_weight >= 0.7:
            return TELESCOPING_MODES["deep"]
        elif am_weight >= 0.35:
            return TELESCOPING_MODES["standard"]
        else:
            return TELESCOPING_MODES["telescoped"]

    def compute_phase_allocation(
        self, total_turns: int, reserved_turns: int = 2
    ) -> Dict[str, int]:
        """Compute turn allocation per narrative phase.

        Reserves last `reserved_turns` for resolution + closing.
        Each phase gets at least 1 turn.

        Returns:
            Dict mapping phase name to number of turns allocated.
        """
        simulation_turns = total_turns - reserved_turns

        # Proportional allocation for non-reserved phases
        # opening: 15%, development: 40%, turning_point: 45%
        ratios = {"opening": 0.15, "development": 0.40, "turning_point": 0.45}
        allocation = {}
        remaining = simulation_turns

        for phase in ["opening", "development", "turning_point"]:
            turns = max(1, round(simulation_turns * ratios[phase]))
            allocation[phase] = turns
            remaining -= turns

        # Distribute any rounding remainder to development
        if remaining > 0:
            allocation["development"] += remaining
        elif remaining < 0:
            allocation["development"] = max(1, allocation["development"] + remaining)

        # Reserved phases
        if reserved_turns >= 2:
            allocation["resolution"] = 1
            allocation["closing"] = 1
        else:
            allocation["resolution"] = max(1, reserved_turns)
            allocation["closing"] = 0

        return allocation

    def compute_phase_boundaries(
        self, phase_allocation: Dict[str, int]
    ) -> Dict[str, int]:
        """Compute cumulative turn index boundaries for each phase.

        Returns:
            Dict mapping phase name to the turn index where that phase ENDS.
        """
        boundaries = {}
        cumulative = 0
        for phase in PHASE_ORDER:
            cumulative += phase_allocation.get(phase, 0)
            boundaries[phase] = cumulative
        return boundaries
