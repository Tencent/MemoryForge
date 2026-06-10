"""Key Life Path Generator — Rule engine + LLM hybrid generation.

Generates year_enrichment_key_life_path_cache.json by:
1. Rule engine: deterministic fields (grade, exam dates, stage transitions)
2. LLM: descriptive fields (descriptions, institution names, research directions)

The rule engine guarantees factual correctness for deterministic information,
while LLM supplements descriptive details with character limits enforced
in the generation prompt (source-control principle).
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from lifelong_synth.simulation_p1_initialisation.key_life_path_models import (
    CareerStatus,
    EducationStageType,
    EducationStatus,
    ExamEntry,
    KeyEventEntry,
    YearKeyLifePath,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Stage name → EducationStageType mapping
# ═══════════════════════════════════════════════════════════════

_STAGE_NAME_MAP: Dict[str, EducationStageType] = {
    # Preschool / Kindergarten
    "preschool": EducationStageType.PRESCHOOL,
    "kindergarten": EducationStageType.PRESCHOOL,
    "幼儿园": EducationStageType.PRESCHOOL,
    # Primary / Elementary  (keys use spaces; underscores normalised before lookup)
    "primary": EducationStageType.PRIMARY,
    "primary school": EducationStageType.PRIMARY,
    "elementary": EducationStageType.PRIMARY,       # matches "elementary school"
    "小学": EducationStageType.PRIMARY,
    # Junior secondary / Middle school
    "junior secondary": EducationStageType.JUNIOR_SECONDARY,
    "junior high": EducationStageType.JUNIOR_SECONDARY,
    "middle school": EducationStageType.JUNIOR_SECONDARY,
    "初中": EducationStageType.JUNIOR_SECONDARY,
    # Senior secondary / High school
    "senior secondary": EducationStageType.SENIOR_SECONDARY,
    "senior high": EducationStageType.SENIOR_SECONDARY,
    "high school": EducationStageType.SENIOR_SECONDARY,
    "高中": EducationStageType.SENIOR_SECONDARY,
    # Undergraduate / College
    "undergraduate": EducationStageType.UNDERGRADUATE,
    "college": EducationStageType.UNDERGRADUATE,
    "本科": EducationStageType.UNDERGRADUATE,
    # Postgraduate
    "master": EducationStageType.MASTER,
    "硕士": EducationStageType.MASTER,
    "doctoral": EducationStageType.DOCTORAL,
    "博士": EducationStageType.DOCTORAL,
    "phd": EducationStageType.DOCTORAL,
    "postdoc": EducationStageType.POSTDOC,
    "博士后": EducationStageType.POSTDOC,
    # Vocational
    "vocational": EducationStageType.VOCATIONAL,
    "职业": EducationStageType.VOCATIONAL,
}


def _map_stage_name_to_enum(stage_name: str) -> EducationStageType:
    """Map a stage name string to EducationStageType enum.

    Normalises underscores to spaces before matching so that values from
    life_plan.json (e.g. ``"elementary_school"``, ``"high_school"``) are
    handled correctly without requiring duplicate keys.
    """
    # Normalise: lowercase, strip whitespace, replace underscores with spaces
    lower = stage_name.lower().strip().replace("_", " ")
    for key, val in _STAGE_NAME_MAP.items():
        if key in lower:
            return val
    return EducationStageType.NONE


# ═══════════════════════════════════════════════════════════════
# Grade label generation
# ═══════════════════════════════════════════════════════════════

_GRADE_LABELS: Dict[EducationStageType, Dict[int, str]] = {
    EducationStageType.PRIMARY: {
        1: "Year 1", 2: "Year 2", 3: "Year 3",
        4: "Year 4", 5: "Year 5", 6: "Year 6",
    },
    EducationStageType.JUNIOR_SECONDARY: {
        1: "Year 7", 2: "Year 8", 3: "Year 9",
    },
    EducationStageType.SENIOR_SECONDARY: {
        1: "Year 10", 2: "Year 11", 3: "Year 12",
    },
    EducationStageType.UNDERGRADUATE: {
        1: "Year 1", 2: "Year 2", 3: "Year 3", 4: "Year 4",
    },
    EducationStageType.MASTER: {
        1: "Year 1", 2: "Year 2", 3: "Year 3",
    },
    EducationStageType.DOCTORAL: {
        1: "Year 1", 2: "Year 2", 3: "Year 3", 4: "Year 4", 5: "Year 5",
        6: "Year 6", 7: "Year 7",
    },
}


def _grade_label(stage_type: EducationStageType, grade_number: int) -> str:
    """Generate a human-readable grade label."""
    labels = _GRADE_LABELS.get(stage_type, {})
    if grade_number in labels:
        return labels[grade_number]
    # Fallback: generic label
    return f"{stage_type.value} Year {grade_number}"


# ═══════════════════════════════════════════════════════════════
# Rule Engine: Education Timeline
# ═══════════════════════════════════════════════════════════════

def _compute_education_timeline(
    birth_year: int,
    education_system: List[Dict[str, Any]],
    pathway_deviations: List[Dict[str, Any]],
) -> Dict[int, EducationStatus]:
    """Compute deterministic education status for every year.

    For each stage in education_system:
        stage_entry_age = stage["typical_entry_age"]
        stage_duration = stage["typical_duration_years"]
        stage_start_year = birth_year + stage_entry_age
        stage_end_year = stage_start_year + stage_duration - 1

        For year in [stage_start_year, stage_end_year]:
            grade_number = year - stage_start_year + 1
            grade_label = _grade_label(stage_type, grade_number)

    Apply pathway deviations for direct-PhD, skip, extend, etc.
    """
    # Deep copy to avoid mutating the original
    edu_system = deepcopy(education_system)
    # Validate entry_month is properly set
    for stage in edu_system:
        em = stage.get("entry_month", 0)
        if em == 0:
            logger.warning(
                f"EducationStage '{stage.get('stage_name', '?')}' has entry_month=0 "
                f"(not inferred). Defaulting to 9 (Northern Hemisphere). "
                f"This may be incorrect for Southern Hemisphere countries."
            )
            stage["entry_month"] = 9

    # Detect direct-PhD pathway
    is_direct_phd = any(
        "direct_phd" in d.get("pathway_type", "").lower()
        or "直博" in d.get("pathway_type", "")
        or "direct_phd" in d.get("description", "").lower()
        or "直博" in d.get("description", "")
        for d in pathway_deviations
        if d.get("dimension", "") in ("education", "")
    )

    if is_direct_phd:
        # Remove master stage
        edu_system = [
            s for s in edu_system
            if "硕士" not in s.get("stage_name", "")
            and "master" not in s.get("stage_name", "").lower()
        ]
        # Find undergraduate end age to set doctoral entry age
        undergrad_end_age = None
        for s in edu_system:
            sn = s.get("stage_name", "").lower()
            if "undergraduate" in sn or "本科" in s.get("stage_name", ""):
                undergrad_end_age = s.get("typical_entry_age", 18) + s.get("typical_duration_years", 4)
        # Extend doctoral duration (typically 5 years for direct PhD)
        # and adjust entry age to right after undergraduate
        for s in edu_system:
            sn = s.get("stage_name", "").lower()
            if "博士" in s.get("stage_name", "") or "doctoral" in sn or "phd" in sn:
                s["typical_duration_years"] = max(
                    s.get("typical_duration_years", 3), 5
                )
                if undergrad_end_age is not None:
                    s["typical_entry_age"] = undergrad_end_age

    result: Dict[int, EducationStatus] = {}

    for stage in edu_system:
        stage_name = stage.get("stage_name", "")
        stage_type = _map_stage_name_to_enum(stage_name)
        entry_age = stage.get("typical_entry_age", 6)
        duration = stage.get("typical_duration_years", 1)

        stage_start_year = birth_year + entry_age
        stage_end_year = stage_start_year + duration - 1

        for year in range(stage_start_year, stage_end_year + 1):
            grade_number = year - stage_start_year + 1
            label = _grade_label(stage_type, grade_number)
            result[year] = EducationStatus(
                stage_type=stage_type,
                grade_label=label,
                institution="",  # Filled by LLM later
            )

    return result


# ═══════════════════════════════════════════════════════════════
# Rule Engine: Exam Timeline
# ═══════════════════════════════════════════════════════════════

def _compute_exams_timeline(
    birth_year: int,
    education_system: List[Dict[str, Any]],
    pathway_deviations: List[Dict[str, Any]],
    country: str = "",
) -> Dict[int, List[ExamEntry]]:
    """Compute exam schedule for every year based on education system.

    Country-aware: only injects China-specific exams if country is China.
    For all other countries, the exam timeline is left empty here and will
    be filled by the LLM enrichment step (_llm_enrich_skeleton) which uses
    the persona's country to infer the correct exam system.

    China-specific exam patterns (only applied when country == "china"):
    - Junior secondary exit exam (中考): last year of junior_secondary, June
    - Senior secondary academic proficiency test (会考): 2nd year of senior_secondary, June
    - University entrance exam (高考): last year of senior_secondary, June 7-8
    - Graduate entrance exam (考研): last year of undergraduate, December
    - Doctoral qualifying exam: 2nd year of doctoral, May-June
    """
    result: Dict[int, List[ExamEntry]] = {}

    # Normalize country
    from lifelong_synth.configs.temporal_context import _normalize_country
    normalized_country = _normalize_country(country) if country else ""
    is_china = normalized_country == "china"

    # Detect direct-PhD (language-agnostic check)
    is_direct_phd = any(
        "direct_phd" in d.get("pathway_type", "").lower()
        or "直博" in d.get("pathway_type", "")
        or "direct_phd" in d.get("description", "").lower()
        or "直博" in d.get("description", "")
        for d in pathway_deviations
        if d.get("dimension", "") in ("education", "")
    )

    if not is_china:
        # For non-China countries, return empty dict.
        # The LLM enrichment step will inject country-appropriate exams
        # based on the persona's country (e.g., NCEA for NZ, A-Levels for UK, SAT for US).
        return result

    # China-specific exam rules
    for stage in education_system:
        stage_name = stage.get("stage_name", "").lower()
        entry_age = stage.get("typical_entry_age", 6)
        duration = stage.get("typical_duration_years", 1)
        stage_start_year = birth_year + entry_age
        stage_end_year = stage_start_year + duration - 1

        # Junior secondary: middle school exit exam in last year
        if _map_stage_name_to_enum(stage_name) == EducationStageType.JUNIOR_SECONDARY:
            exam_year = stage_end_year
            result.setdefault(exam_year, []).append(ExamEntry(
                exam_name="Middle School Exit Exam (中考)",
                date_range=f"{exam_year}-06-20~22",
                description="Junior secondary graduation exam determining high school admission",
            ))

        # Senior secondary: academic proficiency test in 2nd year, university entrance in last year
        if _map_stage_name_to_enum(stage_name) == EducationStageType.SENIOR_SECONDARY:
            if duration >= 2:
                huikao_year = stage_start_year + 1
                result.setdefault(huikao_year, []).append(ExamEntry(
                    exam_name="Academic Proficiency Test (会考)",
                    date_range=f"{huikao_year}-06",
                    description="High school academic proficiency standardized test",
                ))
            exam_year = stage_end_year
            result.setdefault(exam_year, []).append(ExamEntry(
                exam_name="National University Entrance Exam (高考)",
                date_range=f"{exam_year}-06-07~08",
                description="National unified university entrance exam determining university admission",
            ))

        # Undergraduate: graduate entrance exam in last year (only if not direct-PhD)
        if _map_stage_name_to_enum(stage_name) == EducationStageType.UNDERGRADUATE:
            if not is_direct_phd:
                has_postgrad = any(
                    _map_stage_name_to_enum(s.get("stage_name", "")) in (
                        EducationStageType.MASTER, EducationStageType.DOCTORAL
                    )
                    for s in education_system
                )
                if has_postgrad:
                    exam_year = stage_end_year - 1 if duration >= 4 else stage_end_year
                    result.setdefault(exam_year, []).append(ExamEntry(
                        exam_name="Graduate Entrance Exam (考研/推免)",
                        date_range=f"{exam_year}-12-24~25",
                        description="Graduate school entrance exam or recommendation interview",
                    ))

        # Doctoral: qualifying exam in 2nd year
        if _map_stage_name_to_enum(stage_name) == EducationStageType.DOCTORAL:
            if duration >= 2:
                qual_year = stage_start_year + 1
                result.setdefault(qual_year, []).append(ExamEntry(
                    exam_name="Doctoral Qualifying Exam",
                    date_range=f"{qual_year}-05~06",
                    description="Doctoral qualifying assessment; upon passing, formally enters dissertation research phase",
                ))

    return result


# ═══════════════════════════════════════════════════════════════
# Rule Engine: Key Events Timeline
# ═══════════════════════════════════════════════════════════════

def _compute_key_events_timeline(
    birth_year: int,
    education_system: List[Dict[str, Any]],
    pathway_deviations: List[Dict[str, Any]],
) -> Dict[int, List[KeyEventEntry]]:
    """Compute key life events (graduations, enrollments, stage transitions, family events)."""
    result: Dict[int, List[KeyEventEntry]] = {}

    # Detect direct-PhD
    is_direct_phd = any(
        "direct_phd" in d.get("pathway_type", "").lower()
        or "直博" in d.get("pathway_type", "")
        or "direct_phd" in d.get("description", "").lower()
        or "直博" in d.get("description", "")
        for d in pathway_deviations
        if d.get("dimension", "") in ("education", "")
    )

    edu_system = deepcopy(education_system)
    if is_direct_phd:
        edu_system = [
            s for s in edu_system
            if "硕士" not in s.get("stage_name", "")
            and "master" not in s.get("stage_name", "").lower()
        ]
        for s in edu_system:
            sn = s.get("stage_name", "").lower()
            if "博士" in s.get("stage_name", "") or "doctoral" in sn or "phd" in sn:
                s["typical_duration_years"] = max(
                    s.get("typical_duration_years", 3), 5
                )

    for i, stage in enumerate(edu_system):
        stage_name = stage.get("stage_name", "")
        stage_type = _map_stage_name_to_enum(stage_name)
        entry_age = stage.get("typical_entry_age", 6)
        duration = stage.get("typical_duration_years", 1)
        stage_start_year = birth_year + entry_age
        stage_end_year = stage_start_year + duration - 1

        # Enrollment event (first year of stage, except preschool)
        if stage_type != EducationStageType.PRESCHOOL:
            enroll_desc = f"Enrolled in {stage_name}"
            if is_direct_phd and stage_type == EducationStageType.DOCTORAL:
                enroll_desc = "Enrolled in doctoral program directly (skipping master's)"
            result.setdefault(stage_start_year, []).append(KeyEventEntry(
                event_name=f"{stage_name} enrollment",
                date=f"{stage_start_year}-09-01",
                description=enroll_desc,
                is_stage_transition=True,
            ))

        # Graduation event (last year of stage)
        if stage_type not in (EducationStageType.PRESCHOOL, EducationStageType.NONE):
            result.setdefault(stage_end_year, []).append(KeyEventEntry(
                event_name=f"{stage_name} graduation",
                date=f"{stage_end_year}-06",
                description=f"Completed {stage_name}",
                is_stage_transition=True,
            ))

    # ── Family dimension events ──
    # Extract family INSERT / EXTEND deviations and convert to KeyEventEntry objects.
    # This ensures family events (marriage, widowhood, divorce, birth of child) are
    # visible in the key_life_path_cache and therefore to downstream milestone selection.
    _FAMILY_EVENT_PATTERNS: List[Dict[str, Any]] = [
        {
            "keywords": ["marriage", "married", "wedding", "partner", "civil partnership",
                         "结婚", "婚姻", "婚礼"],
            "event_name": "Marriage / Partnership formation",
            "description": "Entered into marriage or long-term partnership",
            "is_transition": True,
        },
        {
            "keywords": ["widow", "widowhood", "bereave", "spousal loss", "spousal_loss",
                         "丧偶", "失去配偶"],
            "event_name": "Spousal loss / Widowhood",
            "description": "Experienced spousal loss (bereavement / widowhood)",
            "is_transition": True,
        },
        {
            "keywords": ["divorc", "separation", "relationship breakdown",
                         "离婚", "分居"],
            "event_name": "Divorce / Relationship separation",
            "description": "Experienced divorce or relationship separation",
            "is_transition": True,
        },
        {
            "keywords": ["birth of child", "first child", "new child", "parenthood",
                         "became parent", "newborn", "childbirth", "adoption",
                         "生子", "生育", "成为父母", "收养"],
            "event_name": "Birth / Adoption of child",
            "description": "Became a parent (birth or adoption of child)",
            "is_transition": True,
        },
        {
            "keywords": ["remarri", "second marriage", "new partner after",
                         "再婚"],
            "event_name": "Remarriage / New partnership",
            "description": "Entered into remarriage or new long-term partnership",
            "is_transition": True,
        },
    ]

    for dim in pathway_deviations:
        if dim.get("dimension", "") != "family":
            continue

        # Check pathway_type and key_stages for family event signals
        pathway_type = dim.get("pathway_type", "").lower()
        key_stages = dim.get("key_stages", [])
        deviations = dim.get("deviations_from_typical", [])

        # Collect all text fields to search for keywords
        searchable_text = " ".join([
            pathway_type,
            " ".join(str(ks) for ks in key_stages),
            " ".join(str(dv) for dv in deviations),
        ]).lower()

        # Try to extract an approximate year from key_stages entries
        # key_stages may contain entries like "spousal_loss(widowhood; ~2015)"
        # or "adult_partnership_or_marriage(inferred from widower status)"
        def _extract_approx_year(text: str, ref_birth_year: int) -> Optional[int]:
            """Try to extract an approximate year from a key_stage string.

            Priority:
            1. Explicit year like (~2015) or (c.2015) or (2015)
            2. Age-based like (age 55) or (at 55)
            3. Fallback: None (caller must decide default)
            """
            import re
            # Explicit year
            year_match = re.search(r'[~c.]?\s*(\d{4})', text)
            if year_match:
                yr = int(year_match.group(1))
                if 1900 <= yr <= 2100:
                    return yr
            # Age-based
            age_match = re.search(r'(?:age|at)\s*(\d{1,2})', text, re.IGNORECASE)
            if age_match:
                age = int(age_match.group(1))
                if 0 <= age <= 120:
                    return ref_birth_year + age
            return None

        for pattern in _FAMILY_EVENT_PATTERNS:
            # Check if any keyword from this pattern appears in searchable text
            if not any(kw in searchable_text for kw in pattern["keywords"]):
                continue

            # Try to find the best approximate year from key_stages
            best_year = None
            for ks in key_stages:
                ks_str = str(ks).lower()
                if any(kw in ks_str for kw in pattern["keywords"]):
                    best_year = _extract_approx_year(ks_str, birth_year)
                    if best_year:
                        break

            # If no year found from key_stages, try deviations
            if best_year is None:
                for dv in deviations:
                    dv_str = str(dv).lower()
                    if any(kw in dv_str for kw in pattern["keywords"]):
                        best_year = _extract_approx_year(dv_str, birth_year)
                        if best_year:
                            break

            # If still no year, try pathway_type itself
            if best_year is None:
                best_year = _extract_approx_year(pathway_type, birth_year)

            if best_year is not None:
                result.setdefault(best_year, []).append(KeyEventEntry(
                    event_name=pattern["event_name"],
                    date=f"{best_year}",
                    description=pattern["description"],
                    is_stage_transition=pattern["is_transition"],
                ))

    return result


# ═══════════════════════════════════════════════════════════════
# LLM Enrichment (optional — can be skipped for smoke test)
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_KEY_LIFE_PATH = (
    "You are a life path data annotation expert. Based on the rule-engine pre-generated annual "
    "path skeleton, fill in descriptive information for each field."
    "\n\nAll output must be in English."
    "\n\n⚠️ Word limits (soft limits; up to 10% over is acceptable):"
    "\n- exam description: ≤80 words"
    "\n- event description: ≤60 words"
    "\n- career description: ≤120 words"
    "\n- pathway_notes: ≤120 words"
    "\n- grade_label: ≤20 words"
    "\n- research_direction: ≤80 words"
    "\n\n⚠️ Critical constraints:"
    "\n1. DO NOT modify deterministic fields generated by the rule engine (year, age, stage_type, grade_label, date_range)"
    "\n2. Only fill in descriptive fields (description, institution, research_direction, pathway_notes)"
    "\n3. Descriptions must be consistent with the inferred path; do NOT fabricate events not indicated by the rule engine"
    "\n4. All information must be based on the REAL education and career system of the persona's country. "
    "Do NOT assume Chinese norms unless the persona is explicitly from China. "
    "Use the correct local exam names, grade systems, and institutional structures for the persona's country."
)


async def _llm_enrich_skeleton(
    skeleton: Dict[int, YearKeyLifePath],
    llm_client,
    persona_config: Dict[str, Any],
    life_plan: Dict[str, Any],
) -> Dict[int, YearKeyLifePath]:
    """Use LLM to enrich descriptive fields in the skeleton.

    If LLM is not available, returns the skeleton as-is (for smoke testing).
    """
    if llm_client is None:
        logger.info("[KeyLifePath] No LLM client provided, returning skeleton without enrichment")
        return skeleton

    # Build a compact representation for the LLM
    years_sorted = sorted(skeleton.keys())
    if not years_sorted:
        return skeleton

    persona_brief = persona_config.get("persona_brief_text", "")
    global_summary = life_plan.get("global_summary", {})
    pathway_info = global_summary.get("inferred_persona_pathway", {})

    skeleton_data = {}
    for year in years_sorted:
        lp = skeleton[year]
        skeleton_data[str(year)] = {
            "year": lp.year,
            "age": lp.age,
            "education_status": {
                "stage_type": lp.education_status.stage_type.value,
                "grade_label": lp.education_status.grade_label,
            "institution": "[TBD]",
                "research_direction": "[TBD]" if lp.education_status.stage_type in (
                    EducationStageType.DOCTORAL, EducationStageType.MASTER, EducationStageType.POSTDOC
                ) else None,
            },
            "exams_this_year": [
                {
                    "exam_name": e.exam_name,
                    "date_range": e.date_range,
                    "description": e.description,
                }
                for e in lp.exams_this_year
            ],
            "key_events": [
                {
                    "event_name": e.event_name,
                    "date": e.date,
                    "description": "[TBD]",
                    "is_stage_transition": e.is_stage_transition,
                }
                for e in lp.key_events
            ],
            "career_status": {
                "is_working": False,
                "description": "[TBD]" if lp.age >= 22 else "",
            },
            "pathway_notes": "[TBD]",
        }

    user_prompt = (
        f"## Persona Brief\n{persona_brief}\n\n"
        f"## Inferred Pathway Info\n{json.dumps(pathway_info, ensure_ascii=False, indent=2)}\n\n"
        f"## Rule-engine pre-generated annual path skeleton for {years_sorted[0]}-{years_sorted[-1]}\n"
        f"Please fill in the descriptive fields marked as [TBD].\n\n"
        f"{json.dumps(skeleton_data, ensure_ascii=False, indent=2)}\n\n"
        f"Output the complete JSON, keeping all deterministic fields unchanged. "
        f"Only fill in the descriptive fields. Output format same as input."
    )

    try:
        from pydantic import BaseModel, Field as PydanticField

        class EnrichedYearEntry(BaseModel):
            """Single year enriched entry."""
            year: int
            age: int
            education_status: Dict[str, Any]
            exams_this_year: List[Dict[str, Any]] = []
            key_events: List[Dict[str, Any]] = []
            career_status: Dict[str, Any] = {}
            pathway_notes: str = ""

        class EnrichedOutput(BaseModel):
            """LLM output for enriched year entries."""
            years: Dict[str, EnrichedYearEntry] = PydanticField(
                ..., description="Mapping from year string to enriched year entry"
            )

        result: EnrichedOutput = await llm_client.generate_structured(
            prompt=user_prompt,
            response_model=EnrichedOutput,
            system_prompt=_SYSTEM_PROMPT_KEY_LIFE_PATH,
            temperature=0.0,
            task_type="p1a_year_enrichment",
        )

        # Merge LLM enrichment back into skeleton
        for year_str, enriched in result.years.items():
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year not in skeleton:
                continue

            lp = skeleton[year]

            # Update institution
            inst = enriched.education_status.get("institution", "")
            if inst and inst != "[TBD]":
                lp.education_status.institution = inst

            # Update research direction
            rd = enriched.education_status.get("research_direction")
            if rd and rd != "[TBD]":
                lp.education_status.research_direction = rd

            # Update key event descriptions
            enriched_events = enriched.key_events or []
            for i, evt in enumerate(lp.key_events):
                if i < len(enriched_events):
                    desc = enriched_events[i].get("description", "")
                    if desc and desc != "[TBD]":
                        lp.key_events[i].description = desc

            # Update career status
            career = enriched.career_status or {}
            if career.get("is_working"):
                lp.career_status = CareerStatus(
                    is_working=True,
                    job_title=career.get("job_title", ""),
                    employer=career.get("employer", ""),
                    industry=career.get("industry", ""),
                    description=career.get("description", ""),
                )

            # Update pathway notes
            notes = enriched.pathway_notes or ""
            if notes and notes != "[TBD]":
                lp.pathway_notes = notes

        logger.info(f"[KeyLifePath] LLM enrichment completed for {len(result.years)} years")

    except Exception as e:
        logger.warning(f"[KeyLifePath] LLM enrichment failed, using skeleton only: {e}")

    return skeleton


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

async def generate_key_life_path_cache(
    life_plan: Dict[str, Any],
    persona_config: Dict[str, Any],
    output_dir: str,
    llm_client=None,
) -> Dict[int, YearKeyLifePath]:
    """Generate the year-level key life path enrichment cache.

    Steps:
    1. Extract inferred_persona_pathway and education_system from life_plan
    2. Run rule engine to generate deterministic fields for all years
    3. Call LLM to enrich descriptive fields (with char limits in prompt)
    4. Validate field lengths (warn but don't truncate — same as Phase 0.3)
    5. Save to year_enrichment_key_life_path_cache.json

    Args:
        life_plan: The complete life plan dict from P1.
        persona_config: The persona configuration dict.
        output_dir: Directory to save the cache file.
        llm_client: Optional AsyncLLMClient for LLM enrichment.

    Returns:
        Dict mapping year -> YearKeyLifePath
    """
    # Step 1: Extract inputs
    global_summary = life_plan.get("global_summary", {})
    social_context = global_summary.get("social_context", {})
    education_system = social_context.get("education_system", [])
    persona_pathway = global_summary.get("inferred_persona_pathway", {})
    dimensions = persona_pathway.get("dimensions", [])

    birth_date_str = (
        persona_config.get("derived_birth_date")
        or persona_config.get("birth_date_str", "")
    )
    if not birth_date_str:
        # Try to derive from timeline_anchor
        timeline_anchor = global_summary.get("timeline_anchor", {})
        birth_date_str = timeline_anchor.get("derived_birth_date", "")

    if not birth_date_str:
        logger.warning("[KeyLifePath] No birth date found, cannot generate key life path cache")
        return {}

    birth_year = int(birth_date_str.split("-")[0])
    logger.info(f"[KeyLifePath] Birth year: {birth_year}")
    logger.info(f"[KeyLifePath] Education system stages: {len(education_system)}")
    logger.info(f"[KeyLifePath] Pathway dimensions: {len(dimensions)}")

    # Step 2: Rule engine — compute deterministic fields
    edu_timeline = _compute_education_timeline(birth_year, education_system, dimensions)
    exams_timeline = _compute_exams_timeline(
        birth_year, education_system, dimensions,
        country=persona_config.get("growing_up_location", {}).get("country", ""),
    )
    events_timeline = _compute_key_events_timeline(birth_year, education_system, dimensions)

    # Build skeleton for all years
    all_years = sorted(set(
        list(edu_timeline.keys())
        + list(exams_timeline.keys())
        + list(events_timeline.keys())
    ))

    if not all_years:
        logger.warning("[KeyLifePath] No years computed from education system")
        return {}

    skeleton: Dict[int, YearKeyLifePath] = {}
    for year in all_years:
        age = year - birth_year
        skeleton[year] = YearKeyLifePath(
            year=year,
            age=age,
            education_status=edu_timeline.get(
                year,
                EducationStatus(stage_type=EducationStageType.NONE, grade_label=""),
            ),
            exams_this_year=exams_timeline.get(year, []),
            key_events=events_timeline.get(year, []),
            career_status=CareerStatus(),
            pathway_notes="",
        )

    logger.info(f"[KeyLifePath] Rule engine generated skeleton for {len(skeleton)} years ({all_years[0]}-{all_years[-1]})")

    # Step 3: LLM enrichment — fill descriptive fields
    enriched = await _llm_enrich_skeleton(skeleton, llm_client, persona_config, life_plan)

    # Step 4: Validate field lengths (warn only, never truncate)
    for year, life_path in enriched.items():
        for exam in life_path.exams_this_year:
            if len(exam.description) > 100 * 1.2:
                logger.warning(
                    f"[KeyLifePath] Year {year}, exam '{exam.exam_name}' "
                    f"description exceeds limit: {len(exam.description)} chars"
                )
        for evt in life_path.key_events:
            if len(evt.description) > 80 * 1.2:
                logger.warning(
                    f"[KeyLifePath] Year {year}, event '{evt.event_name}' "
                    f"description exceeds limit: {len(evt.description)} chars"
                )
        if len(life_path.pathway_notes) > 150 * 1.2:
            logger.warning(
                f"[KeyLifePath] Year {year}, pathway_notes exceeds limit: "
                f"{len(life_path.pathway_notes)} chars"
            )

    # Step 5: Save to cache file
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "year_enrichment_key_life_path_cache.json")
    serializable = {str(year): data.model_dump() for year, data in enriched.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info(f"[KeyLifePath] Saved {len(enriched)} year entries to {cache_path}")

    return enriched
