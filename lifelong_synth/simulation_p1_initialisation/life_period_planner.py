import asyncio
import copy
import json
import logging
import math
import os
from calendar import monthrange
from datetime import date, timedelta
from typing import List, Literal, Dict, Any, Optional, Tuple

# Import data models from centralised definition module
from lifelong_synth.simulation_p1_initialisation.definition import (
    GlobalSummary,
    PeriodDateRange,
    LifePeriod,
    MilestonePlan,
    PlanPatchAction,
    PlanReviewReport,
    PlanDurationValidationReport,
    PlanValidationReport,
    PeriodContentRefresh,
    TransitionDateHint,
    InferredTemporalConstraints,
    TransitionDateHints,
    SocialContextProfile,
    InferredPersonaPathway,
    PathwayDimension,
    CausalAttribution,
    ErrorAttributionReport,
    BackwardPeriodOutput,
    FixImpactAnalysis,
    PeriodJudgement,
    PlanJudgementReport,
)

# Import split LLM client
from llm.client import AsyncLLMClient

# Import persona extension rendering engine
from lifelong_synth.persona_extensions_formatter import format_persona_extensions

logger = logging.getLogger(__name__)

# ==========================================
# Development-Aware Life-Period Planner
# ==========================================
class DevelopmentAwareLifePeriodPlanner:
    """
    Receives target end-state persona constraints and generates a developmentally sound
    life period blueprint from early developmental stages to the target age.
    Serves as the scaffold for downstream Life Simulator and Memory Synthesis.
    """
    
    def __init__(
        self, 
        llm_client: AsyncLLMClient,
        schema_path: str = "",
        enable_general_plan_validator: bool = False,
        validator_model: Optional[str] = None,
        transition_date_model: Optional[str] = None,
        enable_step_print: bool = True,
        institution_naming_policy: str = "real",
        recency_window_years: int = 5,
        childhood_amnesia_age: int = 3,
    ):
        self.llm = llm_client
        if not schema_path:
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "persona_schema.json"
            )
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.enable_general_plan_validator = enable_general_plan_validator
        self.validator_model = validator_model
        self.transition_date_model = transition_date_model
        self.enable_step_print = enable_step_print
        self.institution_naming_policy = institution_naming_policy
        self._recency_window_years = recency_window_years
        self._childhood_amnesia_age = childhood_amnesia_age
        
        self.system_prompt = """
        You are a professional developmental psychologist and senior life trajectory planner.
        Your goal is to map a set of target end-state persona constraints into a highly plausible
        life-stage blueprint that conforms to developmental psychology principles.

        Core objectives:
        1. "End-state constraints -> Stage structure": Given target age, occupation, personality traits,
           and early background (including a rich persona brief), plan a life trajectory that genuinely
           converges to the given end-state.
        2. "Developmental plausibility": Ensure psychological and behavioral traits match the
           developmental age (e.g., childhood emphasizes attachment/care, adolescence emphasizes
           identity/peer comparison, early adulthood emphasizes career path narrowing).
           Do NOT project adult characteristics into childhood.
        3. "Scaffold support": Generate the macro "chapters" of the character's life. Do NOT generate
           daily logs, specific events, or dialogues. Focus on stage divisions, core themes,
           developmental tasks, and typical stressors.
        4. "Identify and model gap periods": If there are waiting periods between school admissions,
           gaps between graduation and employment, career transition buffers, travel breaks, etc.,
           output them as separate stages (stage_label MUST be set to 'gap_transition'), and provide
           that gap stage's dominant theme, tasks, and stress/opportunity profile.
        5. "Autonomously generate stage labels": stage_label is a free-form string field (snake_case).
           Generate the most semantically fitting label based on the actual content of each stage,
           rather than applying fixed developmental psychology categories. For example, if a PhD
           period involves a major research direction shift, split it into 'phd_hypergraph_research'
           and 'phd_llm_pivot' instead of broadly labeling it 'emerging_adulthood'.
           The only hard rule: transition/gap periods MUST use 'gap_transition'.

        Internal workflow:
        - Step 1: Based on early childhood background, infer the "necessary life conditions" required
          to produce the final persona characteristics, values, and specific career outcomes.
        - Step 2: Distribute these conditions across the full life span according to developmental
          psychology principles.
        - Step 3: Formalize these distributions as consecutive Life Periods, seamlessly covering from
          birth to `reference_date` by date.
        - Step 4: Check whether there are explicable gap periods between key anchors, and output
          separate gap stages as necessary.

        You MUST output a valid JSON dictionary strictly following the specified JSON Schema.

        IMPORTANT: All output text fields (title, dominant_theme, developmental_tasks, etc.) must be
        in English. The persona may be from any country; reflect their cultural background accurately
        in the content, but write all field values in English.

        ## Institution naming rules
        """ + self._get_institution_naming_prompt() + """

        ## Stage type annotation rules
        Each stage must annotate three boolean fields:
        - `is_education_stage`: Whether this stage includes educational/learning activities
          (kindergarten, primary school, secondary school, university, graduate school,
          doctoral programs, etc. are all True)
        - `is_work_stage`: Whether this stage includes work/professional activities
          (internships, full-time work, research work, entrepreneurship, etc. are all True)
          Note: Doctoral stages involving research work (publishing papers, conducting experiments)
          should be marked is_education_stage=True AND is_work_stage=True simultaneously
        - `is_transition_stage`: Whether this stage is a transition/gap period
          (waiting for school admission, gap between graduation and employment,
          career transition buffers, etc. are True)
          Note: stages with stage_label='gap_transition' MUST be marked True
        - One stage can have multiple types simultaneously (e.g., doctoral = education + work)
        - Infancy, pure life stages, etc. have all three flags as False

        Special handling for extreme-age personas:
        - For personas aged 10-17 (children/adolescents): ensure life periods reflect age-appropriate cognitive development. Speech patterns should be simpler, more concrete, and peer-focused. Academic performance, family dynamics, and peer relationships are the primary themes. Do NOT project adult concerns (career anxiety, financial stress) into childhood periods.
        - For personas aged 70+ (elderly): ensure at least one life period explicitly covers a formative historical event from their youth (e.g., a WWII veteran's wartime experience, a Cold War-era scientist's career context). These distant memories are often the most vivid and defining for elderly personas — treat them as high-salience events even if they occurred 50+ years ago.

        Early Childhood Period Rule:
        When the persona's birth date precedes the first formal education stage (e.g., elementary school, primary school) by MORE THAN 4 years, you MUST generate a SEPARATE 'early_childhood' period covering birth to school entry. Do NOT merge early childhood (ages 0-4) into the elementary school period. The early_childhood period should have:
          - stage_label: 'early_childhood'
          - period_date_range: birth_date to (first_school_entry_date - 1 day)
          - dominant_theme: pre-linguistic and sensorimotor development, family bonding
          - is_education_stage: false
        This rule applies universally regardless of country or culture. The early_childhood period will receive childhood-amnesia treatment in the memory layer (no episodic events), so it does not add simulation cost.
        """

    def _get_institution_naming_prompt(self) -> str:
        """Return institution naming rules based on the configured policy."""
        if self.institution_naming_policy == "fictional":
            return (
                "- Use fictional names for schools, companies, hospitals, and other institutions "
                "(do not use real institution names)\n"
                "- Fictional names should conform to local naming conventions of the persona's "
                "country/city, sounding plausible but not matching real institutions\n"
                "- For workplaces, if the persona already specifies a real company name, "
                "replace it with a fictional equivalent"
            )
        elif self.institution_naming_policy == "anonymized":
            return (
                "- Use anonymized institution names (e.g., 'a top-ranked university', "
                "'a major tech company', 'a large general hospital')\n"
                "- Do not reveal any identifiable real institution information\n"
                "- For workplaces, if the persona already specifies a real company name, "
                "anonymize it"
            )
        else:  # "real" (default)
            return (
                "- Use real names for schools, companies, hospitals, and other institutions\n"
                "- If the persona mentions a specific city and institution type "
                "(e.g., 'a university in Auckland'), use the real corresponding institution "
                "in that city (e.g., 'University of Auckland')\n"
                "- If the specific institution cannot be determined, use the most well-known "
                "institution of that type in the region\n"
                "- For workplaces, if the persona already specifies a real company name, use it directly"
            )

    def _load_schema(self) -> dict:
        """
        Load the unified persona schema configuration file.
        """
        if os.path.exists(self.schema_path):
            try:
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read schema config file {self.schema_path}: {e}. Returning empty dict.")
                return {}
        else:
            logger.warning(f"Schema config file {self.schema_path} does not exist. Please check the path.")
            return {}

    def _print_step(self, step_no: str, title: str, detail: Optional[str] = None):
        """
        Uniformly print planner execution steps for monitoring current progress.
        """
        if not self.enable_step_print:
            return

        header = f"[Planner Step {step_no}] {title}"
        sep = "=" * len(header)
        logger.info("\n" + sep)
        logger.info(header)
        logger.info(sep)
        if detail:
            logger.info(detail)

    def _get_label_from_schema(self, path_keys: List[str], value_key: str) -> str:
        """
        Helper: look up the Chinese/English label for a value in the schema by nested key path.
        Returns format like "China (China)" or falls back to value_key.
        """
        if not self.schema:
            return str(value_key)
            
        current = self.schema
        try:
            for key in path_keys:
                current = current[key]
                
            if str(value_key) in current:
                item = current[str(value_key)]
                label_zh = item.get("label_zh", "")
                label_en = item.get("label_en", "")
                if label_zh and label_en:
                    return f"{label_zh} ({label_en})"
                return label_zh or label_en or str(value_key)
        except KeyError:
            pass
            
        return str(value_key)

    def _translate_codes(self, constraint_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper: translate constraint_sheet code values to rich text
        labels via the persona schema. Adapted for the simplified field names.
        """
        translated = constraint_sheet.copy()
        
        # 1. Occupation mapping (target_occupation_group -> text)
        occ_val = translated.get("target_occupation_group")
        if occ_val is not None:
            translated["target_occupation_text"] = self._get_label_from_schema(["target_occupation_group", "allowed_values"], occ_val)
            
        # 2. Education mapping (target_education_level, string code "0"-"8")
        edu_val = translated.get("target_education_level")
        if edu_val is not None:
            translated["target_education_text"] = self._get_label_from_schema(["target_education_level", "allowed_values"], str(edu_val))
            
        # 3. Location mapping (nested dict with country code)
        for loc_key in ["growing_up_location", "current_living_location"]:
            loc_data = translated.get(loc_key)
            if isinstance(loc_data, dict) and "country" in loc_data:
                country_code = loc_data["country"]
                loc_data["country_text"] = self._get_label_from_schema([loc_key, "country", "allowed_values"], country_code)
                translated[loc_key] = loc_data

        # 4. Language mapping (codes are already "zh"/"en", matching schema directly)
        primary_lang = translated.get("primary_language")
        if primary_lang is not None:
            translated["primary_language_text"] = self._get_label_from_schema(["language_options", "allowed_values"], primary_lang)
            
        working_langs = translated.get("working_language")
        if isinstance(working_langs, list):
            translated_langs = []
            for lang in working_langs:
                translated_langs.append(self._get_label_from_schema(["language_options", "allowed_values"], lang))
            translated["working_language_text"] = translated_langs
            
        return translated

    @staticmethod
    def _subtract_years(source_date: date, years: int) -> date:
        """
        Subtract N years from the given date, auto-handling leap days (e.g., 2/29 -> 2/28).
        """
        try:
            return source_date.replace(year=source_date.year - years)
        except ValueError:
            return source_date.replace(month=2, day=28, year=source_date.year - years)

    @staticmethod
    def _add_years(source_date: date, years: int) -> date:
        """
        Add N years to the given date, auto-handling leap days (e.g., 2/29 -> 2/28).
        """
        try:
            return source_date.replace(year=source_date.year + years)
        except ValueError:
            return source_date.replace(month=2, day=28, year=source_date.year + years)

    @staticmethod
    def _shift_months(source_date: date, months: int) -> date:
        """
        Shift the given date by N months (positive or negative), auto-clipping to the target month's last day.
        """
        month_index = source_date.year * 12 + (source_date.month - 1) + months
        target_year = month_index // 12
        target_month = month_index % 12 + 1
        target_day = min(source_date.day, monthrange(target_year, target_month)[1])
        return date(target_year, target_month, target_day)

    @staticmethod
    def _parse_iso_date(value: Any) -> Optional[date]:
        """
        Attempt to parse an ISO date string (YYYY-MM-DD); return None on failure.
        """
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None

    @staticmethod
    def _compute_exact_age(birth_date: date, on_date: date) -> int:
        """
        Compute the exact age (in whole years) on the given date.
        """
        years = on_date.year - birth_date.year
        if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
            years -= 1
        return max(0, years)

    @staticmethod
    def _compute_age_display(birth_date: date, on_date: date) -> float:
        """
        Compute a continuous age for display / LLM prompt purposes that never
        rounds UP across a birthday boundary.  For example, if the persona is
        1 day short of turning 22, this returns 21.9 instead of 22.0.
        """
        return math.floor((on_date - birth_date).days / 365.25 * 10) / 10

    def _resolve_timeline_anchor(self, constraint_sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uniformly resolve timeline anchors: supports optional inputs
        - birth_date: date of birth (YYYY-MM-DD)
        - simulation_end_date: simulation end date (YYYY-MM-DD)

        Rule priority:
        1) reference_date prefers simulation_end_date, otherwise today;
        2) If birth_date is given, derive target_age_exact from birth_date + reference_date;
        3) Otherwise use target_age_exact to derive the "earliest birthday semantics" birth_date.
        """
        input_reference = self._parse_iso_date(constraint_sheet.get("simulation_end_date"))
        reference_date = input_reference or date.today()

        input_birth = self._parse_iso_date(constraint_sheet.get("birth_date"))
        input_target_age = constraint_sheet.get("target_age_exact")
        has_valid_target_age = isinstance(input_target_age, int) and input_target_age >= 0

        if input_birth is not None:
            birth_date = input_birth
            if birth_date > reference_date:
                logger.warning("birth_date is later than simulation_end_date, falling back to simulation_end_date.")
                birth_date = reference_date
            resolved_target_age = self._compute_exact_age(birth_date, reference_date)
            if has_valid_target_age and input_target_age != resolved_target_age:
                logger.warning(
                    f"Detected target_age_exact({input_target_age}) inconsistent with birth_date/simulation_end_date derived age({resolved_target_age}), "
                    f"using date-derived age."
                )
        elif has_valid_target_age:
            resolved_target_age = input_target_age
            birth_date = self._subtract_years(reference_date, resolved_target_age + 1) + timedelta(days=1)
        else:
            logger.warning("Missing valid target_age_exact and birth_date, defaulting to age 0 anchor.")
            resolved_target_age = 0
            birth_date = reference_date

        return {
            "reference_date": reference_date,
            "birth_date": birth_date,
            "target_age_exact": resolved_target_age,
            "input_birth_date": constraint_sheet.get("birth_date"),
            "input_simulation_end_date": constraint_sheet.get("simulation_end_date"),
            "input_target_age_exact": input_target_age
        }

    @staticmethod
    def _months_between_dates(later_date: date, earlier_date: date) -> int:
        """
        Compute the full month difference of later_date relative to earlier_date.
        Returns 0 if later_date is earlier than earlier_date.
        """
        if later_date < earlier_date:
            return 0
        months = (later_date.year - earlier_date.year) * 12 + (later_date.month - earlier_date.month)
        if later_date.day < earlier_date.day:
            months -= 1
        return max(0, months)

    def _derive_temporal_constraints_from_hints(
        self,
        hints: TransitionDateHints,
        reference_date: date
    ) -> Dict[str, Any]:
        """
        Normalize month-level temporal constraints from the step-1 LLM transition date output.
        No text-regex-based hard matching is used.
        """
        inferred = hints.inferred_temporal_constraints
        current_role_start_date = self._parse_iso_date(inferred.current_role_start_date)
        transition_anchor_date = self._parse_iso_date(inferred.pre_current_transition_anchor_date)

        current_job_tenure_months: Optional[int] = None
        pre_current_role_gap_months: Optional[int] = None

        if current_role_start_date is not None:
            current_job_tenure_months = self._months_between_dates(reference_date, current_role_start_date)

            if transition_anchor_date is not None:
                pre_current_role_gap_months = self._months_between_dates(current_role_start_date, transition_anchor_date)

        transition_semantic_hint = inferred.transition_semantic_hint or "general"

        return {
            "current_job_tenure_months": current_job_tenure_months,
            "pre_current_role_gap_months": pre_current_role_gap_months,
            "transition_semantic_hint": transition_semantic_hint,
            "current_role_start_date": inferred.current_role_start_date,
            "pre_current_transition_anchor_date": inferred.pre_current_transition_anchor_date,
            "rationale": inferred.rationale
        }

    def _apply_current_job_tenure_constraint(
        self,
        plan_dict: Dict[str, Any],
        current_job_tenure_months: int,
        pre_current_role_gap_months: int = 0,
        transition_semantic_hint: str = "general",
        current_role_start_date: Optional[str] = None,
        pre_current_transition_anchor_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply the "current job tenure + pre-current-role transition duration" constraint to the plan (generic, not limited to any occupation type).
        Notes:
        - Period segmentation is primarily based on `period_date_range`, no longer relying on "whole-year cuts";
        - For the current (last) period, add `role_tenure_range`;
        - For the previous period (if any), add `pre_current_role_transition`, aligning "previous period end -> transition -> job start";
        - Record constraint application info under global_summary for downstream tracking.
        - If step-1 provided precise dates (current_role_start_date / pre_current_transition_anchor_date), prefer those over month-based back-computation.
        """
        if current_job_tenure_months < 0:
            return plan_dict

        if pre_current_role_gap_months < 0:
            pre_current_role_gap_months = 0

        periods = plan_dict.get("life_periods", [])
        if not isinstance(periods, list) or not periods:
            return plan_dict

        timeline_anchor = plan_dict.get("global_summary", {}).get("timeline_anchor", {})
        reference_date_str = timeline_anchor.get("reference_date")
        if isinstance(reference_date_str, str):
            try:
                reference_date = date.fromisoformat(reference_date_str)
            except ValueError:
                reference_date = date.today()
        else:
            reference_date = date.today()

        inferred_tenure_start_date = self._parse_iso_date(current_role_start_date)
        inferred_transition_anchor_date = self._parse_iso_date(pre_current_transition_anchor_date)

        current_period = periods[-1]
        tenure_start_date = inferred_tenure_start_date or self._shift_months(reference_date, -current_job_tenure_months)
        transition_anchor_date = inferred_transition_anchor_date or self._shift_months(tenure_start_date, -pre_current_role_gap_months)

        if inferred_tenure_start_date is not None:
            current_job_tenure_months = self._months_between_dates(reference_date, tenure_start_date)

        if inferred_transition_anchor_date is not None:
            pre_current_role_gap_months = self._months_between_dates(tenure_start_date, transition_anchor_date)

        self._print_step(
            "12.1",
            "Role tenure constraint — date confirmation",
            detail=(
                f"employment_start_date={tenure_start_date.isoformat()} | "
                f"transition_anchor_date={transition_anchor_date.isoformat()} | "
                f"current_job_tenure_months={current_job_tenure_months} | "
                f"pre_current_role_gap_months={pre_current_role_gap_months}"
            )
        )

        current_period["role_tenure_range"] = {
            "start_date": tenure_start_date.isoformat(),
            "end_date": reference_date.isoformat(),
            "tenure_months": current_job_tenure_months,
            "source": "inferred_from_step1_transition_dates"
        }

        semantic_value = "graduation_before_employment" if transition_semantic_hint == "graduation" else "pre_current_role_transition"

        current_period["career_transition"] = {
            "transition_anchor_date": transition_anchor_date.isoformat(),
            "pre_current_role_gap_months": pre_current_role_gap_months,
            "employment_start_date": tenure_start_date.isoformat(),
            "reference_date": reference_date.isoformat(),
            "semantic": semantic_value
        }

        if transition_semantic_hint == "graduation":
            current_period["career_transition"]["graduation_date"] = transition_anchor_date.isoformat()
            current_period["career_transition"]["job_search_gap_months"] = pre_current_role_gap_months

        gap_days = (tenure_start_date - transition_anchor_date).days
        should_insert_gap_stage = gap_days >= 2 and len(periods) >= 2

        # Check if the LLM already generated a gap_transition that covers
        # the transition_anchor_date -> tenure_start_date interval.  If so,
        # we should reuse it instead of inserting a duplicate.
        llm_already_has_gap = False
        if should_insert_gap_stage and len(periods) >= 2:
            candidate = periods[-2]
            if candidate.get("stage_label") == "gap_transition":
                cand_dr = candidate.get("period_date_range")
                if isinstance(cand_dr, dict):
                    cand_start = self._parse_iso_date(cand_dr.get("start_date"))
                    cand_end = self._parse_iso_date(cand_dr.get("end_date"))
                    # The LLM gap roughly covers the transition window
                    if cand_start is not None and cand_end is not None:
                        expected_gap_start = transition_anchor_date + timedelta(days=1)
                        expected_gap_end = tenure_start_date - timedelta(days=1)
                        # Accept if the LLM gap overlaps significantly with the expected gap
                        if cand_start <= expected_gap_start + timedelta(days=30) and cand_end >= expected_gap_end - timedelta(days=30):
                            llm_already_has_gap = True
                            logger.info(
                                f"Detected LLM-generated gap_transition '{candidate.get('period_id')}' "
                                f"({cand_dr.get('start_date')} ~ {cand_dr.get('end_date')}) covering the "
                                f"transition window. Reusing it instead of inserting a new gap."
                            )
                            # Snap the LLM gap to the exact expected date range
                            cand_dr["start_date"] = expected_gap_start.isoformat()
                            cand_dr["end_date"] = expected_gap_end.isoformat()
                            candidate["period_date_range"] = cand_dr
                            candidate["gap_context"] = {
                                "source": "llm_generated_reused_by_postprocess",
                                "transition_anchor_date": transition_anchor_date.isoformat(),
                                "employment_start_date": tenure_start_date.isoformat(),
                                "gap_days": gap_days,
                                "semantic": semantic_value
                            }

        if len(periods) >= 2:
            previous_period = periods[-2] if not llm_already_has_gap else (
                periods[-3] if len(periods) >= 3 else periods[-2]
            )
            previous_period["pre_current_role_transition"] = {
                "transition_anchor_date": transition_anchor_date.isoformat(),
                "next_period_employment_start_date": tenure_start_date.isoformat(),
                "pre_current_role_gap_months": pre_current_role_gap_months,
                "semantic": semantic_value
            }
            if transition_semantic_hint == "graduation":
                previous_period["graduation_transition"] = {
                    "graduation_date": transition_anchor_date.isoformat(),
                    "next_period_employment_start_date": tenure_start_date.isoformat(),
                    "job_search_gap_months": pre_current_role_gap_months
                }

            prev_date_range = previous_period.get("period_date_range")
            if isinstance(prev_date_range, dict):
                prev_start = self._parse_iso_date(prev_date_range.get("start_date"))
                prev_end = self._parse_iso_date(prev_date_range.get("end_date"))
                if prev_start is not None and prev_end is not None:
                    previous_period["pre_current_role_transition"]["anchor_within_previous_period_date_range"] = (
                        prev_start <= transition_anchor_date <= prev_end
                    )

                # Ensure no overlap: the previous period must end before the
                # current (last) period starts.  When transition_anchor_date is
                # later than tenure_start_date (e.g. graduation after employment
                # start), we must cap at tenure_start_date - 1 day so that
                # prev.end_date + 1 day == current.start_date.
                snapped_prev_end = min(transition_anchor_date, tenure_start_date - timedelta(days=1))
                if prev_start is not None and snapped_prev_end < prev_start:
                    snapped_prev_end = prev_start
                prev_date_range["end_date"] = snapped_prev_end.isoformat()
                previous_period["period_date_range"] = prev_date_range

            if should_insert_gap_stage and not llm_already_has_gap:
                gap_period_id = f"LP{len(periods)}"
                gap_start_date = transition_anchor_date + timedelta(days=1)
                gap_end_date = tenure_start_date - timedelta(days=1)
                gap_period = {
                    "period_id": gap_period_id,
                    "stage_label": "gap_transition",
                    "title": "Key Transition Period",
                    "dominant_theme": "Buffer and reorganization transitioning from the previous milestone to the next role",
                    "developmental_tasks": [
                        "Wrap up unfinished business before role transition",
                        "Recalibrate capabilities and goals",
                        "Build psychological and practical readiness for the next stage"
                    ],
                    "stage_goals": [
                        "Complete the stage transition smoothly",
                        "Reduce transition uncertainty",
                        "Establish an execution rhythm for the next stage"
                    ],
                    "salient_pressures": [
                        "Uncertainty during the transition period",
                        "Psychological load from identity shift"
                    ],
                    "salient_opportunities": [
                        "Reflect on and consolidate lessons from the previous stage",
                        "Strategically prepare for the next stage"
                    ],
                    "likely_transition_triggers": [
                        "Formally entering the next role or institution"
                    ],
                    "period_date_range": {
                        "start_date": gap_start_date.isoformat(),
                        "end_date": gap_end_date.isoformat()
                    },
                    "gap_context": {
                        "source": "deterministic_postprocess_from_transition_constraints",
                        "transition_anchor_date": transition_anchor_date.isoformat(),
                        "employment_start_date": tenure_start_date.isoformat(),
                        "gap_days": gap_days,
                        "semantic": semantic_value
                    }
                }
                periods.insert(-1, gap_period)

        current_date_range = current_period.get("period_date_range")
        if isinstance(current_date_range, dict):
            current_date_range["start_date"] = tenure_start_date.isoformat()
            current_date_range["end_date"] = reference_date.isoformat()
            curr_start = self._parse_iso_date(current_date_range.get("start_date"))
            curr_end = self._parse_iso_date(current_date_range.get("end_date"))
            if curr_start is not None and curr_end is not None:
                current_period["career_transition"]["employment_start_within_current_period_date_range"] = (
                    curr_start <= tenure_start_date <= curr_end
                )
            current_period["period_date_range"] = current_date_range
        else:
            current_period["period_date_range"] = {
                "start_date": tenure_start_date.isoformat(),
                "end_date": reference_date.isoformat()
            }

        plan_dict["life_periods"] = periods

        for idx, period in enumerate(periods, start=1):
            period["period_id"] = f"LP{idx}"

        global_summary = plan_dict.get("global_summary", {})
        temporal_constraints = global_summary.get("temporal_constraints", {})
        temporal_constraints["current_job_tenure_months"] = current_job_tenure_months
        temporal_constraints["pre_current_role_gap_months"] = pre_current_role_gap_months
        temporal_constraints["transition_anchor_date"] = transition_anchor_date.isoformat()
        temporal_constraints["employment_start_date"] = tenure_start_date.isoformat()
        temporal_constraints["applied_to_period_id"] = current_period.get("period_id")
        temporal_constraints["reference_date"] = reference_date.isoformat()

        if transition_semantic_hint == "graduation":
            temporal_constraints["post_graduation_job_search_months"] = pre_current_role_gap_months
            temporal_constraints["graduation_date"] = transition_anchor_date.isoformat()

        global_summary["temporal_constraints"] = temporal_constraints
        plan_dict["global_summary"] = global_summary

        return plan_dict



    async def _infer_social_context(
        self,
        constraint_sheet: Dict[str, Any],
        enriched_constraints: Dict[str, Any],
        birth_date: date,
        correction_feedback: Optional[List[str]] = None,
    ) -> SocialContextProfile:
        """
        Step 0: Infer the social context (education system, career norms,
        social institutions) from persona characteristics using LLM.
        The output is used as constraints for all subsequent planning steps.
        """
        self._print_step("0.1", "Constructing social context inference input")

        country = enriched_constraints.get("growing_up_location", {}).get("country_text", "")
        if not country:
            country = constraint_sheet.get("growing_up_location", {}).get("country", "unknown")
        education_text = enriched_constraints.get("target_education_text", "")
        occupation_text = enriched_constraints.get("target_occupation_text", "")
        persona_brief = constraint_sheet.get("persona_brief_text", "")
        birth_year = birth_date.year
        birth_era = f"{(birth_year // 10) * 10}s"  # e.g. "1990s"

        system_prompt = """
        You are a social context and institutional rules inference expert.

        Given a person's basic characteristics (nationality, education level, occupation,
        birth era, etc.), infer the key rules and typical pathways of their social environment.

        Core principles:
        1. Output the "typical rules of that social context", NOT the specific person's actual experience.
        2. These rules will be used as REASONABLENESS BOUNDARIES, not as mandatory pathways.
           The actual pathway for this specific persona will be inferred in a separate step.
        3. Distinguish between hard constraints (rules almost impossible to violate) and soft constraints
           (typical but with possible exceptions).
        4. If the persona_brief_text explicitly describes a non-typical pathway (e.g. child prodigy,
           skipping grades, gap year), note it in persona_specific_adjustments.
        5. Focus on rules directly relevant to life stage planning. Do not output irrelevant cultural details.
        6. All ages and timelines should be based on the actual institutional system of that country/region.
        7. The education_system list should be ordered from earliest stage to latest, and should cover
           ALL stages from primary school up to the person's target education level.
        8. **Academic year start month is country-dependent**. Do NOT assume September.
           - Northern Hemisphere (China, UK, USA, etc.): September (entry_month=9)
           - Southern Hemisphere (New Zealand, Australia, etc.): January-February (entry_month=2)
           - Japan: April (entry_month=4)
           - Korea: March (entry_month=3)
           The entry_month in each EducationStage MUST reflect the actual country's academic calendar.
           This is critical for correctly computing school enrollment dates.
        """

        # Build correction feedback block if provided (from upstream retry)
        correction_block = ""
        if correction_feedback:
            correction_block = (
                "\n\n        [IMPORTANT — Correction signals from previous review]\n"
                "        The previous generation had the following issues. "
                "Please pay special attention and avoid repeating them:\n"
                + "\n".join(f"        - {s}" for s in correction_feedback)
            )
            logger.info(f"Upstream Retry: Injected {len(correction_feedback)} correction signal(s) into social context prompt")

        user_prompt = f"""
        Please infer the social context rules for the following person:

        [Basic characteristics]
        - Country/Region: {country}
        - Birth era: {birth_era} (birth year: {birth_year})
        - Target education level: {education_text}
        - Target occupation: {occupation_text}
        - Persona brief: {persona_brief}

        Please output:
        1. Education system rules (stage structure, entry ages, durations, academic year cycle)
           - Must cover ALL stages from primary school to the target education level
        2. Career path norms (typical entry path and timeline for this occupation)
        3. Key social institutions (e.g. military service, retirement, or other institutions
           that affect life stage planning)
        4. Persona-specific adjustments based on persona_brief_text (if any)
        {correction_block}
        """

        self._print_step(
            "0.2",
            "Calling LLM for social context inference",
            detail=f"country={country} | birth_era={birth_era} | education={education_text}"
        )

        try:
            profile = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=SocialContextProfile,
                system_prompt=system_prompt,
                task_type="p1a_social_context",
                temperature=0.0
            )
            self._print_step(
                "0.3",
                "Social context inference complete",
                detail=(
                    f"education_stages={len(profile.education_system)} | "
                    f"social_institutions={len(profile.social_institutions)} | "
                    f"adjustments={len(profile.persona_specific_adjustments)} | "
                    f"confidence={profile.confidence}"
                )
            )
            return profile
        except Exception as e:
            logger.warning(f"Social context inference failed, using empty fallback: {e}")
            self._print_step("0.3", "Social context inference failed, using fallback", detail=str(e))
            from lifelong_synth.simulation_p1_initialisation.definition import CareerPathNorm
            return SocialContextProfile(
                country=country or "unknown",
                birth_era=birth_era,
                education_system=[],
                career_path_norms=CareerPathNorm(
                    typical_entry_age_min=22,
                    typical_entry_age_max=30,
                    typical_prerequisites=[],
                    notes="fallback_empty"
                ),
                confidence="low",
                summary="fallback_empty_social_context"
            )

    async def _infer_persona_pathway(
        self,
        constraint_sheet: Dict[str, Any],
        social_context: SocialContextProfile,
        birth_date: date,
        reference_date: date,
        correction_feedback: Optional[List[str]] = None,
    ) -> InferredPersonaPathway:
        """
        Step 0b: Infer the actual life pathway for this specific persona,
        combining social norms (Step 0a) with persona_brief_text signals.
        """
        persona_brief = constraint_sheet.get("persona_brief_text", "")
        social_context_text = self._format_social_context_for_prompt(social_context, birth_date=birth_date)
        target_age = self._compute_exact_age(birth_date, reference_date)

        system_prompt = """
You are a persona life pathway inference expert.

Given a person's social context (typical rules and norms) and their
persona_brief_text, infer the ACTUAL life pathway this specific person
most likely followed.

Core principles:

1. persona_brief_text is your PRIMARY input. The social norms are only
   a reference framework.

2. For each life dimension (education, career, social institutions, family),
   determine whether this persona followed the typical pathway or deviated.

3. When analyzing persona_brief_text for signals, apply the THREE-WAY
   absence rule:
   - If a typical stage is NOT mentioned AND there is a POSITIVE signal
     of absence (e.g., "graduated with a bachelor's and went straight to PhD"
     implies no master's),
     → infer the stage does NOT exist (confidence: high)
   - If a typical stage is NOT mentioned AND there is a TIME CONSTRAINT
     that makes it impossible (e.g., only 5 years between bachelor and
     PhD graduation), → infer the stage does NOT exist (confidence: high)
   - If a typical stage is NOT mentioned AND neither positive signal nor
     time constraint exists, → DEFAULT to the typical pathway and mark
     confidence as "medium"
   - If persona_brief_text does not cover a dimension at all (e.g., no
     mention of family), → use typical pathway as default, do NOT infer
     absence (confidence: low)

4. Use deviation type prefixes to classify how the pathway differs:
   SKIP / INSERT / EXTEND / SHORTEN / REORDER / PARALLEL / SPLIT /
   INTERRUPT_RESUME / MERGE

5. Cross-validate with time constraints:
   - Calculate available time windows between known anchors
   - If the typical pathway doesn't fit, choose the most plausible
     alternative

6. Do NOT fabricate deviations without evidence from persona_brief_text.

7. After inferring the pathway dimensions, generate a suggested_periods list.
   This is the MOST IMPORTANT output — it directly controls how many periods
   the backward generation loop will create. Follow these grouping principles:

   a. ALWAYS split at major qualitative transitions, including but not limited to:
      - School change (primary → secondary, secondary → university)
      - Graduation / enrollment
      - Job change / promotion / career pivot
      - Major life event (marriage, civil partnership, birth/adoption of child,
        bereavement / spousal loss, divorce / separation, relocation, illness, retirement)
      - Country move

   b. For long homogeneous stages (>3 years with no qualitative shift):
      Group into 2-3 periods MAX, NOT one per year.
      Examples:
      - 6-year primary school → 2 periods (Years 1-3 foundation, Years 4-6 consolidation)
      - 30-year teaching career → 3-4 periods (early, mid, senior, pre-retirement)
      - 5-year doctoral program → 2 periods (coursework/proposal, dissertation)
      NOT: one period per year.

   c. For stages with a clear internal split (e.g., NCEA Level 1-2 vs Level 3,
      undergraduate exploration vs specialization, early career vs senior role):
      Split at the natural boundary.

   d. For short stages (<2 years): keep as ONE period.

   e. For early childhood (birth to school entry): ONE period only.

   f. For career stages: split at ROLE CHANGES, not annually.
      A 10-year career with no role change = 1-2 periods.
      A 10-year career with 2 promotions = 3 periods.

   g. For retirement: ONE period unless there is a clear active/late split.

   h. approx_start/approx_end: derive from SocialContextProfile education
      timeline anchors (birth_date + typical_entry_age + duration).
      Use YYYY-MM format (no day).

   i. stage_position_hint: for education stages, include grade range
      (e.g. 'Year 4-6 of 6'). For non-education stages, describe the
      career/life phase (e.g. 'early career years 1-3', 'active retirement').
      This field is OPTIONAL — only fill if meaningful.

   j. The suggested_periods list MUST cover the FULL life span from birth
      to reference_date with no gaps. Include early_childhood, all education
      stages, career stages, and any gap/transition periods.

   k. MANDATORY — Family / relationship period materialization:
      When any dimension (especially "family") contains an INSERT deviation
      (e.g., widowhood, divorce, birth of child, remarriage), the
      suggested_periods list MUST include a separate period for that event,
      even if the event's duration is short or its exact timing is uncertain.
      Do NOT absorb family INSERT events into adjacent career or retirement
      periods. Instead, create a dedicated period with:
        - stage_key: descriptive label (e.g., 'spousal_loss_transition',
          'divorce_transition', 'first_child_arrival')
        - label_hint: human-readable label (e.g., 'Transition following
          spousal loss', 'Life restructuring after divorce',
          'New parenthood adjustment')
        - approx_start / approx_end: best estimate; if timing is truly
          unknown, use the midpoint of the adjacent period and mark with
          a ~1-year span.
      This rule applies even if the persona_brief_text does not state the
      exact date — the period exists because the deviation type (INSERT)
      signals a qualitative life shift that MUST be represented.
"""

        # Build correction feedback block if provided (from upstream retry)
        correction_block = ""
        if correction_feedback:
            correction_block = (
                "\n[IMPORTANT — Correction signals from previous review]\n"
                "The previous generation had the following issues. "
                "Please pay special attention and avoid repeating them:\n"
                + "\n".join(f"- {s}" for s in correction_feedback)
            )
            logger.info(f"Upstream Retry: Injected {len(correction_feedback)} correction signal(s) into persona pathway prompt")

        user_prompt = f"""
Please infer the actual life pathway for this persona:

[persona_brief_text]
{persona_brief}

[time_anchors]
birth_date={birth_date.isoformat()}
reference_date={reference_date.isoformat()}
target_age={target_age}

[social_norms_reference]
{social_context_text}

For each dimension (education, career, social_institution, family),
determine:
1. Did this persona follow the typical pathway, or deviate?
2. If deviated, what type of deviation (SKIP/INSERT/EXTEND/etc.)?
3. What is the evidence from persona_brief_text?
4. How confident are you in this inference?
{correction_block}
""" + (f"""

[persona_extensions — Additional Context for Planning]
{format_persona_extensions(constraint_sheet.get('persona_extensions') or {}, stage='planning', heading='Extended Persona Attributes')}
These traits should inform pathway inference decisions.
If "Specific Attitudes" are listed above, treat each attitude as a behavioral/motivational signal:
- Negative attitudes toward academic subjects may correlate with disengagement, lower grades, or avoidance in relevant school periods.
- Positive attitudes toward leisure activities may reinforce non-academic lifestyle choices.
- Record these as explicit evidence signals in persona_brief_signals.

If "Core Identity Traits" are listed above, ensure the life plan includes at least one period where each primary identity trait is meaningfully explored or expressed:
- Sexual orientation (e.g., homosexual, bisexual): include a period covering identity discovery, coming-out, or relationship formation.
- Religious identity (e.g., Atheist, Muslim, devout Catholic): include a period covering a moment of religious questioning, community engagement, or identity-based conflict.
- Racial/ethnic identity (e.g., Black, Latino): include a period covering racial identity formation, discrimination experience, or community belonging.
- Political/cultural identity: include a period covering an event that tests or expresses this identity.
- Gender identity — Transgender: include at least two periods: (1) identity discovery / coming-out; (2) a transition-related period (medical, social, or legal aspects). If the persona is also a healthcare professional or advocate, add a period where they use personal experience to advocate for others.
- Gender identity — Non-binary / Genderqueer: include a period covering identity exploration and a specific moment of asserting this identity in a social or professional context.
- Disability identity — Deaf / Hard of Hearing: include at least one period with a SPECIFIC event involving sign language use (ASL, BSL, or other), navigating a hearing-dominated environment (job interview, medical appointment, public event without interpreters), or connecting with the deaf community. The event should show concrete communication strategies.
- Disability identity — Other physical/cognitive: include a period covering a specific accessibility challenge, an advocacy moment, or a community belonging event that reveals how the disability shapes the character's daily life and worldview.
- Do NOT create a dedicated life period solely for an identity trait — weave it into the theme / pressures / opportunities of naturally occurring periods (e.g., university, early career).

Cultural Grounding (applies when the persona's growing_up_location.country or current_living_location.country is non-Western):
- All events must reflect the persona's actual cultural context — workplace norms, family structures, social expectations, and life milestones should be grounded in their home culture, NOT in Western / American defaults.
- Examples:
  - Malaysian Muslim: work schedule around prayer times, Ramadan observance, family-oriented career decisions.
  - Thai Buddhist monk: monastery hierarchy, alms rounds, dharma study, seasonal retreats.
  - Brazilian: family-centered social life, Carnival, economic inequality context, Portuguese-language media.
  - Japanese retiree: seniority-based career progression, group harmony norms, seasonal cultural events.
  - Polish WWII veteran: post-war Communist Poland context, Catholic cultural background.
- The simulation still produces English text, but embeds culturally authentic details, references, and social dynamics.

Multilingual Persona Guidance:
- If the persona speaks multiple languages or comes from a multilingual background, ensure at least one life period includes a SPECIFIC event where language skills are actively used:
  - A language-learning milestone (first fluent conversation in a new language, a breakthrough moment).
  - A cross-cultural communication event (translating for someone, navigating a situation in a foreign language).
  - A moment where language skills created a specific opportunity or solved a specific problem.
- Include the language name, the context, and what was communicated. For child prodigies with exceptional language skills, include events that show the social and intellectual impact of their abilities.

Cultural / Religious Practice Events:
- If the persona's occupation or background implies regular cultural or religious practices (Buddhist monk → daily meditation; Muslim → daily prayer and Ramadan; indigenous activist → traditional ceremonies; Japanese practitioner → seasonal rituals), ensure at least one life period includes a specific event anchoring the practice. The event should show the practice in action AND reveal what it means to the persona personally. Do NOT create a dedicated period for the practice — weave it into existing periods.

Adversity / Hardship Events:
- If the persona's brief explicitly describes hardship, adversity, or struggle ("working two jobs", "single mother", "war veteran", "fighting against"), ensure at least one life period includes a high-salience adversity event. The event should show the persona actively coping with or confronting the hardship, reveal the emotional and practical stakes, and anchor resilience / bitterness / determination. Do NOT sanitize or resolve the adversity prematurely.

Occupation Episodic Depth:
- For personas with distinctive occupations (truck driver, chef, monk, athlete, artist, etc.), ensure at least one life period includes a SPECIFIC occupation-defining episode:
  - NOT: "worked as a truck driver for 20 years" — YES: "drove a 48-hour cross-country haul through a blizzard, navigating by CB radio".
  - NOT: "practiced meditation daily" — YES: "led a 10-day silent retreat where a participant had a breakthrough experience".
  - NOT: "cooked vegan food" — YES: "developed a signature dish that became the cooking school's most requested recipe".
- These specific episodes are what PersonaGym evaluators use to test persona authenticity.

""" if constraint_sheet.get('persona_extensions') else "")

        self._print_step(
            "0b.1",
            "Calling LLM for persona pathway inference",
            detail=f"persona_brief_length={len(persona_brief)} | target_age={target_age}"
        )

        try:
            pathway = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=InferredPersonaPathway,
                system_prompt=system_prompt,
                task_type="p1a_persona_pathway",
                temperature=0.0
            )
            self._print_step(
                "0b.2",
                "Persona pathway inference complete",
                detail=(
                    f"dimensions={len(pathway.dimensions)} | "
                    f"signals={len(pathway.persona_brief_signals)}"
                )
            )
            return pathway
        except Exception as e:
            logger.warning(f"Persona pathway inference failed, using empty fallback: {e}")
            self._print_step("0b.2", "Persona pathway inference failed, using fallback", detail=str(e))
            return InferredPersonaPathway(
                dimensions=[],
                overall_rationale="fallback_empty",
                persona_brief_signals=[]
            )

    def _format_social_context_for_prompt(self, social_context: SocialContextProfile, persona_pathway: Optional[InferredPersonaPathway] = None, birth_date: Optional[date] = None) -> str:
        """
        Format the SocialContextProfile and optional InferredPersonaPathway into
        a readable text block for injection into LLM prompts.
        Layer 1 = Social Norms (reasonableness boundaries).
        Layer 2 = Inferred Actual Pathway (highest priority reference).
        Layer 3 = Computed Education Timeline Anchors (when birth_date is provided).
        """
        if not social_context.education_system and social_context.confidence == "low":
            if persona_pathway and persona_pathway.dimensions:
                # Even if social context failed, we can still output Layer 2
                pass
            else:
                return "(Social context unavailable — no education system constraints inferred)"

        lines = [f"=== Social Norms (reasonableness boundaries, typical pathway for reference only) ==="]
        lines.append(f"Social Context: {social_context.summary}")
        lines.append(f"Country: {social_context.country} | Birth era: {social_context.birth_era} | Confidence: {social_context.confidence}")

        if social_context.education_system:
            lines.append("\nEducation System (typical pathway):")
            for stage in social_context.education_system:
                mandatory_tag = " [mandatory]" if stage.is_mandatory else ""
                notes_tag = f" — {stage.notes}" if stage.notes else ""
                lines.append(
                    f"  - {stage.stage_name}: entry age ~{stage.typical_entry_age}, "
                    f"duration ~{stage.typical_duration_years} years, "
                    f"starts month {stage.entry_month}{mandatory_tag}{notes_tag}"
                )

            # ── Computed Education Timeline Anchors ──
            # Pre-calculate expected absolute date ranges for each education
            # stage based on birth_date + entry_age + duration.  This gives
            # the LLM concrete date references so it doesn't have to do
            # mental arithmetic during backward generation, which is the
            # root cause of systematic off-by-one cascading errors.
            if birth_date and social_context.education_system:
                anchor_lines = []
                for stage in social_context.education_system:
                    if stage.typical_duration_years <= 0:
                        continue  # skip zero-duration stages like gaokao_transition
                    entry_year = birth_date.year + stage.typical_entry_age
                    entry_month = stage.entry_month or 9  # default to September
                    try:
                        expected_start = date(entry_year, entry_month, 1)
                    except ValueError:
                        continue
                    end_year = entry_year + stage.typical_duration_years
                    # End date is the day before the next stage's entry month
                    try:
                        expected_end = date(end_year, entry_month, 1) - timedelta(days=1)
                    except ValueError:
                        continue
                    anchor_lines.append(
                        f"  - {stage.stage_name}: "
                        f"~{expected_start.isoformat()} to ~{expected_end.isoformat()} "
                        f"(age ~{stage.typical_entry_age} to ~{stage.typical_entry_age + stage.typical_duration_years})"
                    )
                if anchor_lines:
                    lines.append(
                        "\nComputed Education Timeline Anchors "
                        "(derived from birth_date + typical entry age/duration; "
                        "use as STRONG reference for stage boundaries, "
                        "adjust only if persona pathway explicitly deviates):"
                    )
                    lines.extend(anchor_lines)

        cpn = social_context.career_path_norms
        if cpn:
            lines.append(f"\nCareer Path Norms:")
            lines.append(
                f"  - Typical entry age: {cpn.typical_entry_age_min}-{cpn.typical_entry_age_max}"
            )
            if cpn.typical_prerequisites:
                lines.append(f"  - Prerequisites: {', '.join(cpn.typical_prerequisites)}")
            if cpn.notes:
                lines.append(f"  - Notes: {cpn.notes}")

        if social_context.social_institutions:
            lines.append("\nRelevant Social Institutions:")
            for inst in social_context.social_institutions:
                mandatory_tag = " [mandatory]" if inst.is_mandatory else " [optional]"
                lines.append(
                    f"  - {inst.institution_name}: applies to {inst.applies_to}, "
                    f"age {inst.typical_age_min}-{inst.typical_age_max}, "
                    f"duration ~{inst.typical_duration_months} months{mandatory_tag}"
                )
                if inst.notes:
                    lines.append(f"    Notes: {inst.notes}")

        if social_context.persona_specific_adjustments:
            lines.append("\nPersona-Specific Adjustments (deviations from typical pathway):")
            for adj in social_context.persona_specific_adjustments:
                lines.append(f"  - [{adj.adjustment_type}] {adj.description}")
                if adj.affected_stages:
                    lines.append(f"    Affected stages: {', '.join(adj.affected_stages)}")
                if adj.rationale:
                    lines.append(f"    Rationale: {adj.rationale}")

        # === Layer 2: Inferred Actual Pathway (highest priority reference) ===
        if persona_pathway and persona_pathway.dimensions:
            lines.append("")
            lines.append("=== Inferred Actual Pathway (HIGHEST PRIORITY reference — this persona's most likely actual trajectory) ===")
            for dim in persona_pathway.dimensions:
                lines.append(f"\n[{dim.dimension}] pathway_type: {dim.pathway_type} "
                            f"(confidence: {dim.confidence})")
                if dim.key_stages:
                    lines.append(f"  Key stages: {' → '.join(dim.key_stages)}")
                if dim.deviations_from_typical:
                    lines.append(f"  Deviations from typical:")
                    for d in dim.deviations_from_typical:
                        lines.append(f"    - {d}")
                if dim.inference_basis:
                    lines.append(f"  Basis: {dim.inference_basis}")

            lines.append(f"\nOverall rationale: {persona_pathway.overall_rationale}")
            if persona_pathway.persona_brief_signals:
                lines.append(f"Key signals: {', '.join(persona_pathway.persona_brief_signals)}")

        return "\n".join(lines)

    async def _infer_transition_dates_with_llm(
        self,
        constraint_sheet: Dict[str, Any],
        enriched_constraints: Dict[str, Any],
        reference_date: date,
        birth_date: date,
        resolved_target_age: int,
        social_context_text: str = "",
        correction_feedback: Optional[List[str]] = None,
    ) -> TransitionDateHints:
        """
        Step-1 API call: have the model extract "transition dates that should be highlighted in the simulation"
        and directly output the normalized inferred_temporal_constraints.
        """
        self._print_step("4.1", "Build step-1 transition inference input")
        persona_brief_text = constraint_sheet.get("persona_brief_text", "")
        if not isinstance(persona_brief_text, str):
            persona_brief_text = ""

        inference_system_prompt = """
        You are a "key date extractor + temporal constraint normalizer" for life trajectories.
        Given persona constraints and timeline anchors, infer and output the set of transition dates
        that should be highlighted in the simulation (e.g., graduation, job start, career change)
        and supply the inferred_temporal_constraints:
        - current_role_start_date
        - pre_current_transition_anchor_date
        - transition_semantic_hint (general/graduation)

        Key principles:
        - Pay close attention to absolute or relative times mentioned in [persona_brief_text]
        - You MUST prioritize narrative clues in [persona_brief_text] to infer transitions (e.g., graduation, job start, career change);
        - Other structured fields are for calibration only and should NOT override event sequences explicitly stated in persona_brief_text.
        - If [Inferred Actual Pathway] is provided, use it as the primary reference for inferring transition dates.
          For example, if the pathway indicates direct-PhD (no separate master's stage), do NOT infer a master's graduation date.

        Constraints:
        1) Output only structured JSON;
        2) All date fields must be ISO dates YYYY-MM-DD;
        3) Do NOT output transitions obviously later than reference_date;
        4) When no explicit date is given, you may infer, but reduce confidence and explain the basis in rationale;
        5) Do NOT ask the user for additional transition date inputs.
        """
        social_context_block = ""
        if social_context_text:
            social_context_block = f"""

        [social_context]
        {social_context_text}
        """

        inference_prompt = f"""
        [timeline_anchor]
        reference_date={reference_date.isoformat()}
        birth_date={birth_date.isoformat()}
        target_age_exact={resolved_target_age}

        [persona_brief_text]
        {persona_brief_text}

        [constraint_sheet]
        {json.dumps(constraint_sheet, ensure_ascii=False, indent=2)}

        [enriched_constraints]
        {json.dumps(enriched_constraints, ensure_ascii=False, indent=2)}
        {social_context_block}
        """

        # CBQA: inject correction feedback from Error Attribution Agent
        if correction_feedback:
            correction_block = (
                "\n\n        [CBQA Correction Signals — MUST absorb]\n"
                "        The following errors were found in the previous pipeline run; please correct them in this inference:\n"
                + "\n".join(f"        - {s}" for s in correction_feedback)
            )
            inference_prompt += correction_block
            logger.info(f"CBQA: Injected {len(correction_feedback)} correction signal(s) into transition hints prompt")

        self._print_step(
            "4.2",
            "Call step-1 LLM: extract transition dates",
            detail=(
                f"reference_date={reference_date.isoformat()} | "
                f"target_age_exact={resolved_target_age} | "
                f"persona_brief_text_length={len(persona_brief_text)}"
            )
        )

        try:
            hints = await self.llm.generate_structured(
                prompt=inference_prompt,
                response_model=TransitionDateHints,
                system_prompt=inference_system_prompt,
                task_type="p1a_transition_dates",
                temperature=0.0
            )
            self._print_step(
                "4.3",
                "Step-1 LLM completed",
                detail=f"transition_dates_count={len(hints.transition_dates)}"
            )
            return hints
        except Exception as e:
            logger.warning(f"Step-1 transition date extraction failed, falling back to empty hints: {e}")
            self._print_step("4.3", "Step-1 LLM failed, falling back to empty hints", detail=str(e))
            return TransitionDateHints(summary="fallback_empty_hints", transition_dates=[])

    def _transition_hints_to_prompt_text(self, hints: TransitionDateHints) -> str:
        """
        Convert step-1 API results into readable constraint text for the step-2 planning prompt.
        """
        if not hints.transition_dates:
            return "- No explicit transition date hints (you may plan based on persona narrative and age calendar)"

        lines = [f"- summary: {hints.summary}"]
        for item in hints.transition_dates:
            lines.append(
                f"- {item.transition_key}: {item.anchor_date} | label={item.transition_label} | "
                f"hard={item.is_hard_constraint} | confidence={item.confidence} | rationale={item.rationale}"
            )
        return "\n".join(lines)

    def _build_inferred_constraints_hint_text(self, inferred_constraints: Dict[str, Any]) -> str:
        """
        Format the step-1 LLM normalized temporal constraints into readable text for the step-2 prompt.
        """
        lines = [
            f"- current_role_start_date: {inferred_constraints.get('current_role_start_date')}",
            f"- pre_current_transition_anchor_date: {inferred_constraints.get('pre_current_transition_anchor_date')}",
            f"- current_job_tenure_months: {inferred_constraints.get('current_job_tenure_months')}",
            f"- pre_current_role_gap_months: {inferred_constraints.get('pre_current_role_gap_months')}",
            f"- transition_semantic_hint: {inferred_constraints.get('transition_semantic_hint')}",
            f"- rationale: {inferred_constraints.get('rationale')}"
        ]
        return "\n".join(lines)

    def _enforce_childhood_amnesia(
        self,
        plan_dict: Dict[str, Any],
        birth_date: date
    ) -> Dict[str, Any]:
        """DEPRECATED: No longer needed. Childhood amnesia is now handled by
        AutobiographicalMemoryModel in temporal_context.py.
        Kept as no-op for backward compatibility."""
        return plan_dict

    def _ensure_seamless_date_chain(
        self,
        plan_dict: Dict[str, Any],
        birth_date: date,
        reference_date: date
    ) -> Dict[str, Any]:
        """
        Universal post-processing: ensure all life periods form a seamless,
        gap-free, overlap-free date chain from birth_date to reference_date.

        Strategy (forward-pass + boundary anchoring):
        1.  Anchor: LP1.start_date = birth_date, last LP.end_date = reference_date.
        2a. Handle degenerate periods (end < start) — rescue content-bearing
            periods by assigning them the gap between neighbours; remove only
            semantically lightweight periods (e.g. gap_transition).
        2b. Forward-pass: for each adjacent pair (i, i+1), fix gaps/overlaps
            by adjusting period[i].end_date to period[i+1].start_date - 1 day.
        3.  Re-index period_ids as LP1, LP2, ... and synchronise all
            cross-references (applied_to_period_id, last_applied_patches).
        4.  Final validation pass with detailed logging.
        """
        periods = plan_dict.get("life_periods", [])
        if not isinstance(periods, list) or not periods:
            return plan_dict

        alignment_log: List[str] = []

        # --- Step 0: Sort periods by start_date ascending ---
        # Backward generation produces periods in reverse-chronological order
        # (LP1 = most recent, LP_N = oldest). The anchoring logic in Step 1
        # assumes periods[0] is the OLDEST period (birth_date) and periods[-1]
        # is the NEWEST (reference_date). Without sorting first, the anchors
        # are applied to the wrong ends, causing LP1 (most recent) to be
        # stretched back to birth_date and collapsing all other periods to
        # single-day slots.
        def _sort_key(p: Dict[str, Any]) -> date:
            dr = p.get("period_date_range")
            if isinstance(dr, dict):
                d = self._parse_iso_date(dr.get("start_date"))
                if d is not None:
                    return d
            return date.max

        periods_sorted = sorted(periods, key=_sort_key)
        if periods_sorted != periods:
            alignment_log.append(
                "Sorted periods by start_date ascending (backward generation produces reverse order)"
            )
            periods = periods_sorted

        # --- Step 1: Anchor first and last period boundaries ---
        first_dr = periods[0].get("period_date_range")
        if isinstance(first_dr, dict):
            first_start = self._parse_iso_date(first_dr.get("start_date"))
            if first_start != birth_date:
                alignment_log.append(
                    f"Anchored {periods[0].get('period_id')} start_date: "
                    f"{first_dr.get('start_date')} -> {birth_date.isoformat()}"
                )
                first_dr["start_date"] = birth_date.isoformat()
                periods[0]["period_date_range"] = first_dr

        last_dr = periods[-1].get("period_date_range")
        if isinstance(last_dr, dict):
            last_end = self._parse_iso_date(last_dr.get("end_date"))
            if last_end != reference_date:
                alignment_log.append(
                    f"Anchored {periods[-1].get('period_id')} end_date: "
                    f"{last_dr.get('end_date')} -> {reference_date.isoformat()}"
                )
                last_dr["end_date"] = reference_date.isoformat()
                periods[-1]["period_date_range"] = last_dr

        # --- Step 2a: Handle degenerate periods BEFORE forward-pass ---
        # Degenerate periods (end < start) can arise from bad patches or LLM
        # errors.  We handle them here so the forward-pass in Step 2b can
        # automatically fix date gaps around rescued periods.
        SAFE_TO_REMOVE_LABELS = {"gap_transition"}
        valid_periods = []
        removed_content_period_labels = []
        for idx_degen, p in enumerate(periods):
            dr = p.get("period_date_range")
            if isinstance(dr, dict):
                s = self._parse_iso_date(dr.get("start_date"))
                e = self._parse_iso_date(dr.get("end_date"))
                if s is not None and e is not None and e < s:
                    stage_label = p.get("stage_label", "")
                    # Semantically lightweight periods can be safely removed.
                    if stage_label in SAFE_TO_REMOVE_LABELS:
                        alignment_log.append(
                            f"Removed degenerate {stage_label} period "
                            f"{p.get('period_id')} "
                            f"({dr.get('start_date')} ~ {dr.get('end_date')}): "
                            f"end < start"
                        )
                        continue

                    # Content-bearing period: try to rescue by assigning it
                    # the gap between its previous and next neighbours.
                    prev_end = None
                    if valid_periods:
                        prev_dr = valid_periods[-1].get("period_date_range", {})
                        prev_end = self._parse_iso_date(prev_dr.get("end_date"))

                    next_start = None
                    for future_p in periods[idx_degen + 1:]:
                        future_dr = future_p.get("period_date_range")
                        if isinstance(future_dr, dict):
                            ns = self._parse_iso_date(future_dr.get("start_date"))
                            if ns is not None:
                                next_start = ns
                                break

                    rescued = False
                    if prev_end is not None and next_start is not None:
                        new_start = prev_end + timedelta(days=1)
                        new_end = next_start - timedelta(days=1)
                        if new_end >= new_start:
                            dr["start_date"] = new_start.isoformat()
                            dr["end_date"] = new_end.isoformat()
                            p["period_date_range"] = dr
                            alignment_log.append(
                                f"Rescued degenerate period "
                                f"{p.get('period_id')} "
                                f"(stage_label={stage_label}): "
                                f"reassigned to "
                                f"{new_start.isoformat()} ~ "
                                f"{new_end.isoformat()}"
                            )
                            rescued = True
                    elif prev_end is not None:
                        # Last period or no parseable next — give 1-day slot
                        new_start = prev_end + timedelta(days=1)
                        dr["start_date"] = new_start.isoformat()
                        dr["end_date"] = new_start.isoformat()
                        p["period_date_range"] = dr
                        alignment_log.append(
                            f"Rescued degenerate period "
                            f"{p.get('period_id')} "
                            f"(stage_label={stage_label}): "
                            f"assigned minimal slot "
                            f"{new_start.isoformat()}"
                        )
                        rescued = True

                    if not rescued:
                        alignment_log.append(
                            f"WARNING: Removed degenerate content-bearing "
                            f"period {p.get('period_id')} "
                            f"(stage_label={stage_label}, "
                            f"{dr.get('start_date')} ~ "
                            f"{dr.get('end_date')}): "
                            f"end < start and could not rescue"
                        )
                        removed_content_period_labels.append(stage_label)
                        continue
            valid_periods.append(p)
        periods = valid_periods

        # Store removal metadata for downstream invariant validation
        if removed_content_period_labels:
            plan_dict.setdefault("global_summary", {})[
                "_removed_content_periods"
            ] = removed_content_period_labels

        # --- Step 2b: Forward-pass — fix gaps and overlaps between adjacent pairs ---
        for i in range(len(periods) - 1):
            cur_dr = periods[i].get("period_date_range")
            next_dr = periods[i + 1].get("period_date_range")
            if not isinstance(cur_dr, dict) or not isinstance(next_dr, dict):
                continue

            cur_end = self._parse_iso_date(cur_dr.get("end_date"))
            next_start = self._parse_iso_date(next_dr.get("start_date"))
            if cur_end is None or next_start is None:
                continue

            expected_cur_end = next_start - timedelta(days=1)

            if cur_end != expected_cur_end:
                delta_days = (cur_end - expected_cur_end).days
                if delta_days > 0:
                    issue_type = "overlap"
                else:
                    issue_type = "gap"

                # Preserve cur start_date lower bound
                cur_start = self._parse_iso_date(cur_dr.get("start_date"))
                new_cur_end = expected_cur_end
                if cur_start is not None and new_cur_end < cur_start:
                    # Cannot shrink period to negative; instead adjust next_start
                    new_next_start = cur_start + timedelta(days=1)
                    alignment_log.append(
                        f"Date {issue_type} between {periods[i].get('period_id')} and "
                        f"{periods[i+1].get('period_id')}: adjusted {periods[i+1].get('period_id')}.start_date "
                        f"{next_dr.get('start_date')} -> {new_next_start.isoformat()} "
                        f"(cur period too short to shrink)"
                    )
                    next_dr["start_date"] = new_next_start.isoformat()
                    cur_dr["end_date"] = cur_start.isoformat()
                    periods[i + 1]["period_date_range"] = next_dr
                    periods[i]["period_date_range"] = cur_dr
                else:
                    # CBQA Phase 1: Bidirectional adjustment strategy
                    # When shrinking the current period would make it unreasonably
                    # short (< 180 days for large overlaps), prefer adjusting the
                    # next period's start_date instead.
                    cur_duration_if_shrunk = (new_cur_end - cur_start).days if cur_start else None
                    original_cur_duration = (cur_end - cur_start).days if cur_start else None
                    next_end = self._parse_iso_date(next_dr.get("end_date"))
                    next_duration_if_pushed = (next_end - (cur_end + timedelta(days=1))).days if next_end else None

                    use_forward_adjust = False
                    if (
                        issue_type == "overlap"
                        and delta_days > 90
                        and cur_duration_if_shrunk is not None
                        and cur_duration_if_shrunk < 180
                        and original_cur_duration is not None
                        and original_cur_duration > 365
                        and next_duration_if_pushed is not None
                        and next_duration_if_pushed > 90
                    ):
                        # Shrinking current period would lose > 50% of its duration
                        # and the next period can absorb the adjustment
                        use_forward_adjust = True

                    if use_forward_adjust:
                        new_next_start = cur_end + timedelta(days=1)
                        alignment_log.append(
                            f"Date {issue_type} between {periods[i].get('period_id')} and "
                            f"{periods[i+1].get('period_id')}: adjusted {periods[i+1].get('period_id')}.start_date "
                            f"{next_dr.get('start_date')} -> {new_next_start.isoformat()} "
                            f"(bidirectional: preserving cur period duration, {abs(delta_days)} day(s))"
                        )
                        next_dr["start_date"] = new_next_start.isoformat()
                        periods[i + 1]["period_date_range"] = next_dr
                    else:
                        alignment_log.append(
                            f"Date {issue_type} between {periods[i].get('period_id')} and "
                            f"{periods[i+1].get('period_id')}: adjusted {periods[i].get('period_id')}.end_date "
                            f"{cur_dr.get('end_date')} -> {new_cur_end.isoformat()} ({abs(delta_days)} day(s))"
                        )
                        cur_dr["end_date"] = new_cur_end.isoformat()
                        periods[i]["period_date_range"] = cur_dr

        # --- Step 3: Re-index period_ids ---
        old_to_new_id_map = {}
        for idx, period in enumerate(periods, start=1):
            old_id = period.get("period_id")
            new_id = f"LP{idx}"
            if old_id != new_id:
                old_to_new_id_map[old_id] = new_id
            period["period_id"] = new_id

        plan_dict["life_periods"] = periods

        # --- Step 3b: Synchronise cross-references after re-indexing ---
        if old_to_new_id_map:
            alignment_log.append(
                f"Re-indexed period_ids: {old_to_new_id_map}"
            )
            global_summary = plan_dict.get("global_summary", {})

            # Update temporal_constraints.applied_to_period_id
            tc = global_summary.get("temporal_constraints", {})
            old_applied = tc.get("applied_to_period_id")
            if old_applied and old_applied in old_to_new_id_map:
                tc["applied_to_period_id"] = old_to_new_id_map[old_applied]
                alignment_log.append(
                    f"Updated temporal_constraints.applied_to_period_id: "
                    f"{old_applied} -> {old_to_new_id_map[old_applied]}"
                )

            # Fallback: if applied_to_period_id references an ID that was
            # removed (not in map AND not in current periods), point it to
            # the last period.  This is safe because applied_to_period_id
            # always refers to the current-job period which is the last one.
            current_ids = {p.get("period_id") for p in periods}
            if old_applied and old_applied not in current_ids:
                fallback_id = periods[-1]["period_id"] if periods else None
                if fallback_id:
                    tc["applied_to_period_id"] = fallback_id
                    alignment_log.append(
                        f"Fallback: temporal_constraints.applied_to_period_id "
                        f"{old_applied} -> {fallback_id} (original ID removed)"
                    )

            # Update last_applied_patches period_id references
            patches_log = global_summary.get("last_applied_patches", [])
            for p_log in patches_log:
                old_pid = p_log.get("period_id")
                if old_pid and old_pid in old_to_new_id_map:
                    p_log["period_id"] = old_to_new_id_map[old_pid]

            plan_dict["global_summary"] = global_summary

        # --- Step 4: Final validation pass ---
        validation_issues: List[str] = []
        for i in range(len(periods) - 1):
            cur_dr = periods[i].get("period_date_range", {})
            next_dr = periods[i + 1].get("period_date_range", {})
            cur_end = self._parse_iso_date(cur_dr.get("end_date"))
            next_start = self._parse_iso_date(next_dr.get("start_date"))
            if cur_end is not None and next_start is not None:
                if next_start != cur_end + timedelta(days=1):
                    validation_issues.append(
                        f"{periods[i].get('period_id')}.end={cur_end.isoformat()} -> "
                        f"{periods[i+1].get('period_id')}.start={next_start.isoformat()} "
                        f"(expected {(cur_end + timedelta(days=1)).isoformat()})"
                    )

        # Check first/last boundaries
        if periods:
            first_s = self._parse_iso_date(
                periods[0].get("period_date_range", {}).get("start_date")
            )
            last_e = self._parse_iso_date(
                periods[-1].get("period_date_range", {}).get("end_date")
            )
            if first_s is not None and first_s != birth_date:
                validation_issues.append(
                    f"First period starts at {first_s.isoformat()}, expected {birth_date.isoformat()}"
                )
            if last_e is not None and last_e != reference_date:
                validation_issues.append(
                    f"Last period ends at {last_e.isoformat()}, expected {reference_date.isoformat()}"
                )

        # Log results
        if alignment_log:
            logger.info("Seamless date chain alignment applied:")
            for entry in alignment_log:
                logger.info(f"  - {entry}")
        else:
            logger.info("Seamless date chain: all periods already perfectly aligned.")

        if validation_issues:
            logger.warning("Date chain validation issues remain after alignment:")
            for issue in validation_issues:
                logger.warning(f"  ⚠ {issue}")
        else:
            logger.info("Date chain validation passed: all periods seamlessly connected.")

        # Store alignment metadata in global_summary
        plan_dict.setdefault("global_summary", {})["date_chain_alignment"] = {
            "adjustments_made": len(alignment_log),
            "validation_issues_remaining": len(validation_issues),
            "details": alignment_log if alignment_log else ["no adjustments needed"],
            "validation": validation_issues if validation_issues else ["all checks passed"]
        }

        return plan_dict

    # DEPRECATED: These constants were used by _enforce_density_constraints.
    # Kept for backward compatibility but no longer actively used.
    DENSITY_MIN_DAYS: Dict[str, int] = {
        "none": 0,
        "around once a year": 365,
        "once per season": 90,
        "around once a month": 30,
        "around once a week": 7,
    }

    DENSITY_LEVELS = [
        "none",
        "around once a year",
        "once per season",
        "around once a month",
        "around once a week",
    ]

    def _enforce_density_constraints(
        self,
        plan_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """DEPRECATED: No longer needed. Density is now computed by
        AutobiographicalMemoryModel in temporal_context.py at P2 runtime.
        Kept as no-op for backward compatibility."""
        return plan_dict
    def _apply_general_plan_consistency_checks(
        self,
        plan_dict: Dict[str, Any],
        target_age_exact: int,
        current_job_tenure_months: Optional[int]
    ) -> Dict[str, Any]:
        """
        Universal (persona-agnostic) deterministic consistency check.
        Goal: avoid overfitting to a single example's semantics while ensuring usable output.
        """
        periods = plan_dict.get("life_periods", [])
        if not isinstance(periods, list) or not periods:
            return plan_dict

        # 1) Ensure total_periods matches the real period count (date-driven, no age normalization)
        global_summary = plan_dict.get("global_summary", {})
        global_summary["total_periods"] = len(plan_dict.get("life_periods", []))
        plan_dict["global_summary"] = global_summary

        # 2) If tenure constraint exists, ensure the last period has role_tenure_range
        if current_job_tenure_months is not None and current_job_tenure_months >= 0:
            last_period = plan_dict.get("life_periods", [])[-1]
            if "role_tenure_range" not in last_period:
                logger.warning("Detected missing role_tenure_range, flagged for subsequent constraint completion.")

            # 3) Date closed-interval fallback: previous period end = current period start - 1 day
            if len(periods) >= 2:
                previous_period = periods[-2]
                current_range = last_period.get("period_date_range")
                previous_range = previous_period.get("period_date_range")
                if isinstance(current_range, dict) and isinstance(previous_range, dict):
                    current_start = self._parse_iso_date(current_range.get("start_date"))
                    previous_start = self._parse_iso_date(previous_range.get("start_date"))
                    if current_start is not None:
                        repaired_prev_end = current_start - timedelta(days=1)
                        if previous_start is not None and repaired_prev_end < previous_start:
                            repaired_prev_end = previous_start
                        previous_range["end_date"] = repaired_prev_end.isoformat()
                        previous_period["period_date_range"] = previous_range

        return plan_dict

    async def _validate_plan_with_general_llm(
        self,
        plan_dict: Dict[str, Any],
        constraint_sheet: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optional: call a general filtering LLM for a risk report (does not force-rewrite the plan).
        """
        validator_system_prompt = """
        You are a universal life-period planning quality auditor.
        Your task is to check whether the input plan satisfies universal consistency, without relying on any specific example semantics.
        Focus on:
        1) Whether `period_date_range` is continuously covered without overlap or gaps;
        2) Whether date ranges are monotonic without obvious inversions;
        3) If current_job_tenure_months is provided, whether the last period is consistent with the month-level constraint;
        4) If there are gaps (e.g., between graduation and job start), whether they are properly expressed as independent gap periods;
        5) Whether the output contains overly example-specific, non-generalizable field names or semantic bindings.
        Output only structured JSON.
        """
        validator_prompt = f"""
        [constraint_sheet]
        {json.dumps(constraint_sheet, ensure_ascii=False, indent=2)}

        [plan_dict]
        {json.dumps(plan_dict, ensure_ascii=False, indent=2)}
        """

        try:
            report = await self.llm.generate_structured(
                prompt=validator_prompt,
                response_model=PlanValidationReport,
                system_prompt=validator_system_prompt,
                task_type="p1a_plan_validation",
                temperature=0.0
            )
            plan_dict.setdefault("global_summary", {})["validation_report"] = report.model_dump()
        except Exception as e:
            logger.warning(f"General validation model execution failed, skipping this step: {e}")

        return plan_dict

    def _build_plan_generation_prompt(
        self,
        enriched_constraints: Dict[str, Any],
        transition_hints_prompt_text: str,
        inferred_constraints_text: str,
        age_calendar_hint_text: str,
        birth_date: date,
        reference_date: date,
        review_feedback: str = "",
        social_context_text: str = "",
    ) -> str:
        review_feedback_block = ""
        if review_feedback:
            review_feedback_block = f"""

        Additional correction feedback (MUST absorb):
        {review_feedback}
        """

        return f"""
        Please generate a `Life Period Blueprint` for the following persona constraint sheet.
        Ensure life periods continuously cover from birth to reference_date by date.

        Notes:
        1. If 'persona_brief_text' (persona brief) is provided, pay special attention to this information — it determines the underlying narrative of who this person ultimately becomes.
        2. **Date chain constraint (highest priority)**: All periods must be built using `period_date_range` (closed date intervals), and adjacent periods must strictly satisfy:
           - `next_period.start_date == prev_period.end_date + 1 day`
           - First period `start_date` must equal `birth_date`
           - Last period `end_date` must equal `reference_date`
           - **Absolutely no date gaps allowed**
           - **Absolutely no date overlaps allowed**
        3. If there is an interpretable gap between key anchor points, output it as a separate gap period (`stage_label` must be `gap_transition`).
        4. If both `pre_current_transition_anchor_date` and `current_role_start_date` exist, and they differ by at least 2 days, you MUST explicitly output a gap period covering that interval.
        5. Only output date-driven segmentation; do not output any age range fields.
        6. Key date events in the persona (graduation/job start) must be prioritized to align with the corresponding period's `period_date_range`.
        7. Childhood amnesia belongs only to downstream memory segmentation rules and should NOT determine period boundaries; do not split periods for the age-4 boundary, nor shift period order or start/end dates to satisfy that boundary.
        8. `stage_label` is a free-form string (snake_case); please generate the most semantically appropriate label based on the period's actual content, without reusing fixed categories. The only hard convention: gap/transition periods must use 'gap_transition'.
        9. Do not hardcode age rules to satisfy a single example; generate generalizable life plans.
        11. **Social context and pathway reference (high priority)**: If [social_context] is provided:
            - **Prioritize `social_context.education_system` and `persona_pathway`**, using them to determine period order, boundaries, and atypical paths, rather than cutting segments around single age thresholds.
            - **Inferred Actual Pathway (high priority)**: If [Inferred Actual Pathway] is provided,
              this is the person's most likely actual life path. Plan Generation should prioritize following this path.
              Pay special attention to stages marked SKIP (should not be generated) and MERGE (should be merged).
            - **Social Norms hard constraints (plausibility boundaries)**: Cannot be violated (e.g., "impossible to start undergraduate at age 10").
            - **Social Norms typical path (reference only)**: If Inferred Actual Pathway differs from the typical path,
              use Inferred Actual Pathway as the authority.
            - **persona_brief_text (ultimate arbiter)**: When none of the above can cover the case, persona_brief_text takes precedence.
        12. **Period duration vs. content complexity matching (important)**: Each period's duration should be proportional to the complexity of its `developmental_tasks`
            and `stage_goals`. Self-check:
            - For each period, ask yourself: can the tasks described in this period be reasonably completed within the allocated time?
            - When splitting a continuous life experience into multiple sub-periods, ensure the split points reflect real
              rhythm changes, not simply piling most time up front and compressing the remainder.
            - If two adjacent sub-periods belong to the same larger life experience (e.g., same education stage, same job),
              their duration ratio should roughly match the amount of developmental content each carries.
              Extremely uneven splits (e.g., one sub-period is 5x+ the other) usually indicate the split point needs adjustment.
            - Special attention: periods near hard-constraint dates (e.g., `pre_current_transition_anchor_date`)
              are prone to being unreasonably compressed; please carefully check duration allocation in these areas.

        Social context reference (Social Context + Inferred Actual Pathway, follow the priority order above):
        {social_context_text if social_context_text else '(No social context information provided)'}

        Step-1 API transition date hints (please reference explicitly):
        {transition_hints_prompt_text}

        Step-1 API normalized temporal constraints (hard constraints, must be strictly used if non-empty):
        {inferred_constraints_text}

        Age-date reference (must reference explicitly, do not rely solely on vague age common sense):
        {age_calendar_hint_text}

        Hard constraints (must be satisfied):
        - If `current_role_start_date` is non-empty: the last period's `period_date_range.start_date` must equal that date, and `end_date` must equal `reference_date`.
        - If `pre_current_transition_anchor_date` is non-empty: the last period before the current role's `period_date_range.end_date` must equal that date.
        - If both `pre_current_transition_anchor_date` and `current_role_start_date` are non-empty:
          - You only need to generate the period chain up to `pre_current_transition_anchor_date` (satisfying the +1 day constraint)
          - Then directly generate the last period with `start_date` = `current_role_start_date`, `end_date` = `reference_date`
          - The gap between the second-to-last period's `end_date` and the last period's `start_date` will be auto-filled by post-processing; you do not need to handle it
          - Do NOT generate any period between these two dates (including gap_transition)
          - Except for this gap, all other adjacent periods must still satisfy `next.start_date = prev.end_date + 1 day`
        - If only one of these dates is non-empty, all adjacent periods must satisfy `next.start_date = prev.end_date + 1 day`.
        - The last period must end at `reference_date`.

        Output structure requirements:
        - `life_periods` sorted by `period_date_range.start_date` ascending.
        - Do not output age fields.
        - Each gap period must have complete theme, tasks, goals, pressures, and opportunities descriptions; do not leave them empty.{review_feedback_block}

        Timeline anchors:
        - birth_date={birth_date.isoformat()}
        - reference_date={reference_date.isoformat()}

        Enriched persona constraint sheet:
        {json.dumps(enriched_constraints, indent=2, ensure_ascii=False)}
        """ + (f"""

        ### Additional Context for Planning
        {format_persona_extensions(enriched_constraints.get('persona_extensions') or {}, stage='planning', heading='Extended Persona Attributes')}
        These traits should inform planning decisions.
        If "Specific Attitudes" are listed above, ensure they are reflected in the relevant life periods:
        - Negative attitudes toward a subject (e.g. a school subject) should appear as pressures or disengagement signals in the corresponding education periods.
        - Positive attitudes toward leisure activities should shape the persona's goals and opportunities in relevant periods.
        - Do NOT create a dedicated life period for an attitude — instead, weave it into the theme/pressures/opportunities of naturally occurring periods.

        If "Core Identity Traits" are listed above, ensure the life plan includes at least one period where each primary identity trait is meaningfully explored or expressed:
        - Sexual orientation (e.g., homosexual, bisexual): include a period covering identity discovery, coming-out, or relationship formation.
        - Religious identity (e.g., Atheist, Muslim, devout Catholic): include a period covering a moment of religious questioning, community engagement, or identity-based conflict.
        - Racial/ethnic identity (e.g., Black, Latino): include a period covering racial identity formation, discrimination experience, or community belonging.
        - Political/cultural identity: include a period covering an event that tests or expresses this identity.
        - Gender identity — Transgender: include at least two periods: (1) identity discovery / coming-out; (2) a transition-related period (medical, social, or legal aspects). If the persona is also a healthcare professional or advocate, add a period where they use personal experience to advocate for others.
        - Gender identity — Non-binary / Genderqueer: include a period covering identity exploration and a specific moment of asserting this identity in a social or professional context.
        - Disability identity — Deaf / Hard of Hearing: include at least one period with a SPECIFIC event involving sign language use (ASL, BSL, or other), navigating a hearing-dominated environment, or connecting with the deaf community. The event should show concrete communication strategies.
        - Disability identity — Other physical/cognitive: include a period covering a specific accessibility challenge, an advocacy moment, or a community belonging event that reveals how the disability shapes the character's daily life and worldview.
        - Do NOT create a dedicated life period solely for an identity trait — weave it into naturally occurring periods.

        Cultural Grounding (applies when the persona's growing_up_location.country or current_living_location.country is non-Western):
        - All events must reflect the persona's actual cultural context — workplace norms, family structures, social expectations, and life milestones should be grounded in their home culture, NOT in Western / American defaults.
        - Examples: Malaysian Muslim (prayer-time work schedule, Ramadan observance), Thai Buddhist monk (monastery hierarchy, alms rounds), Brazilian (family-centered social life, Carnival), Japanese retiree (seniority-based career progression), Polish WWII veteran (post-war Communist Poland).
        - The simulation still produces English text, but embeds culturally authentic details, references, and social dynamics.

        Multilingual Persona Guidance:
        - If the persona speaks multiple languages or comes from a multilingual background, ensure at least one life period includes a SPECIFIC event where language skills are actively used (language-learning milestone, cross-cultural communication event, moment where skills created opportunity or solved a problem). Include the language name, the context, and what was communicated.

        Cultural / Religious Practice Events:
        - If the persona's occupation or background implies regular cultural or religious practices, ensure at least one life period includes a specific event anchoring the practice. The event should show the practice in action AND reveal what it means to the persona personally.

        Adversity / Hardship Events:
        - If the persona's brief explicitly describes hardship, adversity, or struggle, ensure at least one life period includes a high-salience adversity event. The event should show active coping / confrontation, reveal emotional and practical stakes, and anchor resilience / bitterness / determination. Do NOT sanitize or resolve the adversity prematurely.

        Occupation Episodic Depth:
        - For personas with distinctive occupations, ensure at least one life period includes a SPECIFIC occupation-defining episode (e.g., "drove a 48-hour haul through a blizzard, navigating by CB radio" rather than "worked as a truck driver for 20 years"). These specific episodes are what PersonaGym evaluators use to test persona authenticity.
        """ if enriched_constraints.get('persona_extensions') else "")

    async def _generate_plan_via_llm(
        self,
        enriched_constraints: Dict[str, Any],
        transition_hints: TransitionDateHints,
        inferred_constraints: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
        review_feedback: str = "",
        social_context_text: str = "",
    ) -> Dict[str, Any]:
        transition_hints_prompt_text = self._transition_hints_to_prompt_text(transition_hints)
        inferred_constraints_text = self._build_inferred_constraints_hint_text(inferred_constraints)
        age_calendar_hint_text = self._build_age_calendar_hint(
            reference_date=reference_date,
            birth_date=birth_date,
            target_age_exact=resolved_target_age,
        )
        user_prompt = self._build_plan_generation_prompt(
            enriched_constraints=enriched_constraints,
            transition_hints_prompt_text=transition_hints_prompt_text,
            inferred_constraints_text=inferred_constraints_text,
            age_calendar_hint_text=age_calendar_hint_text,
            birth_date=birth_date,
            reference_date=reference_date,
            review_feedback=review_feedback,
            social_context_text=social_context_text,
        )

        logger.info("\n" + "="*20 + " SYSTEM PROMPT " + "="*20)
        logger.info(self.system_prompt.strip())
        logger.info("="*55)
        logger.info("\n" + "="*20 + " STEP1 TRANSITION HINTS " + "="*20)
        logger.info(json.dumps(transition_hints.model_dump(), indent=2, ensure_ascii=False))
        logger.info("="*59)
        logger.info("\n" + "="*20 + " USER PROMPT " + "="*20)
        logger.info(user_prompt.strip())
        logger.info("="*53 + "\n")
        logger.info("Prompt printed (including Step 1 transition hints), starting plan request...")

        milestone_plan = await self.llm.generate_structured(
            prompt=user_prompt,
            response_model=MilestonePlan,
            system_prompt=self.system_prompt,
            task_type="p1a_milestone_plan",
            temperature=0.0
        )
        return milestone_plan.model_dump()

    def _run_deterministic_postprocess(
        self,
        plan_dict: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
        current_job_tenure_months: Optional[int],
        resolved_pre_current_role_gap_months: Optional[int],
        transition_semantic_hint: str,
        inferred_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan_dict.setdefault("global_summary", {})["timeline_anchor"] = {
            "reference_date": reference_date.isoformat(),
            "target_age_exact": resolved_target_age,
            "derived_birth_date": birth_date.isoformat(),
            "simulation_start_date": birth_date.isoformat(),
            "simulation_end_date": reference_date.isoformat(),
        }

        if current_job_tenure_months is not None and current_job_tenure_months >= 0:
            plan_dict = self._apply_current_job_tenure_constraint(
                plan_dict=plan_dict,
                current_job_tenure_months=current_job_tenure_months,
                pre_current_role_gap_months=resolved_pre_current_role_gap_months or 0,
                transition_semantic_hint=transition_semantic_hint,
                current_role_start_date=inferred_constraints.get("current_role_start_date"),
                pre_current_transition_anchor_date=inferred_constraints.get("pre_current_transition_anchor_date"),
            )

        plan_dict = self._ensure_seamless_date_chain(plan_dict, birth_date, reference_date)

        # Re-anchor applied_to_period_id after date chain alignment.
        # _ensure_seamless_date_chain may have removed/re-indexed periods,
        # so applied_to_period_id (set by _apply_current_job_tenure_constraint)
        # may now be stale.  This is a safety net for the cross-ref sync
        # inside _ensure_seamless_date_chain; if that already corrected it,
        # this block is a no-op (idempotent).
        if current_job_tenure_months is not None and current_job_tenure_months >= 0:
            _periods = plan_dict.get("life_periods", [])
            if _periods:
                _tc = plan_dict.get("global_summary", {}).get(
                    "temporal_constraints", {}
                )
                _current_applied = _tc.get("applied_to_period_id")
                _actual_last_id = _periods[-1].get("period_id")
                if _current_applied != _actual_last_id:
                    logger.info(
                        f"Re-anchored applied_to_period_id: "
                        f"{_current_applied} -> {_actual_last_id}"
                    )
                    _tc["applied_to_period_id"] = _actual_last_id

        plan_dict = self._apply_general_plan_consistency_checks(
            plan_dict,
            target_age_exact=resolved_target_age,
            current_job_tenure_months=current_job_tenure_months,
        )
        plan_dict.setdefault("global_summary", {})["canonical_plan_postprocess"] = {
            "childhood_amnesia_applied_to_canonical_periods": True,
            "note": "childhood amnesia applied at canonical level (v7.3)",
        }
        plan_dict.setdefault("global_summary", {})["total_periods"] = len(plan_dict.get("life_periods", []))
        return plan_dict

    async def _review_plan_reasonableness(
        self,
        persona_config: Dict[str, Any],
        plan_dict: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        inferred_constraints: Optional[Dict[str, Any]] = None,
        social_context_text: str = "",
    ) -> PlanReviewReport:
        social_context_review_block = ""
        if social_context_text:
            social_context_review_block = f"""
        - **Important**: If [social_context] is provided, prioritize reviewing age-stage consistency based on `social_context.education_system` and [Inferred Actual Pathway].
          For each period in the plan, if its stage_label corresponds to a stage with typical_entry_age in social_context.education_system,
          check the deviation between the period's starting age and typical_entry_age. If the deviation exceeds 1 year and persona_brief_text does not provide a reasonable explanation for the atypical path
          (e.g., skipping grades, early enrollment, late enrollment), mark it as a medium or high severity issue.
          This is a general rule applicable to all education stages, not just undergraduate.
          However, if persona_brief_text or persona_pathway explicitly describes an atypical path, the deviation is acceptable.
        - **Important**: Childhood amnesia only affects subsequent derived memory segmentation and is NOT a reason to modify canonical stage boundaries; do not suggest rewriting stage start/end dates simply because a stage spans the age-4 boundary."""

        system_prompt = f"""
        You are a general-purpose life plan plausibility reviewer.

        Your inputs include:
        1. [persona_config]: Target persona constraints
        2. [timeline_anchor]: Timeline anchors (birth_date, reference_date)
        3. [hard_constraints]: Inviolable hard constraints (e.g., current_role_start_date, pre_current_transition_anchor_date)
        4. [plan_dict]: The life period plan to review
        5. [social_context]: Education system and social institution rules for the persona's social background (if provided)

        Your tasks:
        1. Check whether the plan is consistent with common sense in temporal logic, developmental sequence, age semantics, and stage content consistency
        2. Check whether the plan is compatible with hard_constraints
        3. **Prioritize checking stage sequence, stage boundaries, and typical age windows based on `social_context.education_system`**
        4. **Check whether the plan is consistent with [Inferred Actual Pathway] (if provided)**
           - Note specifically: stages marked as SKIP in Inferred Actual Pathway should NOT appear in the plan
        5. **Check whether each stage's duration matches the complexity of its described content**:
           - For each stage, evaluate the complexity of its `developmental_tasks` and `stage_goals`,
             and judge whether the allocated duration is sufficient to reasonably accomplish these tasks
           - When two adjacent stages belong to the same larger life experience (determinable via `stage_label` semantic similarity),
             check whether their duration ratio matches their respective content complexity.
             Extreme imbalance (e.g., >5:1) with the shorter stage describing non-trivial tasks should prompt a suggestion to adjust the split point
           - Pay special attention to whether stages near hard constraint dates are unreasonably compressed
           - Note: this check should be combined with `social_context` — in different cultural/education systems,
             certain stages are naturally shorter (e.g., transition periods, adaptation periods), which is not necessarily a problem
        6. If issues are found, prioritize suggesting local patches
        7. Only return needs_regeneration when structural issues are obvious and local patches are insufficient

        Important notes:
        - Post-processing will automatically insert a gap_transition stage between pre_current_transition_anchor_date and current_role_start_date; you do not need to worry about this gap
        - Post-processing will automatically align the date chain; you do not need to worry about minor date deviations (1-2 days)
        - Your patch suggestions must NOT violate any constraints in [hard_constraints]
        - Focus on life-semantic plausibility, not date precision (dates are guaranteed by post-processing)
        - Do not write special-case checks for specific samples, nor treat a fixed age for primary school as the only standard
        - **Do NOT** use childhood amnesia as a reason to rewrite canonical stage boundaries; if a stage spans the age-4 boundary, treat it as a memory-layer issue, not a stage-boundary issue
        - When suggesting adjustments to sub-stage boundary points, please provide patches for both start_date and end_date (in pairs),
          to avoid degenerate periods caused by modifying only one end
        - Duration balance checks should be evaluated in the context of this persona's cultural background and life path characteristics,
          do not apply any fixed "minimum duration" standard
        {social_context_review_block}

        Only the following patch_types are allowed: update_start_date, update_end_date, update_title, update_stage_label, update_transition_anchor, update_is_education_stage, update_is_work_stage, update_is_transition_stage, update_dominant_theme, update_developmental_tasks, update_stage_goals, update_salient_pressures, update_salient_opportunities, update_likely_transition_triggers.
        field_path can only point to: period_date_range.start_date, period_date_range.end_date, title, stage_label, is_education_stage, is_work_stage, is_transition_stage, dominant_theme, developmental_tasks, stage_goals, salient_pressures, salient_opportunities, likely_transition_triggers, or transition anchor fields starting with career_transition./pre_current_role_transition./graduation_transition.
        **Content field patch usage guide**: After structural operations like split/merge/delete, some period's content fields (dominant_theme, developmental_tasks, stage_goals, etc.) may no longer match their new time range or stage semantics. In such cases, use the corresponding update_* patch to fix these fields. For example, after a split, if the first half's dominant_theme still describes the entire stage, it should be patched to describe only the first half.

        **Stage type annotation consistency** (additional validation dimension):
        1. stage_label ↔ is_education_stage / is_work_stage / is_transition_stage consistency:
           - stage_label contains "school"/"university"/"phd" but is_education_stage=False → implausible
           - stage_label is "gap_transition" but is_transition_stage=False → implausible
           - stage_label contains "career"/"work"/"researcher" but is_work_stage=False → implausible
        2. title ↔ type tag consistency:
           - title describes educational activity (e.g., "Doctoral coursework") but is_education_stage=False → implausible
           - title describes work activity (e.g., "Senior Researcher at Tencent") but is_work_stage=False → implausible
        3. dominant_theme / developmental_tasks ↔ type tag consistency:
           - dominant_theme mentions "research work" but is_work_stage=False → implausible
           - developmental_tasks contains "complete dissertation" but is_education_stage=False → implausible
        4. Cross-period coherence:
           - If adjacent periods switch is_education_stage from True to False, there should be a transition or clear graduation event in between
        """
        # Build hard constraints section
        hard_constraints_text = ""
        if inferred_constraints:
            lines = []
            crs = inferred_constraints.get("current_role_start_date")
            pcta = inferred_constraints.get("pre_current_transition_anchor_date")
            if crs:
                lines.append(
                    f"- current_role_start_date: {crs} (the last period's start_date must equal this value)"
                )
            if pcta:
                lines.append(
                    f"- pre_current_transition_anchor_date: {pcta} (the pre-current-role period's end_date must equal this value)"
                )
            lines.append(
                f"- birth_date: {birth_date.isoformat()} (the first period's start_date must equal this value)"
            )
            lines.append(
                f"- reference_date: {reference_date.isoformat()} (the last period's end_date must equal this value)"
            )
            if crs and pcta:
                lines.append(
                    f"- Post-processing will automatically insert a gap_transition stage between {pcta} and {crs}"
                )
            hard_constraints_text = "\n".join(lines)
        else:
            hard_constraints_text = (
                f"- birth_date: {birth_date.isoformat()}\n"
                f"- reference_date: {reference_date.isoformat()}"
            )

        social_context_section = ""
        if social_context_text:
            social_context_section = f"""

        [social_context]
        {social_context_text}
        """

        user_prompt = f"""
        [persona_config]
        {json.dumps(persona_config, ensure_ascii=False, indent=2)}

        [timeline_anchor]
        birth_date={birth_date.isoformat()}
        reference_date={reference_date.isoformat()}

        [hard_constraints]
        {hard_constraints_text}
        {social_context_section}

        [plan_dict]
        {json.dumps(plan_dict, ensure_ascii=False, indent=2)}

        Please focus on checking:
        - Whether the temporal sequence is reasonable
        - Whether stage labels, titles, developmental tasks, goals, pressures are consistent with date ranges
        - Whether life development between stages follows general common sense
        - Whether there are obviously implausible age-stage combinations
        - Whether any stage has an extremely short/long duration that is semantically unjustified
        - Whether there are cases where dates are coherent but life semantics are incoherent
        - Whether your patch suggestions violate [hard_constraints]
        - **If [social_context] is provided, prioritize checking stage sequence, starting age windows, and typical duration based on `social_context.education_system`**
        - **Pathway consistency check (general)**:
          a. Are all life stages explicitly mentioned in persona_brief_text represented in the plan?
          b. If [Inferred Actual Pathway] is provided, are stages marked as SKIP still generated in the plan?
             If so, mark as high severity issue.
          b2. If [Inferred Actual Pathway] marks INSERT or SPLIT (indicating qualitative sub-stage divisions within a larger stage),
              check whether the plan splits that larger stage into the corresponding multiple periods.
              If a stage marked INSERT/SPLIT is merged into a single period, mark as medium severity issue,
              and suggest splitting it into the corresponding sub-stage periods.
          c. Is the plan consistent with the overall pathway of [Inferred Actual Pathway]?
          d. The reviewer must NOT suggest rewriting stage start/end dates simply because a stage spans the age-4 boundary; this should be treated as a derived memory segmentation issue.
          d. If a stage's duration significantly deviates from the typical duration in its social context,
             and persona_brief_text and Inferred Actual Pathway provide no reasonable explanation,
             mark as an issue and explain the reason. (Do not use hardcoded thresholds; use your own judgment)
        - **stage_label and social_context typical entry age consistency check (general rule)**:
          For each period in the plan, if its stage_label corresponds to a stage with typical_entry_age
          in social_context.education_system, check the deviation between the period's starting age and typical_entry_age.
          If the deviation exceeds 1 year and persona_brief_text does not provide a reasonable explanation for the atypical path
          (e.g., skipping grades, early enrollment, late enrollment), mark as medium or high severity issue.
          This is a general rule applicable to all education stages, not just undergraduate.
        """
        try:
            return await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=PlanReviewReport,
                system_prompt=system_prompt,
                task_type="p1a_plan_review",
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Reviewer failed, falling back to pass: {e}")
            return PlanReviewReport(status="pass", review_summary=f"reviewer_fallback_pass: {e}")

    # ==========================================
    # v3 Judge-Action-Recheck Methods
    # ==========================================

    async def _judge_plan_periods(
        self,
        plan_dict: Dict[str, Any],
        persona_config: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        social_context_text: str = "",
    ) -> PlanJudgementReport:
        """Judge stage: produce a structured verdict for every period.

        One LLM call sees the full plan but outputs per-period judgements
        in reverse order (last period first).
        """
        social_context_block = ""
        if social_context_text:
            social_context_block = f"\n[social_context]\n{social_context_text}\n"

        system_prompt = (
            "You are a structured life-plan reviewer.  You will receive a complete "
            "life-stage plan and must review every period from the LAST period to the "
            "FIRST, outputting one PeriodJudgement per period.\n\n"
            "Verdict types:\n"
            "- pass: the period is correct — persona, social context, adjacent coherence all OK.\n"
            "- rewrite: the content is wrong but the date range is fine — regenerate "
            "stage_label / title / tasks / goals / pressures / opportunities / triggers "
            "(dates stay fixed).\n"
            "- split: the period spans two qualitatively different life phases and should "
            "be split at a specific date.  You MUST provide split_point (ISO date).\n"
            "- merge: the period and an adjacent period are semantically redundant or both "
            "too short (<6 months each with similar content).  Specify merge_with_prev.\n"
            "- delete: the period should not exist (contradicts a SKIP marker in "
            "[Inferred Actual Pathway], or is entirely redundant).\n\n"
            "Key review dimensions:\n"
            "1. stage_label ↔ is_education_stage / is_work_stage / is_transition_stage consistency.\n"
            "2. title ↔ type-tag consistency.\n"
            "3. If [Persona Pathway] marks INSERT or SPLIT for a stage, check that the "
            "plan actually splits it into separate periods.  If not → verdict=split.\n"
            "4. If [Persona Pathway] marks SKIP for a stage, check that the plan does NOT "
            "contain it.  If it does → verdict=delete.\n"
            "5. Period duration vs. content complexity — is the time allocated reasonable?\n"
            "6. Coherence with adjacent periods — smooth life-trajectory transitions.\n"
            "7. Adjacent semantic redundancy: if two neighbouring periods have highly "
            "similar stage_label / title / dominant_theme AND each is shorter than "
            "6 months → verdict=merge on the later one.\n\n"
            "Important:\n"
            "- Do NOT flag childhood-amnesia boundary (age 4) as a stage-boundary issue.\n"
            "- Do NOT apply hard-coded minimum durations; judge relative to social context.\n"
            "- When suggesting split, the split_point MUST fall strictly between the "
            "period's start_date and end_date.\n"
            "- Prefer pass when the period is acceptable.  Only flag real issues.\n"
            "\n8. Grade/position sequence consistency (for education and career stages):\n"
            "   For periods within the same education stage (same stage_label), verify that "
            "any grade labels or year numbers mentioned in their titles form a strictly "
            "non-overlapping, ascending sequence. If two adjacent periods both mention "
            "'Year 1', or if the sequence is non-monotonic (e.g., Year 3 followed by Year 2), "
            "flag the later one with verdict=rewrite and instruct: "
            "'Update title to reflect the correct grade/year based on the period date range "
            "[start ~ end] and the education system entry age from [Social Context].'\n"
            "   Do NOT require grade labels to be present — only check if they ARE present "
            "and are inconsistent with the date range.\n"
            "\n9. Over-segmentation check:\n"
            "   If THREE or more consecutive periods share the SAME stage_label AND each is "
            "shorter than 18 months AND their dominant_themes are highly similar (no "
            "qualitatively distinct shift), flag the MIDDLE ones with verdict=merge. "
            "Prefer merging into 2-period groups (early + late) rather than a single period "
            "to preserve narrative structure.\n"
            "   Exception: do NOT merge if any period contains a major transition event "
            "(graduation, exam, school change, etc.) in its likely_transition_triggers.\n"
            "   Note: this check applies to ALL stage types — education, career, retirement.\n"
        )

        user_prompt = (
            f"[persona_config]\n{json.dumps(persona_config, ensure_ascii=False, indent=2)}\n\n"
            f"[timeline_anchor]\nbirth_date={birth_date.isoformat()}\n"
            f"reference_date={reference_date.isoformat()}\n"
            f"{social_context_block}\n"
            f"[plan_dict]\n{json.dumps(plan_dict, ensure_ascii=False, indent=2)}\n\n"
            "Please review every period from the last to the first and output a "
            "PlanJudgementReport with one PeriodJudgement per period."
        )

        try:
            return await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=PlanJudgementReport,
                system_prompt=system_prompt,
                task_type="p1a_plan_judgement",
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(f"Judge stage failed, falling back to all-pass: {e}")
            periods = plan_dict.get("life_periods", [])
            return PlanJudgementReport(
                judgements=[
                    PeriodJudgement(
                        period_id=p.get("period_id", f"LP{i+1}"),
                        verdict="pass",
                        reason="judge_fallback_pass",
                        instruction="",
                    )
                    for i, p in enumerate(periods)
                ],
                global_assessment=f"judge_fallback_pass: {e}",
            )

    async def _execute_judgements(
        self,
        plan_dict: Dict[str, Any],
        judgement_report: PlanJudgementReport,
        persona_brief_text: str,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
    ) -> tuple:
        """Execute all non-pass verdicts from the judge.

        Execution order: rewrite → merge → split → delete.
        After each merge/split/delete the plan is postprocessed (date-chain
        alignment + period_id renumbering) and the id_tracker is updated so
        that subsequent operations can locate their targets correctly.

        Returns (plan_dict, id_tracker).
        """
        id_tracker: Dict[str, str] = {
            j.period_id: j.period_id for j in judgement_report.judgements
        }

        # --- Phase 1: rewrite (no structural change) ---
        for j in judgement_report.judgements:
            if j.verdict != "rewrite":
                continue
            current_id = id_tracker.get(j.period_id, j.period_id)
            plan_dict = await self._regenerate_specific_periods_v7(
                plan_dict=plan_dict,
                period_ids=[current_id],
                review_feedback=j.instruction,
                persona_brief_text=persona_brief_text,
                social_context_text=social_context_text,
                birth_date=birth_date,
                reference_date=reference_date,
            )

        # --- Phase 2: merge (-1 period each) ---
        for j in judgement_report.judgements:
            if j.verdict != "merge":
                continue
            current_id = id_tracker.get(j.period_id, j.period_id)
            plan_dict = self._merge_period(
                plan_dict=plan_dict,
                period_id=current_id,
                merge_with_prev=j.merge_with_prev,
                instruction=j.instruction,
            )
            plan_dict = self._run_deterministic_postprocess(
                plan_dict=plan_dict,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                current_job_tenure_months=None,
                resolved_pre_current_role_gap_months=None,
                transition_semantic_hint="general",
                inferred_constraints={},
            )
            self._update_id_tracker_after_delete(
                id_tracker, j.period_id,
                plan_dict.get("life_periods", []),
            )

        # Regenerate content for merged periods that need it
        await self._regenerate_needs_content_gen_periods(
            plan_dict, persona_brief_text, social_context_text,
            birth_date, reference_date,
        )

        # --- Phase 3: split (+1 period each) ---
        for j in judgement_report.judgements:
            if j.verdict != "split":
                continue
            current_id = id_tracker.get(j.period_id, j.period_id)
            plan_dict = self._split_period(
                plan_dict=plan_dict,
                period_id=current_id,
                split_point=j.split_point,
                instruction=j.instruction,
            )
            plan_dict = self._run_deterministic_postprocess(
                plan_dict=plan_dict,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                current_job_tenure_months=None,
                resolved_pre_current_role_gap_months=None,
                transition_semantic_hint="general",
                inferred_constraints={},
            )
            self._update_id_tracker_after_split(
                id_tracker, j.period_id,
                plan_dict.get("life_periods", []),
            )

        # Regenerate content for split sub-periods that need it
        await self._regenerate_needs_content_gen_periods(
            plan_dict, persona_brief_text, social_context_text,
            birth_date, reference_date,
        )

        # --- Phase 4: delete (-1 period each) ---
        for j in judgement_report.judgements:
            if j.verdict != "delete":
                continue
            current_id = id_tracker.get(j.period_id, j.period_id)
            plan_dict = self._delete_period(
                plan_dict=plan_dict,
                period_id=current_id,
            )
            plan_dict = self._run_deterministic_postprocess(
                plan_dict=plan_dict,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                current_job_tenure_months=None,
                resolved_pre_current_role_gap_months=None,
                transition_semantic_hint="general",
                inferred_constraints={},
            )
            self._update_id_tracker_after_delete(
                id_tracker, j.period_id,
                plan_dict.get("life_periods", []),
            )

        # Regenerate content for periods that absorbed deleted periods' time ranges
        await self._regenerate_needs_content_gen_periods(
            plan_dict, persona_brief_text, social_context_text,
            birth_date, reference_date,
        )

        return plan_dict, id_tracker

    async def _regenerate_needs_content_gen_periods(
        self,
        plan_dict: Dict[str, Any],
        persona_brief_text: str,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
    ) -> None:
        """Regenerate content for all periods marked with _needs_content_gen."""
        periods = plan_dict.get("life_periods", [])
        for p in periods:
            if not p.get("_needs_content_gen"):
                continue
            pid = p.get("period_id")
            feedback = p.get("_merge_instruction") or p.get("_split_instruction") or ""
            await self._regenerate_specific_periods_v7(
                plan_dict=plan_dict,
                period_ids=[pid],
                review_feedback=feedback,
                persona_brief_text=persona_brief_text,
                social_context_text=social_context_text,
                birth_date=birth_date,
                reference_date=reference_date,
            )
            # Clear the flag after regeneration
            p.pop("_needs_content_gen", None)
            p.pop("_merge_instruction", None)
            p.pop("_split_instruction", None)
            p.pop("_merged_from", None)
            p.pop("_merged_content", None)

    def _split_period(
        self,
        plan_dict: Dict[str, Any],
        period_id: str,
        split_point: Optional[str],
        instruction: str,
    ) -> Dict[str, Any]:
        """Split a period into two sub-periods at the given split_point.

        Does NOT renumber period_ids — caller must invoke
        _run_deterministic_postprocess() afterwards.
        """
        periods = plan_dict.get("life_periods", [])
        idx = next(
            (i for i, p in enumerate(periods) if p.get("period_id") == period_id),
            None,
        )
        if idx is None or not split_point:
            logger.warning(f"[split] Cannot split {period_id}: not found or no split_point")
            return plan_dict

        original = periods[idx]
        split_date = self._parse_iso_date(split_point)
        orig_start = self._parse_iso_date(original["period_date_range"]["start_date"])
        orig_end = self._parse_iso_date(original["period_date_range"]["end_date"])

        if not split_date or not orig_start or not orig_end:
            logger.warning(f"[split] Unparseable dates for {period_id}")
            return plan_dict

        # Clamp split_point into valid range
        if split_date <= orig_start:
            split_date = orig_start + timedelta(days=30)
        if split_date >= orig_end:
            split_date = orig_end - timedelta(days=30)
        if split_date <= orig_start or split_date >= orig_end:
            logger.warning(f"[split] Invalid split_point for {period_id} after clamping")
            return plan_dict

        sub_a = copy.deepcopy(original)
        sub_a["period_id"] = period_id
        sub_a["_original_id"] = period_id
        sub_a["period_date_range"] = {
            "start_date": orig_start.isoformat(),
            "end_date": split_date.isoformat(),
        }
        # Both halves need content regeneration: the original content
        # describes the WHOLE period, not just the first half.
        sub_a["_needs_content_gen"] = True
        sub_a["_split_instruction"] = (
            f"This is the FIRST half after splitting the original period at "
            f"{split_date.isoformat()}. Original instruction: {instruction}. "
            f"Regenerate content to reflect ONLY the first half "
            f"({orig_start.isoformat()} ~ {split_date.isoformat()})."
        )

        sub_b = copy.deepcopy(original)
        sub_b["period_id"] = f"{period_id}_SPLIT"
        sub_b["_original_id"] = f"{period_id}_SPLIT"
        sub_b["period_date_range"] = {
            "start_date": (split_date + timedelta(days=1)).isoformat(),
            "end_date": orig_end.isoformat(),
        }
        sub_b["_needs_content_gen"] = True
        sub_b["_split_instruction"] = (
            f"This is the SECOND half after splitting the original period at "
            f"{split_date.isoformat()}. Original instruction: {instruction}. "
            f"Regenerate content to reflect ONLY the second half "
            f"({(split_date + timedelta(days=1)).isoformat()} ~ {orig_end.isoformat()})."
        )

        # Tag subsequent periods for tracker
        for i in range(idx + 1, len(periods)):
            if "_original_id" not in periods[i]:
                periods[i]["_original_id"] = periods[i]["period_id"]

        periods[idx] = sub_a
        periods.insert(idx + 1, sub_b)

        logger.info(
            f"[split] {period_id} split at {split_date.isoformat()} → "
            f"{sub_a['period_id']} + {sub_b['period_id']}"
        )
        return plan_dict

    def _merge_period(
        self,
        plan_dict: Dict[str, Any],
        period_id: str,
        merge_with_prev: bool = True,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Merge a period with its adjacent neighbour.

        Default direction: merge with the *previous* period.
        Does NOT renumber — caller must invoke _run_deterministic_postprocess().
        """
        periods = plan_dict.get("life_periods", [])
        idx = next(
            (i for i, p in enumerate(periods) if p.get("period_id") == period_id),
            None,
        )
        if idx is None:
            logger.warning(f"[merge] Cannot find {period_id}")
            return plan_dict

        if merge_with_prev and idx > 0:
            prev = periods[idx - 1]
            current = periods[idx]
            # Preserve the absorbed period's key content for LLM context
            absorbed_summary = {
                "period_id": current.get("period_id"),
                "title": current.get("title"),
                "stage_label": current.get("stage_label"),
                "dominant_theme": current.get("dominant_theme"),
                "developmental_tasks": current.get("developmental_tasks", []),
                "stage_goals": current.get("stage_goals", []),
                "date_range": current.get("period_date_range", {}),
            }
            prev["period_date_range"]["end_date"] = current["period_date_range"]["end_date"]
            prev["_needs_content_gen"] = True
            prev["_merge_instruction"] = (
                f"{instruction}\n"
                f"[Absorbed period content for reference]\n"
                f"The following period was merged INTO this one. "
                f"Integrate its key themes into the regenerated content:\n"
                f"  title: {absorbed_summary['title']}\n"
                f"  dominant_theme: {absorbed_summary['dominant_theme']}\n"
                f"  developmental_tasks: {absorbed_summary['developmental_tasks']}\n"
                f"  stage_goals: {absorbed_summary['stage_goals']}"
            )
            for i in range(idx + 1, len(periods)):
                if "_original_id" not in periods[i]:
                    periods[i]["_original_id"] = periods[i]["period_id"]
            periods.pop(idx)
            logger.info(f"[merge] {period_id} merged into {prev.get('period_id')}")

        elif not merge_with_prev and idx < len(periods) - 1:
            current = periods[idx]
            next_p = periods[idx + 1]
            # Preserve the absorbed period's key content for LLM context
            absorbed_summary = {
                "period_id": next_p.get("period_id"),
                "title": next_p.get("title"),
                "stage_label": next_p.get("stage_label"),
                "dominant_theme": next_p.get("dominant_theme"),
                "developmental_tasks": next_p.get("developmental_tasks", []),
                "stage_goals": next_p.get("stage_goals", []),
                "date_range": next_p.get("period_date_range", {}),
            }
            current["period_date_range"]["end_date"] = next_p["period_date_range"]["end_date"]
            current["_needs_content_gen"] = True
            current["_merge_instruction"] = (
                f"{instruction}\n"
                f"[Absorbed period content for reference]\n"
                f"The following period was merged INTO this one. "
                f"Integrate its key themes into the regenerated content:\n"
                f"  title: {absorbed_summary['title']}\n"
                f"  dominant_theme: {absorbed_summary['dominant_theme']}\n"
                f"  developmental_tasks: {absorbed_summary['developmental_tasks']}\n"
                f"  stage_goals: {absorbed_summary['stage_goals']}"
            )
            for i in range(idx + 2, len(periods)):
                if "_original_id" not in periods[i]:
                    periods[i]["_original_id"] = periods[i]["period_id"]
            periods.pop(idx + 1)
            logger.info(f"[merge] {next_p.get('period_id')} merged into {period_id}")

        else:
            logger.warning(f"[merge] Cannot merge {period_id}: no neighbour in requested direction")

        return plan_dict

    def _delete_period(
        self,
        plan_dict: Dict[str, Any],
        period_id: str,
    ) -> Dict[str, Any]:
        """Delete a period; its time range is absorbed by the previous period.

        If the deleted period is the first one, the next period absorbs instead.
        Does NOT renumber — caller must invoke _run_deterministic_postprocess().
        """
        periods = plan_dict.get("life_periods", [])
        idx = next(
            (i for i, p in enumerate(periods) if p.get("period_id") == period_id),
            None,
        )
        if idx is None:
            logger.warning(f"[delete] Cannot find {period_id}")
            return plan_dict

        deleted = periods[idx]
        deleted_start = self._parse_iso_date(deleted["period_date_range"]["start_date"])
        deleted_end = self._parse_iso_date(deleted["period_date_range"]["end_date"])

        for i in range(idx + 1, len(periods)):
            if "_original_id" not in periods[i]:
                periods[i]["_original_id"] = periods[i]["period_id"]

        if idx > 0 and deleted_end:
            absorber = periods[idx - 1]
            absorber["period_date_range"]["end_date"] = deleted_end.isoformat()
            # The absorber's date range expanded; mark for content regeneration
            absorber["_needs_content_gen"] = True
            absorber["_merge_instruction"] = (
                f"This period absorbed the time range of deleted period '{period_id}' "
                f"({deleted_start.isoformat() if deleted_start else '?'} ~ "
                f"{deleted_end.isoformat()}). "
                f"Deleted period info: title='{deleted.get('title', '')}', "
                f"dominant_theme='{deleted.get('dominant_theme', '')}'. "
                f"Update content to cover the expanded date range if needed."
            )
        elif idx < len(periods) - 1 and deleted_start:
            absorber = periods[idx + 1]
            absorber["period_date_range"]["start_date"] = deleted_start.isoformat()
            # The absorber's date range expanded; mark for content regeneration
            absorber["_needs_content_gen"] = True
            absorber["_merge_instruction"] = (
                f"This period absorbed the time range of deleted period '{period_id}' "
                f"({deleted_start.isoformat()} ~ "
                f"{deleted_end.isoformat() if deleted_end else '?'}). "
                f"Deleted period info: title='{deleted.get('title', '')}', "
                f"dominant_theme='{deleted.get('dominant_theme', '')}'. "
                f"Update content to cover the expanded date range if needed."
            )

        periods.pop(idx)
        logger.info(f"[delete] Removed {period_id}")
        return plan_dict

    # ── id_tracker helpers ──

    @staticmethod
    def _update_id_tracker_after_split(
        id_tracker: Dict[str, str],
        split_original_id: str,
        post_periods: list,
    ) -> None:
        """Update id_tracker after a split + postprocess renumber."""
        new_id_map = {
            p.get("_original_id", p["period_id"]): p["period_id"]
            for p in post_periods
        }
        for orig_id in list(id_tracker.keys()):
            current = id_tracker[orig_id]
            if current in new_id_map:
                id_tracker[orig_id] = new_id_map[current]
            elif orig_id in new_id_map:
                id_tracker[orig_id] = new_id_map[orig_id]

    @staticmethod
    def _update_id_tracker_after_delete(
        id_tracker: Dict[str, str],
        deleted_original_id: str,
        post_periods: list,
    ) -> None:
        """Update id_tracker after a delete/merge + postprocess renumber."""
        id_tracker.pop(deleted_original_id, None)
        new_id_map = {
            p.get("_original_id", p["period_id"]): p["period_id"]
            for p in post_periods
        }
        for orig_id in list(id_tracker.keys()):
            current = id_tracker[orig_id]
            if current in new_id_map:
                id_tracker[orig_id] = new_id_map[current]
            elif orig_id in new_id_map:
                id_tracker[orig_id] = new_id_map[orig_id]

    async def _recheck_adjacent_coherence(
        self,
        plan_dict: Dict[str, Any],
        modified_period_ids: List[str],
        persona_config: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        social_context_text: str = "",
    ) -> PlanReviewReport:
        """Lightweight recheck: only the modified periods and their neighbours."""
        periods = plan_dict.get("life_periods", [])
        check_ids: set = set()
        for pid in modified_period_ids:
            idx = next(
                (i for i, p in enumerate(periods) if p.get("period_id") == pid),
                None,
            )
            if idx is not None:
                check_ids.add(pid)
                if idx > 0:
                    check_ids.add(periods[idx - 1].get("period_id"))
                if idx < len(periods) - 1:
                    check_ids.add(periods[idx + 1].get("period_id"))

        filtered_periods = [p for p in periods if p.get("period_id") in check_ids]
        filtered_plan = {**plan_dict, "life_periods": filtered_periods}

        return await self._review_plan_reasonableness(
            persona_config=persona_config,
            plan_dict=filtered_plan,
            birth_date=birth_date,
            reference_date=reference_date,
            social_context_text=social_context_text,
        )

    def _apply_local_plan_patches(
        self,
        plan_dict: Dict[str, Any],
        report: PlanReviewReport,
    ) -> Dict[str, Any]:
        periods = plan_dict.get("life_periods", [])
        if not isinstance(periods, list):
            return plan_dict, []

        by_id = {str(period.get("period_id")): period for period in periods}
        applied = []
        skipped = []

        # ── Fix (Plan-A): Pre-simulate ALL date patches for each period atomically ──
        # Group date patches by period_id so we can validate the *final* state
        # of start_date + end_date together, rather than checking each patch
        # individually against the current (partially-patched) state.
        # This prevents the classic "start > end" false-positive skip that
        # occurs when, e.g., end_date is patched before start_date and the
        # intermediate state looks degenerate.
        from collections import defaultdict as _defaultdict
        date_patches_by_period: Dict[str, Dict[str, str]] = _defaultdict(dict)
        for patch in report.patches:
            if patch.field_path in {"period_date_range.start_date", "period_date_range.end_date"}:
                leaf = patch.field_path.split(".")[-1]
                date_patches_by_period[patch.period_id][leaf] = patch.new_value

        # Validate each period's final simulated date range
        period_date_patch_ok: Dict[str, bool] = {}
        for pid, date_overrides in date_patches_by_period.items():
            period = by_id.get(pid)
            if not period:
                period_date_patch_ok[pid] = False
                continue
            dr = period.get("period_date_range", {})
            simulated_dr = dict(dr)
            simulated_dr.update(date_overrides)
            sim_start = self._parse_iso_date(simulated_dr.get("start_date"))
            sim_end = self._parse_iso_date(simulated_dr.get("end_date"))
            if sim_start is not None and sim_end is not None and sim_end < sim_start:
                logger.warning(
                    f"[patch-A] Rejecting ALL date patches for {pid}: "
                    f"combined result would be degenerate "
                    f"({simulated_dr.get('start_date')} ~ {simulated_dr.get('end_date')})"
                )
                period_date_patch_ok[pid] = False
            else:
                period_date_patch_ok[pid] = True

        for patch in report.patches:
            period = by_id.get(patch.period_id)
            if not period:
                skipped.append({**patch.model_dump(), "skip_reason": f"period_id {patch.period_id} not found"})
                continue

            # Pre-validate date patches using the atomically-simulated result
            if patch.field_path in {"period_date_range.start_date", "period_date_range.end_date"}:
                if not period_date_patch_ok.get(patch.period_id, True):
                    skipped.append({
                        **patch.model_dump(),
                        "skip_reason": (
                            "rejected: combined date patches for this period would "
                            "create a degenerate range (start > end)"
                        ),
                    })
                    continue
                # NOTE: Duration reasonableness is checked by
                # _validate_plan_durations_with_llm() after all patches are applied.
                period.setdefault("period_date_range", {})[patch.field_path.split(".")[-1]] = patch.new_value
            elif patch.field_path in {"title", "stage_label"}:
                period[patch.field_path] = patch.new_value
            elif patch.field_path in {"is_education_stage", "is_work_stage", "is_transition_stage"}:
                period[patch.field_path] = patch.new_value
            elif patch.field_path in {
                "dominant_theme",
                "developmental_tasks",
                "stage_goals",
                "salient_pressures",
                "salient_opportunities",
                "likely_transition_triggers",
            }:
                period[patch.field_path] = patch.new_value
            elif patch.patch_type == "update_transition_anchor" and any(
                patch.field_path.startswith(prefix)
                for prefix in ("career_transition.", "pre_current_role_transition.", "graduation_transition.")
            ):
                root_key, leaf_key = patch.field_path.split(".", 1)
                period.setdefault(root_key, {})[leaf_key] = patch.new_value
            else:
                logger.info(f"Skipping unsupported patch: {patch.model_dump()}")
                skipped.append({**patch.model_dump(), "skip_reason": "unsupported patch type"})
                continue
            applied.append(patch.model_dump())

        plan_dict.setdefault("global_summary", {})["last_applied_patches"] = applied
        if skipped:
            plan_dict.setdefault("global_summary", {})["skipped_patches"] = skipped
            logger.warning(
                f"Skipped {len(skipped)} patch(es) during application: "
                f"{[s.get('skip_reason') for s in skipped]}"
            )
        return plan_dict, applied

    # ----------------------------------------------------------------
    # Content Refresh: re-generate content fields for patched periods
    # ----------------------------------------------------------------

    async def _refresh_patched_period_contents(
        self,
        plan_dict: Dict[str, Any],
        applied_patches: List[Dict[str, Any]],
        persona_brief_text: str,
        social_context_text: str = "",
    ) -> Dict[str, Any]:
        """After stage_label/title patches, re-generate content fields for affected periods.

        When a patch changes stage_label or title, the 6 content fields
        (dominant_theme, developmental_tasks, stage_goals, salient_pressures,
        salient_opportunities, likely_transition_triggers) become stale because
        they still describe the OLD stage semantics. This method calls LLM once
        per affected period to regenerate those fields, keeping each query small
        and precise.

        Args:
            plan_dict: The current plan dictionary.
            applied_patches: List of applied patch dicts from _apply_local_plan_patches.
            persona_brief_text: The persona brief text for context.
            social_context_text: Social context description for context.

        Returns:
            Updated plan_dict with refreshed content fields.
        """
        # 1. Identify periods that had stage_label, title, or is_*_stage patched
        patched_period_ids = set()
        for patch in applied_patches:
            if patch.get("field_path") in {
                "stage_label", "title",
                "is_education_stage", "is_work_stage", "is_transition_stage",
            }:
                patched_period_ids.add(patch.get("period_id"))

        if not patched_period_ids:
            return plan_dict

        # 2. Build period lookup
        periods = plan_dict.get("life_periods", [])
        by_id = {str(p.get("period_id")): p for p in periods}

        # 3. Build context: full plan timeline for reference
        timeline_context = []
        for p in periods:
            pid = p.get("period_id", "?")
            dr = p.get("period_date_range", {})
            timeline_context.append(
                f"  {pid}: {p.get('stage_label', '?')} | "
                f"{p.get('title', '?')} | "
                f"{dr.get('start_date', '?')} ~ {dr.get('end_date', '?')}"
            )
        timeline_text = "\n".join(timeline_context)

        # 4. For each affected period, call LLM to regenerate content fields
        async def refresh_one_period(period_id: str) -> None:
            period = by_id.get(period_id)
            if not period:
                return
            # Skip gap_transition periods — they use fixed template content
            if period.get("stage_label") == "gap_transition":
                return

            dr = period.get("period_date_range", {})
            prompt = f"""You need to regenerate content description fields for a life stage.

[Background]
Persona brief: {persona_brief_text}
{f"Social context: {social_context_text}" if social_context_text else ""}

[Complete life stage timeline]
{timeline_text}

[Stage needing content regeneration]
- period_id: {period_id}
- stage_label: {period.get('stage_label', '')}
- title: {period.get('title', '')}
- Time range: {dr.get('start_date', '?')} ~ {dr.get('end_date', '?')}

[Old content (outdated, does not match the new stage_label/title)]
- dominant_theme: {period.get('dominant_theme', '')}
- developmental_tasks: {json.dumps(period.get('developmental_tasks', []), ensure_ascii=False)}
- stage_goals: {json.dumps(period.get('stage_goals', []), ensure_ascii=False)}

[Task]
Based on the new stage_label and title, combined with the persona brief, social context, and time range, please regenerate the following 9 fields:
1. dominant_theme: The core organizing logic and dominant theme of this stage (one sentence)
2. developmental_tasks: Developmental tasks that need to be accomplished in this stage (3-5 items)
3. stage_goals: What goals the working self tends to organize around in this stage (3-5 items)
4. salient_pressures: Typical pressures faced in this stage (2-4 items)
5. salient_opportunities: Opportunity windows that converge toward the target persona (2-4 items)
6. likely_transition_triggers: Events most likely to push this stage into the next stage (1-3 items)
7. is_education_stage: Whether this stage involves educational/learning activities (bool)
8. is_work_stage: Whether this stage involves work/career activities (bool)
9. is_transition_stage: Whether this stage is a transition/gap period (bool)

Requirements:
- Content must be semantically fully consistent with the new stage_label ({period.get('stage_label', '')}) and title ({period.get('title', '')})
- Content must match the age range corresponding to the time range ({dr.get('start_date', '?')} ~ {dr.get('end_date', '?')})
- Content must be consistent with specific information in the persona brief (e.g., school, major, research direction, etc.)
- Do not copy old content; write fresh based on the new stage semantics
"""
            system_prompt = (
                "You are a developmental psychology expert. Your task is to generate precise content descriptions for a life stage. "
                "The output must fully match the stage's stage_label, title, and time range."
            )

            try:
                refreshed = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=PeriodContentRefresh,
                    system_prompt=system_prompt,
                    task_type="p1a_period_refresh",
                    temperature=0.0,
                )
                # Apply refreshed content to the period
                period["dominant_theme"] = refreshed.dominant_theme
                period["developmental_tasks"] = refreshed.developmental_tasks
                period["stage_goals"] = refreshed.stage_goals
                period["salient_pressures"] = refreshed.salient_pressures
                period["salient_opportunities"] = refreshed.salient_opportunities
                period["likely_transition_triggers"] = refreshed.likely_transition_triggers
                # Sync stage type tags
                period["is_education_stage"] = refreshed.is_education_stage
                period["is_work_stage"] = refreshed.is_work_stage
                period["is_transition_stage"] = refreshed.is_transition_stage
                logger.info(
                    f"Content refresh for {period_id}: "
                    f"dominant_theme='{refreshed.dominant_theme[:50]}...'"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to refresh content for {period_id}: {e}. "
                    f"Keeping old content fields."
                )

        # 5. Execute refreshes sequentially (each is a separate LLM call)
        self._print_step("10b", f"Refreshing content fields for {len(patched_period_ids)} patched period(s)")
        for pid in sorted(patched_period_ids):
            await refresh_one_period(pid)

        return plan_dict

    def _validate_plan_invariants(
        self,
        plan_dict: Dict[str, Any],
        birth_date: date,
        reference_date: date,
    ) -> List[str]:
        issues: List[str] = []
        periods = plan_dict.get("life_periods", [])
        if not isinstance(periods, list) or not periods:
            return ["life_periods is empty"]

        valid_densities = set(self.DENSITY_LEVELS)
        previous_end: Optional[date] = None
        previous_age: Optional[int] = None

        for index, period in enumerate(periods, start=1):
            period_id = period.get("period_id")
            expected_period_id = f"LP{index}"
            if period_id != expected_period_id:
                issues.append(f"period_id not sequential: expected {expected_period_id}, got {period_id}")

            dr = period.get("period_date_range")
            if not isinstance(dr, dict):
                issues.append(f"{period_id}: missing period_date_range")
                continue

            start = self._parse_iso_date(dr.get("start_date"))
            end = self._parse_iso_date(dr.get("end_date"))
            if start is None or end is None:
                issues.append(f"{period_id}: unparseable start_date/end_date")
                continue
            if start > end:
                issues.append(f"{period_id}: start_date > end_date")
            # Minimum duration: each period should be at least ~1 month (30 days)
            period_duration_days = (end - start).days + 1
            if period_duration_days < 30:
                issues.append(
                    f"{period_id}: period too short ({period_duration_days} days, "
                    f"{start.isoformat()} ~ {end.isoformat()}). "
                    f"Minimum is 30 days."
                )
            if index == 1 and start != birth_date:
                issues.append(f"{period_id}: first period start {start.isoformat()} != birth_date {birth_date.isoformat()}")
            if index == len(periods) and end != reference_date:
                issues.append(f"{period_id}: last period end {end.isoformat()} != reference_date {reference_date.isoformat()}")
            if previous_end is not None:
                expected_start = previous_end + timedelta(days=1)
                if start != expected_start:
                    issues.append(f"{period_id}: expected start_date {expected_start.isoformat()}, got {start.isoformat()}")

            # NOTE: default_memory_density validation removed.
            # Density is now computed at P2 runtime by AutobiographicalMemoryModel.

            derived_age = self._compute_exact_age(birth_date, start)
            if previous_age is not None and derived_age < previous_age:
                issues.append(f"{period_id}: derived age regressed from {previous_age} to {derived_age}")
            previous_age = derived_age
            previous_end = end

            # Validate transition anchors — but skip cross-period reference
            # fields that are written by _apply_current_job_tenure_constraint()
            # and legitimately reference dates outside this period.
            CROSS_PERIOD_CONTAINERS = {
                "career_transition",
                "pre_current_role_transition",
                "graduation_transition",
                "gap_context",
                "role_tenure_range",
            }
            for key, value in period.items():
                if key in CROSS_PERIOD_CONTAINERS:
                    continue  # skip entire sub-dict
                if isinstance(value, str) and (
                    key.endswith("anchor_date")
                    or key in {"graduation_date", "employment_start_date"}
                ):
                    anchor_date = self._parse_iso_date(value)
                    if anchor_date is not None and not (start <= anchor_date <= end):
                        issues.append(
                            f"{period_id}: transition anchor {key}={value} "
                            f"falls outside {start.isoformat()}~{end.isoformat()}"
                        )

        # --- Duration statistical anomaly detection ---
        # NOTE: Duration-related checks (IQR outlier detection, adjacent ratio,
        # education stage duration) have been moved to the LLM-based
        # _validate_plan_durations_with_llm() method for more flexible,
        # context-aware judgment. Only deterministic checks remain here.

        # --- Education stage total duration check (deterministic) ---
        # When social_context is available, aggregate periods belonging to
        # the same education stage and verify the total duration roughly
        # matches the typical_duration_years from social_context.
        # This catches cases where the LLM splits an education stage into
        # sub-periods but the total duration is wrong (e.g. 5-year primary
        # school instead of 6-year).
        social_ctx = plan_dict.get("global_summary", {}).get("social_context", {})
        edu_stages = social_ctx.get("education_system", [])
        if edu_stages and periods:
            last_period_id = periods[-1].get("period_id") if periods else None
            for stage_info in edu_stages:
                stage_name = stage_info.get("stage_name", "")
                typical_dur = stage_info.get("typical_duration_years")
                typical_entry_age = stage_info.get("typical_entry_age")
                if not stage_name or not typical_dur or typical_dur <= 0:
                    continue
                # Find all periods whose stage_label contains this stage_name
                # AND whose age range is consistent with the expected entry age.
                # This avoids false matches (e.g. "early_primary_school_adjustment"
                # matching "primary_school" when it's actually kindergarten).
                matching_periods = []
                for p in periods:
                    label = p.get("stage_label", "")
                    if stage_name not in label:
                        continue
                    dr = p.get("period_date_range", {})
                    s = self._parse_iso_date(dr.get("start_date"))
                    e = self._parse_iso_date(dr.get("end_date"))
                    if not s or not e:
                        continue
                    # Check age at start — if typical_entry_age is known,
                    # exclude periods that start too early (>2 years before
                    # expected entry age), which likely belong to a different
                    # stage despite the label containing the stage_name.
                    if typical_entry_age is not None:
                        age_at_start = (s - birth_date).days / 365.25
                        if age_at_start < typical_entry_age - 2:
                            continue  # too young, likely a different stage
                    matching_periods.append((p.get("period_id"), s, e))
                if not matching_periods:
                    continue
                # Skip if the only matching period is the terminal (last)
                # period — it may be an ongoing stage (e.g. a 10yo child
                # still in primary school) whose duration is naturally
                # shorter than typical.
                if all(pid == last_period_id for pid, _, _ in matching_periods):
                    continue
                # Calculate total duration
                earliest = min(s for _, s, _ in matching_periods)
                latest = max(e for _, _, e in matching_periods)
                total_days = (latest - earliest).days + 1
                total_years = total_days / 365.25
                # Allow ±1 year tolerance for stages ≥3 years, or ±0.5 year
                # for shorter stages. This catches meaningful compression
                # (e.g. 5-year primary school when typical is 6) while
                # allowing minor boundary alignment differences.
                if typical_dur >= 3:
                    tolerance = 1.0
                else:
                    tolerance = 0.5
                if total_years < typical_dur - tolerance:
                    pids = [pid for pid, _, _ in matching_periods]
                    issues.append(
                        f"Education stage '{stage_name}' total duration "
                        f"({total_years:.1f}y across {pids}) is significantly "
                        f"shorter than typical {typical_dur}y from social context"
                    )

        # Check if _ensure_seamless_date_chain removed any content-bearing
        # periods.  This catches cases where a bad patch or LLM error caused
        # a degenerate period that could not be rescued.
        removed_content = plan_dict.get("global_summary", {}).get(
            "_removed_content_periods", []
        )
        if removed_content:
            issues.append(
                f"Content-bearing period(s) were removed as degenerate and "
                f"could not be rescued: {removed_content}"
            )

        plan_dict.setdefault("global_summary", {})["invariant_validation"] = {
            "is_valid": not issues,
            "issues": issues,
        }
        return issues

    async def _validate_plan_durations_with_llm(
        self,
        plan_dict: Dict[str, Any],
        social_context_text: str = "",
    ) -> List[str]:
        """
        LLM-based duration reasonableness validation.

        Replaces the previous hardcoded threshold checks (IQR outlier,
        adjacent ratio 5:1, education duration 70% tolerance) with a single
        LLM call that can make context-aware judgments about whether each
        period's duration is reasonable given the persona's social context,
        education system, and life trajectory.

        This method is only called during P1a life plan initialization and
        the QA loop — never during simulation — so the extra LLM call is
        acceptable for better flexibility.

        Returns a list of issue strings (empty if no issues found).
        """
        periods = plan_dict.get("life_periods", [])
        if not periods:
            return []

        # Build a concise period summary table for the LLM
        period_lines = []
        for p in periods:
            dr = p.get("period_date_range", {})
            start = dr.get("start_date", "?")
            end = dr.get("end_date", "?")
            pid = p.get("period_id", "?")
            label = p.get("stage_label", "")
            title = p.get("title", "")
            dev_tasks = p.get("developmental_tasks", [])
            stage_goals = p.get("stage_goals", [])
            # Calculate duration in days and years
            s = self._parse_iso_date(start)
            e = self._parse_iso_date(end)
            dur_str = ""
            if s and e:
                dur_days = (e - s).days + 1
                dur_years = round(dur_days / 365, 1)
                dur_str = f", duration={dur_days}d (~{dur_years}y)"
            tasks_str = ""
            if dev_tasks:
                tasks_str = f"\n    developmental_tasks: {dev_tasks}"
            goals_str = ""
            if stage_goals:
                goals_str = f"\n    stage_goals: {stage_goals}"
            period_lines.append(
                f"  {pid}: stage_label={label}, title=\"{title}\""
                f", dates={start}~{end}{dur_str}"
                f"{tasks_str}{goals_str}"
            )
        period_table = "\n".join(period_lines)

        social_ctx_section = ""
        if social_context_text:
            social_ctx_section = f"""
## Social Context
{social_context_text}
"""

        system_prompt = (
            "You are a life-plan duration reasonableness validator.\n\n"
            "You will receive a list of life periods with their stage labels, "
            "titles, date ranges, durations, developmental tasks, and stage "
            "goals, along with the persona's social context (education system, "
            "career norms, social institutions, etc.).\n\n"
            "Your task is to check whether each period's duration is reasonable "
            "given the context. Specifically, look for:\n\n"
            "1. **Education/training stage compression**: Is any education or "
            "training-related period significantly shorter than the typical "
            "duration for that stage **as specified in the social context**? "
            "The social context provides the authoritative reference for "
            "typical durations in the persona's country and era — always use "
            "it as your primary calibration source rather than any default "
            "assumptions about a particular country's system. "
            "Only flag cases where the actual duration is clearly too short "
            "to complete the described developmental tasks.\n"
            "   **IMPORTANT**: An education stage may be split into multiple "
            "sub-periods (e.g. 'primary_school_early' + 'primary_school_late',"
            " or 'junior_secondary_entry' + 'junior_secondary_consolidation')."
            " When checking duration, you MUST aggregate all sub-periods that "
            "belong to the same education stage and compare the TOTAL duration "
            "against the social context's typical_duration_years. For example, "
            "if social context says primary_school is 6 years, and the plan "
            "has two periods covering primary school totaling only 5 years, "
            "that is a problem even though each individual period looks fine.\n\n"
            "2. **Extreme adjacent imbalance**: Are there adjacent non-gap "
            "periods where one is dramatically shorter than the other, and "
            "the short one describes non-trivial developmental tasks that "
            "cannot realistically be completed in that time? Note: some "
            "imbalance is natural — a short transition or adaptation period "
            "next to a long education or career period is expected and fine.\n\n"
            "3. **Content-duration mismatch**: Is any period's duration "
            "clearly insufficient for the complexity of its developmental "
            "tasks and stage goals? Conversely, do NOT flag periods that are "
            "short but have appropriately simple or transitional content.\n\n"
            "4. **Education stage over-extension with entry age mismatch**: "
            "Is any education stage's duration significantly LONGER than the "
            "social context's typical_duration_years for that stage, AND does "
            "the period start at an age well before the typical_entry_age? "
            "This combination suggests the period may have incorrectly merged "
            "a pre-education phase (e.g., family-reared infancy/toddlerhood "
            "before formal schooling, pre-apprenticeship period before formal "
            "training) with an education stage. Flag this as a medium-to-high "
            "severity issue.\n\n"
            "Important guidelines:\n"
            "- **Culture-agnostic**: Do NOT assume any specific country's "
            "education system or career norms as default. Always rely on the "
            "provided social context. Education systems vary widely: some "
            "countries have 6-year primary school, others have 5 or 4; some "
            "have 2-year or 3-year secondary stages; undergraduate programs "
            "range from 3 to 5 years; vocational tracks, gap years, military "
            "service, and apprenticeships are common in some cultures but not "
            "others. Judge each period against the social context provided, "
            "not against any single country's norms.\n"
            "- The LAST period (terminal period) is externally constrained "
            "(e.g. by job tenure or current enrollment) — do NOT flag it.\n"
            "- Periods with stage_label='gap_transition' are expected to be "
            "short — do NOT flag them.\n"
            "- Be flexible: minor deviations (e.g. a few months shorter than "
            "typical) are acceptable. Only flag clear problems where the "
            "duration is fundamentally incompatible with the described content.\n"
            "- Consider the persona's specific pathway — if the social context "
            "mentions accelerated programs, grade skipping, direct PhD "
            "admission, or other non-standard paths, shorter durations for "
            "affected stages are justified.\n"
            "- For non-education stages (career, family, retirement, etc.), "
            "there are no universal 'correct' durations — only flag if the "
            "duration is clearly absurd relative to the described content.\n\n"
            "Return is_valid=true if all durations are reasonable, or "
            "is_valid=false with specific issues if problems are found."
        )

        user_prompt = f"""## Period Duration Table
{period_table}
{social_ctx_section}
Please validate whether each period's duration is reasonable given the social context above.
"""

        try:
            report: PlanDurationValidationReport = await self.llm.generate_structured(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_model=PlanDurationValidationReport,
                task_type="p1a_duration_validation",
                temperature=0.0,
            )
        except Exception as e:
            logger.warning(
                f"LLM duration validation failed ({e}); "
                f"falling back to pass (no issues reported)"
            )
            return []

        if report.is_valid:
            logger.info(f"LLM duration validation passed: {report.summary}")
            return []

        # Convert LLM issues to string format compatible with the QA loop
        issue_strings = []
        for issue in report.issues:
            issue_strings.append(
                f"[LLM-duration-check] {issue.period_id} ({issue.issue_type}, "
                f"severity={issue.severity}): {issue.description}"
                + (f" Suggestion: {issue.suggestion}" if issue.suggestion else "")
            )
        logger.warning(
            f"LLM duration validation found {len(issue_strings)} issue(s): "
            f"{report.summary}"
        )
        return issue_strings

    async def _regenerate_plan_with_review_feedback(
        self,
        enriched_constraints: Dict[str, Any],
        transition_hints: TransitionDateHints,
        inferred_constraints: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        review_report: PlanReviewReport,
        invariant_issues: Optional[List[str]] = None,
        previous_plan_dict: Optional[Dict[str, Any]] = None,
        social_context_text: str = "",
    ) -> Dict[str, Any]:
        feedback_parts = []

        # 1. Review summary
        if review_report.review_summary:
            feedback_parts.append(f"Review summary: {review_report.review_summary}")

        # 2. Reviewer issues
        if review_report.issues:
            feedback_parts.append("Issues found in review:")
            for issue in review_report.issues:
                feedback_parts.append(
                    f"  - [{issue.severity}] {issue.issue_type}: {issue.reason}"
                )

        # 3. Invariant validation issues
        if invariant_issues:
            feedback_parts.append("Structural validation failures:")
            for inv_issue in invariant_issues:
                feedback_parts.append(f"  - {inv_issue}")

        # 4. Previous plan structure summary (for targeted correction)
        if previous_plan_dict:
            feedback_parts.append(
                "\nPreviously generated stage structure (for reference, please correct the issues):"
            )
            for lp in previous_plan_dict.get("life_periods", []):
                dr = lp.get("period_date_range", {})
                feedback_parts.append(
                    f"  {lp.get('period_id')}: {lp.get('stage_label')} "
                    f"({dr.get('start_date')} ~ {dr.get('end_date')})"
                )

        review_feedback = "\n".join(feedback_parts) or "Need to regenerate the life stage plan to fix structural semantic issues."

        return await self._generate_plan_via_llm(
            enriched_constraints=enriched_constraints,
            transition_hints=transition_hints,
            inferred_constraints=inferred_constraints,
            birth_date=birth_date,
            reference_date=reference_date,
            resolved_target_age=enriched_constraints.get("resolved_target_age_exact", self._compute_exact_age(birth_date, reference_date)),
            review_feedback=review_feedback,
            social_context_text=social_context_text,
        )


    def _build_age_calendar_hint(self, reference_date: date, birth_date: date, target_age_exact: int) -> str:
        """
        Build an "age -> date closed interval" mapping text for clarifying time boundaries in the prompt.
        """
        lines = [
            f"reference_date={reference_date.isoformat()}",
            f"derived_birth_date={birth_date.isoformat()}",
            "Age to date mapping (closed interval, inclusive):"
        ]

        for age in range(0, target_age_exact + 1):
            age_start = self._add_years(birth_date, age)
            age_end = self._add_years(birth_date, age + 1) - timedelta(days=1)
            if age == target_age_exact:
                age_end = min(age_end, reference_date)
            lines.append(f"- Age {age}: {age_start.isoformat()} ~ {age_end.isoformat()}")

        return "\n".join(lines)

    def _compile_memory_segments_from_plan(
        self,
        canonical_plan: Dict[str, Any],
        birth_date: date,
        reference_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Build a derived memory plan from canonical life periods.

        Generalized resolution-boundary splitting: when a period spans
        multiple temporal resolutions (none/year/season/month), it is
        automatically split into sub-segments at each resolution boundary.
        Each sub-segment has a uniform resolution, so compute_period_density()
        can use the finest resolution (which equals the only resolution).

        This replaces the old age-4-only splitting logic with a fully
        generalized approach that handles all resolution transitions.
        """
        periods = canonical_plan.get("life_periods", [])
        if not isinstance(periods, list) or not periods:
            return {
                "policy_version": "memory_segmentation_v3",
                "segments": [],
                "metadata": {
                    "source": "canonical_life_plan",
                    "notes": ["no canonical life periods available"],
                },
            }

        segments: List[Dict[str, Any]] = []

        # Import AM Model once outside the loop
        from lifelong_synth.configs.temporal_context import (
            AutobiographicalMemoryModel,
            compute_period_density,
        )
        ref_date_str = (reference_date or birth_date).isoformat()
        am_model = AutobiographicalMemoryModel(
            birth_date=birth_date.isoformat(),
            reference_date=ref_date_str,
            country="",  # Will use default preset
        )

        for period in periods:
            period_id = str(period.get("period_id", ""))
            dr = period.get("period_date_range", {})
            if not isinstance(dr, dict):
                continue

            period_start = self._parse_iso_date(dr.get("start_date"))
            period_end = self._parse_iso_date(dr.get("end_date"))
            if period_start is None or period_end is None or period_start > period_end:
                continue

            stage_label = period.get("stage_label", "")
            title = period.get("title", "")

            # ── Compute per-age resolutions ──
            start_age = max(0, int((period_start - birth_date).days / 365.25))
            end_age = int((period_end - birth_date).days / 365.25)

            age_resolutions = []
            for age in range(start_age, end_age + 1):
                profile = am_model.compute_age_profile(age)
                age_resolutions.append((age, profile.temporal_resolution))

            if not age_resolutions:
                # Fallback: single segment with computed density
                density_dict = compute_period_density(
                    period, am_model, birth_date.isoformat()
                )
                segments.append({
                    "segment_id": f"{period_id}_M1",
                    "parent_period_id": period_id,
                    "stage_label": stage_label,
                    "title": title,
                    "start_date": period_start.isoformat(),
                    "end_date": period_end.isoformat(),
                    "time_unit": "module",            # always module (v3 speed optimisation)
                    "max_detail_events": 0,           # placeholder; overwritten by AM-weight budget below
                    "max_outline_events": 0,          # placeholder; overwritten by AM-weight budget below
                    "max_medium_events_per_unit": 2,  # General-Event seeds per time unit
                    "segmentation_reason": "canonical_period_passthrough",
                    "am_weight": 0.0,                 # placeholder; filled in AM-weight budget step
                })
                continue

            # ── Find resolution boundaries and split ──
            sub_ranges: List[tuple] = []
            current_res = age_resolutions[0][1]
            seg_start = period_start

            for age, res in age_resolutions:
                if res != current_res:
                    # Split at the age boundary
                    boundary = self._add_years(birth_date, age)
                    # Clamp boundary to period range
                    if boundary < period_start:
                        boundary = period_start
                    if boundary > period_end:
                        boundary = period_end
                    seg_end = boundary - timedelta(days=1)
                    if seg_end >= seg_start:
                        sub_ranges.append((seg_start, seg_end, current_res))
                    seg_start = boundary
                    current_res = res

            # Last sub-range
            sub_ranges.append((seg_start, period_end, current_res))

            # ── v2 optimisation: split sub-ranges at recency boundary ──
            # Instead of using AM weight threshold (0.25), we use a simple
            # recency rule: years_ago > RECENCY_YEAR_THRESHOLD → module.
            # This ensures all distant periods (>5 years from target age)
            # are treated as module density, significantly reducing events.
            RECENCY_YEAR_THRESHOLD = 5
            current_age = (reference_date - birth_date).days / 365.25 if reference_date else 0

            am_split_ranges: List[tuple] = []
            for sub_start_sr, sub_end_sr, res_sr in sub_ranges:
                # Only split sub-ranges with resolution "none" or "year"
                # (higher resolutions like "season" are recent and won't
                # be merged into modules)
                if res_sr not in ("none", "year"):
                    am_split_ranges.append((sub_start_sr, sub_end_sr, res_sr))
                    continue

                sr_start_age = max(0, int((sub_start_sr - birth_date).days / 365.25))
                sr_end_age = int((sub_end_sr - birth_date).days / 365.25)

                if sr_start_age >= sr_end_age:
                    # Single-year sub-range, no need to split
                    am_split_ranges.append((sub_start_sr, sub_end_sr, res_sr))
                    continue

                # Walk through ages and split at recency boundary crossings
                chunk_start = sub_start_sr
                chunk_is_distant = (current_age - sr_start_age) > RECENCY_YEAR_THRESHOLD

                for check_age in range(sr_start_age + 1, sr_end_age + 1):
                    age_is_distant = (current_age - check_age) > RECENCY_YEAR_THRESHOLD

                    if age_is_distant != chunk_is_distant:
                        # Recency boundary crossed — split here
                        boundary = self._add_years(birth_date, check_age)
                        boundary = max(boundary, sub_start_sr)
                        boundary = min(boundary, sub_end_sr)
                        chunk_end = boundary - timedelta(days=1)
                        if chunk_end >= chunk_start:
                            am_split_ranges.append((chunk_start, chunk_end, res_sr))
                        chunk_start = boundary
                        chunk_is_distant = age_is_distant

                # Last chunk
                if chunk_start <= sub_end_sr:
                    am_split_ranges.append((chunk_start, sub_end_sr, res_sr))

            sub_ranges = am_split_ranges

            # ── Merge consecutive distant sub-ranges into modules ──
            merged_ranges: List[tuple] = []
            i_sr = 0
            while i_sr < len(sub_ranges):
                sub_start_sr, sub_end_sr, res_sr = sub_ranges[i_sr]

                # Check if this sub-range is distant (years_ago > threshold)
                sr_start_age = max(0, int((sub_start_sr - birth_date).days / 365.25))
                sr_end_age = int((sub_end_sr - birth_date).days / 365.25)

                all_distant = True
                for check_age in range(sr_start_age, sr_end_age + 1):
                    if (current_age - check_age) <= RECENCY_YEAR_THRESHOLD:
                        all_distant = False
                        break

                if all_distant and res_sr in ("none", "year"):
                    # Try to merge with subsequent distant sub-ranges
                    module_start = sub_start_sr
                    module_end = sub_end_sr
                    j_sr = i_sr + 1
                    while j_sr < len(sub_ranges):
                        next_start, next_end, next_res = sub_ranges[j_sr]
                        next_start_age = max(0, int((next_start - birth_date).days / 365.25))
                        next_end_age = int((next_end - birth_date).days / 365.25)

                        next_all_distant = True
                        for check_age in range(next_start_age, next_end_age + 1):
                            if (current_age - check_age) <= RECENCY_YEAR_THRESHOLD:
                                next_all_distant = False
                                break

                        if next_all_distant and next_res in ("none", "year"):
                            module_end = next_end
                            j_sr += 1
                        else:
                            break

                    merged_ranges.append((module_start, module_end, "module"))
                    i_sr = j_sr
                else:
                    merged_ranges.append((sub_start_sr, sub_end_sr, res_sr))
                    i_sr += 1

            sub_ranges = merged_ranges

            # ── Create segments for each sub-range ──
            for idx, (sub_start, sub_end, res) in enumerate(sub_ranges):
                if res == "module":
                    # Module segment: merged low-AM years, minimal events
                    module_years = max(0.25, (sub_end - sub_start).days / 365.25)
                    # v7: Proportional detail_cap — consistent with compute_period_density
                    ref = reference_date or date.today()
                    boundary_5yr = ref - timedelta(days=5 * 365)
                    boundary_1yr = ref - timedelta(days=365)
                    boundary_3yr = ref - timedelta(days=3 * 365)
                    module_days_f = max(1.0, float((sub_end - sub_start).days))

                    # High-res gate: no high-res for segments that ended > 5 years ago
                    if sub_end <= boundary_5yr:
                        detail_cap = 0
                        outline_cap = 0
                    else:
                        def _overlap(s: date, e: date, ws: date, we: date) -> float:
                            return max(0.0, (min(e, we) - max(s, ws)).days)

                        ov_w1 = _overlap(sub_start, sub_end, boundary_1yr, ref)
                        ov_w2 = _overlap(sub_start, sub_end, boundary_3yr, boundary_1yr)
                        detail_cap = int(round((ov_w1 * 3.0 + ov_w2 * 2.0) / module_days_f))
                        # outline_cap: scale by fraction within 5-year window
                        ov_5yr = _overlap(sub_start, sub_end, boundary_5yr, ref)
                        fraction_in_window = ov_5yr / module_days_f
                        outline_cap_by_length = max(0, min(round(module_years * 0.3), 2))
                        outline_cap_by_length = max(0, round(outline_cap_by_length * fraction_in_window))
                        # outline_cap must be >= detail_cap (Conway & Pleydell-Pearce, 2000:
                        # general events are prerequisites for episodic memories)
                        outline_cap = max(outline_cap_by_length, detail_cap)

                    segments.append({
                        "segment_id": f"{period_id}_M{idx + 1}",
                        "parent_period_id": period_id,
                        "stage_label": stage_label,
                        "title": title,
                        "start_date": sub_start.isoformat(),
                        "end_date": sub_end.isoformat(),
                        "time_unit": "module",
                        "max_detail_events": 0,          # placeholder; overwritten by AM-weight budget below
                        "max_outline_events": 0,          # placeholder; overwritten by AM-weight budget below
                        "max_medium_events_per_unit": 2,  # General-Event seeds per time unit
                        "segmentation_reason": "low_am_module_merge",
                        "am_weight": 0.0,                 # placeholder; filled in AM-weight budget step
                    })
                    continue

                pseudo_period = {
                    "period_date_range": {
                        "start_date": sub_start.isoformat(),
                        "end_date": sub_end.isoformat(),
                    }
                }
                density_dict = compute_period_density(
                    pseudo_period, am_model, birth_date.isoformat()
                )

                # Determine segmentation reason
                if res == "none":
                    reason = "childhood_amnesia_before_encoding"
                elif len(sub_ranges) > 1:
                    reason = "resolution_boundary_split"
                else:
                    reason = "canonical_period_passthrough"

                # childhood_amnesia segments get 0 medium events (no habitual memories for infants)
                medium_count = 0 if reason == "childhood_amnesia_before_encoding" else 2
                segments.append({
                    "segment_id": f"{period_id}_M{idx + 1}",
                    "parent_period_id": period_id,
                    "stage_label": stage_label,
                    "title": title,
                    "start_date": sub_start.isoformat(),
                    "end_date": sub_end.isoformat(),
                    "time_unit": "module",            # always module (v3 speed optimisation)
                    "max_detail_events": 0,           # placeholder; overwritten by AM-weight budget below
                    "max_outline_events": 0,          # placeholder; overwritten by AM-weight budget below
                    "max_medium_events_per_unit": medium_count,  # General-Event seeds per time unit
                    "segmentation_reason": reason,
                    "am_weight": 0.0,                 # placeholder; filled in AM-weight budget step
                })

        # ── v3 (speed): AM-weight-based budget assignment ──
        # Rule:
        #   1. Compute am_weight for each segment at its midpoint.
        #   2. Last segment (chronologically latest) → detail=2, outline=2.
        #   3. Among all OTHER segments (all except the last):
        #      - Segments whose end_date falls within the 20yr window
        #        (i.e. seg_end > boundary_20yr) → outline=1.
        #        If more than MAX_OUTLINE_SEGMENTS qualify, keep only the
        #        top-MAX_OUTLINE_SEGMENTS by am_weight (prevents budget explosion
        #        for personas with many recent periods).
        #      - Segments outside the 20yr window → detail=0, outline=0.
        #   4. All remaining non-last segments → detail=0, outline=0.
        #
        # Rationale (psychology):
        #   - Last segment: Conway & Pleydell-Pearce (2000) — most recent chapter
        #     is most behaviourally relevant, always gets full episodic detail.
        #   - Recency window (default 5yr): high-res events concentrated here.
        #   - 20yr window: Singer & Salovey (1993) self-defining memories span
        #     the last ~15-20 years; career-formative events in this window are
        #     behaviourally relevant and should appear as general-event memories.
        #   - Beyond 20yr but not childhood amnesia: still gets 1 general event.
        #   - Childhood amnesia: no events at all.
        MAX_OUTLINE_SEGMENTS = 8  # raised cap to allow more general events
        if segments:
            # Step 1: Compute am_weight for each segment at its midpoint
            ref_d = reference_date or date.today()
            boundary_recency_d = ref_d - timedelta(days=self._recency_window_years * 365)
            boundary_20yr_d = ref_d - timedelta(days=20 * 365)

            for seg in segments:
                seg_start_d = date.fromisoformat(seg["start_date"])
                seg_end_d = date.fromisoformat(seg["end_date"])
                mid_d = seg_start_d + (seg_end_d - seg_start_d) / 2
                mid_age = (mid_d - birth_date).days / 365.25
                profile = am_model.compute_age_profile(int(mid_age))
                seg["am_weight"] = round(profile.am_weight, 4)
                # Reset all budgets to 0 (will be set below)
                seg["max_detail_events"] = 0
                seg["max_outline_events"] = 0

            # Step 2: Last segment → detail=2, outline=2
            last_seg = segments[-1]
            last_seg["max_detail_events"] = 2
            last_seg["max_outline_events"] = 2

            # Step 3: Non-last segments — budget based on recency and amnesia
            non_last = segments[:-1]
            for i, seg in enumerate(non_last):
                seg_end_d = date.fromisoformat(seg["end_date"])
                reason = seg.get("segmentation_reason", "")

                if reason == "childhood_amnesia_before_encoding":
                    # Childhood amnesia: no events at all
                    seg["max_detail_events"] = 0
                    seg["max_outline_events"] = 0
                    seg["max_medium_events_per_unit"] = 0  # No habitual memories for infants
                elif seg_end_d > boundary_recency_d:
                    # Within recency window: gets both detail and general-event
                    seg["max_detail_events"] = 1
                    seg["max_outline_events"] = 2
                elif seg_end_d > boundary_20yr_d:
                    # Within 20yr but outside recency: general-event only
                    seg["max_detail_events"] = 0
                    seg["max_outline_events"] = 1
                else:
                    # Beyond 20yr but not amnesia: still gets 1 general event
                    seg["max_detail_events"] = 0
                    seg["max_outline_events"] = 1

        return {
            "policy_version": "memory_segmentation_v4_memoryforge",
            "segments": segments,
            "metadata": {
                "source": "canonical_life_plan",
                "notes": [
                    "all segments use time_unit=module (speed optimisation v3)",
                    "last segment: detail=2, outline=2",
                    f"non-last within recency window ({self._recency_window_years}yr): detail=1, outline=2",
                    "non-last within 20yr window: detail=0, outline=1",
                    "non-last beyond 20yr (non-amnesia): detail=0, outline=1",
                    "childhood amnesia segments: detail=0, outline=0",
                ],
            },
        }

    def _enrich_plan_with_date_ranges(
        self,
        plan_dict: Dict[str, Any],
        target_age_exact: int,
        reference_date: date,
        birth_date: date,
        input_birth_date: Any = None,
        input_simulation_end_date: Any = None
    ) -> Dict[str, Any]:
        """
        Convert each age_range to precise dates based on reference_date + birth_date + target_age_exact.
        For [start_age, end_age]:
          period_start_date = birth_date + start_age years
          period_end_date = birth_date + (end_age + 1) years - 1 day
        Truncate the last stage to reference_date.
        """
        global_summary = plan_dict.get("global_summary", {})
        global_summary["target_age_exact"] = target_age_exact
        global_summary["timeline_anchor"] = {
            "reference_date": reference_date.isoformat(),
            "target_age_exact": target_age_exact,
            "derived_birth_date": birth_date.isoformat(),
            "input_birth_date": input_birth_date,
            "input_simulation_end_date": input_simulation_end_date,
            "simulation_start_date": birth_date.isoformat(),
            "simulation_end_date": reference_date.isoformat(),
        }
        plan_dict["global_summary"] = global_summary

        for lp in plan_dict.get("life_periods", []):
            age_range = lp.get("age_range")
            if not isinstance(age_range, (list, tuple)) or len(age_range) != 2:
                continue

            start_age, end_age = age_range
            if not isinstance(start_age, int) or not isinstance(end_age, int):
                continue

            period_start = self._add_years(birth_date, start_age)
            nominal_period_end = self._add_years(birth_date, end_age + 1) - timedelta(days=1)

            period_end = nominal_period_end
            if end_age >= target_age_exact:
                period_end = min(period_end, reference_date)

            if period_end < period_start:
                period_end = period_start

            lp["date_range"] = {
                "start_date": period_start.isoformat(),
                "end_date": period_end.isoformat()
            }

        return plan_dict

    # ==========================================
    # CBQA (Causal Backpropagation QA) Methods
    # ==========================================

    def _validate_transition_hints_arithmetic(
        self,
        hints: TransitionDateHints,
        birth_date: date,
    ) -> List[str]:
        """
        Deterministic arithmetic checks on transition hints.
        Catches LLM arithmetic errors before they propagate to Plan Generation.
        Returns a list of correction signal strings (empty if no issues).
        """
        issues = []
        birth_year = birth_date.year

        for td in hints.transition_dates:
            anchor = self._parse_iso_date(td.anchor_date)
            if anchor is None:
                continue

            age_at_anchor = (anchor - birth_date).days / 365.25

            # Check education milestones against typical age ranges
            key_lower = td.transition_key.lower()
            if "undergrad" in key_lower and "graduation" in key_lower:
                expected_min, expected_max = 20.5, 24.5
                if not (expected_min <= age_at_anchor <= expected_max):
                    # Compute the most likely correct date
                    correct_year = birth_year + 22
                    correct_date = date(correct_year, 6, 30)
                    issues.append(
                        f"[arithmetic-check] {td.transition_key}: age at anchor "
                        f"{td.anchor_date} = {age_at_anchor:.1f}y, expected "
                        f"{expected_min}-{expected_max}y. Likely arithmetic error "
                        f"({birth_year}+22={birth_year+22}, not {anchor.year}). "
                        f"Suggested correction: {correct_date.isoformat()}"
                    )

            if "phd" in key_lower and "graduation" in key_lower:
                expected_min, expected_max = 25.0, 33.0
                if not (expected_min <= age_at_anchor <= expected_max):
                    issues.append(
                        f"[arithmetic-check] {td.transition_key}: age at anchor "
                        f"{td.anchor_date} = {age_at_anchor:.1f}y, expected "
                        f"{expected_min}-{expected_max}y."
                    )

        # Cross-check: undergrad_end < phd_start
        undergrad_end = None
        phd_start = None
        for td in hints.transition_dates:
            key_lower = td.transition_key.lower()
            if "undergrad" in key_lower and "graduation" in key_lower:
                undergrad_end = self._parse_iso_date(td.anchor_date)
            if "phd" in key_lower and ("entry" in key_lower or "start" in key_lower):
                phd_start = self._parse_iso_date(td.anchor_date)

        if undergrad_end and phd_start and phd_start < undergrad_end:
            issues.append(
                f"[cross-check] phd_start ({phd_start.isoformat()}) < "
                f"undergrad_end ({undergrad_end.isoformat()}): "
                f"PhD cannot start before undergraduate graduation."
            )

        return issues

    def _collect_pipeline_stage_summaries(
        self,
        social_context: SocialContextProfile,
        persona_pathway: InferredPersonaPathway,
        transition_hints: TransitionDateHints,
        inferred_constraints: Dict[str, Any],
        plan_dict: Dict[str, Any],
        birth_date: date,
    ) -> str:
        """
        Collect concise summaries of each pipeline stage's output
        for the Error Attribution Agent.
        """
        # Step 0a: Social Context
        edu_stages = ", ".join(
            f"{s.stage_name}({s.typical_duration_years}y, entry_age={s.typical_entry_age})"
            for s in social_context.education_system
        )
        step_0a_summary = f"Country={social_context.country}, education_system=[{edu_stages}]"

        # Step 0b: Persona Pathway
        pathway_parts = []
        for dim in persona_pathway.dimensions:
            stages_str = " → ".join(dim.key_stages)
            devs = "; ".join(dim.deviations_from_typical) if dim.deviations_from_typical else "none"
            pathway_parts.append(f"{dim.dimension}: {stages_str} | deviations: {devs}")
        step_0b_summary = "\n    ".join(pathway_parts)

        # Step 4: Transition Hints
        hints_parts = []
        for td in transition_hints.transition_dates:
            age_at = ""
            anchor = self._parse_iso_date(td.anchor_date)
            if anchor and birth_date:
                age_at = f", age={self._compute_age_display(birth_date, anchor):.1f}"
            hints_parts.append(
                f"{td.transition_key}={td.anchor_date} "
                f"(conf={td.confidence}, hard={td.is_hard_constraint}{age_at})"
            )
        step_4_summary = "; ".join(hints_parts)

        # Step 5: Inferred Constraints
        step_5_summary = (
            f"current_role_start_date={inferred_constraints.get('current_role_start_date')}, "
            f"pre_current_transition_anchor_date={inferred_constraints.get('pre_current_transition_anchor_date')}"
        )

        # Step 7+8: Plan (post-processed)
        plan_parts = []
        for lp in plan_dict.get("life_periods", []):
            dr = lp.get("period_date_range", {})
            plan_parts.append(
                f"{lp.get('period_id')}: {lp.get('stage_label')} "
                f"({dr.get('start_date')} ~ {dr.get('end_date')})"
            )
        step_7_summary = "\n    ".join(plan_parts)

        return f"""## Pipeline Stage Outputs

### Step 0a — Social Context Inference (LLM, mutable)
{step_0a_summary}

### Step 0b — Persona Pathway Inference (LLM, mutable)
{step_0b_summary}

### Step 4 — Transition Hints Inference (LLM, mutable)
{step_4_summary}

### Step 5 — Temporal Constraints (deterministic, auto-cascade)
{step_5_summary}

### Step 7+8 — Plan Generation + Postprocess (LLM + deterministic)
{step_7_summary}
"""

    async def _run_error_attribution_agent(
        self,
        validation_failures: List[str],
        pipeline_stages_text: str,
        social_context_text: str,
        birth_date: date,
    ) -> ErrorAttributionReport:
        """
        CBQA Error Attribution Agent: analyse validation failures and
        attribute each to the most upstream causal pipeline stage.
        """
        system_prompt = """You are a Pipeline Error Attribution Expert.

Your task is to analyse validation failures from a multi-step LLM pipeline
and attribute each failure to the **most upstream causal root stage**.

[Pipeline Dependency Structure]
Step 0a (Social Context, LLM) → Step 0b (Persona Pathway, LLM)
→ Step 4 (Transition Hints, LLM) → Step 5 (Temporal Constraints, deterministic)
→ Step 7 (Plan Generation, LLM) → Step 8 (Postprocess, deterministic)
→ Step 9 (Validation)

[Attribution Principles]
1. **Trace the causal chain**: If a Plan (Step 7) date error faithfully followed
   an incorrect Transition Hint (Step 4), the root cause is Step 4, not Step 7.
2. **Most-upstream rule**: If Step 4's error originated from Step 0b's incorrect
   inference, attribute to Step 0b.
3. **Deterministic steps are transparent**: Steps 5 and 8 do not introduce new
   errors; they only propagate upstream errors. If Step 8's date alignment caused
   a problem, check whether Step 7's output already had overlapping dates.
4. **Arithmetic verification**: If an LLM step's output contains verifiable
   arithmetic (e.g., birth_year + age = graduation_year), check correctness.
5. **correction_signal must be specific**: Do NOT say "please fix the date".
   Instead say "undergrad_graduation should be 2020-06-30 (1998+22=2020),
   not the current 2019-06-30".
6. **attributed_stage_id** must be one of: step_0a, step_0b, step_4, step_7.
"""

        user_prompt = f"""[Pipeline Stage Outputs]
{pipeline_stages_text}

[Birth Date]
{birth_date.isoformat()}

[Validation Failures]
{chr(10).join(f'- {f}' for f in validation_failures)}

Please output an ErrorAttributionReport with:
1. One CausalAttribution per validation failure
2. A summary of the overall attribution analysis
3. recommended_re_execution_stages: ordered list of stage IDs to re-run
"""

        try:
            report = await self.llm.generate_structured(
                prompt=user_prompt,
                system_prompt=system_prompt,
                response_model=ErrorAttributionReport,
                task_type="p1a_error_attribution",
                temperature=0.0,
            )
            logger.info(
                f"CBQA Error Attribution: {len(report.attributions)} attribution(s), "
                f"re-execute: {report.recommended_re_execution_stages}, "
                f"summary: {report.summary}"
            )
            return report
        except Exception as e:
            logger.warning(f"CBQA Error Attribution Agent failed: {e}; returning empty report")
            return ErrorAttributionReport(
                attributions=[],
                summary=f"Attribution failed: {e}",
                recommended_re_execution_stages=[],
            )

    async def _cbqa_re_execute_upstream(
        self,
        attribution_report: ErrorAttributionReport,
        constraint_sheet: Dict[str, Any],
        enriched_constraints: Dict[str, Any],
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
        social_context: SocialContextProfile,
        persona_pathway: InferredPersonaPathway,
        social_context_text: str,
        transition_hints: TransitionDateHints,
        inferred_constraints: Dict[str, Any],
    ) -> Tuple[
        SocialContextProfile,
        InferredPersonaPathway,
        str,
        TransitionDateHints,
        Dict[str, Any],
        Dict[str, Any],
    ]:
        """
        CBQA: Selectively re-execute upstream stages based on attribution report.
        Returns updated (social_context, persona_pathway, social_context_text,
                         transition_hints, inferred_constraints, raw_plan).
        """
        stages_to_rerun = set(attribution_report.recommended_re_execution_stages)

        # Build correction context for each stage
        correction_context: Dict[str, List[str]] = {}
        for attr in attribution_report.attributions:
            stage_id = attr.attributed_stage_id
            correction_context.setdefault(stage_id, []).append(attr.correction_signal)

        # Re-execute in topological order with cascade
        if "step_0a" in stages_to_rerun:
            logger.info("CBQA: Re-executing Step 0a (Social Context Inference)")
            social_context = await self._infer_social_context(
                constraint_sheet=constraint_sheet,
                enriched_constraints=enriched_constraints,
                birth_date=birth_date,
            )
            stages_to_rerun.update(["step_0b", "step_4", "step_7"])

        if "step_0b" in stages_to_rerun:
            logger.info("CBQA: Re-executing Step 0b (Persona Pathway Inference)")
            persona_pathway = await self._infer_persona_pathway(
                constraint_sheet=constraint_sheet,
                social_context=social_context,
                birth_date=birth_date,
                reference_date=reference_date,
            )
            social_context_text = self._format_social_context_for_prompt(social_context, persona_pathway, birth_date=birth_date)
            stages_to_rerun.update(["step_4", "step_7"])

        if "step_4" in stages_to_rerun:
            logger.info(
                f"CBQA: Re-executing Step 4 (Transition Hints) with "
                f"{len(correction_context.get('step_4', []))} correction signal(s)"
            )
            transition_hints = await self._infer_transition_dates_with_llm(
                constraint_sheet=constraint_sheet,
                enriched_constraints=enriched_constraints,
                reference_date=reference_date,
                birth_date=birth_date,
                resolved_target_age=resolved_target_age,
                social_context_text=social_context_text,
                correction_feedback=correction_context.get("step_4"),
            )
            # Step 5 is deterministic — auto-cascade
            inferred_constraints = self._derive_temporal_constraints_from_hints(
                hints=transition_hints,
                reference_date=reference_date,
            )
            stages_to_rerun.add("step_7")

        raw_plan = None
        if "step_7" in stages_to_rerun:
            # Update enriched_constraints with new inferred values
            cjt = inferred_constraints.get("current_job_tenure_months")
            rpg = inferred_constraints.get("pre_current_role_gap_months")
            tsh = inferred_constraints.get("transition_semantic_hint") or "general"
            if cjt is not None:
                enriched_constraints["current_job_tenure_months"] = cjt
            if rpg is not None:
                enriched_constraints["pre_current_role_gap_months"] = rpg
            enriched_constraints["transition_semantic_hint"] = tsh

            step7_feedback = "\n".join(correction_context.get("step_7", []))
            logger.info("CBQA: Re-executing Step 7 (Plan Generation)")
            raw_plan = await self._generate_plan_via_llm(
                enriched_constraints=enriched_constraints,
                transition_hints=transition_hints,
                inferred_constraints=inferred_constraints,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                social_context_text=social_context_text,
                review_feedback=step7_feedback,
            )

        return (
            social_context,
            persona_pathway,
            social_context_text,
            transition_hints,
            inferred_constraints,
            raw_plan,
        )

    # ==========================================
    # AFBE (Anchor-First Backward Expansion) Methods
    # ==========================================

    # Threshold: education stages with typical_entry_age < this value
    # are treated as "pre-higher-education" (deterministic dates).
    # Stages at or above this age are "higher-education" (backward from graduation).
    _HIGHER_ED_ENTRY_AGE_THRESHOLD = 17

    @staticmethod
    def _classify_education_stages(
        education_system: List,
        threshold: int = 17,
    ) -> tuple:
        """Dynamically classify education stages into pre-higher-ed and higher-ed
        based on typical_entry_age, not hardcoded stage names.

        This ensures the system works for ANY cultural context (US, UK, JP, DE, etc.)
        where LLM-inferred stage_name values may differ.

        Returns (pre_higher_ed_stages, higher_ed_stages) — both lists of EducationStage.
        """
        pre_higher = []
        higher = []
        for s in education_system:
            if s.typical_entry_age < threshold:
                pre_higher.append(s)
            else:
                higher.append(s)
        return pre_higher, higher

    @staticmethod
    def _parse_pathway_stage_key(stage_str: str) -> str:
        """Extract the stage key from a pathway key_stages string.

        e.g. 'primary_school(6y, age6-12, city key primary likely)' -> 'primary_school'
             'doctoral(5y, age22-27, direct_phd...)' -> 'doctoral'
        """
        paren_idx = stage_str.find("(")
        if paren_idx > 0:
            return stage_str[:paren_idx].strip()
        return stage_str.strip()

    def _compute_early_stage_dates(
        self,
        social_context: SocialContextProfile,
        birth_date: date,
        reference_date: date,
        first_higher_ed_start: Optional[date] = None,
        skipped_stages: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Deterministically compute dates for pre-higher-education stages
        based on social_context.education_system.

        Uses dynamic classification (entry_age < threshold) instead of
        hardcoded stage names, so it works for any cultural context.

        Returns a list of dicts ordered by start_date, each with:
          stage_key, start_date, end_date, source, typical_duration_years
        """
        birth_year = birth_date.year
        stages: List[Dict[str, Any]] = []
        skipped = skipped_stages or set()

        pre_higher_ed, _ = self._classify_education_stages(
            social_context.education_system,
            self._HIGHER_ED_ENTRY_AGE_THRESHOLD,
        )

        for edu_stage in pre_higher_ed:
            if edu_stage.stage_name in skipped:
                continue

            entry_date = date(
                birth_year + edu_stage.typical_entry_age,
                edu_stage.entry_month, 1,
            )
            end_date = date(
                birth_year + edu_stage.typical_entry_age + edu_stage.typical_duration_years,
                edu_stage.entry_month, 1,
            ) - timedelta(days=1)

            stages.append({
                "stage_key": edu_stage.stage_name,
                "start_date": entry_date,
                "end_date": end_date,
                "source": "social_context_deterministic",
                "typical_duration_years": edu_stage.typical_duration_years,
            })

        # Sort by start_date
        stages.sort(key=lambda s: s["start_date"])

        # If we know when higher-ed starts, clamp the last pre-higher-ed stage
        if first_higher_ed_start and stages:
            last = stages[-1]
            expected_end = first_higher_ed_start - timedelta(days=1)
            if last["end_date"] > expected_end:
                last["end_date"] = expected_end
            elif last["end_date"] < expected_end:
                # There's a gap — extend the last stage to fill it
                last["end_date"] = expected_end

        # Ensure adjacent stages are seamless
        for i in range(len(stages) - 1):
            expected_end = stages[i + 1]["start_date"] - timedelta(days=1)
            stages[i]["end_date"] = expected_end

        # Prepend early_childhood: birth_date to first education stage
        if stages:
            first_edu_start = stages[0]["start_date"]
            if first_edu_start > birth_date:
                stages.insert(0, {
                    "stage_key": "early_childhood",
                    "start_date": birth_date,
                    "end_date": first_edu_start - timedelta(days=1),
                    "source": "birth_date_to_first_education",
                    "typical_duration_years": None,
                })
        else:
            # No pre-higher-ed stages found; create a single early_childhood
            end = (first_higher_ed_start - timedelta(days=1)) if first_higher_ed_start else reference_date
            stages.append({
                "stage_key": "early_childhood",
                "start_date": birth_date,
                "end_date": end,
                "source": "fallback",
                "typical_duration_years": None,
            })

        return stages

    def _derive_anchor_sequence(
        self,
        social_context: SocialContextProfile,
        persona_pathway: InferredPersonaPathway,
        birth_date: date,
        reference_date: date,
        inferred_constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Derive an ordered anchor sequence from social context, persona pathway,
        and inferred constraints.

        Returns a list of anchor dicts ordered from earliest to latest, each with:
          stage_key, start_date (Optional), end_date (Optional),
          is_hard_constraint, source, needs_llm_content, needs_llm_dates
        """
        birth_year = birth_date.year

        # ── 1. Extract hard constraints ──
        graduation_date_str = inferred_constraints.get("pre_current_transition_anchor_date")
        employment_start_str = inferred_constraints.get("current_role_start_date")
        graduation_date = self._parse_iso_date(graduation_date_str) if graduation_date_str else None
        employment_start = self._parse_iso_date(employment_start_str) if employment_start_str else None

        # ── 2. Extract education pathway stages & detect skipped stages ──
        edu_dim = next(
            (d for d in persona_pathway.dimensions if d.dimension == "education"),
            None,
        )
        # Dynamically classify stages by entry_age threshold
        all_pre_higher, all_higher = self._classify_education_stages(
            social_context.education_system,
            self._HIGHER_ED_ENTRY_AGE_THRESHOLD,
        )
        all_stage_names = {s.stage_name for s in social_context.education_system}

        skipped_stages = set()
        if edu_dim:
            for dev in edu_dim.deviations_from_typical:
                if dev.upper().startswith("SKIP"):
                    # e.g. "SKIP: No independent master stage"
                    dev_lower = dev.lower()
                    for stage_name in all_stage_names:
                        if stage_name.lower() in dev_lower:
                            skipped_stages.add(stage_name)

        # ── 3. Compute higher-ed dates backward from hard constraints ──
        higher_ed_anchors: List[Dict[str, Any]] = []

        # Find higher-ed stages from education_system (dynamic classification)
        higher_ed_stages = [
            s for s in all_higher
            if s.stage_name not in skipped_stages
        ]

        # Sort by entry_age descending (we build backward)
        higher_ed_stages.sort(key=lambda s: s.typical_entry_age, reverse=True)

        # The last higher-ed stage ends at graduation_date
        cursor_end = graduation_date  # e.g. 2025-06-30
        for edu_stage in higher_ed_stages:
            if cursor_end is None:
                # Fallback: compute from birth_year + entry_age + duration
                entry = date(birth_year + edu_stage.typical_entry_age, edu_stage.entry_month, 1)
                end = date(
                    birth_year + edu_stage.typical_entry_age + edu_stage.typical_duration_years,
                    edu_stage.entry_month, 1,
                ) - timedelta(days=1)
                higher_ed_anchors.append({
                    "stage_key": edu_stage.stage_name,
                    "start_date": entry,
                    "end_date": end,
                    "is_hard_constraint": False,
                    "source": "social_context_fallback",
                    "needs_llm_content": True,
                    "needs_llm_dates": False,
                })
                cursor_end = entry - timedelta(days=1)
            else:
                # Compute start from typical duration
                start = date(
                    cursor_end.year - edu_stage.typical_duration_years,
                    edu_stage.entry_month, 1,
                )
                # But also check against birth_year + entry_age
                expected_start = date(birth_year + edu_stage.typical_entry_age, edu_stage.entry_month, 1)
                # Use the later of the two (to avoid impossibly early starts)
                if expected_start > start:
                    start = expected_start

                higher_ed_anchors.append({
                    "stage_key": edu_stage.stage_name,
                    "start_date": start,
                    "end_date": cursor_end,
                    "is_hard_constraint": edu_stage == higher_ed_stages[0],  # last stage has hard end
                    "source": "backward_from_graduation",
                    "needs_llm_content": True,
                    "needs_llm_dates": False,
                })
                cursor_end = start - timedelta(days=1)

        # Reverse to get chronological order
        higher_ed_anchors.reverse()

        # ── 4. Compute pre-higher-ed dates deterministically ──
        first_higher_ed_start = higher_ed_anchors[0]["start_date"] if higher_ed_anchors else None
        early_stages = self._compute_early_stage_dates(
            social_context=social_context,
            birth_date=birth_date,
            reference_date=reference_date,
            first_higher_ed_start=first_higher_ed_start,
            skipped_stages=skipped_stages,
        )
        # Mark early stages
        for s in early_stages:
            s["is_hard_constraint"] = False
            s["needs_llm_content"] = True
            s["needs_llm_dates"] = False

        # ── 5. Build post-graduation anchors ──
        post_grad_anchors: List[Dict[str, Any]] = []

        if graduation_date and employment_start:
            # Gap between graduation and employment
            gap_start = graduation_date + timedelta(days=1)
            gap_end = employment_start - timedelta(days=1)
            if gap_end >= gap_start:
                post_grad_anchors.append({
                    "stage_key": "gap_transition",
                    "start_date": gap_start,
                    "end_date": gap_end,
                    "is_hard_constraint": True,
                    "source": "graduation_to_employment_gap",
                    "needs_llm_content": True,
                    "needs_llm_dates": False,
                })

        if employment_start:
            post_grad_anchors.append({
                "stage_key": "career_current",
                "start_date": employment_start,
                "end_date": reference_date,
                "is_hard_constraint": True,
                "source": "employment_to_reference",
                "needs_llm_content": True,
                "needs_llm_dates": False,
            })
        elif graduation_date:
            # No employment info — extend last period to reference_date
            post_grad_anchors.append({
                "stage_key": "post_graduation",
                "start_date": graduation_date + timedelta(days=1),
                "end_date": reference_date,
                "is_hard_constraint": True,
                "source": "graduation_to_reference",
                "needs_llm_content": True,
                "needs_llm_dates": False,
            })

        # ── 6. Combine all anchors in chronological order ──
        all_anchors = early_stages + higher_ed_anchors + post_grad_anchors

        # Final safety: ensure last anchor ends at reference_date
        if all_anchors and all_anchors[-1]["end_date"] != reference_date:
            all_anchors[-1]["end_date"] = reference_date

        # Ensure first anchor starts at birth_date
        if all_anchors and all_anchors[0]["start_date"] != birth_date:
            all_anchors[0]["start_date"] = birth_date

        logger.info(
            f"AFBE: Derived {len(all_anchors)} anchors: "
            + ", ".join(f"{a['stage_key']}({a['start_date']}~{a['end_date']})" for a in all_anchors)
        )
        return all_anchors

    async def _generate_single_period_via_llm(
        self,
        anchor: Dict[str, Any],
        period_index: int,
        total_periods: int,
        persona_brief_text: str,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
        already_generated_summaries: str = "",
    ) -> Dict[str, Any]:
        """Generate content fields for a single life period via LLM.

        The dates are already determined by the anchor sequence.
        LLM only needs to generate semantic content fields.
        """
        stage_key = anchor["stage_key"]
        start_date = anchor["start_date"]
        end_date = anchor["end_date"]
        duration_days = (end_date - start_date).days
        duration_years = duration_days / 365.25
        age_at_start = self._compute_age_display(birth_date, start_date)
        age_at_end = self._compute_age_display(birth_date, end_date)

        system_prompt = (
            "You are a life stage content generation expert. Given a life stage's date range and basic information, "
            "please generate semantic content fields for that stage. Output only structured JSON."
        )

        user_prompt = f"""Generate content for period {period_index}/{total_periods} of the following persona.

[Stage basic information — dates are fixed and cannot be changed]
- period_id: LP{period_index}
- stage_key: {stage_key}
- start_date: {start_date.isoformat()}
- end_date: {end_date.isoformat()}
- Duration: {duration_years:.1f} years ({duration_days} days)
- Age range: {age_at_start:.1f} ~ {age_at_end:.1f} years old
- Birth date: {birth_date.isoformat()}
- Total periods: {total_periods}

[Persona background]
{persona_brief_text}

[Social context]
{social_context_text}

[Summaries of other already-generated stages (for narrative coherence reference)]
{already_generated_summaries if already_generated_summaries else '(This is the first generated stage)'}

Please generate the following fields:
1. stage_label: Semantic label in snake_case format (e.g., 'early_childhood', 'undergraduate_exploration'). Transition/gap periods must use 'gap_transition'.
2. title: A life summary title for this stage
3. dominant_theme: The core organizing logic and dominant theme of this stage
4. developmental_tasks: Developmental tasks that need to be accomplished in this stage (3-5 items)
5. stage_goals: What goals the working self tends to organize around in this stage (3-5 items)
6. salient_pressures: Typical pressures faced in this stage (2-4 items)
7. salient_opportunities: Opportunity windows that converge toward the target persona (2-4 items)
8. likely_transition_triggers: Events most likely to push this stage into the next stage (1-3 items)
9. is_education_stage: bool
11. is_work_stage: bool
12. is_transition_stage: bool
"""

        try:
            period_content = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=LifePeriod,
                system_prompt=system_prompt,
                task_type="p1a_forward_period",
                temperature=0.0,
            )
            # Override dates with anchor values (LLM might hallucinate different dates)
            result = period_content.model_dump()
            result["period_id"] = f"LP{period_index}"
            result["period_date_range"] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            return result
        except Exception as e:
            logger.warning(f"AFBE: Failed to generate period {period_index} ({stage_key}): {e}")
            # Return a minimal fallback period
            return {
                "period_id": f"LP{period_index}",
                "stage_label": stage_key,
                "title": f"{stage_key} stage",
                "dominant_theme": f"Development during {stage_key}",
                "developmental_tasks": ["Age-appropriate development"],
                "stage_goals": ["Growth and adaptation"],
                "salient_pressures": ["Typical age-related pressures"],
                "salient_opportunities": ["Learning and growth"],
                "likely_transition_triggers": ["Natural progression"],
                "period_date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "is_education_stage": stage_key not in ("early_childhood", "gap_transition", "career_current", "post_graduation"),
                "is_work_stage": stage_key in ("career_current",),
                "is_transition_stage": stage_key in ("gap_transition",),
            }

    async def _generate_plan_backward(
        self,
        enriched_constraints: Dict[str, Any],
        social_context: SocialContextProfile,
        persona_pathway: InferredPersonaPathway,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
        inferred_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """AFBE: Generate the complete life plan using Anchor-First Backward Expansion.

        1. Derive anchor sequence (deterministic dates)
        2. Generate content for each period via LLM (parallelized)
        3. Assemble into MilestonePlan format
        """
        self._print_step("AFBE-A", "Deriving anchor sequence (deterministic date calculation)")
        anchors = self._derive_anchor_sequence(
            social_context=social_context,
            persona_pathway=persona_pathway,
            birth_date=birth_date,
            reference_date=reference_date,
            inferred_constraints=inferred_constraints,
        )

        total_periods = len(anchors)
        persona_brief_text = enriched_constraints.get("persona_brief_text", "")
        if not isinstance(persona_brief_text, str):
            persona_brief_text = ""

        self._print_step(
            "AFBE-B",
            f"Generating content for {total_periods} period(s) in parallel",
            detail=", ".join(f"{a['stage_key']}" for a in anchors),
        )

        # Generate all periods in parallel (dates are already determined)
        tasks = []
        for idx, anchor in enumerate(anchors, start=1):
            tasks.append(
                self._generate_single_period_via_llm(
                    anchor=anchor,
                    period_index=idx,
                    total_periods=total_periods,
                    persona_brief_text=persona_brief_text,
                    social_context_text=social_context_text,
                    birth_date=birth_date,
                    reference_date=reference_date,
                    already_generated_summaries="",  # All parallel, no prior summaries
                )
            )

        period_results = await asyncio.gather(*tasks)

        # Assemble into plan dict
        plan_dict = {
            "global_summary": {
                "target_age_exact": resolved_target_age,
                "total_periods": total_periods,
                "planning_logic": (
                    f"AFBE (Anchor-First Backward Expansion): "
                    f"{total_periods} stages derived from social context and persona pathway, "
                    f"dates computed deterministically, content generated in parallel."
                ),
            },
            "life_periods": list(period_results),
        }

        self._print_step(
            "AFBE-C",
            "Anchor sequence generation complete",
            detail=f"{total_periods} periods assembled",
        )

        return plan_dict

    # ==========================================
    # v7: Pure LLM Backward Sequential Generation
    # ==========================================

    MAX_SUMMARY_WINDOW = 5  # Sliding window for summaries (Section 6.1)
    MAX_PERIODS_V7 = 20     # Safety upper limit (Section 5, Principle 4)

    async def _generate_single_period_backward_v7(
        self,
        end_date: date,
        birth_date: date,
        reference_date: date,
        age_at_end: float,
        persona_brief_text: str,
        social_context_text: str,
        already_generated_summaries: str,
        step_number: int,
        progress_note: str = "",
        review_feedback: str = "",
    ) -> Dict[str, Any]:
        """v7: Generate a single life period via LLM in backward direction.

        The end_date is fixed; LLM decides start_date and all content fields.
        Anti-overfitting: prompt does NOT use culture-specific examples.
        """
        system_prompt = (
            "You are a life-stage planning expert. You are building a person's "
            "life-stage blueprint step by step from the most recent period backward.\n"
            "Each call you generate exactly ONE period. The end_date is already fixed; "
            "you decide the start_date and all content fields.\n\n"
            "Key principles:\n"
            "1. Refer to [Social Context] education system rules to decide typical durations.\n"
            "2. Refer to [Persona Pathway] to decide stage divisions.\n"
            "3. If [Persona Pathway] marks SKIP for a stage, do NOT generate that stage.\n"
            "4. Periods must be seamless: your start_date minus 1 day = previous period's end_date.\n"
            "5. If the age at end is very young (suggesting early childhood), "
            "evaluate whether the social context defines a pre-education phase "
            "before the current stage's typical_entry_age. If so, generate the "
            "current period with a start_date that aligns with the current stage's "
            "typical_entry_age, and let the next backward step generate the earlier "
            "pre-education period. Only set is_first_period=True when you are certain "
            "the current period reaches back to birth.\n"
            "6. All dates must be ISO format YYYY-MM-DD.\n"
            "7. Do NOT assume any specific education system structure — "
            "always derive stage names and durations from [Social Context].\n"
            "8. CRITICAL: start_date MUST be strictly BEFORE end_date. "
            "If the given end_date falls in the middle of a natural stage boundary "
            "(e.g. end_date is Aug 31 but the stage you want starts Sep 1), "
            "you must still set start_date < end_date. "
            "The period covers whatever portion of the stage falls within [start_date, end_date].\n"
            "9. Stage granularity principle:\n"
            "   a. Do NOT merge two qualitatively different life phases into one period. "
            "If a period's start would cover ages well before the typical_entry_age "
            "of its stage_label (as defined in [Social Context]), you should generate "
            "a separate earlier period for that pre-stage phase instead.\n"
            "   b. If [Persona Pathway] marks INSERT or SPLIT for a stage, you MUST "
            "generate separate periods for the sub-phases rather than merging them "
            "into a single period. For example, if a doctoral stage has an INSERT "
            "marker for a research direction shift, generate one period for the "
            "early research phase and another for the later phase — do NOT combine "
            "them into one long period.\n"
            "   c. When a single long phase (>5 years) contains a significant qualitative "
            "shift (direction change, role transition, environment change), consider "
            "splitting it into separate periods — unless the shift is minor and the "
            "phase remains essentially continuous.\n"
            "   d. Transition periods between major life phases may be short but should not "
            "be omitted if they contain significant life events (graduation, job change, "
            "relocation, role transition, etc.). Generate a brief period for such "
            "transitions rather than skipping them.\n"
            "10. Period granularity rules (CRITICAL — prevents over-segmentation):\n"
            "   a. For homogeneous education stages (primary school, lower secondary, etc.) "
            "with no qualitative shift: generate ONE period per 2-3 years, NOT one per year. "
            "A 6-year primary school should produce at most 2 periods.\n"
            "   b. For stages with a clear internal split (e.g., NCEA Level 1-2 vs Level 3, "
            "undergraduate exploration vs specialization): split at the natural boundary.\n"
            "   c. For early childhood (birth to school entry): ONE period only.\n"
            "   d. For career stages: split at ROLE CHANGES, not annually. "
            "A 10-year career with no role change = 1-2 periods.\n"
            "   e. NEVER generate a period shorter than 1 year unless it is a gap_transition "
            "or a major life event (graduation, job change, relocation).\n"
            "   f. When [Social Context] shows a stage with typical_duration_years >= 4 "
            "and no INSERT/SPLIT marker in [Persona Pathway], generate at most 2 periods "
            "for that stage.\n"
            "   g. For retirement stages: ONE period unless there is a clear active/late split."
        )

        summaries_block = already_generated_summaries or "(This is the first period being generated — the most recent/current life stage)"

        # Build review feedback block if provided (from upstream retry)
        review_feedback_block = ""
        if review_feedback:
            review_feedback_block = (
                "\n[IMPORTANT — Correction signals from previous review]\n"
                "The previous plan generation had the following issues. "
                "Please pay special attention and avoid repeating them:\n"
                f"{review_feedback}\n"
            )

        user_prompt = f"""Please generate the current life period (this is backward step #{step_number}).

[Hard Constraints — DO NOT change]
- This period's end_date: {end_date.isoformat()}
- Birth date: {birth_date.isoformat()}
- Simulation end date: {reference_date.isoformat()}
- Age at end of this period: {age_at_end:.1f} years old

[Persona Background]
{persona_brief_text}

[Social Context & Persona Pathway]
{social_context_text}

[Already generated subsequent periods (from recent to distant)]
{progress_note}
{summaries_block}
{review_feedback_block}
"""

        try:
            period_output = await self.llm.generate_structured(
                prompt=user_prompt,
                response_model=BackwardPeriodOutput,
                system_prompt=system_prompt,
                task_type="p1a_backward_period",
                temperature=0.0,
            )
            return period_output.model_dump()
        except Exception as e:
            logger.warning(f"v7 backward generation step {step_number} failed: {e}")
            # Fallback: generate a minimal period
            fallback_start = self._subtract_years(end_date, 3)
            if fallback_start < birth_date:
                fallback_start = birth_date
            return {
                "start_date": fallback_start.isoformat(),
                "stage_label": "unknown_stage",
                "title": f"Life period ending at age {age_at_end:.0f}",
                "dominant_theme": "Development and growth",
                "developmental_tasks": ["Age-appropriate development"],
                "stage_goals": ["Growth and adaptation"],
                "salient_pressures": ["Typical pressures"],
                "salient_opportunities": ["Learning and growth"],
                "likely_transition_triggers": ["Natural progression"],
                "is_education_stage": False,
                "is_work_stage": False,
                "is_transition_stage": False,
                "is_first_period": fallback_start <= birth_date,
            }

    async def _generate_plan_backward_sequential(
        self,
        enriched_constraints: Dict[str, Any],
        social_context_text: str,
        birth_date: date,
        reference_date: date,
        resolved_target_age: int,
        review_feedback: str = "",
        persona_pathway: Optional[InferredPersonaPathway] = None,
    ) -> Dict[str, Any]:
        """v7: Pure LLM Backward Sequential Generation.

        Generate periods from the last one backward to the first,
        using a sliding window of summaries (Section 6.1) and
        enriched summaries with theme+triggers (Section 6.2).
        """
        persona_brief_text = enriched_constraints.get("persona_brief_text", "")
        if not isinstance(persona_brief_text, str):
            persona_brief_text = ""

        # ── Extract pathway granularity hints (INSERT/SPLIT markers) ──
        # ── Plan-C: Enhanced pathway granularity hints ──
        # Include not just INSERT/SPLIT markers but also the full key_stages
        # list for each dimension, so the LLM has a complete picture of what
        # sub-phases exist and cannot silently skip early ones.
        pathway_granularity_hints = ""
        if persona_pathway and persona_pathway.dimensions:
            insert_split_items = []
            key_stages_lines = []
            for dim in persona_pathway.dimensions:
                # Collect INSERT/SPLIT deviations
                for dev in dim.deviations_from_typical:
                    dev_upper = dev.strip().upper()
                    if dev_upper.startswith("INSERT") or dev_upper.startswith("SPLIT"):
                        insert_split_items.append(f"[{dim.dimension}] {dev}")
                # Collect key_stages for all dimensions (not just deviating ones)
                if dim.key_stages:
                    stages_str = " → ".join(dim.key_stages)
                    key_stages_lines.append(
                        f"  [{dim.dimension}] {stages_str}"
                    )

            hint_parts = []
            if key_stages_lines:
                hint_parts.append(
                    "[Persona Pathway — Key Stages (HIGHEST PRIORITY)]\n"
                    "This persona's life follows these specific stage sequences. "
                    "You MUST generate separate periods for EACH stage listed. "
                    "Do NOT merge multiple stages into one period.\n"
                    + "\n".join(key_stages_lines)
                )
            if insert_split_items:
                hint_parts.append(
                    "[Pathway Granularity Hints — IMPORTANT]\n"
                    "The inferred persona pathway indicates the following sub-phase "
                    "splits or insertions within major stages:\n"
                    + "\n".join(f"- {item}" for item in insert_split_items)
                    + "\nIMPORTANT: When generating a period that falls within one of these "
                    "major stages, you MUST generate separate periods for each "
                    "sub-phase. Do NOT merge the entire stage into a single period. "
                    "Generate only the sub-phase that ends at the current cursor, "
                    "and let the next backward step generate the earlier sub-phase."
                )
            if hint_parts:
                pathway_granularity_hints = "\n\n".join(hint_parts)

        periods = []
        cursor_end = reference_date
        generated_summaries = []  # newest first

        # ── Target-Driven Mode ──
        # When persona_pathway has suggested_periods, iterate over them
        # instead of using the pure LLM while loop. This gives the LLM
        # concrete period anchors to target, improving coherence.
        if persona_pathway and persona_pathway.suggested_periods:
            target_periods = list(reversed(persona_pathway.suggested_periods))
            # target_periods are now in reverse-chronological order (latest first)

            for sp_idx, sp in enumerate(target_periods):
                step = sp_idx + 1
                age_at_end = self._compute_age_display(birth_date, cursor_end)

                # Sliding window summaries (Section 6.1)
                recent_summaries = generated_summaries[:self.MAX_SUMMARY_WINDOW]
                summaries_text = "\n".join(
                    f"  LP(+{i+1}): {s}" for i, s in enumerate(recent_summaries)
                ) if recent_summaries else ""

                # Global progress note (Section 6.1)
                progress_note = ""
                if generated_summaries:
                    progress_note = (
                        f"[Progress: {len(generated_summaries)} period(s) generated so far, "
                        f"covering age ~{age_at_end:.0f} onward to age ~{resolved_target_age}. "
                        f"Now generating the period ending at age ~{age_at_end:.1f}.]"
                    )

                # Append pathway granularity hints to progress note
                if pathway_granularity_hints:
                    progress_note = (
                        f"{progress_note}\n{pathway_granularity_hints}"
                        if progress_note
                        else pathway_granularity_hints
                    )

                # ── [Period Anchor] Target-driven prompt injection ──
                period_anchor = (
                    f"\n\n[Period Anchor — Target-Driven Mode]\n"
                    f"This period MUST correspond to the following suggested life stage:\n"
                    f"  - stage_key: {sp.stage_key}\n"
                    f"  - label_hint: {sp.label_hint}\n"
                    f"  - approx_start: {sp.approx_start}\n"
                    f"  - approx_end: {sp.approx_end}\n"
                )
                if sp.stage_position_hint:
                    period_anchor += f"  - stage_position_hint: {sp.stage_position_hint}\n"
                if sp.consolidation_rationale:
                    period_anchor += f"  - consolidation_rationale: {sp.consolidation_rationale}\n"
                period_anchor += (
                    "\nGenerate exactly ONE period matching this anchor. "
                    "Use the approx_start/approx_end as strong guidance for date boundaries, "
                    "but adjust slightly if needed for calendar consistency.\n"
                )
                progress_note = (progress_note + period_anchor) if progress_note else period_anchor

                self._print_step(
                    f"v7-{step}",
                    f"Target-Driven Backward step {step} [{sp.stage_key}]",
                    detail=f"end_date={cursor_end.isoformat()}, age≈{age_at_end:.1f}",
                )

                period_output = await self._generate_single_period_backward_v7(
                    end_date=cursor_end,
                    birth_date=birth_date,
                    reference_date=reference_date,
                    age_at_end=age_at_end,
                    persona_brief_text=persona_brief_text,
                    social_context_text=social_context_text,
                    already_generated_summaries=summaries_text,
                    step_number=step,
                    progress_note=progress_note,
                    review_feedback=review_feedback if step == 1 else "",
                )

                # Record end_date (not in LLM output, we track it)
                period_output["_end_date"] = cursor_end.isoformat()

                periods.append(period_output)

                # Enriched summary with theme + triggers (Section 6.2)
                triggers = period_output.get("likely_transition_triggers", [])
                triggers_str = ", ".join(triggers) if triggers else "N/A"
                generated_summaries.append(
                    f"{period_output.get('stage_label', '?')}: {period_output.get('title', '?')} "
                    f"({period_output.get('start_date', '?')} ~ {cursor_end.isoformat()}) "
                    f"[theme: {period_output.get('dominant_theme', '?')}] "
                    f"[triggers: {triggers_str}]"
                )

                start = self._parse_iso_date(period_output.get("start_date"))
                if start is None:
                    start = birth_date  # fallback

                # Termination: reached birth
                if start <= birth_date or period_output.get("is_first_period"):
                    period_output["start_date"] = birth_date.isoformat()
                    break

                cursor_end = start - timedelta(days=1)

        else:
            # ── Fallback: Pure LLM Backward Sequential ──
            # step counts total iterations (including guard-discarded ones);
            # len(periods) counts effective periods.  We cap *effective* periods
            # at MAX_PERIODS_V7 and add a generous safety margin for total steps
            # to avoid infinite loops when guards keep discarding.
            step = 0
            max_total_steps = self.MAX_PERIODS_V7 * 2  # safety ceiling

            while len(periods) < self.MAX_PERIODS_V7 and step < max_total_steps:
                step += 1
                age_at_end = self._compute_age_display(birth_date, cursor_end)

                # Sliding window summaries (Section 6.1)
                recent_summaries = generated_summaries[:self.MAX_SUMMARY_WINDOW]
                summaries_text = "\n".join(
                    f"  LP(+{i+1}): {s}" for i, s in enumerate(recent_summaries)
                ) if recent_summaries else ""

                # Global progress note (Section 6.1)
                progress_note = ""
                if generated_summaries:
                    progress_note = (
                        f"[Progress: {len(generated_summaries)} period(s) generated so far, "
                        f"covering age ~{age_at_end:.0f} onward to age ~{resolved_target_age}. "
                        f"Now generating the period ending at age ~{age_at_end:.1f}.]"
                    )

                # Append pathway granularity hints to progress note
                if pathway_granularity_hints:
                    progress_note = (
                        f"{progress_note}\n{pathway_granularity_hints}"
                        if progress_note
                        else pathway_granularity_hints
                    )

                self._print_step(
                    f"v7-{step}",
                    f"Backward generation step {step}",
                    detail=f"end_date={cursor_end.isoformat()}, age≈{age_at_end:.1f}",
                )

                period_output = await self._generate_single_period_backward_v7(
                    end_date=cursor_end,
                    birth_date=birth_date,
                    reference_date=reference_date,
                    age_at_end=age_at_end,
                    persona_brief_text=persona_brief_text,
                    social_context_text=social_context_text,
                    already_generated_summaries=summaries_text,
                    step_number=step,
                    progress_note=progress_note,
                    review_feedback=review_feedback if step == 1 else "",
                )

                # Record end_date (not in LLM output, we track it)
                period_output["_end_date"] = cursor_end.isoformat()

                periods.append(period_output)

                # Enriched summary with theme + triggers (Section 6.2)
                triggers = period_output.get("likely_transition_triggers", [])
                triggers_str = ", ".join(triggers) if triggers else "N/A"
                generated_summaries.append(
                    f"{period_output.get('stage_label', '?')}: {period_output.get('title', '?')} "
                    f"({period_output.get('start_date', '?')} ~ {cursor_end.isoformat()}) "
                    f"[theme: {period_output.get('dominant_theme', '?')}] "
                    f"[triggers: {triggers_str}]"
                )

                start = self._parse_iso_date(period_output.get("start_date"))
                if start is None:
                    start = birth_date  # fallback

                # ── Guard: start_date > end_date means LLM is confused ──
                if start > cursor_end:
                    logger.warning(
                        f"[v7] Backward step {step}: LLM returned start_date={start} > "
                        f"end_date={cursor_end}. Discarding this period and forcing cursor backward."
                    )
                    periods.pop()  # remove the invalid period
                    generated_summaries.pop()  # remove its summary
                    forced_end = min(start - timedelta(days=1), cursor_end - timedelta(days=365))
                    if forced_end < birth_date:
                        forced_end = birth_date
                    cursor_end = forced_end
                    if cursor_end <= birth_date:
                        break
                    continue

                # ── Guard: period too short (< 30 days) ──
                period_duration_days = (cursor_end - start).days + 1
                if period_duration_days < 30 and start > birth_date:
                    logger.warning(
                        f"[v7] Backward step {step}: period too short "
                        f"({period_duration_days} days, {start} ~ {cursor_end}). "
                        f"Discarding and letting next step absorb this range."
                    )
                    periods.pop()  # remove the too-short period
                    generated_summaries.pop()  # remove its summary
                    cursor_end = start - timedelta(days=1)
                    if cursor_end <= birth_date:
                        break
                    continue

                # Termination conditions
                if start <= birth_date or period_output.get("is_first_period"):
                    period_output["start_date"] = birth_date.isoformat()
                    break

                cursor_end = start - timedelta(days=1)

            # Reverse to chronological order and assign period_ids (fallback mode)
            periods.reverse()
            for i, p in enumerate(periods, 1):
                p["period_id"] = f"LP{i}"
                p["period_date_range"] = {
                    "start_date": p.pop("start_date"),
                    "end_date": p.pop("_end_date"),
                }
                p.pop("is_first_period", None)

        # ── Shared: assign period_ids + build period_date_range for target-driven mode ──
        # Target-driven mode generates periods in chronological order but does NOT
        # assemble period_date_range (only fallback mode does that in its own loop).
        # Fix: mirror the same start_date/_end_date → period_date_range assembly here.
        if persona_pathway and persona_pathway.suggested_periods:
            for i, p in enumerate(periods, 1):
                if "period_id" not in p:
                    p["period_id"] = f"LP{i}"
                # Assemble period_date_range from flat start_date / _end_date fields
                # (identical to what the fallback branch does)
                if "period_date_range" not in p:
                    p["period_date_range"] = {
                        "start_date": p.pop("start_date", ""),
                        "end_date": p.pop("_end_date", ""),
                    }
                p.pop("is_first_period", None)

        return {
            "global_summary": {
                "target_age_exact": resolved_target_age,
                "total_periods": len(periods),
                "planning_logic": (
                    f"v7 Target-Driven Backward: "
                    f"{len(periods)} periods generated from InferredPersonaPathway.suggested_periods."
                    if (persona_pathway and persona_pathway.suggested_periods)
                    else f"v7 Pure LLM Backward Sequential: "
                    f"{len(periods)} periods generated from last to first."
                ),
            },
            "life_periods": periods,
        }

    # ==========================================
    # v7: Impact Propagation Analysis (Section 4.3)
    # ==========================================

    def _analyze_fix_impact(
        self,
        plan_dict: Dict[str, Any],
        review_report: PlanReviewReport,
    ) -> FixImpactAnalysis:
        """Analyze which periods need fixing and whether fixes propagate."""
        directly_affected = set()
        has_date_issues = False

        for issue in review_report.issues:
            for pid in issue.affected_period_ids:
                directly_affected.add(pid)
            # Check if this is a date-related issue
            if any(kw in issue.issue_type.lower() for kw in
                   ["date", "duration", "boundary", "missing_stage", "extra_stage"]):
                has_date_issues = True

        for patch in review_report.patches:
            directly_affected.add(patch.period_id)
            if patch.patch_type in ("update_start_date", "update_end_date"):
                has_date_issues = True

        if not directly_affected:
            return FixImpactAnalysis(fix_strategy="isolated")

        # If no date issues, fixes are isolated
        if not has_date_issues:
            return FixImpactAnalysis(
                directly_affected=sorted(directly_affected),
                fix_strategy="isolated",
                fix_groups=[sorted(directly_affected)],
            )

        # Trace propagation for date issues
        periods = plan_dict.get("life_periods", [])
        all_affected = set(directly_affected)
        period_ids = [p.get("period_id") for p in periods]

        # Simple propagation: if a period's dates change, check neighbors
        frontier = set(directly_affected)
        while frontier:
            new_frontier = set()
            for pid in frontier:
                if pid not in period_ids:
                    continue
                idx = period_ids.index(pid)
                # Check neighbors
                for neighbor_idx in [idx - 1, idx + 1]:
                    if 0 <= neighbor_idx < len(periods):
                        neighbor_id = period_ids[neighbor_idx]
                        if neighbor_id not in all_affected:
                            new_frontier.add(neighbor_id)
            all_affected |= new_frontier
            frontier = new_frontier
            # Only propagate one level for date issues
            break

        propagation_affected = all_affected - directly_affected
        total_periods = len(periods)

        # Determine strategy
        strategy = self._determine_fix_strategy(all_affected, total_periods)

        return FixImpactAnalysis(
            directly_affected=sorted(directly_affected),
            propagation_affected=sorted(propagation_affected),
            fix_strategy=strategy,
            fix_groups=[sorted(all_affected)],
        )

    @staticmethod
    def _determine_fix_strategy(
        all_affected: set,
        total_periods: int,
    ) -> str:
        """Determine fix strategy based on number of affected periods."""
        if total_periods == 0:
            return "isolated"
        ratio = len(all_affected) / total_periods
        if len(all_affected) <= 2:
            return "isolated"
        elif ratio <= 0.5:
            return "cascading"
        else:
            return "full_regen"

    async def _regenerate_specific_periods_v7(
        self,
        plan_dict: Dict[str, Any],
        period_ids: List[str],
        review_feedback: str,
        persona_brief_text: str,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
    ) -> Dict[str, Any]:
        """v7: Regenerate only the specified periods, keeping all others unchanged.

        Dates are constrained by adjacent periods.

        Fix (Plan-B): Build rich neighbor context so the LLM cannot generate
        a semantically wrong stage type.  When a period is the result of a
        split, the adjacent periods provide strong anchors (e.g. "the next
        period is career_entry_adaptation, so this period must still be inside
        the PhD stage").
        """
        periods = plan_dict.get("life_periods", [])

        for pid in period_ids:
            idx = next((i for i, p in enumerate(periods) if p.get("period_id") == pid), None)
            if idx is None:
                continue

            # Determine date constraints from neighbors
            if idx > 0:
                prev_end = periods[idx - 1]["period_date_range"]["end_date"]
                start_date = self._parse_iso_date(prev_end) + timedelta(days=1)
            else:
                start_date = birth_date

            if idx < len(periods) - 1:
                next_start = periods[idx + 1]["period_date_range"]["start_date"]
                end_date = self._parse_iso_date(next_start) - timedelta(days=1)
            else:
                end_date = reference_date

            age_at_end = self._compute_age_display(birth_date, end_date)
            age_at_start = self._compute_age_display(birth_date, start_date)

            # ── Plan-B: Build neighbor context summary ──
            neighbor_context_parts = []
            if idx > 0:
                prev_p = periods[idx - 1]
                prev_dr = prev_p.get("period_date_range", {})
                neighbor_context_parts.append(
                    f"PRECEDING period (immediately before this one): "
                    f"stage_label={prev_p.get('stage_label', '?')}, "
                    f"title={prev_p.get('title', '?')!r}, "
                    f"dates={prev_dr.get('start_date', '?')} ~ {prev_dr.get('end_date', '?')}"
                )
            if idx < len(periods) - 1:
                next_p = periods[idx + 1]
                next_dr = next_p.get("period_date_range", {})
                neighbor_context_parts.append(
                    f"FOLLOWING period (immediately after this one): "
                    f"stage_label={next_p.get('stage_label', '?')}, "
                    f"title={next_p.get('title', '?')!r}, "
                    f"dates={next_dr.get('start_date', '?')} ~ {next_dr.get('end_date', '?')}"
                )

            neighbor_context = ""
            if neighbor_context_parts:
                neighbor_context = (
                    "\n[Neighbor Context — CRITICAL]\n"
                    "The period you are generating sits between the following fixed periods. "
                    "Your stage_label and title MUST be semantically consistent with these neighbors. "
                    "Do NOT generate a stage type that contradicts the surrounding life context.\n"
                    + "\n".join(f"  - {c}" for c in neighbor_context_parts)
                    + f"\n  - THIS period covers: {start_date.isoformat()} ~ {end_date.isoformat()} "
                    f"(age ~{age_at_start:.1f} ~ ~{age_at_end:.1f})"
                )

            # Combine review feedback with neighbor context
            combined_feedback = ""
            if review_feedback:
                combined_feedback += f"[Review feedback for targeted fix] {review_feedback}"
            if neighbor_context:
                combined_feedback += neighbor_context

            # Regenerate this period with fixed dates + enriched context
            new_period = await self._generate_single_period_backward_v7(
                end_date=end_date,
                birth_date=birth_date,
                reference_date=reference_date,
                age_at_end=age_at_end,
                persona_brief_text=persona_brief_text,
                social_context_text=social_context_text,
                already_generated_summaries=combined_feedback or "[Targeted regeneration]",
                step_number=0,  # special: targeted regen
            )

            # Override dates (fixed by neighbors)
            new_period["period_id"] = pid
            new_period["period_date_range"] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            new_period.pop("is_first_period", None)
            new_period.pop("_end_date", None)
            # Preserve stage type tags from original if not in new output
            old_period = periods[idx]
            for tag in ("is_education_stage", "is_work_stage", "is_transition_stage"):
                if tag not in new_period:
                    new_period[tag] = old_period.get(tag, False)
            periods[idx] = new_period

        return plan_dict

    async def _cascading_fix_v7(
        self,
        plan_dict: Dict[str, Any],
        affected_ids: set,
        review_feedback: str,
        persona_brief_text: str,
        social_context_text: str,
        birth_date: date,
        reference_date: date,
    ) -> Dict[str, Any]:
        """v7: Fix a group of affected periods by regenerating them backward.

        Unaffected periods on both sides serve as anchors.
        """
        periods = plan_dict.get("life_periods", [])
        affected_indices = sorted(
            [i for i, p in enumerate(periods) if p.get("period_id") in affected_ids]
        )

        if not affected_indices:
            return plan_dict

        first_idx = affected_indices[0]
        last_idx = affected_indices[-1]

        # End anchor: from the period AFTER the last affected, or reference_date
        if last_idx < len(periods) - 1:
            end_anchor = (
                self._parse_iso_date(periods[last_idx + 1]["period_date_range"]["start_date"])
                - timedelta(days=1)
            )
        else:
            end_anchor = reference_date

        # Start anchor: from the period BEFORE the first affected, or birth_date
        if first_idx > 0:
            start_anchor = (
                self._parse_iso_date(periods[first_idx - 1]["period_date_range"]["end_date"])
                + timedelta(days=1)
            )
        else:
            start_anchor = birth_date

        # Regenerate affected periods backward (from last to first)
        cursor_end = end_anchor
        new_periods = []

        for i in range(len(affected_indices) - 1, -1, -1):
            idx = affected_indices[i]
            is_first_in_group = (i == 0)
            age_at_end = self._compute_age_display(birth_date, cursor_end)

            period_output = await self._generate_single_period_backward_v7(
                end_date=cursor_end,
                birth_date=birth_date,
                reference_date=reference_date,
                age_at_end=age_at_end,
                persona_brief_text=persona_brief_text,
                social_context_text=social_context_text,
                already_generated_summaries=f"[Cascading fix] {review_feedback}",
                step_number=0,
            )

            start = self._parse_iso_date(period_output.get("start_date"))
            if start is None:
                start = start_anchor if is_first_in_group else birth_date

            # If this is the first period in the group, anchor start_date
            if is_first_in_group and start_anchor:
                start = start_anchor

            period_output["period_id"] = periods[idx]["period_id"]
            period_output["period_date_range"] = {
                "start_date": start.isoformat(),
                "end_date": cursor_end.isoformat(),
            }
            period_output.pop("is_first_period", None)
            period_output.pop("_end_date", None)

            new_periods.append((idx, period_output))
            cursor_end = start - timedelta(days=1)

        # Replace affected periods in plan
        for idx, new_period in new_periods:
            periods[idx] = new_period

        return plan_dict

    async def generate_plan(self, constraint_sheet: dict) -> Dict[str, Any]:
        """
        v7: Generate a complete life-stage blueprint using Pure LLM Backward Sequential Generation.

        Flow:
          Step 0a: Social Context Inference
          Step 0b: Persona Pathway Inference
          Step 1:  Backward Sequential Generation (N steps)
          Step 2:  Initial Postprocess
          Step 3:  Review + Invariant Validation
          Step 4:  Impact Propagation Analysis (if issues found)
          Step 5:  Targeted Fix (≤2 rounds)
          Step 6:  Upstream Retry (rare)
          Step 7:  Final Postprocess (always)
          Finalize
        """
        target_age = constraint_sheet.get('target_age_exact', 'unknown')
        name = constraint_sheet.get('persona_name_text', 'persona')
        logger.info(f"[v7] Starting life stage blueprint generation... target_age={target_age}, name={name}")

        self._print_step("1", "Receiving input constraints", detail=f"name={name} | target_age={target_age}")
        self._print_step("2", "Translating schema-encoded fields")
        enriched_constraints = self._translate_codes(constraint_sheet)

        self._print_step("3", "Resolving timeline anchors (birth_date / simulation_end_date)")
        timeline_anchor = self._resolve_timeline_anchor(constraint_sheet)
        reference_date: date = timeline_anchor["reference_date"]
        birth_date: date = timeline_anchor["birth_date"]
        resolved_target_age: int = timeline_anchor["target_age_exact"]

        enriched_constraints["resolved_reference_date"] = reference_date.isoformat()
        enriched_constraints["resolved_birth_date"] = birth_date.isoformat()
        enriched_constraints["resolved_target_age_exact"] = resolved_target_age

        # ── Step 0a: Social Context Inference ──
        self._print_step("0a", "Inferring social context rules (Social Norms Inference)")
        social_context = await self._infer_social_context(
            constraint_sheet=constraint_sheet,
            enriched_constraints=enriched_constraints,
            birth_date=birth_date,
        )

        # ── Step 0b: Persona Pathway Inference ──
        self._print_step("0b", "Inferring persona's actual life pathway (Persona Pathway Inference)")
        persona_pathway = await self._infer_persona_pathway(
            constraint_sheet=constraint_sheet,
            social_context=social_context,
            birth_date=birth_date,
            reference_date=reference_date,
        )
        social_context_text = self._format_social_context_for_prompt(social_context, persona_pathway, birth_date=birth_date)

        # ── Step 1: v7 Backward Sequential Generation ──
        self._print_step("v7-GEN", "Generating life stage blueprint via v7 pure LLM backward sequential")
        raw_plan = await self._generate_plan_backward_sequential(
            enriched_constraints=enriched_constraints,
            social_context_text=social_context_text,
            birth_date=birth_date,
            reference_date=reference_date,
            resolved_target_age=resolved_target_age,
            persona_pathway=persona_pathway,
        )
        logger.info(f"[v7] Backward generation complete. Periods: {len(raw_plan.get('life_periods', []))}")

        # ── Step 2: Initial Postprocess ──
        self._print_step("v7-PP1", "Initial Postprocess (first pass)")
        plan_dict = self._run_deterministic_postprocess(
            plan_dict=raw_plan,
            birth_date=birth_date,
            reference_date=reference_date,
            resolved_target_age=resolved_target_age,
            current_job_tenure_months=None,
            resolved_pre_current_role_gap_months=None,
            transition_semantic_hint="general",
            inferred_constraints={},
        )

        # ── Step 3: Structured Judge ──
        self._print_step("v7-JUDGE", "Structured Judge: review every period with structured verdicts")

        persona_brief_text = constraint_sheet.get("persona_brief_text", "")
        if not isinstance(persona_brief_text, str):
            persona_brief_text = ""

        judgement_report = await self._judge_plan_periods(
            plan_dict=plan_dict,
            persona_config=constraint_sheet,
            birth_date=birth_date,
            reference_date=reference_date,
            social_context_text=social_context_text,
        )
        plan_dict.setdefault("global_summary", {})["last_judgement_report"] = judgement_report.model_dump()

        non_pass = [j for j in judgement_report.judgements if j.verdict != "pass"]
        if non_pass:
            # ── Step 4: Execute Actions (rewrite → merge → split → delete) ──
            self._print_step(
                "v7-ACTION",
                f"Execute {len(non_pass)} action(s): "
                + ", ".join(f"{j.period_id}:{j.verdict}" for j in non_pass),
            )
            plan_dict, id_tracker = await self._execute_judgements(
                plan_dict=plan_dict,
                judgement_report=judgement_report,
                persona_brief_text=persona_brief_text,
                social_context_text=social_context_text,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
            )

            # ── Step 4.5: Final deterministic postprocess + clean up internal fields ──
            plan_dict = self._run_deterministic_postprocess(
                plan_dict=plan_dict,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                current_job_tenure_months=None,
                resolved_pre_current_role_gap_months=None,
                transition_semantic_hint="general",
                inferred_constraints={},
            )
            for period in plan_dict.get("life_periods", []):
                period.pop("_original_id", None)
                period.pop("_needs_content_gen", None)
                period.pop("_split_instruction", None)
                period.pop("_merge_instruction", None)
                period.pop("_merged_from", None)
                period.pop("_merged_content", None)

            # ── Step 5: Recheck adjacent coherence ──
            recheck_ids = [
                id_tracker.get(j.period_id, j.period_id) for j in non_pass
                if id_tracker.get(j.period_id) is not None
            ]
            self._print_step("v7-RECHECK", f"Recheck: {recheck_ids} and their neighbours")
            review_report = await self._recheck_adjacent_coherence(
                plan_dict=plan_dict,
                modified_period_ids=recheck_ids,
                persona_config=constraint_sheet,
                birth_date=birth_date,
                reference_date=reference_date,
                social_context_text=social_context_text,
            )
            plan_dict.setdefault("global_summary", {})["last_review_report"] = review_report.model_dump()

            # One-shot patch if recheck found issues (no loop)
            if review_report.status != "pass" and review_report.patches:
                plan_dict, _ = self._apply_local_plan_patches(plan_dict, review_report)
                plan_dict = self._run_deterministic_postprocess(
                    plan_dict=plan_dict,
                    birth_date=birth_date,
                    reference_date=reference_date,
                    resolved_target_age=resolved_target_age,
                    current_job_tenure_months=None,
                    resolved_pre_current_role_gap_months=None,
                    transition_semantic_hint="general",
                    inferred_constraints={},
                )
        else:
            # All periods passed — create a minimal review_report for downstream
            review_report = PlanReviewReport(
                status="pass",
                review_summary="All periods passed structured judge — no action needed.",
            )
            plan_dict.setdefault("global_summary", {})["last_review_report"] = review_report.model_dump()

        # ── Invariant Validation + Upstream Retry Loop ──
        # Keep retrying until no fatal (high-severity) issues remain.
        MAX_UPSTREAM_RETRIES = 8
        upstream_retry_count = 0
        while True:
            invariant_issues = self._validate_plan_invariants(plan_dict, birth_date, reference_date)
            duration_issues = await self._validate_plan_durations_with_llm(
                plan_dict, social_context_text=social_context_text,
            )
            invariant_issues.extend(duration_issues)

            # ── Graceful Degradation ──
            fatal_issues = [i for i in invariant_issues if "severity=high" in i]
            warning_issues = [i for i in invariant_issues if "severity=high" not in i]
            if warning_issues:
                logger.warning(f"[v7] Non-fatal plan issues (proceeding): " + " | ".join(warning_issues))
                plan_dict.setdefault("global_summary", {})["warnings"] = warning_issues

            if not fatal_issues and review_report.status != "needs_regeneration":
                # No fatal issues and no needs_regeneration — exit loop
                break

            # ── Step 6: Upstream Retry (re-run 0a + 0b + backward generation) ──
            upstream_retry_count += 1
            if upstream_retry_count > MAX_UPSTREAM_RETRIES:
                logger.warning(
                    f"[v7-UPSTREAM] Max retry limit ({MAX_UPSTREAM_RETRIES}) reached — "
                    f"exiting retry loop with {len(fatal_issues)} fatal issue(s) remaining. "
                    f"Proceeding with best-effort plan."
                )
                plan_dict.setdefault("global_summary", {})["upstream_retry_exhausted"] = True
                break
            self._print_step(
                "v7-UPSTREAM",
                f"Upstream Retry #{upstream_retry_count}: re-run 0a + 0b + backward generation"
                + (f" (fatal issues: {len(fatal_issues)})" if fatal_issues else "")
            )

            # ── Build correction feedback from review issues + invariant issues ──
            upstream_feedback_parts: List[str] = []
            if review_report.review_summary:
                upstream_feedback_parts.append(f"Review summary: {review_report.review_summary}")
            for issue in review_report.issues:
                upstream_feedback_parts.append(
                    f"[{issue.severity}] {issue.issue_type}: {issue.reason}"
                    + (f" (affected periods: {', '.join(issue.affected_period_ids)})" if issue.affected_period_ids else "")
                )
            for inv_issue in invariant_issues:
                upstream_feedback_parts.append(f"Invariant violation: {inv_issue}")

            upstream_feedback = upstream_feedback_parts  # List[str] for 0a/0b
            upstream_feedback_text = "\n".join(upstream_feedback_parts)  # str for backward gen
            logger.info(
                f"[v7-UPSTREAM] Retry #{upstream_retry_count}: Built {len(upstream_feedback_parts)} correction signal(s) "
                f"for upstream retry"
            )

            social_context = await self._infer_social_context(
                constraint_sheet=constraint_sheet,
                enriched_constraints=enriched_constraints,
                birth_date=birth_date,
                correction_feedback=upstream_feedback,
            )
            persona_pathway = await self._infer_persona_pathway(
                constraint_sheet=constraint_sheet,
                social_context=social_context,
                birth_date=birth_date,
                reference_date=reference_date,
                correction_feedback=upstream_feedback,
            )
            social_context_text = self._format_social_context_for_prompt(social_context, persona_pathway, birth_date=birth_date)
            raw_plan = await self._generate_plan_backward_sequential(
                enriched_constraints=enriched_constraints,
                social_context_text=social_context_text,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                review_feedback=upstream_feedback_text,
                persona_pathway=persona_pathway,
            )
            plan_dict = self._run_deterministic_postprocess(
                plan_dict=raw_plan,
                birth_date=birth_date,
                reference_date=reference_date,
                resolved_target_age=resolved_target_age,
                current_job_tenure_months=None,
                resolved_pre_current_role_gap_months=None,
                transition_semantic_hint="general",
                inferred_constraints={},
            )
            review_report = await self._review_plan_reasonableness(
                persona_config=constraint_sheet,
                plan_dict=plan_dict,
                birth_date=birth_date,
                reference_date=reference_date,
                inferred_constraints={},
                social_context_text=social_context_text,
            )
            plan_dict.setdefault("global_summary", {})["last_review_report"] = review_report.model_dump()
            # Loop continues — will re-validate and retry if still broken

        if upstream_retry_count > 0:
            retry_exhausted = plan_dict.get("global_summary", {}).get("upstream_retry_exhausted", False)
            if retry_exhausted:
                logger.warning(
                    f"[v7] Upstream retry loop exhausted after {upstream_retry_count} retry(s) — "
                    f"fatal issues may still remain (proceeding with best-effort plan)"
                )
            else:
                logger.info(
                    f"[v7] Upstream retry loop completed after {upstream_retry_count} retry(s) — "
                    f"all fatal issues resolved"
                )

        # ── Step 7: Final Postprocess (always, Section 4.4) ──
        self._print_step("v7-PP-FINAL", "Final Postprocess (unconditional, always executed)")
        plan_dict = self._run_deterministic_postprocess(
            plan_dict=plan_dict,
            birth_date=birth_date,
            reference_date=reference_date,
            resolved_target_age=resolved_target_age,
            current_job_tenure_months=None,
            resolved_pre_current_role_gap_months=None,
            transition_semantic_hint="general",
            inferred_constraints={},
        )
        # Final invariant check (record only, do not trigger new fix cycle)
        final_issues = self._validate_plan_invariants(plan_dict, birth_date, reference_date)
        if final_issues:
            plan_dict.setdefault("global_summary", {})["final_postprocess_warnings"] = final_issues
            logger.warning(f"[v7] Final postprocess found {len(final_issues)} residual issue(s): {final_issues}")

        # ── Finalize ──
        plan_dict["derived_memory_plan"] = self._compile_memory_segments_from_plan(
            canonical_plan=plan_dict,
            birth_date=birth_date,
            reference_date=reference_date,
        )

        # Store social context and inferred persona pathway in the plan output for downstream modules
        plan_dict.setdefault("global_summary", {})["social_context"] = social_context.model_dump()
        plan_dict.setdefault("global_summary", {})["inferred_persona_pathway"] = persona_pathway.model_dump()

        logger.info(
            f"[v7] Life stage planning complete: status={review_report.status}, "
            f"periods={len(plan_dict.get('life_periods', []))}, "
            f"derived_segments={len(plan_dict.get('derived_memory_plan', {}).get('segments', []))}"
        )
        return plan_dict

    def extract_and_print_plan(self, plan_dict: Dict[str, Any]):
        """
        Helper function: read and print the output dict in a clear format.
        You can reference this function's logic to connect the dict to downstream modules.
        """
        summary = plan_dict.get('global_summary', {})
        logger.info("\n=== Global Life Overview ===")
        logger.info(f"Target age: {summary.get('target_age_exact')}")
        logger.info(f"Total periods: {summary.get('total_periods')}")
        logger.info(f"Planning logic: {summary.get('planning_logic')}")
        
        periods = plan_dict.get('life_periods', [])
        logger.info("\n=== Detailed Life Stages ===")
        for lp in periods:
            logger.info(f"\n[{lp.get('period_id')}] {lp.get('title')} ({lp.get('stage_label')})")
            logger.info(f"  Dominant theme: {lp.get('dominant_theme')}")
            logger.info(f"  Developmental tasks: {', '.join(lp.get('developmental_tasks', []))}")
            logger.info(f"  Stage goals: {', '.join(lp.get('stage_goals', []))}")
            logger.info(f"  Salient pressures: {', '.join(lp.get('salient_pressures', []))}")
            logger.info(f"  Salient opportunities: {', '.join(lp.get('salient_opportunities', []))}")
            logger.info(f"  Transition triggers: {', '.join(lp.get('likely_transition_triggers', []))}")
            period_date_range = lp.get('period_date_range', {})
            if period_date_range:
                logger.info(f"  Stage date range: {period_date_range.get('start_date')} ~ {period_date_range.get('end_date')}")
            role_tenure_range = lp.get('role_tenure_range', {})
            if role_tenure_range:
                logger.info(
                    f"  Current role tenure window: {role_tenure_range.get('start_date')} ~ {role_tenure_range.get('end_date')} "
                    f"({role_tenure_range.get('tenure_months')} months)"
                )


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import asyncio
    
    # Configure basic logging output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    async def main():
        # Read API config from environment variables
        API_BASE = os.environ.get("OPENAI_API_BASE", "")
        API_KEY = os.environ.get("OPENAI_API_KEY", "")
        MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")
        if not API_BASE or not API_KEY:
            print("Set OPENAI_API_BASE and OPENAI_API_KEY environment variables")
            return
        
        client = AsyncLLMClient(
            default_model=MODEL_NAME,
            api_base=API_BASE,
            api_key=API_KEY
        )
        
        planner = DevelopmentAwareLifePeriodPlanner(
            llm_client=client,
        )
        
        # 2. Read the specified test sample
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_p0_persona_settings", "test_sample.json")
        
        if os.path.exists(sample_path):
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                
            # 3. Generate plan (will trigger prompt printing)
            if client is not None:
                plan = await planner.generate_plan(sample_data)
                if plan is None:
                    logger.error("Returned plan is None, exiting.")
                    return

                output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_plan.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(plan, f, indent=4, ensure_ascii=False)
                logger.info(f"Plan saved to: {output_path}")

                logger.info(plan)
                # Print the full model output JSON to verify all required fields are present
                logger.info("\n" + "="*20 + " Full Model Output JSON Validation " + "="*20)
                logger.info(json.dumps(plan, indent=4, ensure_ascii=False))
                logger.info("="*66 + "\n")
                
                # Also keep the manual extraction print method
                planner.extract_and_print_plan(plan)
            else:
                # Prompt-print-only test (will error at generate_structured since client is None)
                try:
                    await planner.generate_plan(sample_data)
                except AttributeError:
                    logger.warning("LLM request skipped because client is None. The prompt above should have been printed.")
        else:
            logger.error(f"Test file not found: {sample_path}")

    asyncio.run(main())