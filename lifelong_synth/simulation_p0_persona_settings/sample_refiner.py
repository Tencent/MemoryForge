"""
P0.5 — Sample Refiner: Validate & Auto-Refine Persona Config (v3 Two-Phase Strategy)
=====================================================================================

Two-phase approach:
  Phase 1 (v1): One-shot generate_structured with Pydantic Literal constraints
  Phase 2 (v2): Field-by-field validation + per-field retry with precise feedback
  Phase 3: Final full Participant model validation

Must-provide field: persona_brief_text (the only truly required input)
All other fields are optional — missing or invalid ones are auto-inferred by LLM.
"""

import json
import logging
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type

from pydantic import BaseModel, Field

from lifelong_synth.simulation_p1_initialisation.definition import (
    Participant,
    LocationInfo,
    SelfSystemAnchor,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

MUST_PROVIDE_FIELDS = [
    "persona_brief_text",
]

MAX_RETRY_PER_FIELD = 1000


# ═══════════════════════════════════════════════════════════════
# Field Type Enum
# ═══════════════════════════════════════════════════════════════

class FieldType(str, Enum):
    ENUM = "enum"
    LIST_ENUM = "list_enum"
    STRUCT = "struct"
    INT = "int"
    DATE_STR = "date_str"


# ═══════════════════════════════════════════════════════════════
# Field Issue
# ═══════════════════════════════════════════════════════════════

class IssueKind(str, Enum):
    MISSING = "missing"
    INVALID_ENUM = "invalid_enum"
    INVALID_STRUCT = "invalid_struct"
    INVALID_TYPE = "invalid_type"


@dataclass
class FieldIssue:
    field_name: str
    kind: IssueKind
    detail: str
    current_value: Any = None


# ═══════════════════════════════════════════════════════════════
# Field Registry
# ═══════════════════════════════════════════════════════════════

@dataclass
class FieldSpec:
    name: str
    type: FieldType
    allowed_values: Optional[List[str]] = None
    element_allowed_values: Optional[List[str]] = None
    fallback: Any = None
    description: str = ""
    description_zh: str = ""
    int_min: Optional[int] = None
    int_max: Optional[int] = None
    should_infer: bool = True  # False = sensitive field, skip auto-inference


FIELD_REGISTRY: Dict[str, FieldSpec] = {
    # ── Language & Name (auto-inferrable from persona_brief_text) ──
    "primary_language": FieldSpec(
        name="primary_language",
        type=FieldType.ENUM,
        allowed_values=["en"],
        fallback="en",
        description="Primary language code",
        description_zh="主要语言编码",
    ),
    "working_language": FieldSpec(
        name="working_language",
        type=FieldType.LIST_ENUM,
        element_allowed_values=["en"],
        fallback=["en"],
        description="Working language code list",
        description_zh="工作语言编码列表",
    ),
    "persona_name_text": FieldSpec(
        name="persona_name_text",
        type=FieldType.ENUM,  # free-text, but use ENUM type for simple string validation
        allowed_values=None,  # no enum constraint — any non-empty string is valid
        fallback=None,
        description="Persona display name",
        description_zh="角色名称",
    ),

    # ── Basic Identity ──
    "gender_identity_code": FieldSpec(
        name="gender_identity_code",
        type=FieldType.ENUM,
        allowed_values=["1_male", "2_female", "3_non_binary", "4_other", "Z_not_stated"],
        fallback=None,
        description="Gender identity code",
        description_zh="性别编码",
    ),
    "target_age_exact": FieldSpec(
        name="target_age_exact",
        type=FieldType.INT,
        int_min=0,
        int_max=120,
        fallback=None,
        description="Target age in years",
        description_zh="目标年龄",
    ),

    # ── Geography & Background ──
    "growing_up_location": FieldSpec(
        name="growing_up_location",
        type=FieldType.STRUCT,
        fallback=None,
        description="Where the persona primarily grew up",
        description_zh="成长地",
    ),
    "current_living_location": FieldSpec(
        name="current_living_location",
        type=FieldType.STRUCT,
        fallback=None,
        description="Residence at simulation END (target terminal state). Format: {country, province, city}",
        description_zh="模拟结束时的居住地（终态）",
    ),
    "sim_start_living_location": FieldSpec(
        name="sim_start_living_location",
        type=FieldType.STRUCT,
        fallback=None,
        description=(
            "Residence at simulation START. For birth-start simulations this is typically the "
            "growing-up location (hometown). May differ from current_living_location (end state). "
            "Format: {country, province, city}"
        ),
        description_zh="模拟开始时的居住地（初态，通常为成长地）",
    ),
    # ── Simulation-start state (inferred at P0.5, used directly in P1) ──
    "sim_start_occupation_group": FieldSpec(
        name="sim_start_occupation_group",
        type=FieldType.ENUM,
        fallback=None,
        description=(
            "Occupation category at simulation START. null if infant/toddler/not yet working. "
            "For birth-start simulations this is almost always null."
        ),
        description_zh="模拟开始时的职业分类（初态）",
        allowed_values=[
            "student", "manager_executive", "professional_finance_law_consulting",
            "professional_tech_research", "professional_health_education",
            "office_admin_support", "sales_customer_service", "service_hospitality_retail",
            "skilled_trades_technical_ops", "manual_logistics_transport",
            "public_service_military", "self_employed_creator",
            "retired", "unemployed", "homemaker",
        ],
    ),
    "sim_start_education_level": FieldSpec(
        name="sim_start_education_level",
        type=FieldType.ENUM,
        fallback=None,
        description=(
            "Highest completed education level (ISCED 0-8) at simulation START. "
            "For birth-start simulations use '0' (pre-primary). "
            "ISCED: 0=pre-primary, 1=primary, 2=lower-secondary, 3=upper-secondary, "
            "4=post-secondary non-tertiary, 5=short-cycle tertiary, "
            "6=bachelor, 7=master, 8=doctoral."
        ),
        description_zh="模拟开始时的最高完成教育等级（初态，ISCED 0-8）",
        allowed_values=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
    ),
    "sim_start_self_system_anchor": FieldSpec(
        name="sim_start_self_system_anchor",
        type=FieldType.STRUCT,
        fallback=None,
        description=(
            "Core values (Schwartz top-3) at simulation START. "
            "For birth-start simulations, reflect basic temperament tendencies "
            "(e.g. security, benevolence, stimulation) rather than adult values. "
            "Format: {core_values_schwartz_top3: [exactly 3 from: self_direction, stimulation, "
            "hedonism, achievement, power, security, conformity, tradition, benevolence, universalism]}"
        ),
        description_zh="模拟开始时的价值观锚点（初态，Schwartz top-3）",
    ),
    "childhood_primary_residential_context": FieldSpec(
        name="childhood_primary_residential_context",
        type=FieldType.ENUM,
        allowed_values=["cities", "towns_and_suburbs", "rural_areas"],
        fallback=None,
        description="Childhood residential context (DEGURBA)",
        description_zh="童年居住环境类型",
    ),
    "childhood_living_arrangement": FieldSpec(
        name="childhood_living_arrangement",
        type=FieldType.ENUM,
        allowed_values=[
            "two_married_parents", "two_cohabiting_parents",
            "single_parent", "other",
        ],
        fallback=None,
        description="Childhood living arrangement (OECD)",
        description_zh="童年居住安排",
        should_infer=False,  # [SENSITIVE] Never auto-infer from persona_brief_text
    ),

    # ── Occupation & Education ──
    "target_occupation_group": FieldSpec(
        name="target_occupation_group",
        type=FieldType.ENUM,
        allowed_values=[
            "student", "manager_executive",
            "professional_finance_law_consulting", "professional_tech_research",
            "professional_health_education", "office_admin_support",
            "sales_customer_service", "service_hospitality_retail",
            "skilled_trades_technical_ops", "manual_logistics_transport",
            "public_service_military", "self_employed_creator",
            "retired", "unemployed", "homemaker",
        ],
        fallback=None,
        description="Simplified occupation group code",
        description_zh="职业分组编码",
    ),
    "target_education_level": FieldSpec(
        name="target_education_level",
        type=FieldType.ENUM,
        allowed_values=["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        fallback=None,
        description="ISCED 2011 education level code",
        description_zh="教育水平编码（ISCED 2011）",
    ),

    # ── Psychological & Values ──
    "self_system_anchor": FieldSpec(
        name="self_system_anchor",
        type=FieldType.STRUCT,
        fallback=None,
        description="Self-system value anchor",
        description_zh="自我系统价值锚点",
    ),
    "caregiver_bond_pbi": FieldSpec(
        name="caregiver_bond_pbi",
        type=FieldType.ENUM,
        allowed_values=[
            "optimal_parenting", "affectionate_constraint",
            "affectionless_control", "neglectful_parenting",
        ],
        fallback=None,
        description="Caregiver bond type (PBI)",
        description_zh="照料者依恋类型（PBI）",
        should_infer=False,  # [SENSITIVE] Never auto-infer from persona_brief_text
    ),
    "adult_attachment_rq4cat": FieldSpec(
        name="adult_attachment_rq4cat",
        type=FieldType.ENUM,
        allowed_values=["secure", "fearful", "preoccupied", "dismissing"],
        fallback=None,
        description="Adult attachment style (RQ 4-category)",
        description_zh="成人依恋风格（RQ四分类）",
        should_infer=False,  # [SENSITIVE] Never auto-infer from persona_brief_text
    ),
    "childhood_adversity_aceiq_13": FieldSpec(
        name="childhood_adversity_aceiq_13",
        type=FieldType.LIST_ENUM,
        element_allowed_values=[
            "parental_separation_or_divorce",
            "one_or_no_parents",
            "parental_substance_abuse",
            "parental_mental_illness",
            "parental_incarnation",
            "parental_death",
            "household_domestic_violence",
            "physical_abuse",
            "emotional_abuse",
            "sexual_abuse",
            "physical_neglect",
            "emotional_neglect",
            "bullying",
        ],
        fallback=[],
        description="Childhood adversity items (ACE-IQ 13)",
        description_zh="童年逆境项目（ACE-IQ 13）",
        should_infer=False,  # [SENSITIVE] Never auto-infer from persona_brief_text
    ),

}

# Processing order for Phase 2 per-field retry
# NOTE: Language fields (primary_language, working_language) removed — hardcoded in refine()
# NOTE: Sensitive fields (should_infer=False) ARE included here because:
#   - If user provides a MISSING value → not inferred (skipped by _scan_fields)
#   - If user provides an INVALID value → per-field LLM retry to fix it to a valid value
FIELD_PROCESSING_ORDER: List[str] = [
    # Round 0: Name
    "persona_name_text",
    # Round 1: Basic Identity
    "gender_identity_code",
    "target_age_exact",
    # Round 2: Geography & Background
    "growing_up_location",
    "current_living_location",
    "sim_start_living_location",
    "childhood_primary_residential_context",
    # Round 2b: Simulation-start state
    "sim_start_occupation_group",
    "sim_start_education_level",
    "sim_start_self_system_anchor",
    "childhood_living_arrangement",   # sensitive: only repaired if user provided invalid value
    # Round 3: Occupation & Education
    "target_occupation_group",
    "target_education_level",
    # Round 4: Psychological & Values
    "self_system_anchor",
    "caregiver_bond_pbi",             # sensitive: only repaired if user provided invalid value
    "adult_attachment_rq4cat",        # sensitive: only repaired if user provided invalid value
    "childhood_adversity_aceiq_13",   # sensitive: only repaired if user provided invalid value
]


# ═══════════════════════════════════════════════════════════════
# Pydantic Response Model for Phase 1
# ═══════════════════════════════════════════════════════════════

class RefinedPersonaConfig(BaseModel):
    """LLM structured output schema — all fields Optional.
    
    Note: Sensitive fields (childhood_living_arrangement, caregiver_bond_pbi,
    adult_attachment_rq4cat, childhood_adversity_aceiq_13) and language fields
    (primary_language, working_language) are intentionally excluded from this
    model. They are never auto-inferred by the LLM.
    """
    persona_name_text: Optional[str] = Field(
        None, description="Persona display name",
    )
    gender_identity_code: Optional[Literal[
        "1_male", "2_female", "3_non_binary", "4_other", "Z_not_stated"
    ]] = Field(None, description="Gender identity code")
    target_age_exact: Optional[int] = Field(
        None, description="Target age in years (0-120)", ge=0, le=120,
    )
    growing_up_location: Optional[LocationInfo] = Field(
        None, description="Where the persona primarily grew up",
    )
    current_living_location: Optional[LocationInfo] = Field(
        None, description="Residence at simulation END (target terminal state). Format: {country, province, city}",
    )
    sim_start_living_location: Optional[LocationInfo] = Field(
        None,
        description=(
            "Residence at simulation START. For birth-start simulations this is typically the "
            "growing-up location (hometown). May differ from current_living_location (end state). "
            "Format: {country, province, city}"
        ),
    )
    sim_start_occupation_group: Optional[str] = Field(
        None,
        description=(
            "Occupation category at simulation START. null if infant/toddler/not yet working. "
            "For birth-start simulations this is almost always null."
        ),
    )
    sim_start_education_level: Optional[str] = Field(
        None,
        description=(
            "Highest completed education level (ISCED 0-8) at simulation START. "
            "For birth-start simulations use '0'."
        ),
    )
    sim_start_self_system_anchor: Optional[SelfSystemAnchor] = Field(
        None,
        description=(
            "Core values (Schwartz top-3) at simulation START. "
            "For birth-start simulations, reflect basic temperament tendencies."
        ),
    )
    childhood_primary_residential_context: Optional[Literal[
        "cities", "towns_and_suburbs", "rural_areas"
    ]] = Field(None, description="Childhood residential context (DEGURBA)")
    target_occupation_group: Optional[Literal[
        "student", "manager_executive",
        "professional_finance_law_consulting", "professional_tech_research",
        "professional_health_education", "office_admin_support",
        "sales_customer_service", "service_hospitality_retail",
        "skilled_trades_technical_ops", "manual_logistics_transport",
        "public_service_military", "self_employed_creator",
        "retired", "unemployed", "homemaker",
    ]] = Field(None, description="Simplified occupation group code")
    target_education_level: Optional[Literal[
        "0", "1", "2", "3", "4", "5", "6", "7", "8"
    ]] = Field(None, description="ISCED 2011 education level")
    self_system_anchor: Optional[SelfSystemAnchor] = Field(
        None, description="Self-system anchor (core values Schwartz top 3)",
    )


# ═══════════════════════════════════════════════════════════════# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Pydantic Response Models for P0.6 Attitude Extraction
# ═══════════════════════════════════════════════════════════════

class SpecificAttitudeItem(BaseModel):
    topic: str = Field(
        ...,
        description="The topic or domain of the attitude (e.g. 'Native American history', 'nightclub culture')",
    )
    valence: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="Attitude valence toward the topic",
    )
    intensity: Literal["mild", "moderate", "strong"] = Field(
        ...,
        description="Intensity of the attitude",
    )
    evidence: str = Field(
        ...,
        description="The exact phrase from persona_brief_text that signals this attitude",
    )


class ExtractedAttitudes(BaseModel):
    specific_attitudes: List[SpecificAttitudeItem] = Field(
        default_factory=list,
        description="List of specific topic-directed attitudes explicitly stated in the persona brief",
    )


# ═══════════════════════════════════════════════════════════════
# Pydantic Response Models for P0.7 Identity Trait Extraction
# ═══════════════════════════════════════════════════════════════

class IdentityTraitItem(BaseModel):
    trait_category: Literal[
        "sexual_orientation",   # e.g., homosexual, bisexual, asexual
        "gender_identity",      # e.g., transgender, non-binary, genderqueer
        "religious_identity",   # e.g., Atheist, Muslim, devout Catholic, Jewish
        "racial_ethnic",        # e.g., Black, Latino, Asian-American, Indigenous
        "political_identity",   # e.g., conservative, progressive, libertarian
        "cultural_identity",    # e.g., immigrant, first-generation, diaspora
        "disability_identity",  # e.g., deaf, autistic, physically disabled
        "other_identity",       # any other stable identity descriptor
    ] = Field(..., description="Category of the identity trait")
    trait_value: str = Field(
        ...,
        description="The specific identity trait value (e.g., 'homosexual', 'Atheist', 'Black')",
    )
    salience: Literal["primary", "secondary"] = Field(
        ...,
        description="'primary' if explicitly named in the brief as a defining characteristic; 'secondary' if implied",
    )
    evidence: str = Field(
        ...,
        description="The exact phrase from persona_brief_text that signals this identity trait",
    )


class ExtractedIdentityTraits(BaseModel):
    identity_traits: List[IdentityTraitItem] = Field(
        default_factory=list,
        description="List of stable identity traits explicitly stated in the persona brief",
    )


# ═══════════════════════════════════════════════════════════════
# Pydantic Response Models for P0.8 Occupational Register Extraction
# ═══════════════════════════════════════════════════════════════

class OccupationalRegisterItem(BaseModel):
    register_type: Literal[
        "technical_jargon",       # domain-specific technical terms
        "professional_shorthand", # abbreviations and shorthand used in the field
        "workplace_culture",      # workplace-specific expressions and norms
        "social_register",        # class/education-level speech markers
        "cultural_speech",        # culturally-specific patterns (Southern US, working-class UK, ...)
    ] = Field(..., description="Type of speech register signal")
    description: str = Field(
        ...,
        description="Brief description of the speech register (e.g., 'uses trucking CB radio slang')",
    )
    example_expressions: List[str] = Field(
        default_factory=list,
        description="2-3 example expressions or vocabulary items typical of this register",
    )


class ExtractedOccupationalRegister(BaseModel):
    occupational_register: List[OccupationalRegisterItem] = Field(
        default_factory=list,
        description="List of occupation/background-specific speech register signals",
    )


def _validate_field_value(field_name: str, value: Any, spec: FieldSpec) -> Optional[FieldIssue]:
    """Validate a single field value against its spec. Returns None if valid."""
    if value is None:
        return FieldIssue(
            field_name=field_name,
            kind=IssueKind.MISSING,
            detail=f"Field '{field_name}' is None/missing",
            current_value=None,
        )

    if spec.type == FieldType.ENUM:
        if spec.allowed_values and str(value) not in spec.allowed_values:
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_ENUM,
                detail=f"Value '{value}' not in {spec.allowed_values}",
                current_value=value,
            )

    elif spec.type == FieldType.LIST_ENUM:
        if not isinstance(value, list):
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Expected list, got {type(value).__name__}",
                current_value=value,
            )
        if spec.element_allowed_values:
            for elem in value:
                if str(elem) not in spec.element_allowed_values:
                    return FieldIssue(
                        field_name=field_name,
                        kind=IssueKind.INVALID_ENUM,
                        detail=f"Element '{elem}' not in {spec.element_allowed_values}",
                        current_value=value,
                    )

    elif spec.type == FieldType.STRUCT:
        if not isinstance(value, dict):
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_STRUCT,
                detail=f"Expected dict, got {type(value).__name__}",
                current_value=value,
            )
        # Try Pydantic model construction for LocationInfo / SelfSystemAnchor
        try:
            if field_name in ("growing_up_location", "current_living_location", "sim_start_living_location"):
                LocationInfo(**value)
            elif field_name in ("self_system_anchor", "sim_start_self_system_anchor"):
                SelfSystemAnchor(**value)
        except Exception as e:
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_STRUCT,
                detail=f"Struct validation failed: {e}",
                current_value=value,
            )

    elif spec.type == FieldType.INT:
        if not isinstance(value, int):
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Expected int, got {type(value).__name__}",
                current_value=value,
            )
        if spec.int_min is not None and value < spec.int_min:
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Value {value} < min {spec.int_min}",
                current_value=value,
            )
        if spec.int_max is not None and value > spec.int_max:
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Value {value} > max {spec.int_max}",
                current_value=value,
            )

    elif spec.type == FieldType.DATE_STR:
        if not isinstance(value, str):
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Expected date string, got {type(value).__name__}",
                current_value=value,
            )
        # Basic format check
        try:
            parts = value.split("-")
            if len(parts) != 3:
                raise ValueError
            int(parts[0]); int(parts[1]); int(parts[2])
        except (ValueError, IndexError):
            return FieldIssue(
                field_name=field_name,
                kind=IssueKind.INVALID_TYPE,
                detail=f"Invalid date format: '{value}', expected YYYY-MM-DD",
                current_value=value,
            )

    return None  # valid


# ═══════════════════════════════════════════════════════════════
# Prompt Builders
# ═══════════════════════════════════════════════════════════════

ONE_SHOT_SYSTEM_PROMPT = (
    "You are a persona config auto-refinement system. "
    "Given a character profile and basic info, infer missing persona attributes. "
    "All enum fields MUST be selected from the provided allowed value set. "
    "Only output fields that need to be inferred; do not modify already-provided fields."
)

FIELD_INFERENCE_SYSTEM_PROMPT = (
    "You are a persona field inference assistant. "
    "Given the character's known info and a specific field to infer, "
    "output ONLY the field's value — no explanation, no markdown, no quotes. "
    "For enum fields, you MUST choose from the allowed values list."
)


def _build_one_shot_prompt(
    missing_fields: List[str],
    current_config: dict,
) -> str:
    """Build the Phase 1 one-shot prompt."""
    lines = ["## Known Information"]
    lines.append(f"- Name: {current_config.get('persona_name_text', 'Unknown')}")
    lines.append(f"- Brief: {current_config.get('persona_brief_text', '')}")

    # Provided valid fields
    provided_lines = []
    for fname, spec in FIELD_REGISTRY.items():
        if not spec.should_infer:
            continue  # Skip sensitive fields — never include in prompt
        if fname not in missing_fields and fname in current_config:
            val = current_config[fname]
            if val is not None:
                provided_lines.append(f"  - {fname}: {json.dumps(val, ensure_ascii=False)}")

    if provided_lines:
        lines.append("\n## Already-provided valid fields (do NOT modify these)")
        lines.extend(provided_lines)

    # Missing fields with descriptions (filtered: no sensitive fields)
    inferrable_missing = [
        f for f in missing_fields
        if FIELD_REGISTRY.get(f, FieldSpec(name="", type=FieldType.ENUM)).should_infer
    ]
    lines.append("\n## Fields to infer")
    for fname in inferrable_missing:
        spec = FIELD_REGISTRY.get(fname)
        if not spec:
            continue
        lines.append(f"\n**{fname}** ({spec.description})")
        lines.append(f"  Description: {spec.description}")
        if spec.allowed_values:
            lines.append(f"  Allowed values: {spec.allowed_values}")
        if spec.element_allowed_values:
            lines.append(f"  Element allowed values: {spec.element_allowed_values}")
        if spec.type == FieldType.STRUCT:
            if fname in ("growing_up_location", "current_living_location", "sim_start_living_location"):
                lines.append("  Format: {\"country\": \"...\", \"province\": \"...\", \"city\": \"...\"}")
            elif fname in ("self_system_anchor", "sim_start_self_system_anchor"):
                lines.append("  Format: {\"core_values_schwartz_top3\": [\"val1\", \"val2\", \"val3\"]}")
                lines.append(f"  Allowed core values: {SelfSystemAnchor.model_fields['core_values_schwartz_top3'].annotation}")
        if spec.int_min is not None:
            lines.append(f"  Range: {spec.int_min}-{spec.int_max}")

    lines.append("\n## Inference Rules")
    lines.append("1. All inferences must be consistent with the persona brief")
    lines.append("2. Enum fields MUST be selected from allowed values — never invent new values")
    lines.append("3. If insufficient info in brief, choose the most reasonable default")
    lines.append("4. If persona_name_text is missing, generate a culturally appropriate name consistent with the persona brief")
    lines.append("5. For gender_identity_code: Actively infer from available cues. "
             "Use the persona's name (many names are strongly gendered in their culture), "
             "occupation (some occupations have strong gender associations), "
             "and any gendered language in the brief (e.g., 'widower'->1_male, 'widow'->2_female). "
             "Do NOT default to Z_not_stated when reasonable gender cues exist -- "
             "Z_not_stated should only be used when the brief is genuinely gender-ambiguous "
             "(e.g., gender-neutral name + gender-neutral occupation + no gendered pronouns).")
    lines.append("6. Before assigning gender_identity_code, reason step-by-step: "
             "What is the cultural origin of the name? Is it a gendered name in that culture? "
             "Does the occupation or description contain gender cues? "
             "Only if all cues are genuinely ambiguous should you select Z_not_stated.")
    lines.append("7. For sim_start_living_location (residence at simulation START): "
             "This is where the persona BEGINS their life simulation, which is typically at birth or early childhood. "
             "For birth-start simulations, sim_start_living_location should be the hometown/growing-up location. "
             "It may differ from current_living_location (end state). "
             "If the persona grew up in City A but now lives in City B, "
             "sim_start_living_location = City A (hometown), current_living_location = City B (end state).")
    lines.append("8. For current_living_location (residence at simulation END): "
             "This is the persona's residence at the TARGET terminal state (simulation end). "
             "Infer from the persona brief — where does this person live as an adult/at their described life stage?")
    lines.append("9. For sim_start_occupation_group (occupation at simulation START): "
             "This is the persona's occupation at the very beginning of the simulation (typically birth or early childhood). "
             "For birth-start simulations, this is almost always null. "
             "Only set a non-null value if the simulation starts when the persona is already working/studying.")
    lines.append("10. For sim_start_education_level (education level at simulation START): "
             "This is the highest completed education level at the very beginning of the simulation. "
             "For birth-start simulations, use '0' (pre-primary/no education yet). "
             "ISCED: 0=pre-primary, 1=primary, 2=lower-secondary, 3=upper-secondary, "
             "4=post-secondary non-tertiary, 5=short-cycle tertiary, 6=bachelor, 7=master, 8=doctoral.")
    lines.append("11. For sim_start_self_system_anchor (core values at simulation START): "
             "This is the persona's Schwartz top-3 values at the very beginning of the simulation. "
             "For birth-start simulations, reflect basic temperament tendencies (e.g. security, benevolence, stimulation) "
             "rather than adult values like hedonism, achievement, or power. "
             "These may differ significantly from self_system_anchor (end-state values).")
    lines.append("")
    lines.append("## Schwartz Value Theory Reference (for self_system_anchor inference)")
    lines.append("")
    lines.append("The 10 basic values are organized in a circular structure with compatibility/conflict relationships:")
    lines.append("")
    lines.append("Compatible clusters (values that typically co-occur):")
    lines.append("- OPENNESS TO CHANGE: self_direction + stimulation + hedonism")
    lines.append("- SELF-ENHANCEMENT: achievement + power + hedonism")
    lines.append("- CONSERVATION: security + conformity + tradition")
    lines.append("- SELF-TRANSCENDENCE: benevolence + universalism")
    lines.append("")
    lines.append("Conflict relationships (values that rarely co-occur in the same person):")
    lines.append("- self_direction <-> conformity, tradition")
    lines.append("- stimulation <-> security, conformity")
    lines.append("- achievement <-> benevolence (can coexist but represent different orientations)")
    lines.append("- power <-> universalism")
    lines.append("")
    lines.append("When inferring core_values_schwartz_top3:")
    lines.append("1. First identify the persona's LIFE ORIENTATION: Is this person primarily")
    lines.append("   self-focused (achievement/power) or other-focused (benevolence/universalism)?")
    lines.append("   Open to change (self_direction/stimulation) or tradition-bound (conformity/tradition)?")
    lines.append("2. Select values that form a COHERENT cluster -- avoid contradictory combinations")
    lines.append("3. Consider the persona's OCCUPATION and LIFE EXPERIENCE as value indicators:")
    lines.append("   - Entrepreneurs -> achievement, self_direction, stimulation")
    lines.append("   - Healthcare workers -> benevolence, universalism")
    lines.append("   - Military/law enforcement -> security, conformity, tradition")
    lines.append("   - Artists/creators -> self_direction, stimulation, hedonism")
    lines.append("   - Factory owners/managers -> achievement, power, security")
    lines.append("4. The top 3 should include at most one value from each major cluster")
    lines.append("   (Openness, Enhancement, Conservation, Transcendence)")

    return "\n".join(lines)


def _build_field_prompt(
    field_name: str,
    field_spec: FieldSpec,
    current_config: dict,
    invalid_value: Any = None,
) -> str:
    """Build the Phase 2 per-field inference prompt.

    Works for both inferrable fields (should_infer=True) and sensitive fields
    (should_infer=False) when the user provided an invalid value that needs repair.
    For sensitive fields, the prompt makes clear this is a correction task, not
    a free inference — the LLM must pick the closest valid value.
    """
    lines = ["## Known persona info"]
    lines.append(f"- Name: {current_config.get('persona_name_text', 'Unknown')}")
    lines.append(f"- Brief: {current_config.get('persona_brief_text', '')}")

    # Context from already-known fields
    for fname in FIELD_PROCESSING_ORDER:
        if fname == field_name:
            break
        if fname in current_config and current_config[fname] is not None:
            val = current_config[fname]
            lines.append(f"- {fname}: {json.dumps(val, ensure_ascii=False)}")

    lines.append(f"\n## Field to infer")
    lines.append(f"**Field name**: {field_name}")
    lines.append(f"**Description**: {field_spec.description}")

    # For sensitive fields being repaired, add a correction note
    if not field_spec.should_infer and invalid_value is not None:
        lines.append(
            f"\n⚠️ The user provided '{invalid_value}' for this field, which is not a valid value. "
            f"Please select the closest valid value from the allowed list below."
        )

    if field_spec.allowed_values:
        lines.append("**Allowed values** (must choose one):")
        for v in field_spec.allowed_values:
            lines.append(f"  - {v}")

    if field_spec.element_allowed_values:
        lines.append("**Element allowed values** (each element must be from this list):")
        for v in field_spec.element_allowed_values:
            lines.append(f"  - {v}")

    if field_spec.type == FieldType.STRUCT:
        if field_name in ("growing_up_location", "current_living_location", "sim_start_living_location"):
            lines.append('**Format**: {"country": "...", "province": "...", "city": "..."}')
        elif field_name in ("self_system_anchor", "sim_start_self_system_anchor"):
            lines.append('**Format**: {"core_values_schwartz_top3": ["val1", "val2", "val3"]}')

    if field_spec.int_min is not None:
        lines.append(f"**Range**: {field_spec.int_min} to {field_spec.int_max}")

    lines.append("\n## Output format")
    lines.append("Output ONLY the field value, nothing else.")

    return "\n".join(lines)


def _parse_field_response(field_name: str, field_spec: FieldSpec, raw: str) -> Any:
    """Parse LLM text response into a field value."""
    raw = raw.strip()

    # Remove markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last ``` lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    if field_spec.type == FieldType.ENUM:
        # Try direct match
        if raw in (field_spec.allowed_values or []):
            return raw
        # Try stripping quotes
        cleaned = raw.strip("'\"")
        if cleaned in (field_spec.allowed_values or []):
            return cleaned
        return raw  # will be caught by validation

    elif field_spec.type == FieldType.LIST_ENUM:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try comma-separated
        return [v.strip().strip("'\"") for v in raw.split(",")]

    elif field_spec.type == FieldType.STRUCT:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw  # will be caught by validation

    elif field_spec.type == FieldType.INT:
        try:
            return int(raw)
        except ValueError:
            return raw

    elif field_spec.type == FieldType.DATE_STR:
        return raw

    return raw


# ═══════════════════════════════════════════════════════════════
# SampleRefiner — Main Class
# ═══════════════════════════════════════════════════════════════

class SampleRefiner:
    """
    P0.5: Validate and auto-refine persona config.

    Two-phase strategy:
      Phase 1: One-shot generate_structured (fast path)
      Phase 2: Field-by-field validation + per-field retry (safety net)
      Phase 3: Final full Participant validation
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: AsyncLLMClient instance for LLM calls.
        """
        self.llm = llm_client

    # ── Phase 0: Pre-Check ──

    def _validate_must_provide(self, config: dict) -> None:
        """Check that persona_brief_text (the only must-provide field) is present and non-empty."""
        val = config.get("persona_brief_text")
        if val is None or (isinstance(val, str) and not val.strip()):
            raise ValueError(
                "Missing required field: 'persona_brief_text'. "
                "This is the only field that MUST be provided in the input JSON."
            )

    def _scan_fields(self, config: dict) -> List[FieldIssue]:
        """Scan all refiner-managed fields for missing or invalid values.
        
        Fields with should_infer=False are only validated if the user
        explicitly provided a value; None is silently accepted (not treated
        as a MISSING issue that needs LLM inference).
        """
        issues = []
        for fname, spec in FIELD_REGISTRY.items():
            value = config.get(fname)

            if not spec.should_infer:
                # Sensitive field: only validate if user explicitly provided a value
                if value is not None:
                    issue = _validate_field_value(fname, value, spec)
                    if issue is not None:
                        issues.append(issue)
                # If None and should_infer=False: silently accept, no MISSING issue
                continue

            if value is None:
                issues.append(FieldIssue(
                    field_name=fname,
                    kind=IssueKind.MISSING,
                    detail=f"Field '{fname}' is missing",
                    current_value=None,
                ))
            else:
                issue = _validate_field_value(fname, value, spec)
                if issue is not None:
                    issues.append(issue)
        return issues

    # ── Phase 1: One-Shot Generation ──

    async def _phase1_one_shot_generate(
        self,
        missing_fields: List[str],
        current_config: dict,
    ) -> Optional[dict]:
        """
        Phase 1: One-shot LLM structured output.

        Returns:
            LLM inference result dict, or None if generate_structured fails entirely.
        """
        try:
            result = await self.llm.generate_structured(
                prompt=_build_one_shot_prompt(missing_fields, current_config),
                system_prompt=ONE_SHOT_SYSTEM_PROMPT,
                response_model=RefinedPersonaConfig,
                temperature=0.0,
                task_type="p0_refinement",
            )
            return result.model_dump(exclude_none=True)
        except Exception as e:
            logger.warning(
                f"[P0.5] Phase 1 generate_structured failed: {e}. "
                f"Will fall back to Phase 2 per-field retry."
            )
            return None

    def _merge_phase1_result(
        self,
        raw_config: dict,
        phase1_result: dict,
        issues: List[FieldIssue],
    ) -> dict:
        """Merge Phase 1 results, only filling fields that had issues."""
        config = dict(raw_config)
        fields_needing_fix = {i.field_name for i in issues}

        # Clear all fields with issues
        for issue in issues:
            config.pop(issue.field_name, None)

        # Fill with Phase 1 results
        for field, value in phase1_result.items():
            if field in fields_needing_fix and value is not None:
                config[field] = value

        return config

    # ── Phase 2: Field-by-Field Validation & Repair ──

    async def _phase2_validate_and_repair(
        self,
        config: dict,
        issues: List[FieldIssue],
    ) -> dict:
        """Phase 2: Validate Phase 1 outputs, per-field retry for invalid ones.

        Behaviour by field type and issue kind:

        Inferrable fields (should_infer=True):
          - MISSING → per-field LLM retry to infer a value
          - INVALID → per-field LLM retry to fix to a valid value

        Sensitive fields (should_infer=False):
          - MISSING (None) → leave as None, do NOT LLM-infer
          - INVALID (user provided a bad value) → per-field LLM retry to correct
            to the nearest valid value (the LLM sees the invalid value and picks
            the closest allowed one)
        """
        fields_to_check = {i.field_name for i in issues}
        # Map field_name → original invalid value (for prompt context)
        invalid_values: Dict[str, Any] = {
            i.field_name: i.current_value for i in issues
        }

        repair_list = []

        for field_name in fields_to_check:
            spec = FIELD_REGISTRY.get(field_name)
            value = config.get(field_name)

            if spec and not spec.should_infer:
                # Sensitive field:
                #   - If None (missing) → skip, do NOT infer
                #   - If invalid (non-None bad value) → add to repair_list
                if value is None:
                    logger.info(
                        f"[P0.5] Phase 2: Sensitive field '{field_name}' is missing — "
                        f"leaving as None (not inferred)."
                    )
                    continue
                else:
                    # User provided a value but it's invalid → repair it
                    repair_list.append(field_name)
                    logger.warning(
                        f"[P0.5] Phase 2: Sensitive field '{field_name}' has invalid value "
                        f"'{value}' — will LLM-repair to nearest valid value."
                    )
                continue

            # Inferrable field: check if still missing/invalid after Phase 1
            if value is None:
                repair_list.append(field_name)
                continue
            if spec and _validate_field_value(field_name, value, spec) is not None:
                repair_list.append(field_name)
                logger.warning(
                    f"[P0.5] Phase 2: Field '{field_name}' still invalid after Phase 1"
                )

        if not repair_list:
            logger.info("[P0.5] Phase 2: All fields valid. No per-field retry needed.")
            return config

        logger.info(
            f"[P0.5] Phase 2: {len(repair_list)} field(s) need per-field retry: {repair_list}"
        )

        for field_name in FIELD_PROCESSING_ORDER:
            if field_name in repair_list:
                fixed_value = await self._fix_single_field(
                    field_name=field_name,
                    field_spec=FIELD_REGISTRY[field_name],
                    current_config=config,
                    invalid_value=invalid_values.get(field_name),
                )
                config[field_name] = fixed_value

        return config

    async def _fix_single_field(
        self,
        field_name: str,
        field_spec: FieldSpec,
        current_config: dict,
        invalid_value: Any = None,
    ) -> Any:
        """Per-field retry until valid or exhausted retries → fallback.

        T39: Uses generate_structured instead of generate_text + _parse_field_response
        to avoid brittle string parsing.
        """
        prompt = _build_field_prompt(field_name, field_spec, current_config, invalid_value=invalid_value)

        last_invalid_value = None
        last_error_msg = None

        # Build a dynamic Pydantic model for structured output
        from pydantic import BaseModel as _BM, Field as _F
        import json as _json

        if field_spec.type == FieldType.ENUM and field_spec.allowed_values:
            allowed_str = ", ".join(f"'{v}'" for v in field_spec.allowed_values)

            class _FieldOutput(_BM):
                value: str = _F(
                    ...,
                    description=f"Must be one of: {allowed_str}"
                )
        elif field_spec.type == FieldType.LIST_ENUM and field_spec.element_allowed_values:
            allowed_str = ", ".join(f"'{v}'" for v in field_spec.element_allowed_values)

            class _FieldOutput(_BM):
                value: list = _F(
                    ...,
                    description=f"List of values, each must be one of: {allowed_str}"
                )
        elif field_spec.type == FieldType.INT:
            range_str = ""
            if field_spec.int_min is not None:
                range_str = f" Range: {field_spec.int_min} to {field_spec.int_max}."

            class _FieldOutput(_BM):
                value: int = _F(
                    ...,
                    description=f"Integer value.{range_str}"
                )
        elif field_spec.type == FieldType.DATE_STR:
            class _FieldOutput(_BM):
                value: str = _F(
                    ...,
                    description="Date string in YYYY-MM-DD format"
                )
        elif field_spec.type == FieldType.STRUCT:
            class _FieldOutput(_BM):
                value: dict = _F(
                    ...,
                    description=f"Structured value for {field_name}"
                )
        else:
            class _FieldOutput(_BM):
                value: Any = _F(
                    ...,
                    description=f"Value for {field_name}"
                )

        for attempt in range(1, MAX_RETRY_PER_FIELD + 1):
            retry_prompt = prompt
            if attempt > 1:
                retry_prompt += (
                    f"\n\n⚠️ Last answer '{last_invalid_value}' was invalid: {last_error_msg}\n"
                    f"Please choose again from the allowed values."
                )

            try:
                result: _FieldOutput = await self.llm.generate_structured(
                    prompt=retry_prompt,
                    response_model=_FieldOutput,
                    system_prompt=FIELD_INFERENCE_SYSTEM_PROMPT,
                    temperature=0.0,
                    task_type="p0_refinement",
                )
                parsed_value = result.value
            except Exception as e:
                logger.warning(
                    f"[P0.5] Phase 2: LLM call failed for '{field_name}' attempt {attempt}: {e}"
                )
                continue

            issue = _validate_field_value(field_name, parsed_value, field_spec)
            if issue is None:
                logger.info(
                    f"[P0.5] Phase 2: Field '{field_name}' fixed: {parsed_value} "
                    f"(attempt {attempt}/{MAX_RETRY_PER_FIELD})"
                )
                return parsed_value
            else:
                last_invalid_value = parsed_value
                last_error_msg = issue.detail
                logger.warning(
                    f"[P0.5] Phase 2: Field '{field_name}' invalid on attempt {attempt}: "
                    f"{parsed_value} → {issue.detail}"
                )

        # Exhausted retries → fallback
        fallback = field_spec.fallback
        if fallback is not None:
            logger.warning(
                f"[P0.5] Phase 2: Field '{field_name}' exhausted retries. Using fallback: {fallback}"
            )
            return fallback
        else:
            logger.warning(
                f"[P0.5] Phase 2: Field '{field_name}' exhausted retries. No fallback — None."
            )
            return None

    # ── Phase 3: Final Full Validation ──

    def _phase3_final_validation(self, config: dict) -> None:
        """Validate using P1 Participant model. Raises on failure.
        
        Sensitive fields (should_infer=False) are excluded from validation
        because they may contain user-provided values that are not in the
        allowed set (e.g. legacy data), and the refiner is not responsible
        for fixing them.
        """
        try:
            test_payload = {
                "participant_id": "P_VALIDATION_TEST",
                "role": "target",
                "persona_name_text": config.get("persona_name_text", ""),
                "persona_brief_text": config.get("persona_brief_text", ""),
            }
            # Only include fields that should be inferred (skip sensitive fields)
            for field_name, spec in FIELD_REGISTRY.items():
                if not spec.should_infer:
                    continue  # Skip sensitive fields in final validation
                if field_name in config and config[field_name] is not None:
                    test_payload[field_name] = config[field_name]
            Participant(**test_payload)
            logger.info("[P0.5] Phase 3: Final validation passed.")
        except Exception as e:
            logger.error(f"[P0.5] Phase 3: Final validation FAILED: {e}")
            raise

    # ── P0.6: Specific Attitude Extraction ──

    async def _extract_specific_attitudes(self, persona_brief: str) -> "Optional[List[dict]]":
        """P0.6: Extract specific topic-directed attitudes from persona_brief_text.

        Only extracts attitudes that are CLEARLY signaled by the brief text.
        Returns a list of dicts, or None if extraction fails / no attitudes found.
        """
        prompt = (
            "## Task\n"
            "Extract specific attitudes explicitly stated or strongly implied in the persona brief below.\n"
            "Only extract attitudes that are CLEARLY signaled — do NOT invent attitudes not present in the text.\n\n"
            f"## Persona Brief\n{persona_brief}\n\n"
            "## Extraction Rules\n"
            "1. Each attitude MUST have a direct textual anchor in the brief — quote it verbatim in the 'evidence' field.\n"
            '2. \"hates X\" → valence=negative, intensity=strong\n'
            '3. \"loves X\" / \"enjoys X\" → valence=positive, intensity=moderate or strong\n'
            '4. \"spends weekends at X\" / \"obsessed with X\" → valence=positive, intensity=moderate (behavioral signal)\n'
            '5. \"dislikes X\" / \"avoids X\" → valence=negative, intensity=mild or moderate\n'
            "6. General personality descriptors (e.g. 'shallow-minded', 'introverted') are NOT topic-directed attitudes — skip them.\n"
            "7. Output an EMPTY list if no specific topic-directed attitudes are found.\n"
            "8. Do NOT infer attitudes beyond what is explicitly stated."
        )
        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=(
                    "You are a persona attribute extractor. "
                    "Extract only topic-directed attitudes that are explicitly stated in the brief. "
                    "Never invent or infer attitudes beyond what the text clearly signals."
                ),
                response_model=ExtractedAttitudes,
                temperature=0.0,
                task_type="p0_refinement",
            )
            attitudes = [item.model_dump() for item in result.specific_attitudes]
            logger.info(
                f"[P0.6] Extracted {len(attitudes)} specific attitude(s) from persona brief."
            )
            return attitudes if attitudes else None
        except Exception as e:
            logger.warning(f"[P0.6] specific_attitudes extraction failed: {e}")
            return None

    # ── P0.7: Identity Trait Extraction ──

    async def _extract_identity_traits(self, persona_brief: str) -> "Optional[List[dict]]":
        """P0.7: Extract stable identity traits from persona_brief_text.

        Identity traits describe WHO the person IS (not what they like/dislike),
        complementing P0.6 which captures topic-directed attitudes. Returned list
        is written into persona_extensions.identity_traits (see refine()).
        """
        prompt = (
            "## Task\n"
            "Extract stable identity traits explicitly stated in the persona brief below.\n"
            "Identity traits describe WHO the person IS (not what they like/dislike).\n\n"
            f"## Persona Brief\n{persona_brief}\n\n"
            "## Extraction Rules\n"
            "1. Each trait MUST have a direct textual anchor in the brief — quote it verbatim in 'evidence'.\n"
            "2. Extract ONLY stable, defining identity characteristics:\n"
            "   - Sexual orientation: 'homosexual', 'gay', 'lesbian', 'bisexual', 'asexual'\n"
            "   - Gender identity: 'transgender', 'non-binary', 'genderqueer'\n"
            "   - Religious identity: 'Atheist', 'Muslim', 'Catholic', 'Jewish', 'Buddhist', 'agnostic'\n"
            "   - Racial/ethnic identity: 'Black', 'Latino', 'Asian-American', 'Indigenous', 'White'\n"
            "   - Political identity: 'conservative', 'progressive', 'libertarian'\n"
            "   - Cultural identity: 'immigrant', 'first-generation', 'diaspora'\n"
            "   - Disability identity: 'deaf', 'autistic', 'physically disabled', 'blind'\n"
            "3. Do NOT extract:\n"
            "   - Occupation, age, location (handled elsewhere)\n"
            "   - Personality traits ('shallow-minded', 'introverted')\n"
            "   - Topic-directed attitudes ('hates X', 'loves Y')\n"
            "4. Output an EMPTY list if no identity traits are found.\n"
            "5. Do NOT infer traits beyond what is explicitly stated."
        )
        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=(
                    "You are a persona attribute extractor. "
                    "Extract only stable identity traits that are explicitly stated in the brief. "
                    "Never invent or infer traits beyond what the text clearly signals."
                ),
                response_model=ExtractedIdentityTraits,
                temperature=0.0,
                task_type="p0_refinement",
            )
            traits = [item.model_dump() for item in result.identity_traits]
            logger.info(
                f"[P0.7] Extracted {len(traits)} identity trait(s) from persona brief."
            )
            return traits if traits else None
        except Exception as e:
            logger.warning(f"[P0.7] identity_traits extraction failed: {e}")
            return None

    # ── P0.8: Occupational Register Extraction ──

    async def _extract_occupational_register(self, persona_brief: str) -> "Optional[List[dict]]":
        """P0.8: Extract occupation-specific and background-specific speech-register signals.

        Captures HOW the person speaks (vocabulary, jargon, cadence), not what they
        believe. Feeds P3 simulation and M2P style sections.
        """
        prompt = (
            "## Task\n"
            "Extract occupation-specific and background-specific speech register signals "
            "from the persona brief below.\n"
            "These signals describe HOW the person speaks, not what they believe.\n\n"
            f"## Persona Brief\n{persona_brief}\n\n"
            "## Extraction Rules\n"
            "1. Extract signals ONLY when occupation or background strongly implies a distinctive register:\n"
            "   - Blue-collar workers (truck driver, factory worker, farmer): working-class speech, trade jargon\n"
            "   - Religious practitioners (monk, imam, priest): religious terminology, contemplative register\n"
            "   - Medical professionals: clinical terminology\n"
            "   - Chefs/culinary: culinary terminology, kitchen culture\n"
            "   - Artists/musicians: creative vocabulary, scene-specific slang\n"
            "   - Athletes/coaches: sports terminology, motivational language\n"
            "   - Academics/researchers: academic register, hedged language\n"
            "2. Include 2-3 example expressions typical of this register.\n"
            "3. Do NOT extract for generic white-collar professionals (engineer, teacher, scientist) "
            "unless the brief specifies a distinctive cultural background.\n"
            "4. Output an EMPTY list if no distinctive register is implied.\n"
            "5. Do NOT invent expressions — only suggest plausible examples."
        )
        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                system_prompt=(
                    "You are a sociolinguistics expert. Extract only occupation-specific "
                    "speech register signals clearly implied by the brief. Focus on "
                    "distinctive registers that differ from standard educated speech."
                ),
                response_model=ExtractedOccupationalRegister,
                temperature=0.0,
                task_type="p0_refinement",
            )
            registers = [item.model_dump() for item in result.occupational_register]
            logger.info(
                f"[P0.8] Extracted {len(registers)} occupational register signal(s) from persona brief."
            )
            return registers if registers else None
        except Exception as e:
            logger.warning(f"[P0.8] occupational_register extraction failed: {e}")
            return None

        # ── Main Entry Point ──

    async def refine(self, raw_config: dict) -> dict:
        """
        Refine a persona config: validate, auto-fill, and return a complete config.

        Args:
            raw_config: The raw persona config dict (from JSON).

        Returns:
            Refined config dict with all fields validated/filled.
        """
        config = dict(raw_config)

        # ── Defensive SimulatorArena bypass (LLM-classified) ──
        # SimulatorArena personas have no demographic anchors. Running the full P0
        # inference would hallucinate values. The supported path for SimulatorArena
        # is memory_to_prompt/run_m2p_simulator_arena.py. We DELEGATE the format
        # decision to an LLM classifier to avoid brittle string heuristics.
        try:
            from memory_to_prompt.simulator_arena_adapter import classify_persona_format
            _fmt = await classify_persona_format(self.llm, config)
            _is_simulator_arena = (_fmt.format == "simulator_arena")
        except Exception as e:
            logger.warning(
                f"[P0] SimulatorArena format classification failed: {e}; proceeding with full refine."
            )
            _is_simulator_arena = False
        if _is_simulator_arena:
            logger.warning(
                "[P0] SimulatorArena-shaped persona routed to full refine(); returning early. "
                "Use memory_to_prompt/run_m2p_simulator_arena.py instead."
            )
            config["_bypass_simulation"] = True
            config.setdefault("primary_language", "en")
            config.setdefault("working_language", ["en"])
            return config

        # ── Harmonize extra_context → persona_extensions ──
        # persona_to_json outputs "extra_context"; the pipeline uses "persona_extensions"
        if "extra_context" in config and "persona_extensions" not in config:
            config["persona_extensions"] = config.pop("extra_context")
        elif "extra_context" in config and "persona_extensions" in config:
            # If both exist, merge extra_context into persona_extensions
            ext = config.pop("extra_context")
            for k, v in ext.items():
                if k not in config["persona_extensions"]:
                    config["persona_extensions"][k] = v

        # ── Extract persona_extensions before validation (it's not in FIELD_REGISTRY) ──
        extensions = config.pop("persona_extensions", None)

        # ── Hardcode language fields for English-only simulation ──
        config["primary_language"] = "en"
        config["working_language"] = ["en"]

        # ── Phase 0: Pre-Check ──
        self._validate_must_provide(config)
        issues = self._scan_fields(config)

        if not issues:
            logger.info("[P0.5] All fields valid. No LLM calls needed.")
            # Re-insert persona_extensions
            if extensions is not None:
                config["persona_extensions"] = extensions
            return config

        logger.info(
            f"[P0.5] Found {len(issues)} field(s) to fix: "
            f"{[i.field_name for i in issues]}"
        )

        # ── Phase 1: One-Shot Generation ──
        phase1_result = await self._phase1_one_shot_generate(
            missing_fields=[i.field_name for i in issues],
            current_config=config,
        )

        if phase1_result is not None:
            config = self._merge_phase1_result(config, phase1_result, issues)
            logger.info("[P0.5] Phase 1 complete. Checking Phase 1 output validity...")
        else:
            logger.warning("[P0.5] Phase 1 failed entirely. All issues go to Phase 2.")

        # ── Phase 2: Field-by-Field Validation & Repair ──
        config = await self._phase2_validate_and_repair(config, issues)

        # ── Phase 3: Final Full Validation ──
        self._phase3_final_validation(config)

        # ── Re-insert persona_extensions after validation ──
        if extensions is not None:
            config["persona_extensions"] = extensions

        # ── P0.6: Extract specific_attitudes into persona_extensions ──
        persona_brief = config.get("persona_brief_text", "")
        if persona_brief:
            attitudes = await self._extract_specific_attitudes(persona_brief)
            if attitudes:
                if "persona_extensions" not in config or config["persona_extensions"] is None:
                    config["persona_extensions"] = {}
                config["persona_extensions"]["specific_attitudes"] = {
                    "_display_name": "Specific Attitudes",
                    "_render_hint": "named_items",
                    "_domain": "personality",
                    "_item_name_key": "topic",
                    "_item_value_key": "valence",
                    "items": attitudes,
                }

        # ── P0.7: Extract identity_traits into persona_extensions ──
        if persona_brief:
            identity_traits = await self._extract_identity_traits(persona_brief)
            if identity_traits:
                if "persona_extensions" not in config or config["persona_extensions"] is None:
                    config["persona_extensions"] = {}
                config["persona_extensions"]["identity_traits"] = {
                    "_display_name": "Core Identity Traits",
                    "_render_hint": "named_items",
                    "_domain": "personality",
                    "_item_name_key": "trait_category",
                    "_item_value_key": "trait_value",
                    "items": identity_traits,
                }

        # ── P0.8: Extract occupational_register into persona_extensions ──
        if persona_brief:
            occ_register = await self._extract_occupational_register(persona_brief)
            if occ_register:
                if "persona_extensions" not in config or config["persona_extensions"] is None:
                    config["persona_extensions"] = {}
                config["persona_extensions"]["occupational_register"] = {
                    "_display_name": "Occupational Speech Register",
                    "_render_hint": "named_items",
                    "_domain": "communication",
                    "_item_name_key": "register_type",
                    "_item_value_key": "description",
                    "items": occ_register,
                }

        # ── Post-inference validation: education level consistency ──
        config = self._post_validate_education_level(config)

        return config

    def _post_validate_education_level(self, config: dict) -> dict:
        """Post-inference validation: ensure target_education_level is consistent
        with education-related keywords in persona_brief_text.

        Key rules:
        - "college dropout" / "university dropout" / "dropped out of college" →
          minimum level "3" (upper secondary, i.e. completed high school to enter college)
        - "high school dropout" / "dropped out of high school" →
          maximum level "2" (lower secondary completed)
        - "college graduate" / "university graduate" / "bachelor" →
          minimum level "6"
        - "master" / "graduate degree" → minimum level "7"
        - "PhD" / "doctoral" → minimum level "8"
        """
        brief = config.get("persona_brief_text", "").lower()
        edu = config.get("target_education_level")
        if edu is None:
            return config

        edu_int = int(edu) if str(edu).isdigit() else -1

        # College/university dropout → must have completed high school (≥ "3")
        college_dropout_patterns = [
            "college dropout", "university dropout", "dropped out of college",
            "dropped out of university", "college drop-out", "university drop-out",
        ]
        if any(p in brief for p in college_dropout_patterns):
            if edu_int < 3:
                logger.warning(
                    f"[P0.5 post-validate] target_education_level='{edu}' is inconsistent with "
                    f"'college dropout' in persona_brief_text. Correcting to '3' (upper secondary)."
                )
                config["target_education_level"] = "3"
            return config

        # High school dropout → must not exceed lower secondary ("2")
        hs_dropout_patterns = [
            "high school dropout", "dropped out of high school",
            "high-school dropout", "high school drop-out",
        ]
        if any(p in brief for p in hs_dropout_patterns):
            if edu_int > 2:
                logger.warning(
                    f"[P0.5 post-validate] target_education_level='{edu}' is inconsistent with "
                    f"'high school dropout' in persona_brief_text. Correcting to '2'."
                )
                config["target_education_level"] = "2"
            return config

        # College/university graduate → minimum "6"
        college_grad_patterns = [
            "college graduate", "university graduate", "bachelor's degree",
            "bachelor degree", "graduated from college", "graduated from university",
        ]
        if any(p in brief for p in college_grad_patterns):
            if edu_int < 6:
                logger.warning(
                    f"[P0.5 post-validate] target_education_level='{edu}' is inconsistent with "
                    f"college graduate in persona_brief_text. Correcting to '6'."
                )
                config["target_education_level"] = "6"
            return config

        return config
