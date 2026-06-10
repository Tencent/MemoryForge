from typing import TypedDict, List, Dict, Literal, Union, Optional, Any

# ==========================================
# Part 1: Type Hints
# ==========================================

# 1. Self-System Anchor internal structure
class SelfSystemAnchor(TypedDict, total=False):
    core_values_schwartz_top3: List[Literal[
        "self_direction", "stimulation", "hedonism", "achievement",
        "power", "security", "conformity", "tradition", "benevolence", "universalism"
    ]]

# 2. Location structure (country + optional province/city)
class LocationInfo(TypedDict, total=False):
    country: str  # Country code or name, e.g. 'china', 'uk', 'us', 'japan', etc.
    province: Optional[str]
    city: Optional[str]

# 3. Full input schema type definition
class PersonaInputSchema(TypedDict, total=False):
    # --- [Must-Provide] Only persona_brief_text is truly required by P0.5 ---
    persona_brief_text: Optional[str]                                   # Persona brief description (REQUIRED)

    # --- [Auto-Inferral] All other fields are optional — P0.5 SampleRefiner auto-infers if missing/invalid ---
    primary_language: Optional[Literal["zh", "en"]]                     # Primary language code
    working_language: Optional[List[Literal["zh", "en"]]]               # Working language code list
    persona_name_text: Optional[str]                                    # Persona name
    target_age_exact: int
    growing_up_location: LocationInfo                                    # Where the agent primarily grew up
    current_living_location: LocationInfo                                # Current city of residence
    target_occupation_group: Literal[                                    # Simplified occupation group code
        "student", "manager_executive",
        "professional_finance_law_consulting", "professional_tech_research",
        "professional_health_education", "office_admin_support",
        "sales_customer_service", "service_hospitality_retail",
        "skilled_trades_technical_ops", "manual_logistics_transport",
        "public_service_military", "self_employed_creator",
        "retired", "unemployed", "homemaker"
    ]
    self_system_anchor: SelfSystemAnchor
    gender_identity_code: Optional[Literal["1_male", "2_female", "3_non_binary", "4_other", "Z_not_stated"]]
    target_education_level: Optional[Literal[
        "0", "1", "2", "3", "4", "5", "6", "7", "8"
    ]]
    childhood_primary_residential_context: Optional[Literal["cities", "towns_and_suburbs", "rural_areas"]]
    childhood_living_arrangement: Optional[Literal["two_married_parents", "two_cohabiting_parents", "single_parent", "other"]]
    caregiver_bond_pbi: Optional[Literal["optimal_parenting", "affectionate_constraint", "affectionless_control", "neglectful_parenting"]]
    childhood_adversity_aceiq_13: Optional[List[str]]
    adult_attachment_rq4cat: Optional[Literal["secure", "fearful", "preoccupied", "dismissing"]]
    current_job_tenure_months: Optional[int]
    pre_current_role_gap_months: Optional[int]
    post_graduation_job_search_months: Optional[int]
    birth_date: Optional[str]                                            # YYYY-MM-DD
    simulation_end_date: Optional[str]                                   # YYYY-MM-DD
    persona_extensions: Optional[Dict[str, Any]]                         # Benchmark-specific extension attributes (preserved as-is, not validated)


# ==========================================
# Part 2: Instance template (for runtime construction)
# ==========================================

persona_input_template: PersonaInputSchema = {
    # ---------------------------------------------------------
    # 🔴 Category 1: Required
    # ---------------------------------------------------------
    "target_age_exact": 35,
    "growing_up_location": {                                             # Where the agent primarily grew up
        "country": "china",
        "province": "广西",
        "city": "南宁",
    },
    "current_living_location": {                                         # Current city of residence
        "country": "china",
        "province": "广东",
        "city": "深圳",
    },
    "target_occupation_group": "professional_tech_research",             # Simplified occupation group code
    
    "self_system_anchor": {
        # Schwartz core values Top 3 (must contain exactly 3 values)
        "core_values_schwartz_top3": ["self_direction", "benevolence", "security"]
    },
    "primary_language": "en",                                             # Primary language code (hardcoded "en" for English simulation)
    "working_language": ["en"],                                          # Working language code list (hardcoded ["en"] for English simulation)

    # ---------------------------------------------------------
    # 🟡 Category 2: Inferral (auto-inferred if not specified)
    # ---------------------------------------------------------
    "target_education_level": "7",                                       # ISCED 2011 level code (0-8)
    "childhood_primary_residential_context": "cities",                   # cities / towns_and_suburbs / rural_areas
    "childhood_living_arrangement": "other",                             # OECD living arrangement

    # ---------------------------------------------------------
    # 🟢 Category 3: Optional (sensitive, should not be hard-inferred)
    # ---------------------------------------------------------
    "gender_identity_code": "1_male",
    "caregiver_bond_pbi": "affectionless_control",
    "childhood_adversity_aceiq_13": [
        "one_or_no_parents_or_parental_separation_divorce"
    ],
    "adult_attachment_rq4cat": "fearful",
    "current_job_tenure_months": 4,
    "pre_current_role_gap_months": 0,
    "post_graduation_job_search_months": 0,
    "birth_date": "1998-03-25",
    "simulation_end_date": "2026-03-24",

    # ---------------------------------------------------------
    # 🔵 Category 4: Free-text
    # ---------------------------------------------------------
    "persona_name_text": "Ethan Carter",
    "persona_brief_text": "A guarded but highly responsible man..."
}