"""Data models for year-level key life path enrichment cache.

This module defines the per-year structure that expands the cross-year
inferred_persona_pathway into deterministic, year-level life path anchors
for P2/P3 prompt injection.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class EducationStageType(str, Enum):
    """Education stage type labels."""
    PRESCHOOL = "preschool"
    PRIMARY = "primary"
    JUNIOR_SECONDARY = "junior_secondary"
    SENIOR_SECONDARY = "senior_secondary"
    UNDERGRADUATE = "undergraduate"
    MASTER = "master"
    DOCTORAL = "doctoral"
    POSTDOC = "postdoc"
    VOCATIONAL = "vocational"
    GAP = "gap"
    NONE = "none"


class ExamEntry(BaseModel):
    """A single exam entry for the year."""
    exam_name: str = Field(
        ..., description="Exam name, e.g. 'College Entrance Exam', 'Graduate Entrance Exam', 'Doctoral Qualifying Exam'"
    )
    date_range: str = Field(
        ..., description="Exam date or date range, e.g. '2016-06-07~08', '2020-12-26~27'"
    )
    description: str = Field(
        ...,
        description="Brief exam description, 1 sentence. MUST NOT exceed 100 Chinese characters."
    )


class KeyEventEntry(BaseModel):
    """A key life event for the year."""
    event_name: str = Field(
        ..., description="Event name, e.g. 'High School Graduation', 'Joined XX Company', 'Got Married'"
    )
    date: str = Field(
        ..., description="Event date or approximate time, e.g. '2016-06', '2016-09-01'"
    )
    description: str = Field(
        ...,
        description="Brief event description, 1 sentence. MUST NOT exceed 80 Chinese characters."
    )
    is_stage_transition: bool = Field(
        default=False, description="Whether this is a stage transition event"
    )


class EducationStatus(BaseModel):
    """Education status for a specific year."""
    stage_type: EducationStageType = Field(
        ...,
        description=(
            "Education stage type. Must be one of: "
            "'preschool' (kindergarten/preschool, age ~3-6), "
            "'primary' (primary/elementary school, age ~6-12), "
            "'junior_secondary' (middle school/junior high, age ~12-15), "
            "'senior_secondary' (high school/senior secondary, age ~15-18), "
            "'undergraduate' (university/college), "
            "'master' (master's program), "
            "'doctoral' (PhD program), "
            "'postdoc' (postdoctoral), "
            "'vocational' (vocational training), "
            "'gap' (gap year/between stages), "
            "'none' (NOT enrolled in any formal education — use ONLY if genuinely not in school). "
            "Do NOT default to 'none' unless the person is genuinely not enrolled."
        )
    )
    grade_label: str = Field(
        ...,
        description="Grade label, e.g. 'Freshman', '3rd year PhD', 'Senior year'. MUST NOT exceed 30 characters."
    )
    institution: str = Field(
        default="", description="School/institution name"
    )
    research_direction: Optional[str] = Field(
        default=None,
        description="Research direction (doctoral/research stages only). MUST NOT exceed 100 characters."
    )


class CareerStatus(BaseModel):
    """Career status for a specific year."""
    is_working: bool = Field(default=False, description="Whether working")
    job_title: str = Field(default="", description="Job title")
    employer: str = Field(default="", description="Employer/company name")
    industry: str = Field(default="", description="Industry")
    description: str = Field(
        default="",
        description="Career status description. MUST NOT exceed 150 Chinese characters."
    )


class YearKeyLifePath(BaseModel):
    """Year-level key life path data for enrichment injection.

    This is the per-year expansion of inferred_persona_pathway,
    providing deterministic life path anchors for P2/P3 prompt injection.

    All text fields have character limits enforced in the generation prompt
    and Pydantic descriptions — no truncation at usage time.
    """
    year: int = Field(..., description="Year")
    age: int = Field(..., description="Age in this year")

    education_status: EducationStatus = Field(
        ..., description="Education status for this year"
    )
    exams_this_year: List[ExamEntry] = Field(
        default_factory=list, description="List of exams for this year"
    )
    key_events: List[KeyEventEntry] = Field(
        default_factory=list, description="List of key life events for this year"
    )
    career_status: CareerStatus = Field(
        default_factory=CareerStatus, description="Career status for this year"
    )
    pathway_notes: str = Field(
        default="",
        description="Pathway notes: deviations from standard path or special circumstances. "
        "MUST NOT exceed 150 Chinese characters."
    )
