"""Write-Time Pipeline — Pre-compute retrieval fields at event storage time.

Design philosophy (v6 / STARE-E v2):
  - All LLM calls happen HERE (write time), never during retrieval
  - Every operation has a rule-based fallback (zero LLM cost mode)
  - Summaries are SCENE-ORIENTED: different fields for different use cases
  - Extractive summarization replaces simple truncation
  - ensure_length_budget() provides graduated compression as safety net
  - 🆕 v2: Embedding vectors computed at write time using local model
  - 🆕 v2: Causal parent IDs computed using structural heuristics + embedding
  - Inspired by Concordia's AssociativeMemoryBank.add() pattern

Usage:
  pipeline = WriteTimePipeline(llm_client=llm, embedding_engine=emb)
  record = await pipeline.enrich_event(record, event_type="milestone")
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_event_id(eid: str) -> Tuple[int, int]:
    """Parse 'LP{lp}_E{seq}' into (lp_num, seq_num) for correct numeric ordering.

    Falls back to (0, 0) for unrecognised formats so comparisons degrade
    gracefully rather than raising.

    Examples:
        'LP2_E001'  -> (2, 1)
        'LP10_E003' -> (10, 3)
    """
    try:
        lp_part, e_part = eid.split("_E", 1)
        lp_num = int(lp_part.replace("LP", "").replace("lp", ""))
        e_num = int(e_part)
        return (lp_num, e_num)
    except (ValueError, AttributeError, IndexError):
        return (0, 0)


def _event_id_lt(a: str, b: str) -> bool:
    """Return True if event *a* was written strictly before event *b*."""
    return _parse_event_id(a) < _parse_event_id(b)


def _event_id_le(a: str, b: str) -> bool:
    """Return True if event *a* was written at the same time as or before *b*."""
    return _parse_event_id(a) <= _parse_event_id(b)

class WriteTimePipeline:
    """Pre-compute all retrieval-relevant fields at write time."""

    def __init__(self, llm_client=None, embedding_engine=None):
        """
        Args:
            llm_client: Optional async LLM client for summary generation.
                        If None, only rule-based processing is used.
            embedding_engine: Optional EmbeddingEngine for computing
                        embedding vectors at write time. If None, embedding
                        fields are left empty.
        """
        self._llm = llm_client
        self._emb = embedding_engine

    # ── Importance Scoring (Rule-Based, Zero LLM) ───────────────

    EVENT_TYPE_WEIGHTS = {
        "milestone": 0.90,      # graduation, marriage, birth
        "career": 0.80,         # job change, promotion
        "academic": 0.75,       # exam, enrollment
        "family": 0.70,         # family events
        "health": 0.70,         # illness, recovery
        "social": 0.55,         # friendship, social activities
        "daily_routine": 0.30,  # eating, commuting
    }

    def compute_importance(
        self,
        summary: str,
        event_type: str = "",
        resolution_level: str = "low",
        participants: Optional[List[str]] = None,
    ) -> float:
        """Compute importance score using deterministic rules.

        Formula:
          score = base(event_type) + resolution_boost + participant_bonus + length_bonus

        Resolution hierarchy: low-res < outline < high-res
          - low-res:  period-level summary, covers all participants by default → no boost
          - outline:  hand-picked key moment from within a low-res period → moderate boost
          - high-res: P3-simulated with full interaction sequence → highest boost

        Participant bonus only applies to outline/high-res events because low-res
        events naturally list ALL participants in the period (not a signal of importance).

        Returns:
            Importance score in [0.1, 1.0]
        """
        base = self.EVENT_TYPE_WEIGHTS.get(event_type.lower(), 0.50) if event_type else 0.50
        # v9: resolution boost reflects information density, not just simulation depth.
        # outline > low-res because outline events are explicitly selected as important moments.
        RESOLUTION_BOOST = {
            "high": 0.20,   # P3 simulated, full interaction sequence — highest fidelity
            "outline": 0.12,            # Explicitly selected key moment within a period — above low-res
            "low": 0.0,      # Period-level summary only — baseline, no boost
        }
        resolution_boost = RESOLUTION_BOOST.get(resolution_level, 0.0)
        # Participant bonus: only meaningful for outline/high-res where participants are
        # specifically chosen for the event. low-res lists all period participants by default,
        # so counting them would unfairly inflate low-res scores.
        if resolution_level in ("outline", "high"):
            n = len(participants) if participants else 0
            participant_bonus = min(n * 0.03, 0.10)
        else:
            participant_bonus = 0.0
        length_bonus = min(len(summary) / 2000, 0.05)
        score = base + resolution_boost + participant_bonus + length_bonus
        return round(max(0.1, min(score, 1.0)), 3)

    # ── Sentence Splitting Utility ──────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences at Chinese/English sentence boundaries."""
        if not text:
            return []
        # Split at sentence-ending punctuation, keeping the delimiter
        parts = re.split(r'(?<=[。！？；.!?;])', text)
        return [p.strip() for p in parts if p.strip()]

    # ── Extractive Summarization ────────────────────────────────

    # Causal markers: sentences containing these are high-priority
    CAUSAL_MARKERS = ['therefore', 'so', 'caused', 'led to', 'as a result', 'consequently', 'ultimately', 'influenced', 'changed']
    TEMPORAL_MARKERS = ['later', 'afterward', 'subsequently', 'then', 'finally']

    def _extractive_compress(self, text: str, budget: int) -> str:
        """Extract most important sentences to fit within budget.

        Priority order:
          1. First sentence (core event description)
          2. Sentences with causal markers (consequences/impact)
          3. Sentences with temporal markers (progression)
          4. Remaining sentences by position

        This is fundamentally different from truncation: it preserves
        the MOST IMPORTANT information rather than the FIRST N characters.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self.ensure_length_budget(text, budget)

        # Always include first sentence (core event)
        result = [sentences[0]]
        remaining_budget = budget - len(sentences[0])

        if remaining_budget <= 0:
            return self.ensure_length_budget(sentences[0], budget)

        # Prioritize causal sentences
        for sent in sentences[1:]:
            if remaining_budget <= 0:
                break
            if any(m in sent for m in self.CAUSAL_MARKERS) and len(sent) <= remaining_budget:
                result.append(sent)
                remaining_budget -= len(sent)

        # Then temporal sentences
        for sent in sentences[1:]:
            if remaining_budget <= 0:
                break
            if sent in result:
                continue
            if any(m in sent for m in self.TEMPORAL_MARKERS) and len(sent) <= remaining_budget:
                result.append(sent)
                remaining_budget -= len(sent)

        return ''.join(result)

    def generate_context_summary(self, summary: str, participants: Optional[List[str]] = None) -> str:
        """Generate summary_for_context — compact version for build_context() display.

        No post-processing truncation. If summary is already short, return as-is.
        For longer summaries, extract the most important sentences.
        """
        if len(summary) <= 150:
            return summary
        return self._extractive_compress(summary, 150)

    def generate_prompt_summary(self, summary: str, participants: Optional[List[str]] = None) -> str:
        """Return summary as-is for P3 scene planning. Length controlled by LLM prompt."""
        return summary

    def generate_oneliner(self, summary: str) -> str:
        """Generate summary_oneliner — extract the first sentence as a compact form.

        No hard truncation. Returns the first sentence regardless of length.
        """
        if len(summary) <= 50:
            return summary
        sentences = self._split_sentences(summary)
        if sentences:
            return sentences[0]
        return summary

    # ── Scene-Oriented Summary Generation ───────────────────────

    def generate_scene_summaries_rule_based(
        self, summary: str, participants: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """Generate all three summary variants using pure rules.

        Zero LLM cost. Always available as fallback.

        Returns:
            (summary_for_context, summary_for_prompt, summary_oneliner)
        """
        for_context = self.generate_context_summary(summary, participants)
        for_prompt = self.generate_prompt_summary(summary, participants)
        oneliner = self.generate_oneliner(summary)
        return for_context, for_prompt, oneliner

    async def generate_scene_summaries_with_llm(
        self, summary: str, participants: Optional[List[str]] = None
    ) -> Tuple[str, str, str, str]:
        """Generate all three summary variants + event_theme_category using LLM.

        Higher quality than rule-based. Called at write time only.
        Falls back to rule-based if LLM is unavailable or fails.

        Returns:
            (summary_for_context, summary_for_prompt, summary_oneliner, event_theme_category)
        """
        if not self._llm or len(summary) <= 120:
            for_context, for_prompt, oneliner = self.generate_scene_summaries_rule_based(summary, participants)
            return for_context, for_prompt, oneliner, ""  # category left empty, fallback later

        # Category options for event_theme_category
        CATEGORY_OPTIONS = [
            "career_transition",  # graduation, hiring, resignation, promotion, defense, interview
            "eval_debugging",     # evaluation, metrics, debugging, ablation, baseline
            "meeting",            # conference, group meeting, report, review, discussion
            "social",             # dinner, chat, friends, family, birthday, travel
            "academic",           # exam, course, paper, contest, homework
            "milestone",          # major life event
            "daily_routine",      # daily, commute, meal, rest
            "other",              # none of the above
        ]

        prompt = (
            f"Generate summaries and theme classification for:\n\n{summary}"
        )
        try:
            from pydantic import BaseModel as _BaseModel, Field as _Field

            class _MultiSummaryOutput(_BaseModel):
                context_summary: str = _Field(
                    ...,
                    description="100-120 words, preserving core events and causal relationships. Plain text only."
                )
                detailed_summary: str = _Field(
                    ...,
                    description="200-300 words, preserving time, place, people, causal chain. Plain text only."
                )
                oneliner: str = _Field(
                    ...,
                    description="≤50 words, only the most critical information. Plain text only."
                )
                theme_category: str = _Field(
                    ...,
                    description=(
                        "Theme classification. Must be exactly one of the following: "
                        "career_transition (job changes, promotions, business expansion, starting a company, "
                        "scaling operations, professional milestones — NOT academic activities); "
                        "eval_debugging (evaluation, metrics, debugging, ablation, baseline); "
                        "meeting (conference, group meeting, report, review, discussion); "
                        "social (dinner, chat, friends, family, birthday, travel, community events); "
                        "academic (formal education only: exam, course, paper, thesis defense, "
                        "academic contest, homework — NOT business or career events); "
                        "milestone (major life event: marriage, birth, death, moving, identity milestone); "
                        "daily_routine (daily commute, meal, rest, routine work); "
                        "other (none of the above). "
                        "IMPORTANT: Business expansion, scaling operations, and professional growth "
                        "are career_transition, NOT academic."
                    )
                )

            result: _MultiSummaryOutput = await self._llm.generate_structured(
                prompt=prompt,
                response_model=_MultiSummaryOutput,
                max_tokens=550,
                temperature=0.0,
                task_type="multi_summary",
            )
            for_context = result.context_summary or ""
            for_prompt = result.detailed_summary or ""
            oneliner = result.oneliner or ""
            category = result.theme_category if result.theme_category in CATEGORY_OPTIONS else "other"

            # Fallback for missing fields
            if not for_context:
                for_context = self.generate_context_summary(summary, participants)
            if not for_prompt:
                for_prompt = self.generate_prompt_summary(summary, participants)
            if not oneliner:
                oneliner = self.generate_oneliner(summary)
            if not category:
                category = "other"
            return for_context, for_prompt, oneliner, category
        except Exception:
            for_context, for_prompt, oneliner = self.generate_scene_summaries_rule_based(summary, participants)
            return for_context, for_prompt, oneliner, ""

    # ── Graduated Length Enforcement (safety net) ────────────────

    @staticmethod
    def ensure_length_budget(text: str, budget: int) -> str:
        """Ensure text fits within budget using graduated compression.

        Strategy (graduated, not abrupt):
          1. If within budget: return as-is
          2. If within 1.2x budget (slight overflow): truncate at sentence boundary
          3. If >1.2x budget (significant overflow): hard cut with ellipsis

        This is the ONLY place truncation logic exists in the entire system.
        """
        if not text or len(text) <= budget:
            return text

        # Try sentence boundaries (Chinese + English)
        for sep in ['。', '；', '！', '？', '. ', '; ', '! ', '? ']:
            idx = text.rfind(sep, 0, budget)
            if idx > budget * 0.5:
                return text[:idx + len(sep)]

        # Try clause boundaries as fallback
        for sep in ['，', '、', ', ']:
            idx = text.rfind(sep, 0, budget)
            if idx > budget * 0.6:
                return text[:idx + len(sep)] + "…"

        # Last resort: hard cut with ellipsis
        return text[:budget - 1] + "…"

    # ── Date Normalization ──────────────────────────────────────

    @staticmethod
    def normalize_event_date(time_period: str) -> str:
        """Extract normalized date from time_period string.

        Handles formats:
          - "from 2003-03-15 morning to 2003-03-15 afternoon" -> "2003-03-15"
          - "2003-03-27 to 2007-03-26" -> "2003-03-27"
        """
        matches = re.findall(r'(\d{4}-\d{2}-\d{2})', time_period)
        return matches[0] if matches else ""

    # ── Period Summary Brief Generation ─────────────────────────

    def generate_period_summary_brief(self, period_summary: str) -> str:
        """Generate period_summary_brief — extract key sentences from period summary.

        No hard truncation. Returns the most important sentences.
        """
        if len(period_summary) <= 150:
            return period_summary
        return self._extractive_compress(period_summary, 150)

    # ── 🆕 STARE-E v2: Embedding Computation ────────────────────

    def compute_embedding(
        self, summary: str, event_type: str = "", period_theme: str = ""
    ) -> List[float]:
        """Compute embedding vector at write time. Uses local model, zero API cost.

        Embedding input strategy:
          Concatenate structured metadata with summary text to create a
          semantically rich representation that captures both content and context.
          Uses 'passage:' prefix for E5 model's asymmetric retrieval convention.
        """
        if not self._emb:
            return []

        parts = []
        if event_type:
            parts.append(f"Event type: {event_type}")
        if period_theme:
            parts.append(f"Period theme: {period_theme}")
        parts.append(summary)

        enriched_text = ". ".join(parts)
        embedding = self._emb.encode_passage(enriched_text)
        return embedding.tolist()

    def compute_causal_parents(
        self,
        new_event_summary: str,
        new_event_period_id: str,
        existing_events: Dict,
        new_event_id: str = "",
    ) -> List[str]:
        """Identify causal parent events using structural heuristics + embedding similarity.

        Uses event_id lexicographic ordering as the primary time-order gate:
        any existing event whose ID is >= new_event_id is considered a future
        event and is excluded.  LP*_E* IDs sort naturally (LP6_E001 < LP6_E002
        < LP7_E001) so this is both simple and correct.

        When new_event_id is not provided we fall back to period-number filtering
        to preserve backward compatibility.
        """
        parents = []

        try:
            current_period_num = int(new_event_period_id.replace("LP", ""))
        except (ValueError, AttributeError):
            current_period_num = None

        new_emb = None
        if self._emb:
            new_emb = self._emb.encode_passage(new_event_summary)

        for eid, ev in existing_events.items():
            ev_period_id = eid.split("_")[0] if "_" in eid else ""

            # ── Primary gate: event_id ordering ────────────────────────────
            # If new_event_id is known, exclude any event written at the same
            # time or later.  Uses numeric LP/E parsing so LP10 > LP9 correctly.
            if new_event_id and not _event_id_lt(eid, new_event_id):
                continue

            # ── Period-range filter (skip very old periods) ─────────────────
            if current_period_num is not None:
                try:
                    ev_period_num = int(ev_period_id.replace("LP", ""))
                except (ValueError, AttributeError):
                    continue
                if ev_period_num < current_period_num - 2:
                    continue

            # ── Heuristic 1: Same period → direct predecessor ───────────────
            if ev_period_id == new_event_period_id:
                parents.append(eid)
                continue

            if current_period_num is None:
                continue
            try:
                ev_period_num = int(ev_period_id.replace("LP", ""))
            except (ValueError, AttributeError):
                continue

            # ── Heuristic 2: Previous period, high importance ───────────────
            if ev_period_num == current_period_num - 1 and ev.importance_score >= 0.7:
                parents.append(eid)
                continue

            # ── Heuristic 3: Semantic similarity as soft causal signal ───────
            if new_emb is not None and hasattr(ev, 'embedding_vector') and ev.embedding_vector:
                ev_emb = np.array(ev.embedding_vector, dtype=np.float32)
                sim = float(np.dot(new_emb, ev_emb))
                if sim > 0.7 and ev_period_num >= current_period_num - 2:
                    parents.append(eid)

        return parents

    # ── Full Pipeline Entry Point ───────────────────────────────

    async def enrich_event(
        self,
        record,  # EventRecord
        event_type: str = "",
        period_theme: str = "",
        existing_events: Optional[Dict] = None,
        use_llm_summary: bool = False,
        event_id: str = "",
    ):
        """Run the full write-time enrichment pipeline (STARE-E v2).

        Args:
            record: The EventRecord to enrich (mutated in place)
            event_type: Event type tag for importance scoring
            period_theme: Theme of the current life period (for enriched embedding)
            existing_events: Existing events dict for causal parent computation
            use_llm_summary: Whether to use LLM for summary generation

        Returns:
            The enriched EventRecord
        """
        # 1. Importance scoring (always rule-based, deterministic)
        record.importance_score = self.compute_importance(
            summary=record.summary,
            event_type=event_type,
            resolution_level=record.resolution_level,
            participants=record.participants,
        )

        # 2. Scene-oriented summaries + event_theme_category (extractive, not truncation)
        llm_category = ""
        if use_llm_summary and self._llm and len(record.summary) > 120:
            (
                record.summary_for_context,
                record.summary_for_prompt,
                record.summary_oneliner,
                llm_category,
            ) = await self.generate_scene_summaries_with_llm(record.summary, record.participants)
        else:
            record.summary_for_context, record.summary_for_prompt, record.summary_oneliner = (
                self.generate_scene_summaries_rule_based(record.summary, record.participants)
            )

        # Step 7: Assign event_theme_category (merged with summary call, zero extra LLM cost)
        if llm_category and not record.event_theme_category:
            record.event_theme_category = llm_category
        elif not record.event_theme_category:
            record.event_theme_category = self._classify_theme_category_rule_based(record)

        # 3. Date normalization
        record.event_date = self.normalize_event_date(record.time_period)

        # 4. 🆕 Embedding vector (local model, zero API cost)
        if self._emb and not record.embedding_vector:
            record.embedding_vector = self.compute_embedding(
                record.summary, event_type, period_theme
            )

        # 5. 🆕 Causal parent IDs
        if existing_events and not record.causal_parent_ids:
            period_id = ""
            # Try to extract period_id from context
            if hasattr(record, '_period_id'):
                period_id = record._period_id
            record.causal_parent_ids = self.compute_causal_parents(
                record.summary, period_id, existing_events,
                new_event_id=event_id or getattr(record, 'event_id', ''),
            )

        # 6. v6: behavioral_style extraction (only for high-res events)
        if (record.resolution_level == "high"
                and hasattr(record, 'interaction_details')
                and record.interaction_details):
            target_turns = [
                t for t in record.interaction_details
                if (t.speaker_id if hasattr(t, 'speaker_id') else t.get("speaker_id", "")) == "P_TARGET"
            ]
            if target_turns:
                # Extract linguistic style features
                contents = [
                    (t.content if hasattr(t, 'content') else t.get("content", ""))
                    for t in target_turns[:5]
                ]
                thoughts = [
                    (t.internal_thought if hasattr(t, 'internal_thought') else t.get("internal_thought", ""))
                    for t in target_turns[:5]
                    if (t.internal_thought if hasattr(t, 'internal_thought') else t.get("internal_thought", ""))
                ]
                style_markers = []
                all_content = "".join(contents)
                # Hesitant/filler words
                if any(m in all_content for m in ["um", "uh", "like", "you know"]):
                    style_markers.append("hesitant_fillers")
                # Colloquial hedging
                if any(m in all_content for m in ["I think", "I feel", "actually"]):
                    style_markers.append("colloquial_hedging")
                # Inner monologue presence
                if thoughts:
                    style_markers.append("has_inner_monologue")

                record.behavioral_style = {
                    "sample_utterances": contents[:3],
                    "style_markers": style_markers,
                    "interaction_turn_count": len(target_turns),
                }

        return record

    def _classify_theme_category_rule_based(self, record) -> str:
        """Rule-based fallback for event_theme_category (used when LLM is unavailable)."""
        text = " ".join([
            record.summary or "",
            getattr(record, 'initial_summary', '') or "",
        ]).lower()
        
        if any(kw in text for kw in ["graduation", "interview", "promotion", "resign",
                                      "hiring", "defense", "resignation"]):
            return "career_transition"
        if any(kw in text for kw in ["eval", "metric", "reward",
                                      "debug", "ablation", "baseline"]):
            return "eval_debugging"
        if any(kw in text for kw in ["meeting", "review", "seminar",
                                      "conference", "discussion"]):
            return "meeting"
        if any(kw in text for kw in ["dinner", "chat", "party", "travel",
                                      "friends", "family", "birthday"]):
            return "social"
        if any(kw in text for kw in ["exam", "course", "paper", "contest",
                                      "homework"]):
            return "academic"
        if getattr(record, 'event_type', '') in ("milestone",):
            return "milestone"
        if getattr(record, 'event_type', '') in ("daily_routine",):
            return "daily_routine"
        return "other"
