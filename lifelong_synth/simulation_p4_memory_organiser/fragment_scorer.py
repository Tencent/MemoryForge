"""
Memory Fragment Scorer — Importance-weighted scoring for in-memory post-scene memories.

Used by P3 (high_res_event_simulator) and P2 (event_organiser) to select
post-scene memory fragments for refined summary generation.

Theoretical basis:
  - Conway (2000): Self-reference effect — protagonist memories are prioritized
  - Park et al. (2023): Importance scoring via emotional intensity
  - Ebbinghaus (1885): Not applicable here (same-scene memories have uniform recency)

Design note:
  post_memories are NOT stored in MemoryBase; they are in-memory dicts generated
  during the current scene. Therefore they cannot use STARE-E v2 (which operates
  on EventRecord objects). This module provides a lightweight alternative.
"""

# High-intensity emotion keywords (strongly memorable events)
_HIGH_INTENSITY_KEYWORDS = [
    "tense", "excited", "fear", "sad", "angry", "desperate", "terrified", "thrilled", "furious", "devastated",
]

# Medium-intensity emotion keywords (moderately memorable events)
_MEDIUM_INTENSITY_KEYWORDS = [
    "worried", "happy", "uneasy", "expectant", "touched", "anxious", "relieved", "nostalgic",
]


def score_memory_fragment(mem: dict) -> float:
    """Score a post-scene memory fragment for selection priority.

    Three-dimensional scoring (recency omitted: all same-scene memories
    have identical timestamps, so recency provides zero discriminative power):

      score = protagonist_bonus + emotion_bonus + richness_bonus

    Args:
        mem: A post-scene memory dict with keys:
            participant_id, summary, emotional_tone

    Returns:
        Float score (higher = more important to include)
    """
    score = 0.0
    pid = mem.get("participant_id", "")

    # Signal 1: Protagonist priority
    # Conway (2000) — Self-reference effect: self-related memories
    # are encoded and retrieved with priority
    if pid == "P_TARGET":
        score += 3.0

    # Signal 2: Emotional intensity
    # Park et al. (2023) — Importance/poignancy scoring:
    # emotionally intense events are more memorable
    tone = mem.get("emotional_tone", "")
    if any(kw in tone for kw in _HIGH_INTENSITY_KEYWORDS):
        score += 2.0
    elif any(kw in tone for kw in _MEDIUM_INTENSITY_KEYWORDS):
        score += 1.0

    # Signal 3: Content richness
    # Longer summaries contain more detail, contributing more to the refined summary
    summary = mem.get("summary", "")
    score += min(len(summary) / 100.0, 1.5)

    return score


# ────────────────────────────────────────────────────────────────
# Role Type Importance Scorer
# ────────────────────────────────────────────────────────────────

# Role type importance weights for scene completion
# Inspired by Social Network Analysis (Newman, 2004) —
# structural holes: roles with higher "betweenness" in the
# character network are more critical for narrative coherence.
ROLE_TYPE_IMPORTANCE: dict[str, float] = {
    # Authority figures (high structural importance)
    "teacher": 9, "parent": 9, "boss": 9, "employer": 9,
    "supervisor": 9, "mentor": 8,
    # Close relationships (high narrative importance)
    "spouse": 9, "partner": 9, "best_friend": 8, "sibling": 8,
    # Peer relationships
    "friend": 6, "classmate": 5, "colleague": 5, "roommate": 5,
    # Acquaintances
    "neighbor": 4, "stranger": 2, "acquaintance": 3,
    # Professional
    "doctor": 6, "client": 5, "customer": 3,
}


def score_role_type_importance(role_type: str, event_summary: str = "") -> float:
    """Score a missing role type by its narrative importance.

    Two-dimensional scoring:
    1. Role type importance (structural importance in social network)
    2. Event relevance (whether the event specifically requires this role)

    Based on:
    - Newman (2004): Betweenness centrality in social networks
    - Park et al. (2023): Importance-weighted selection
    """
    score = 0.0

    # Dimension 1: Base importance by role type
    rt_lower = role_type.lower().strip()
    score = ROLE_TYPE_IMPORTANCE.get(rt_lower, 3.0)  # default = 3
    if score == 3.0:  # no exact match, try partial match
        for key, val in ROLE_TYPE_IMPORTANCE.items():
            if key in rt_lower or rt_lower in key:
                score = val
                break

    # Dimension 2: Event-specific relevance
    # If the event summary mentions the role type, boost its score
    if event_summary and rt_lower in event_summary.lower():
        score += 3.0

    return score
