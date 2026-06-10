"""
Persona Extensions Formatter — Benchmark-Agnostic Extension Rendering Engine
=============================================================================

Converts the structured `persona_extensions` dict into human-readable prompt
sections. This module is benchmark-agnostic — it only reads `_render_hint` and
`_display_name` metadata, not specific field names.

Supported render hints:
  - named_items / feature_list: List of {name_key: name, value_key: value} dicts
  - grouped_list: Dict of {group_name: [items]}
  - key_value: Flat key-value pairs
  - free_text: Plain string
  - auto: Auto-detect format (default)

Usage:
    from lifelong_synth.persona_extensions_formatter import format_persona_extensions

    text = format_persona_extensions(persona_extensions)
    # → human-readable string for prompt injection
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Default Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_CHAR_BUDGET = 1500  # Max chars per extension value in prompt
DEFAULT_HEADING = "Extended Persona Attributes"


# ═══════════════════════════════════════════════════════════════
# Auto-Detection Logic
# ═══════════════════════════════════════════════════════════════

def _detect_render_hint(value: Any) -> str:
    """Auto-detect the best render hint based on value structure."""
    if isinstance(value, str):
        return "free_text"
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            return "named_items"
        return "free_text"  # list of strings
    if isinstance(value, dict):
        # Check for structured format with metadata keys
        if "_render_hint" in value:
            return value["_render_hint"]
        # Check for grouped structure
        non_meta_keys = [k for k in value.keys() if not k.startswith("_")]
        if non_meta_keys:
            first_val = value[non_meta_keys[0]]
            if isinstance(first_val, list):
                return "grouped_list"
            if isinstance(first_val, str):
                return "key_value"
        return "free_text"
    return "free_text"


def _detect_item_keys(items: List[Dict[str, Any]]) -> tuple:
    """Auto-detect name/value key names from a list of item dicts.

    Looks for common key patterns:
      - feature_name / feature_answer (SimulatorArena style)
      - name / answer / value / description
    """
    if not items:
        return ("name", "value")

    first = items[0]
    name_key = None
    value_key = None

    # Common name-key candidates
    for candidate in ["feature_name", "name", "title", "label", "key", "category"]:
        if candidate in first:
            name_key = candidate
            break

    # Common value-key candidates
    for candidate in ["feature_answer", "answer", "value", "description", "content", "text"]:
        if candidate in first:
            value_key = candidate
            break

    if name_key is None:
        # Fall back to first non-meta string key
        for k, v in first.items():
            if not k.startswith("_") and isinstance(v, str):
                name_key = k
                break
        if name_key is None:
            name_key = list(first.keys())[0] if first else "name"

    if value_key is None:
        # Fall back to second non-meta key or first non-name key
        keys = [k for k in first.keys() if not k.startswith("_") and k != name_key]
        value_key = keys[0] if keys else "value"

    return (name_key, value_key)


# ═══════════════════════════════════════════════════════════════
# Individual Renderers
# ═══════════════════════════════════════════════════════════════

def _render_named_items(
    value: Any,
    display_name: str,
    char_budget: int,
    item_name_key: Optional[str] = None,
    item_value_key: Optional[str] = None,
) -> str:
    """Render named_items format: list of {name_key: name, value_key: value}."""
    items = value
    # Handle wrapped format: {"items": [...], "features": [...]}
    if isinstance(value, dict):
        if "items" in value:
            items = value["items"]
        elif "features" in value:
            items = value["features"]
        else:
            # Might be a single item dict
            items = [value]

    if not isinstance(items, list) or len(items) == 0:
        return ""

    # Auto-detect item keys if not provided
    if item_name_key is None or item_value_key is None:
        detected_name, detected_value = _detect_item_keys(items)
        item_name_key = item_name_key or detected_name
        item_value_key = item_value_key or detected_value

    lines = [f"### {display_name}"]
    used_chars = len(lines[0])
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get(item_name_key, "Unknown"))
        val = str(item.get(item_value_key, ""))
        line = f"- **{name}**: {val}"
        if used_chars + len(line) > char_budget:
            lines.append(f"- ... ({len(items) - items.index(item)} more items truncated)")
            break
        lines.append(line)
        used_chars += len(line)

    return "\n".join(lines)


def _render_grouped_list(
    value: Any,
    display_name: str,
    char_budget: int,
) -> str:
    """Render grouped_list format: dict of {group_name: [items]}."""
    if isinstance(value, dict):
        # Extract groups, skipping meta keys
        groups = {k: v for k, v in value.items() if not k.startswith("_")}
    else:
        return ""

    if not groups:
        return ""

    lines = [f"### {display_name}"]
    used_chars = len(lines[0])

    for group_name, group_items in groups.items():
        if isinstance(group_items, list):
            items_str = ", ".join(str(i) for i in group_items[:10])
            if len(group_items) > 10:
                items_str += f" ... (+{len(group_items) - 10} more)"
        else:
            items_str = str(group_items)

        line = f"**{group_name}**: {items_str}"
        if used_chars + len(line) > char_budget:
            break
        lines.append(line)
        used_chars += len(line)

    return "\n".join(lines)


def _render_key_value(
    value: Any,
    display_name: str,
    char_budget: int,
) -> str:
    """Render key_value format: flat dict of key-value pairs."""
    if isinstance(value, dict):
        pairs = {k: v for k, v in value.items() if not k.startswith("_")}
    else:
        return ""

    if not pairs:
        return ""

    lines = [f"### {display_name}"]
    used_chars = len(lines[0])

    for k, v in pairs.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        line = f"- **{k}**: {val_str}"
        if used_chars + len(line) > char_budget:
            break
        lines.append(line)
        used_chars += len(line)

    return "\n".join(lines)


def _render_free_text(
    value: Any,
    display_name: str,
    char_budget: int,
) -> str:
    """Render free_text format: plain string."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) > char_budget:
        text = text[:char_budget] + "..."
    return f"### {display_name}\n{text}"


# ═══════════════════════════════════════════════════════════════
# Main Public API
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Domain-Relevance Routing
# ═══════════════════════════════════════════════════════════════

# Default domain sets for each pipeline stage (per design doc §6.3.3)
STAGE_DOMAINS: Dict[str, set] = {
    "planning":    {"knowledge", "demographic", "personality"},
    "event":       None,  # all domains (summary level)
    "simulation":  {"communication", "behavioral", "personality"},
    "consistency": {"communication", "behavioral", "personality"},
    "full":        None,  # all domains
}


def format_persona_extensions(
    extensions: Dict[str, Any],
    heading: str = DEFAULT_HEADING,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    stage: str = "full",
    include_domains: Optional[set] = None,
) -> str:
    """Format persona_extensions dict into a human-readable prompt section.

    This is the **core rendering engine** for extensions. It converts the
    structured `persona_extensions` dict into a human-readable prompt section.
    It is benchmark-agnostic — it only reads `_render_hint` and `_display_name`
    metadata, not specific field names.

    Args:
        extensions: The persona_extensions dict from persona config.
        heading: Section heading for the entire extension block.
        char_budget: Maximum characters per individual extension value.
        stage: Pipeline stage hint — controls domain filtering.
               One of: "planning", "event", "simulation", "consistency", "full".
               When not "full" or "event", only extensions whose ``_domain``
               tag matches the stage's relevant domains are rendered.
               Extensions without ``_domain`` default to ``"other"`` and are
               included at "event"/"full" stages but excluded elsewhere.
        include_domains: Optional explicit set of domain strings to include.
               If provided, overrides the ``stage``-based filtering.

    Returns:
        Formatted string ready for prompt injection, or empty string if
        extensions is None or empty.
    """
    if not extensions:
        return ""

    # Resolve domain filter
    if include_domains is not None:
        allowed_domains = include_domains
    else:
        allowed_domains = STAGE_DOMAINS.get(stage, None)

    parts: List[str] = []

    for key, value in extensions.items():
        # Skip meta keys at the top level
        if key.startswith("_"):
            continue

        # Skip None values
        if value is None:
            continue

        # Domain filtering
        if allowed_domains is not None and isinstance(value, dict):
            ext_domain = value.get("_domain", "other")
            if ext_domain not in allowed_domains:
                continue

        # Determine display name
        display_name = key.replace("_", " ").title()
        if isinstance(value, dict) and "_display_name" in value:
            display_name = value["_display_name"]

        # Determine render hint
        render_hint = "auto"
        if isinstance(value, dict) and "_render_hint" in value:
            render_hint = value["_render_hint"]

        if render_hint == "auto":
            render_hint = _detect_render_hint(value)

        # Extract item keys if specified
        item_name_key = None
        item_value_key = None
        if isinstance(value, dict):
            item_name_key = value.get("_item_name_key")
            item_value_key = value.get("_item_value_key")

        # Render based on hint
        # "feature_list" is an alias for "named_items"
        if render_hint in ("named_items", "feature_list"):
            section = _render_named_items(
                value, display_name, char_budget,
                item_name_key=item_name_key,
                item_value_key=item_value_key,
            )
        elif render_hint == "grouped_list":
            section = _render_grouped_list(value, display_name, char_budget)
        elif render_hint == "key_value":
            section = _render_key_value(value, display_name, char_budget)
        elif render_hint == "free_text":
            section = _render_free_text(value, display_name, char_budget)
        else:
            # Fallback: render as JSON
            section = _render_free_text(value, display_name, char_budget)

        if section:
            parts.append(section)

    if not parts:
        return ""

    return f"## {heading}\n\n" + "\n\n".join(parts)
