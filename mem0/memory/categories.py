"""Shared memory category taxonomy.

`category` is a first-class payload field: how long a fact stays true, then topic.
Decay is one consumer (it maps each class to a curve). Filters, dream, and
analytics can reuse the same six values without importing decay.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

CATEGORY_PERSONAL_CORE = "personal_core"
CATEGORY_PREFERENCES = "preferences"
CATEGORY_INTERESTS = "interests"
CATEGORY_STATE = "state"
CATEGORY_KNOWLEDGE = "knowledge"
CATEGORY_MISC = "misc"

MEMORY_CATEGORIES = frozenset(
    {
        CATEGORY_PERSONAL_CORE,
        CATEGORY_PREFERENCES,
        CATEGORY_INTERESTS,
        CATEGORY_STATE,
        CATEGORY_KNOWLEDGE,
        CATEGORY_MISC,
    }
)

_STATE_RE = re.compile(r"正在|计划|等待|对比")
_CORE_RE = re.compile(r"家庭|健康|过敏|职业|家人|父亲|母亲")


def infer_category(
    category: Optional[str] = None,
    attributed_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a taxonomy value.

    A valid six-class value is kept. Missing or non-enum values fall back by
    attribution (user → preferences, assistant → knowledge) and otherwise misc.
    """
    if category in MEMORY_CATEGORIES:
        return category
    if metadata:
        meta_category = metadata.get("category")
        if meta_category in MEMORY_CATEGORIES:
            return meta_category
        if attributed_to is None:
            attributed_to = metadata.get("attributed_to")
    if attributed_to == "user":
        return CATEGORY_PREFERENCES
    if attributed_to == "assistant":
        return CATEGORY_KNOWLEDGE
    return CATEGORY_MISC


def resolve_category(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return CATEGORY_MISC
    category = payload.get("category")
    if category in MEMORY_CATEGORIES:
        return category
    return CATEGORY_MISC


def assign_inferred_category(extracted: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Write category onto an infer=True payload using LLM output plus attribution fallback."""
    extracted = extracted or {}
    metadata["category"] = infer_category(
        extracted.get("category"),
        extracted.get("attributed_to") or metadata.get("attributed_to"),
        metadata,
    )


def assign_direct_category(metadata: Dict[str, Any]) -> None:
    """Fill category on infer=False writes. Keep a caller-supplied value, including custom strings."""
    existing = metadata.get("category")
    if existing in MEMORY_CATEGORIES:
        return
    if existing:
        return
    metadata["category"] = CATEGORY_MISC


def prelabel_category(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Deterministic backfill guess. None means 'leave for the LLM'."""
    if not payload:
        return None
    existing = payload.get("category")
    if existing in MEMORY_CATEGORIES:
        return None
    if payload.get("memory_kind") == "pattern":
        return CATEGORY_INTERESTS
    text = payload.get("data") or ""
    if _STATE_RE.search(text):
        return CATEGORY_STATE
    if _CORE_RE.search(text):
        return CATEGORY_PERSONAL_CORE
    if payload.get("attributed_to") == "assistant":
        return CATEGORY_KNOWLEDGE
    return None
