"""Read-side helpers for Dream governance metadata.

Mem0 stores every field in a flat vector-store payload. Governance tags
(`governance_status`, `merged_into`, ...) sit alongside `data` / `hash` /
`text_lemmatized`. Missing `governance_status` is treated as active so
pre-Dream memories keep working.
"""

from typing import Any, Dict, Optional

GOVERNANCE_STATUS_ACTIVE = "active"
GOVERNANCE_STATUS_MERGED = "merged"
GOVERNANCE_STATUS_SUPERSEDED = "superseded"
GOVERNANCE_STATUS_ARCHIVED = "archived"

MEMORY_KIND_PATTERN = "pattern"

GOVERNANCE_PAYLOAD_KEYS = (
    "governance_status",
    "merged_into",
    "superseded_by",
    "synthesized_from",
    "memory_kind",
    "governance_pass_id",
    "governance_timestamp",
    "synthesis_evidence_hash",
)

PROMOTED_PAYLOAD_KEYS = (
    "user_id",
    "agent_id",
    "run_id",
    "actor_id",
    "role",
    "attributed_to",
    "expiration_date",
    *GOVERNANCE_PAYLOAD_KEYS,
)

CORE_AND_PROMOTED_KEYS = frozenset(
    {
        "data",
        "hash",
        "created_at",
        "updated_at",
        "id",
        "text_lemmatized",
        "attributed_to",
        *PROMOTED_PAYLOAD_KEYS,
    }
)


def memory_status(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return GOVERNANCE_STATUS_ACTIVE
    return payload.get("governance_status") or GOVERNANCE_STATUS_ACTIVE


def memory_is_active(payload: Optional[Dict[str, Any]]) -> bool:
    return memory_status(payload) == GOVERNANCE_STATUS_ACTIVE


def is_pattern_memory(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    kind = payload.get("memory_kind")
    if kind is None:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        kind = metadata.get("memory_kind")
    return kind == MEMORY_KIND_PATTERN


def should_include_memory(
    payload: Optional[Dict[str, Any]],
    *,
    latest_only: bool = False,
    include_merged: bool = False,
) -> bool:
    status = memory_status(payload)
    if latest_only:
        return status == GOVERNANCE_STATUS_ACTIVE
    if status == GOVERNANCE_STATUS_MERGED:
        return include_merged
    if status == GOVERNANCE_STATUS_ARCHIVED:
        return include_merged
    return True


def fetch_limit_for_filters(
    limit: int,
    *,
    latest_only: bool = False,
    include_merged: bool = False,
    show_expired: bool = False,
) -> int:
    needs_overfetch = latest_only or (not include_merged) or (not show_expired)
    if not needs_overfetch:
        return limit
    return max(limit * 4, 60)
