"""Apply non-destructive Dream actions via Memory.update / Memory.add."""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mem0.memory.governance_filters import (
    GOVERNANCE_STATUS_ACTIVE,
    GOVERNANCE_STATUS_MERGED,
    GOVERNANCE_STATUS_SUPERSEDED,
    MEMORY_KIND_PATTERN,
)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evidence_hash(evidence_ids: List[str]) -> str:
    return hashlib.sha256("|".join(sorted(str(i) for i in evidence_ids)).encode("utf-8")).hexdigest()


def pick_canonical(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prefer longer text, then later created_at."""
    return max(
        items,
        key=lambda item: (len(item.get("memory") or ""), item.get("created_at") or ""),
    )


def apply_merge(
    memory,
    *,
    source_ids: List[str],
    canonical_id: str,
    pass_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = _now()
    for source_id in source_ids:
        if source_id == canonical_id:
            continue
        memory.update(
            source_id,
            metadata={
                "governance_status": GOVERNANCE_STATUS_MERGED,
                "merged_into": canonical_id,
                "governance_pass_id": pass_id,
                "governance_timestamp": timestamp,
            },
        )
    return {
        "type": "merge",
        "source_ids": list(source_ids),
        "canonical_id": canonical_id,
        "reason": reason,
    }


def apply_supersede(
    memory,
    *,
    old_id: str,
    new_id: str,
    pass_id: str,
    reason: Optional[str] = None,
    old_content: Optional[str] = None,
    new_content: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = _now()
    memory.update(
        old_id,
        metadata={
            "governance_status": GOVERNANCE_STATUS_SUPERSEDED,
            "superseded_by": new_id,
            "governance_pass_id": pass_id,
            "governance_timestamp": timestamp,
        },
    )
    return {
        "type": "supersede",
        "old_id": old_id,
        "new_id": new_id,
        "source_ids": [old_id, new_id],
        "reason": reason,
        "old_content": old_content,
        "new_content": new_content,
    }


def apply_update(memory, *, memory_id: str, new_content: str, pass_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
    timestamp = _now()
    existing = memory.get(memory_id) or {}
    old_content = existing.get("memory")
    memory.update(
        memory_id,
        text=new_content,
        metadata={
            "governance_pass_id": pass_id,
            "governance_timestamp": timestamp,
        },
    )
    return {
        "type": "update",
        "id": memory_id,
        "old_content": old_content,
        "new_content": new_content,
        "reason": reason,
    }


def apply_synthesize(
    memory,
    *,
    text: str,
    evidence_ids: List[str],
    user_id: str,
    pass_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = _now()
    hashed = evidence_hash(evidence_ids)
    result = memory.add(
        [{"role": "user", "content": text}],
        user_id=user_id,
        infer=False,
        metadata={
            "governance_status": GOVERNANCE_STATUS_ACTIVE,
            "memory_kind": MEMORY_KIND_PATTERN,
            "synthesized_from": list(evidence_ids),
            "synthesis_evidence_hash": hashed,
            "governance_pass_id": pass_id,
            "governance_timestamp": timestamp,
        },
    )
    new_id = None
    if isinstance(result, dict):
        rows = result.get("results") or []
        if rows:
            new_id = rows[0].get("id")
    return {
        "type": "synthesize",
        "id": new_id,
        "source_ids": list(evidence_ids),
        "evidence_ids": list(evidence_ids),
        "reason": reason,
        "new_content": text,
    }
