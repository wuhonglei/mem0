"""Category-aware search-time decay.

A ranking bias only: never deletes, never filters, and does not change
add / get_all semantics. Access bookkeeping writes payload fields directly
on the vector store so Memory.update (re-embed + history + updated_at) is
not on the hot path.

Category values come from ``mem0.memory.categories``. This module only maps
those values onto decay curves.
"""

from __future__ import annotations

import logging
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from math import exp
from typing import Any, Dict, List, Optional

from mem0.memory.categories import (
    CATEGORY_INTERESTS,
    CATEGORY_KNOWLEDGE,
    CATEGORY_MISC,
    CATEGORY_PERSONAL_CORE,
    CATEGORY_PREFERENCES,
    CATEGORY_STATE,
    resolve_category,
)

logger = logging.getLogger(__name__)

DECAY_MODE_OFF = "off"
DECAY_MODE_COLLECT = "collect"
DECAY_MODE_ENFORCE = "enforce"
VALID_DECAY_MODES = frozenset({DECAY_MODE_OFF, DECAY_MODE_COLLECT, DECAY_MODE_ENFORCE})

# category -> (floor, half_life_days)
DECAY_CURVES = {
    CATEGORY_PERSONAL_CORE: (0.9, 365.0),
    CATEGORY_PREFERENCES: (0.85, 180.0),
    CATEGORY_INTERESTS: (0.8, 90.0),
    CATEGORY_STATE: (0.3, 14.0),
    CATEGORY_KNOWLEDGE: (0.3, 30.0),
    CATEGORY_MISC: (0.5, 60.0),
}

FRESH_SCALE = 1.5
SECONDS_PER_DAY = 86400.0
DEFAULT_ACCESS_LOG_MAX = 20

ENV_DECAY_MODE = "MEM0_DECAY_MODE"
ENV_ACCESS_LOG_MAX = "MEM0_DECAY_ACCESS_LOG_MAX"


def decay_mode() -> str:
    raw = (os.environ.get(ENV_DECAY_MODE) or DECAY_MODE_OFF).strip().lower()
    if raw not in VALID_DECAY_MODES:
        logger.warning("Invalid %s=%r; falling back to %s", ENV_DECAY_MODE, raw, DECAY_MODE_OFF)
        return DECAY_MODE_OFF
    return raw


def access_log_max() -> int:
    raw = os.environ.get(ENV_ACCESS_LOG_MAX)
    if raw is None or raw == "":
        return DEFAULT_ACCESS_LOG_MAX
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_ACCESS_LOG_MAX
    return max(1, value)


def should_apply_scaling(override: Optional[bool] = None) -> bool:
    if override is False:
        return False
    if override is True:
        return True
    return decay_mode() == DECAY_MODE_ENFORCE


def should_record_access(override: Optional[bool] = None, mode: Optional[str] = None) -> bool:
    """Skip writes during A/B (override set) so both arms do not pollute access history."""
    if override is not None:
        return False
    resolved = mode if mode is not None else decay_mode()
    return resolved in (DECAY_MODE_COLLECT, DECAY_MODE_ENFORCE)


def search_rank_pool_size(limit: int, internal_limit: int, decay_override: Optional[bool] = None) -> int:
    if should_apply_scaling(decay_override):
        return internal_limit
    return limit


def finalize_search_scores(
    scored_results: List[Dict[str, Any]],
    *,
    limit: int,
    explain: bool = False,
    decay_override: Optional[bool] = None,
    vector_store: Any = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Step 8.5 / 8.6: optional re-rank, then fire-and-forget access logging."""
    if should_apply_scaling(decay_override):
        scored_results = apply_decay_rerank(scored_results, now=now, limit=limit, explain=explain)
    if should_record_access(decay_override) and vector_store is not None:
        schedule_record_access(vector_store, scored_results[:limit])
    return scored_results


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def decay_scaling(payload: Optional[Dict[str, Any]], now: Optional[datetime] = None) -> float:
    """Return the category-aware multiplier in [floor, 1.5]."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    category = resolve_category(payload)
    floor, half_life_days = DECAY_CURVES[category]
    half_life_seconds = half_life_days * SECONDS_PER_DAY

    payload = payload or {}
    stamp = _parse_timestamp(payload.get("last_accessed_at") or payload.get("updated_at"))
    if stamp is None:
        age_seconds = 0.0
    else:
        age_seconds = max(0.0, (now - stamp).total_seconds())

    return floor + (FRESH_SCALE - floor) * exp(-age_seconds / half_life_seconds)


def append_access_event(
    payload: Optional[Dict[str, Any]],
    ts: Any,
    max_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a copy of payload with the access ring buffer and counters updated."""
    updated = deepcopy(payload) if payload else {}
    limit = max_n if max_n is not None else access_log_max()
    if isinstance(ts, datetime):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    log = list(updated.get("access_log") or [])
    log.append(ts_str)
    if len(log) > limit:
        log = log[-limit:]

    updated["access_log"] = log
    updated["last_accessed_at"] = ts_str
    try:
        count = int(updated.get("access_count") or 0)
    except (TypeError, ValueError):
        count = 0
    updated["access_count"] = count + 1
    return updated


def record_access(vector_store: Any, results: List[Dict[str, Any]], now: Optional[datetime] = None) -> None:
    """Write access fields onto each returned memory. Failures are logged only."""
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat() if isinstance(now, datetime) else now
    for result in results or []:
        mem_id = result.get("id") if isinstance(result, dict) else None
        if not mem_id:
            continue
        try:
            existing = vector_store.get(vector_id=mem_id)
            if existing is None:
                continue
            payload = existing.payload if hasattr(existing, "payload") else None
            if payload is None and isinstance(existing, dict):
                payload = existing.get("payload")
            new_payload = append_access_event(payload, ts)
            vector_store.update(vector_id=mem_id, payload=new_payload, vector=None)
        except Exception:
            logger.exception("decay record_access failed for %s", mem_id)


def schedule_record_access(vector_store: Any, results: List[Dict[str, Any]]) -> None:
    snapshot = [{"id": row.get("id")} for row in results or [] if isinstance(row, dict) and row.get("id")]
    if not snapshot:
        return
    thread = threading.Thread(
        target=record_access,
        args=(vector_store, snapshot),
        daemon=True,
        name="mem0-decay-access",
    )
    thread.start()


def apply_decay_rerank(
    scored: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    limit: int,
    explain: bool = False,
) -> List[Dict[str, Any]]:
    """Multiply combined scores by decay, sort on the unclamped product, then clamp."""
    now = now or datetime.now(timezone.utc)
    working = list(scored or [])
    for row in working:
        scale = decay_scaling(row.get("payload"), now=now)
        row["_decay_scale"] = scale
        row["score"] = float(row.get("score") or 0.0) * scale

    working.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    trimmed = working[:limit]
    for row in trimmed:
        scale = row.pop("_decay_scale", 1.0)
        if explain:
            details = row.setdefault("score_details", {})
            details["decay_scale"] = scale
        row["score"] = min(float(row["score"]), 1.0)
    return trimmed
