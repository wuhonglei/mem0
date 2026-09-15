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

# One global floor bounds how far any memory can be demoted; the category only
# decides how fast its weight falls off. A per-category floor reads like
# protection but behaves like a standing bonus for the categories that hold it:
# measured, the 0.9 floor on personal_core kept low-relevance facts (0.175-0.41)
# inside the top-10 of unrelated queries, and a single 0.7 floor with
# per-category half-lives scored better on every axis (junk intrusions 3 -> 2,
# knowledge share 47% -> 53%, perturbation 78% -> 67% at alpha=0.3).
DECAY_FLOOR = 0.7
DECAY_HALF_LIVES = {
    CATEGORY_PERSONAL_CORE: 365.0,
    CATEGORY_PREFERENCES: 180.0,
    CATEGORY_INTERESTS: 90.0,
    # state holds in-progress plans, and the queries that ask about them
    # ("what am I working on", "how did that interview go") are precisely the
    # ones a 14-day half-life punished: measured, the best state memory fell
    # from rank 1-2 to outside the top 10 in 10 of 23 regression queries while
    # the category-blind control kept it at 2-7.
    CATEGORY_STATE: 45.0,
    CATEGORY_KNOWLEDGE: 30.0,
    CATEGORY_MISC: 60.0,
}
DEFAULT_HALF_LIFE = DECAY_HALF_LIVES[CATEGORY_MISC]


def half_life_days(category: Optional[str]) -> float:
    """Half-life in days for a resolved category (unknown values use misc)."""
    return DECAY_HALF_LIVES.get(category or CATEGORY_MISC, DEFAULT_HALF_LIFE)


FRESH_SCALE = 1.5
SECONDS_PER_DAY = 86400.0
DEFAULT_ACCESS_LOG_MAX = 20

ENV_DECAY_MODE = "MEM0_DECAY_MODE"
ENV_ACCESS_LOG_MAX = "MEM0_DECAY_ACCESS_LOG_MAX"
ENV_DECAY_STRENGTH = "MEM0_DECAY_STRENGTH"


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


def decay_strength() -> float:
    """How much of the decay deviation to apply, in [0, 1].

    ``1.0`` is the full bias; lower values keep the ordering pressure but
    compress it, which is how the effect gets tuned against the 10-30 %
    perturbation budget instead of flipping enforce on and off.
    """
    raw = os.environ.get(ENV_DECAY_STRENGTH)
    if raw is None or raw == "":
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to 1.0", ENV_DECAY_STRENGTH, raw)
        return 1.0
    if not 0.0 <= value <= 1.0:
        clamped = min(1.0, max(0.0, value))
        logger.warning("Out-of-range %s=%r; clamping to %s", ENV_DECAY_STRENGTH, raw, clamped)
        return clamped
    return value


def apply_strength(scale: float, strength: Optional[float] = None) -> float:
    """Compress a scaling factor's deviation from 1.0 by the configured strength.

    ``1 + a * (scale - 1)``: a=1 is the raw curve, a=0 leaves relevance alone.
    """
    alpha = decay_strength() if strength is None else strength
    return 1.0 + alpha * (scale - 1.0)


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
    half_life = half_life_days(category)
    half_life_seconds = half_life * SECONDS_PER_DAY

    payload = payload or {}
    # Recency, not "last touched": updated_at is bumped by governance bookkeeping
    # (merge / supersede / archive) as well as by content edits, so on this store
    # 1326 of 1327 UPDATE events changed no text at all. Prefer the access log,
    # then the last content edit, then creation.
    stamp = _parse_timestamp(
        payload.get("last_accessed_at")
        or payload.get("content_updated_at")
        or payload.get("created_at")
        or payload.get("updated_at")
    )
    if stamp is None:
        age_seconds = 0.0
    else:
        age_seconds = max(0.0, (now - stamp).total_seconds())

    return DECAY_FLOOR + (FRESH_SCALE - DECAY_FLOOR) * exp(-age_seconds / half_life_seconds)


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
    strength = decay_strength()
    working = list(scored or [])
    for row in working:
        curve = decay_scaling(row.get("payload"), now=now)
        scale = apply_strength(curve, strength)
        row["_decay_scale"] = curve
        row["_decay_effective"] = scale
        row["score"] = float(row.get("score") or 0.0) * scale

    working.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    trimmed = working[:limit]
    for row in trimmed:
        scale = row.pop("_decay_scale", 1.0)
        effective = row.pop("_decay_effective", scale)
        if explain:
            details = row.setdefault("score_details", {})
            details["decay_scale"] = scale
            details["decay_effective_scale"] = effective
            details["decay_strength"] = strength
        row["score"] = min(float(row["score"]), 1.0)
    return trimmed
