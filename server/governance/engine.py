"""Orient / Gather / Consolidate / Prune / Synthesize Dream pass."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mem0.memory.governance_filters import (
    is_pattern_memory,
    memory_is_active,
    memory_status,
)
from mem0.memory.main import _build_session_scope
from mem0.memory.utils import extract_json, remove_code_blocks

from governance.actions import (
    apply_absorb,
    apply_merge,
    apply_supersede,
    apply_synthesize,
    apply_update,
    evidence_hash,
    pick_canonical,
)
from governance.config import (
    DREAM_LIST_HARD_CAP,
    DREAM_LIST_TOP_K,
    LLM_CANDIDATE_MIN_SCORE,
    SYNTHESIS_MIN_MEMORIES,
    dream_on_add_enabled,
    absorb_enabled,
    similarity_threshold,
)
from governance.dedup import search_similar, split_auto_and_llm_candidates
from governance.prompts import (
    CONSOLIDATE_SYSTEM_PROMPT,
    ON_ADD_SYSTEM_PROMPT,
    SYNTHESIS_COVERAGE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_consolidate_user_prompt,
    build_coverage_user_prompt,
    build_on_add_user_prompt,
    build_synthesis_user_prompt,
)

logger = logging.getLogger(__name__)


def _new_pass_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"pass-{stamp}-{uuid.uuid4().hex[:8]}"


def _entity_filters(user_id=None, agent_id=None, run_id=None) -> Dict[str, str]:
    return {k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v}


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    text = remove_code_blocks(raw)
    try:
        parsed = json.loads(text, strict=False)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        try:
            parsed = json.loads(extract_json(text), strict=False)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.warning("Dream LLM response was not valid JSON")
            return {}


def _llm_json(memory, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    response = memory.llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return _parse_llm_json(response if isinstance(response, str) else str(response))


def _empty_stats() -> Dict[str, int]:
    return {
        "memories_scanned": 0,
        "created": 0,
        "updated": 0,
        "merged": 0,
        "superseded": 0,
        "synthesized": 0,
    }


def _report(pass_id: str, source: str, stats: Dict[str, int], actions: List[Dict[str, Any]], duration_ms: float, summary: str, **extra) -> Dict[str, Any]:
    payload = {
        "pass_id": pass_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "source": source,
        "stats": stats,
        "actions": actions,
        "summary": summary,
    }
    payload.update(extra)
    return payload


def _memory_id(item: Dict[str, Any]) -> Optional[str]:
    return item.get("id") or item.get("memory_id")


def _memory_text(item: Dict[str, Any]) -> str:
    return item.get("memory") or item.get("text") or item.get("data") or ""


_LLM_ID_FIELDS = ("canonical_id", "old_id", "new_id", "id")


def _resolve_index(token: Any, index_to_real: Dict[str, str]) -> Optional[str]:
    if token is None:
        return None
    return index_to_real.get(str(token).strip())


def _resolve_indexes(tokens: Any, index_to_real: Dict[str, str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    if not isinstance(tokens, list):
        return resolved
    for token in tokens:
        real_id = _resolve_index(token, index_to_real)
        if real_id and real_id not in seen:
            seen.add(real_id)
            resolved.append(real_id)
    return resolved


def remap_llm_action(action: Dict[str, Any], index_to_real: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Translate LLM index ids back to real memory ids. Unknown indexes drop the action."""
    if not isinstance(action, dict):
        return None
    kind = (action.get("type") or "").lower()
    if kind in {"", "none"}:
        return None
    remapped = dict(action)
    for field in _LLM_ID_FIELDS:
        if remapped.get(field) in (None, ""):
            continue
        real_id = _resolve_index(remapped[field], index_to_real)
        if real_id is None:
            logger.info("Dropping Dream action %s: unknown index %r for %s", kind, remapped[field], field)
            return None
        remapped[field] = real_id
    if "source_ids" in remapped:
        remapped["source_ids"] = _resolve_indexes(remapped.get("source_ids"), index_to_real)
    if kind == "merge":
        sources = list(remapped.get("source_ids") or [])
        canonical = remapped.get("canonical_id")
        if canonical and canonical not in sources:
            sources = [canonical, *sources]
            remapped["source_ids"] = sources
        if not canonical or len(sources) < 2:
            return None
    elif kind == "supersede":
        if not remapped.get("old_id") or not remapped.get("new_id"):
            return None
    elif kind == "update":
        if not remapped.get("id"):
            return None
    return remapped


def _remap_llm_actions(actions: Any, index_to_real: Dict[str, str]) -> List[Dict[str, Any]]:
    remapped = []
    for action in actions or []:
        item = remap_llm_action(action, index_to_real)
        if item:
            remapped.append(item)
    return remapped


def _indexed_rows(items: List[Dict[str, Any]], *, fields: tuple) -> tuple:
    """Build prompt rows with compact integer ids and an index->real-id map."""
    rows: List[Dict[str, Any]] = []
    index_to_real: Dict[str, str] = {}
    for item in items:
        real_id = _memory_id(item)
        if not real_id:
            continue
        idx_n = len(index_to_real)
        idx = str(idx_n)
        index_to_real[idx] = str(real_id)
        row = {"id": idx_n}
        for field in fields:
            if field == "id":
                continue
            if field in item and item[field] is not None:
                row[field] = item[field]
        rows.append(row)
    return rows, index_to_real


def is_synthesis_eligible(item: Dict[str, Any]) -> bool:
    if item.get("agent_id") or item.get("run_id"):
        return False
    if not item.get("user_id"):
        return False
    if is_pattern_memory(item):
        return False
    return memory_is_active(item)


def _list_memories_all(memory, filters: Dict[str, str], *, include_merged: bool = False) -> List[Dict[str, Any]]:
    """Walk the full memory set for a filter scope using keyset pagination.

    ``get_all`` caps at ``DREAM_LIST_TOP_K`` per call. Instead of naive OFFSET
    pagination (whose skips count the UNFILTERED row set, so heavy governance
    tagging makes pages return fewer rows than requested and terminates the
    walk early), we pass the oldest ``created_at`` of the previous page as a
    cursor. Rows are fetched strictly older than the cursor, so:

    - no row is fetched twice (except ties on the exact same timestamp, which
      ``seen_ids`` dedup absorbs),
    - the walk terminates only when a fetch returns nothing, independent of
      how many rows the governance/expiry read filters remove per page.
    """
    collected: List[Dict[str, Any]] = []
    seen_ids: set = set()
    cursor: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"offset": 0}
        if cursor is not None:
            kwargs["before_created_at"] = cursor
        result = memory.get_all(
            filters=filters,
            top_k=DREAM_LIST_TOP_K,
            latest_only=False,
            include_merged=include_merged,
            **kwargs,
        )
        items = result.get("results") if isinstance(result, dict) else result
        if not items:
            break
        page_new = 0
        for it in items:
            mid = str(it.get("id"))
            if mid not in seen_ids:
                seen_ids.add(mid)
                collected.append(it)
                page_new += 1
        oldest = min((str(it.get("created_at") or "") for it in items), default=None)
        if oldest is None or page_new == 0:
            break
        cursor = oldest
        if len(collected) >= DREAM_LIST_HARD_CAP:
            break
    return collected


def _list_memories(memory, filters: Dict[str, str], *, include_merged: bool = False) -> List[Dict[str, Any]]:
    result = memory.get_all(
        filters=filters,
        top_k=DREAM_LIST_TOP_K,
        latest_only=False,
        include_merged=include_merged,
    )
    if isinstance(result, dict):
        return list(result.get("results") or [])
    if isinstance(result, list):
        return result
    return []


def _existing_evidence_hashes(memories: List[Dict[str, Any]]) -> set:
    hashes = set()
    for item in memories:
        if not is_pattern_memory(item):
            continue
        meta = item.get("metadata") if isinstance(
            item.get("metadata"), dict) else {}
        hashed = item.get("synthesis_evidence_hash") or meta.get(
            "synthesis_evidence_hash")
        if hashed:
            hashes.add(hashed)
        evidence = item.get("synthesized_from") or meta.get(
            "synthesized_from") or []
        if evidence:
            hashes.add(evidence_hash([str(i) for i in evidence]))
    return hashes


def _apply_llm_actions(memory, actions: List[Dict[str, Any]], pass_id: str, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    applied = []
    for action in actions or []:
        kind = (action.get("type") or "").lower()
        try:
            if kind == "merge":
                source_ids = action.get("source_ids") or []
                canonical_id = action.get("canonical_id")
                if not canonical_id or len(source_ids) < 2:
                    continue
                applied.append(
                    apply_merge(
                        memory,
                        source_ids=[str(i) for i in source_ids],
                        canonical_id=str(canonical_id),
                        pass_id=pass_id,
                        reason=action.get("reason"),
                    )
                )
                stats["merged"] += max(0, len(source_ids) - 1)
            elif kind == "supersede":
                old_id = action.get("old_id")
                new_id = action.get("new_id")
                if not old_id or not new_id:
                    continue
                applied.append(
                    apply_supersede(
                        memory,
                        old_id=str(old_id),
                        new_id=str(new_id),
                        pass_id=pass_id,
                        reason=action.get("reason"),
                        old_content=action.get("old_content"),
                        new_content=action.get("new_content"),
                    )
                )
                stats["superseded"] += 1
            elif kind == "update":
                memory_id = action.get("id")
                new_content = action.get("new_content")
                if not memory_id or not new_content:
                    continue
                applied.append(
                    apply_update(
                        memory,
                        memory_id=str(memory_id),
                        new_content=new_content,
                        pass_id=pass_id,
                        reason=action.get("reason"),
                    )
                )
                stats["updated"] += 1
        except Exception:
            logger.exception("Failed to apply Dream action %s", kind)
    return applied


def consolidate_new_memory(memory, new_item: Dict[str, Any], filters: Dict[str, str], pass_id: str, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    new_id = _memory_id(new_item)
    text = _memory_text(new_item)
    if not new_id or not text:
        return []
    candidates = search_similar(memory, text, filters, exclude_id=new_id, vector_id=new_id)
    auto, llm_band = split_auto_and_llm_candidates(candidates)
    applied: List[Dict[str, Any]] = []

    if auto:
        cluster = [
            {"id": new_id, "memory": text, "created_at": new_item.get("created_at") or ""},
            *auto,
        ]
        canonical = pick_canonical(cluster)
        source_ids = []
        seen = set()
        for item in cluster:
            mid = item.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            source_ids.append(mid)
        applied.append(
            apply_merge(
                memory,
                source_ids=source_ids,
                canonical_id=canonical["id"],
                pass_id=pass_id,
                reason=f"cosine>{similarity_threshold():.2f} auto-merge",
            )
        )
        stats["merged"] += max(0, len(source_ids) - 1)

    handled_ids = set()
    for action in applied:
        handled_ids.update(filter(
            None, [action.get("canonical_id"), action.get("old_id"), action.get("new_id")]))
        handled_ids.update(action.get("source_ids") or [])
    remaining = [c for c in llm_band if c["id"] not in handled_ids]
    if remaining:
        index_to_real = {"0": str(new_id)}
        candidate_rows = []
        for candidate in remaining:
            idx_n = len(index_to_real)
            index_to_real[str(idx_n)] = str(candidate["id"])
            candidate_rows.append({"id": idx_n, "memory": candidate["memory"], "score": candidate["score"]})
        parsed = _llm_json(
            memory,
            ON_ADD_SYSTEM_PROMPT,
            build_on_add_user_prompt({"id": 0, "memory": text}, candidate_rows),
        )
        applied.extend(
            _apply_llm_actions(
                memory,
                _remap_llm_actions(parsed.get("actions") or [], index_to_real),
                pass_id,
                stats,
            )
        )
    return applied


def run_on_add(
    memory,
    add_result: Any,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not dream_on_add_enabled():
        return None
    filters = _entity_filters(user_id, agent_id, run_id)
    if not filters:
        return None
    rows = add_result.get("results") if isinstance(
        add_result, dict) else add_result
    if not rows:
        return None
    new_items = [r for r in rows if isinstance(
        r, dict) and r.get("event") == "ADD" and _memory_id(r)]
    if not new_items:
        return None

    pass_id = _new_pass_id()
    stats = _empty_stats()
    stats["memories_scanned"] = len(new_items)
    started = time.perf_counter()
    actions: List[Dict[str, Any]] = []
    for item in new_items:
        actions.extend(consolidate_new_memory(
            memory, item, filters, pass_id, stats))
    duration_ms = (time.perf_counter() - started) * 1000
    if not actions:
        return None
    return _report(
        pass_id,
        "on_add",
        stats,
        actions,
        duration_ms,
        f"On-add consolidate: merged={stats['merged']} superseded={stats['superseded']}",
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
    )


def _gather_last_messages(memory, filters: Dict[str, str]) -> List[Dict[str, Any]]:
    try:
        scope = _build_session_scope(filters)
        return memory.db.get_last_messages(scope, limit=10) or []
    except Exception:
        logger.debug(
            "Dream gather: no session messages available", exc_info=True)
        return []


def _cluster_active(memory, items: List[Dict[str, Any]], filters: Dict[str, str]) -> List[List[Dict[str, Any]]]:
    active = [i for i in items if memory_is_active(i) and _memory_id(i)]
    parent = {str(_memory_id(i)): str(_memory_id(i)) for i in active}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_id = {str(_memory_id(i)): i for i in active}
    for item in active:
        mid = str(_memory_id(item))
        text = _memory_text(item)
        for candidate in search_similar(memory, text, filters, exclude_id=mid, vector_id=mid):
            if candidate["score"] < LLM_CANDIDATE_MIN_SCORE:
                continue
            other = candidate["id"]
            if other in by_id:
                union(mid, other)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in active:
        mid = str(_memory_id(item))
        buckets.setdefault(find(mid), []).append(item)
    return [group for group in buckets.values() if len(group) >= 2]


def run_synthesis(memory, items: List[Dict[str, Any]], *, user_id: str, pass_id: str, stats: Dict[str, int], force: bool = False) -> List[Dict[str, Any]]:
    eligible = [i for i in items if is_synthesis_eligible(i)]
    if len(eligible) < SYNTHESIS_MIN_MEMORIES and not force:
        return []
    if not eligible:
        return []
    existing_hashes = _existing_evidence_hashes(items)
    prompt_rows, index_to_real = _indexed_rows(
        [{"id": _memory_id(i), "memory": _memory_text(i)} for i in eligible],
        fields=("memory",),
    )
    parsed = _llm_json(
        memory,
        SYNTHESIS_SYSTEM_PROMPT,
        build_synthesis_user_prompt(prompt_rows),
    )
    by_real_id = {str(_memory_id(i)): i for i in eligible}
    applied = []
    for pattern in parsed.get("patterns") or []:
        evidence = _resolve_indexes(pattern.get("evidence_ids"), index_to_real)
        if len(evidence) < 2:
            continue
        text = (pattern.get("text") or "").strip()
        if not text:
            continue
        hashed = evidence_hash(evidence)
        if hashed in existing_hashes:
            continue
        action = apply_synthesize(
            memory,
            text=text,
            evidence_ids=evidence,
            user_id=user_id,
            pass_id=pass_id,
            reason=pattern.get("reason"),
        )
        applied.append(action)
        existing_hashes.add(hashed)
        stats["synthesized"] += 1
        stats["created"] += 1

        # Coverage pass: archive source memories fully absorbed by the new
        # pattern. A separate conservative LLM judgement per pattern; any
        # source with unique details is kept. Off by default.
        if absorb_enabled():
            new_id = action.get("id")
            sources = [
                {"id": str(idx), "memory": _memory_text(by_real_id[rid])}
                for idx, rid in zip(pattern.get("evidence_ids") or [], evidence)
                if rid in by_real_id
            ]
            if len(sources) >= 2:
                try:
                    judged = _llm_json(
                        memory,
                        SYNTHESIS_COVERAGE_SYSTEM_PROMPT,
                        build_coverage_user_prompt(text, sources),
                    )
                except Exception as e:
                    logger.warning("Synthesis coverage LLM call failed: %s", e)
                    judged = {}
                allowed = {str(idx) for idx in pattern.get("evidence_ids") or []}
                for j in judged.get("judgements") or []:
                    jid = str(j.get("id"))
                    verdict = j.get("verdict")
                    if jid in allowed and verdict == "absorb":
                        real_id = None
                        for idx, rid in zip(pattern.get("evidence_ids") or [], evidence):
                            if str(idx) == jid:
                                real_id = rid
                                break
                        if real_id and real_id in by_real_id:
                            applied.append(
                                apply_absorb(
                                    memory,
                                    source_id=real_id,
                                    pattern_id=new_id,
                                    pass_id=pass_id,
                                    reason=j.get("reason"),
                                )
                            )
                            stats["absorbed"] = stats.get("absorbed", 0) + 1
    return applied


def run_dream(
    memory,
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    synthesize: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    filters = _entity_filters(user_id, agent_id, run_id)
    if not filters:
        raise ValueError(
            "filters must contain at least one of: user_id, agent_id, run_id")

    pass_id = _new_pass_id()
    stats = _empty_stats()
    started = time.perf_counter()

    items = _list_memories_all(memory, filters, include_merged=False)
    stats["memories_scanned"] = len(items)
    last_messages = _gather_last_messages(memory, filters)

    actions: List[Dict[str, Any]] = []
    for cluster in _cluster_active(memory, items, filters):
        cluster_rows, index_to_real = _indexed_rows(
            [{"id": _memory_id(i), "memory": _memory_text(i), "status": memory_status(i)} for i in cluster],
            fields=("memory", "status"),
        )
        parsed = _llm_json(
            memory,
            CONSOLIDATE_SYSTEM_PROMPT,
            build_consolidate_user_prompt(cluster_rows, last_messages),
        )
        actions.extend(
            _apply_llm_actions(
                memory,
                _remap_llm_actions(parsed.get("actions") or [], index_to_real),
                pass_id,
                stats,
            )
        )

    if synthesize:
        if not user_id:
            logger.info("Skipping synthesis: user_id is required")
        else:
            # Re-list so merge/supersede from this pass is visible. Skip merged so they
            # do not occupy the get_all top_k; patterns are still excluded by eligibility.
            items_after = _list_memories_all(memory, filters, include_merged=False)
            actions.extend(
                run_synthesis(memory, items_after, user_id=user_id,
                              pass_id=pass_id, stats=stats, force=force)
            )

    duration_ms = (time.perf_counter() - started) * 1000
    summary = (
        f"Scanned {stats['memories_scanned']}. "
        f"merged={stats['merged']} superseded={stats['superseded']} "
        f"updated={stats['updated']} synthesized={stats['synthesized']}"
    )
    return _report(
        pass_id,
        "manual",
        stats,
        actions,
        duration_ms,
        summary,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
    )
