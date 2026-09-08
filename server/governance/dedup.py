"""Hash + vector similarity candidate discovery for Dream governance."""

from typing import Any, Dict, List, Optional

from mem0.memory.governance_filters import GOVERNANCE_STATUS_MERGED, memory_status

from governance.config import LLM_CANDIDATE_MIN_SCORE, ON_ADD_SEARCH_TOP_K, similarity_threshold


def _hit_id(hit: Any) -> str:
    if hasattr(hit, "id"):
        return str(hit.id)
    if isinstance(hit, dict):
        return str(hit.get("id", ""))
    return ""


def _hit_payload(hit: Any) -> Dict[str, Any]:
    if hasattr(hit, "payload") and isinstance(hit.payload, dict):
        return hit.payload
    if isinstance(hit, dict):
        return hit.get("payload") or {}
    return {}


def _hit_score(hit: Any) -> float:
    if hasattr(hit, "score") and hit.score is not None:
        try:
            return float(hit.score)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(hit, dict) and hit.get("score") is not None:
        try:
            return float(hit["score"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _hit_text(payload: Dict[str, Any]) -> str:
    return payload.get("data") or payload.get("memory") or ""


def _stored_query_vector(memory, vector_id: Optional[str]) -> Optional[List[float]]:
    if not vector_id:
        return None
    try:
        row = memory.vector_store.get(vector_id=vector_id)
    except TypeError:
        row = memory.vector_store.get(vector_id)
    except Exception:
        return None
    if row is None:
        return None
    raw = getattr(row, "vector", None)
    if raw is None and isinstance(row, dict):
        raw = row.get("vector")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def search_similar(
    memory,
    text: str,
    filters: Dict[str, Any],
    *,
    exclude_id: Optional[str] = None,
    vector_id: Optional[str] = None,
    top_k: int = ON_ADD_SEARCH_TOP_K,
) -> List[Dict[str, Any]]:
    """Return similar memories using the vector store's cosine-like score."""
    if not filters:
        return []
    embedding = _stored_query_vector(memory, vector_id)
    if embedding is None:
        if not text:
            return []
        embedding = memory.embedding_model.embed(text, "search")
    raw = memory.vector_store.search(query=text or "", vectors=embedding, top_k=top_k, filters=filters)
    if raw is None:
        return []
    try:
        hits = list(raw)
    except TypeError:
        return []

    out: List[Dict[str, Any]] = []
    for hit in hits:
        hid = _hit_id(hit)
        if not hid or hid == exclude_id:
            continue
        payload = _hit_payload(hit)
        if memory_status(payload) == GOVERNANCE_STATUS_MERGED:
            continue
        out.append(
            {
                "id": hid,
                "memory": _hit_text(payload),
                "score": _hit_score(hit),
                "created_at": payload.get("created_at"),
                "payload": payload,
            }
        )
    return out


def split_auto_and_llm_candidates(
    candidates: List[Dict[str, Any]],
    *,
    auto_threshold: Optional[float] = None,
    llm_min: float = LLM_CANDIDATE_MIN_SCORE,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    threshold = similarity_threshold() if auto_threshold is None else auto_threshold
    auto = [c for c in candidates if c.get("score", 0) > threshold]
    llm = [c for c in candidates if llm_min <= c.get("score", 0) <= threshold]
    return auto, llm
