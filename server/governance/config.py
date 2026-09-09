"""Dream governance configuration (self-hosted server)."""

import os

DEFAULT_SIMILARITY_THRESHOLD = 0.95
LLM_CANDIDATE_MIN_SCORE = 0.70
SYNTHESIS_MIN_MEMORIES = 20
ON_ADD_SEARCH_TOP_K = 10
DREAM_LIST_TOP_K = 1000
# Hard cap for paginated full-listing inside a dream pass (safety valve).
DREAM_LIST_HARD_CAP = 50_000


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def dream_on_add_enabled() -> bool:
    return _env_bool("MEM0_DREAM_ON_ADD", True)


def similarity_threshold() -> float:
    raw = os.environ.get("MEM0_DREAM_SIMILARITY_THRESHOLD")
    if raw is None:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_SIMILARITY_THRESHOLD
