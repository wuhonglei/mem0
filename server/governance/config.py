"""Dream governance configuration (self-hosted server)."""

import os

DEFAULT_SIMILARITY_THRESHOLD = 0.95
LLM_CANDIDATE_MIN_SCORE = 0.70
SYNTHESIS_MIN_MEMORIES = 20
ON_ADD_SEARCH_TOP_K = 10
DREAM_LIST_TOP_K = 1000
# Hard cap for paginated full-listing inside a dream pass (safety valve).
DREAM_LIST_HARD_CAP = 50_000
# Synthesis batching: per-batch character budget for the user prompt. With
# ~170 chars of JSON overhead per memory this keeps a batch inside a ~30k
# token window, safe for the configured LLM regardless of store size.
SYNTHESIS_BATCH_CHAR_BUDGET = 100_000
# Pattern governance: similarity above which a new/old pattern pair is sent
# for LLM coverage judgement (pattern-on-pattern near-duplicate cleanup).
PATTERN_DEDUP_SIMILARITY = 0.90


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


def absorb_enabled() -> bool:
    """Coverage-absorb: archive source memories fully covered by a pattern.

    Off by default — enabling it makes synthesis lossy (sources are archived),
    gated behind an explicit opt-in per deployment.
    """
    return _env_bool("MEM0_DREAM_ABSORB", False)
