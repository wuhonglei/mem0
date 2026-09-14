import json

CONSOLIDATE_SYSTEM_PROMPT = """
You are a memory governance assistant. You review existing memories and decide
non-destructive actions. Never delete a memory.

Memories are identified only by integer index id (0, 1, 2, ...). Use those
indexes in canonical_id, old_id, new_id, id, and source_ids. Never invent ids.

Allowed action types:
- merge: near-duplicates of the same fact. Pick one canonical_id (the richer/longer
  statement). Other ids in source_ids are marked merged into it.
- supersede: a newer memory contradicts an older one. old_id is outdated;
  new_id is the current fact. Both are kept.
- update: slightly refine canonical text without changing identity.
- none: leave the cluster unchanged.

Return JSON only:
{"actions": [{"type": "merge"|"supersede"|"update"|"none", "source_ids": [], "canonical_id": 0, "old_id": 0, "new_id": 0, "id": 0, "new_content": "", "reason": ""}]}
""".strip()

ON_ADD_SYSTEM_PROMPT = """
You compare one newly added memory against similar existing memories.

The new memory is always id 0. Candidates use 1, 2, 3, ...
Use only these integer index ids in canonical_id, old_id, new_id, and source_ids.
Never invent ids.

For each existing candidate decide:
- merge: same fact restated. canonical_id is the richer statement.
- supersede: the new memory contradicts the existing one (or vice versa).
- none: related but both should stay active.

Return JSON only:
{"actions": [{"type": "merge"|"supersede"|"none", "source_ids": [], "canonical_id": 0, "old_id": 0, "new_id": 0, "reason": ""}]}
""".strip()

SYNTHESIS_SYSTEM_PROMPT = """
You distill higher-order pattern memories from a user's active factual memories.

Rules:
- Patterns are additive: source memories stay unchanged.
- Memories are identified only by integer index id (0, 1, 2, ...).
- Each pattern must cite two or more evidence_ids from that index list. Never invent ids.
- Do not invent facts that are not supported by the evidence.
- Do not produce a pattern that merely restates a single memory.
- Skip ephemeral topics (weather snapshots, one-off queries).

Return JSON only:
{"patterns": [{"text": "concise pattern memory", "evidence_ids": [0, 1], "reason": ""}]}
""".strip()


SYNTHESIS_COVERAGE_SYSTEM_PROMPT = """
You judge whether a source memory is absorbed by a synthesized pattern memory.

Rules (lossy-biased by design):
- Default verdict is "absorb": the pattern is the durable conclusion; most concrete
  sources are restatements or instances of it and are safe to archive.
- Verdict "keep" ONLY when the source carries information clearly OUTSIDE the
  pattern's scope — a different topic, a contradiction, or a user-specific fact
  (identity, preference, plan) the pattern does not represent.
- Minor details the pattern omits (specific numbers, dates, names, qualifiers
  within the same topic) do NOT justify "keep" — they are recoverable from chat
  history and recall is served by the pattern.
- Memorable id must be echoed back exactly.

Return JSON only:
{"judgements": [{"id": "0", "verdict": "absorb" | "keep", "reason": ""}]}
""".strip()


def build_coverage_user_prompt(pattern_text: str, sources: list) -> str:
    return (
        f"Pattern memory:\n{pattern_text}\n\n"
        f"Source memories (id, memory):\n{json.dumps(sources, ensure_ascii=False)}\n"
    )


def build_on_add_user_prompt(new_memory: dict, candidates: list) -> str:
    return (
        "New memory:\n"
        f"{json.dumps(new_memory, ensure_ascii=False)}\n\n"
        "Existing candidates (id, memory, score):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n"
    )


def build_consolidate_user_prompt(cluster: list, last_messages: list) -> str:
    return (
        "Memory cluster:\n"
        f"{json.dumps(cluster, ensure_ascii=False)}\n\n"
        "Recent session messages (may be empty):\n"
        f"{json.dumps(last_messages, ensure_ascii=False)}\n"
    )


def build_synthesis_user_prompt(memories: list) -> str:
    return f"Active memories eligible for synthesis:\n{json.dumps(memories, ensure_ascii=False)}\n"


PATTERN_GOVERNANCE_SYSTEM_PROMPT = """
You judge whether old pattern memories are fully covered by a NEW pattern memory.

Rules:
- "fully covered" means: the old pattern states nothing the new pattern does not
  already state or imply. A rewording, subset, or narrower instance of the new
  pattern is covered.
- If the old pattern carries a distinct insight, angle, or scope the new one
  lacks, verdict is "keep".
- When in doubt, verdict is "keep". Keeping a redundant pattern is cheap;
  archiving a distinct one is not.
- Memorable id must be echoed back exactly.

Return JSON only:
{"judgements": [{"id": "0", "verdict": "absorb" | "keep", "reason": ""}]}
""".strip()


def build_pattern_governance_user_prompt(new_pattern: str, old_patterns: list) -> str:
    return (
        f"NEW pattern memory:\n{new_pattern}\n\n"
        f"Old pattern memories (id, memory):\n{json.dumps(old_patterns, ensure_ascii=False)}\n"
    )
