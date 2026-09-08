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
