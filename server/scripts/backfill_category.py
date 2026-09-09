#!/usr/bin/env python3
"""Backfill payload.category for existing memories.

Usage:
  docker exec mem0-dev-mem0-1 python3 /app/scripts/backfill_category.py --dry-run
  docker exec mem0-dev-mem0-1 python3 /app/scripts/backfill_category.py --llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import psycopg
from psycopg.rows import dict_row

from mem0.memory.categories import MEMORY_CATEGORIES, prelabel_category


def _build_dsn() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "wuhonglei")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def fetch_unlabeled(conn, limit: int, offset: int) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, payload
            FROM memories
            WHERE payload->>'data' IS NOT NULL
              AND (
                payload->>'category' IS NULL
                OR payload->>'category' = ''
                OR NOT (payload->>'category' = ANY(%s))
              )
            ORDER BY id
            LIMIT %s OFFSET %s
            """,
            (list(MEMORY_CATEGORIES), limit, offset),
        )
        return cur.fetchall()


def count_unlabeled(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM memories
            WHERE payload->>'data' IS NOT NULL
              AND (
                payload->>'category' IS NULL
                OR payload->>'category' = ''
                OR NOT (payload->>'category' = ANY(%s))
              )
            """,
            (list(MEMORY_CATEGORIES),),
        )
        return cur.fetchone()["cnt"]


def set_category(conn, record_id: str, category: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE memories
            SET payload = jsonb_set(payload, '{category}', %s::jsonb, true)
            WHERE id = %s
            """,
            (json.dumps(category), record_id),
        )


def classify_with_llm(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    items = []
    for row in rows:
        payload = row["payload"] or {}
        items.append({"id": str(row["id"]), "text": (
            payload.get("data") or "")[:400]})
    prompt = (
        "Classify each memory into exactly one category: "
        "personal_core, preferences, interests, state, knowledge, misc. "
        "Persistence of the fact first, topic second. Prefer misc over a risky guess.\n"
        'Return JSON: {"labels": [{"id": "...", "category": "..."}]}\n\n' +
        json.dumps(items, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=os.environ.get("MEM0_DEFAULT_LLM_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    labels = parsed.get("labels") or parsed.get("results") or []
    out = {}
    for item in labels:
        category = item.get("category")
        if category in MEMORY_CATEGORIES and item.get("id"):
            out[str(item["id"])] = category
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory category")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm", action="store_true",
                        help="Classify leftovers with the LLM")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conn = psycopg.connect(_build_dsn(), autocommit=False)
    conn.row_factory = dict_row
    distribution: Counter = Counter()
    try:
        remaining = count_unlabeled(conn)
        print(f"Unlabeled (non-enum category): {remaining}")
        if remaining == 0:
            print("Nothing to do.")
            return

        offset = 0
        processed = 0
        prelabeled = 0
        llm_labeled = 0
        leftover = 0
        to_process = args.limit or remaining
        while processed < to_process:
            batch = fetch_unlabeled(
                conn, min(args.batch_size, to_process - processed), offset)
            if not batch:
                break
            needs_llm: List[Dict[str, Any]] = []
            for row in batch:
                processed += 1
                guess = prelabel_category(row["payload"] or {})
                if guess:
                    distribution[guess] += 1
                    prelabeled += 1
                    if not args.dry_run:
                        set_category(conn, str(row["id"]), guess)
                else:
                    needs_llm.append(row)

            if args.llm and needs_llm:
                labels = classify_with_llm(needs_llm)
                for row in needs_llm:
                    category = labels.get(str(row["id"]))
                    if category:
                        distribution[category] += 1
                        llm_labeled += 1
                        if not args.dry_run:
                            set_category(conn, str(row["id"]), category)
                    else:
                        leftover += 1
            else:
                leftover += len(needs_llm)

            if not args.dry_run:
                conn.commit()
            offset += len(batch)

        print("Distribution:", dict(distribution))
        print(
            f"Prelabeled: {prelabeled}  LLM: {llm_labeled}  leftover: {leftover}")
        if args.dry_run:
            print("DRY RUN — no rows written")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
