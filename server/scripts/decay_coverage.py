#!/usr/bin/env python3
"""Daily Memory Decay observation report.

Gates for leaving collect:
  - at least 7 days of collect (operator-enforced)
  - >=30% of memories have last_accessed_at
  - personal_core touch rate > 0
  - cap 14 days; do not switch to enforce if gates fail
"""

from __future__ import annotations

import os
from typing import Any, Dict

import psycopg
from psycopg.rows import dict_row


def _build_dsn() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "wuhonglei")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def report(conn) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE payload->>'data' IS NOT NULL")
        total = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM memories
            WHERE payload->>'last_accessed_at' IS NOT NULL
              AND payload->>'last_accessed_at' <> ''
            """
        )
        accessed = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM memories
            WHERE payload->>'category' = 'personal_core'
            """
        )
        core_total = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM memories
            WHERE payload->>'category' = 'personal_core'
              AND payload->>'last_accessed_at' IS NOT NULL
              AND payload->>'last_accessed_at' <> ''
            """
        )
        core_touched = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT payload->>'category' AS category, COUNT(*) AS n
            FROM memories
            GROUP BY 1
            ORDER BY n DESC
            """
        )
        by_category = {row["category"]
                       or "(none)": row["n"] for row in cur.fetchall()}

    coverage = (accessed / total) if total else 0.0
    core_rate = (core_touched / core_total) if core_total else 0.0
    ready = coverage >= 0.30 and core_touched > 0
    return {
        "total": total,
        "accessed": accessed,
        "coverage": coverage,
        "personal_core_total": core_total,
        "personal_core_touched": core_touched,
        "personal_core_touch_rate": core_rate,
        "by_category": by_category,
        "ready_for_enforce": ready,
    }


def main() -> None:
    conn = psycopg.connect(_build_dsn())
    conn.row_factory = dict_row
    try:
        data = report(conn)
    finally:
        conn.close()

    print(f"total                 {data['total']}")
    print(f"accessed              {data['accessed']}")
    print(f"coverage              {data['coverage']:.1%}")
    print(f"personal_core total   {data['personal_core_total']}")
    print(f"personal_core touched {data['personal_core_touched']}")
    print(f"personal_core rate    {data['personal_core_touch_rate']:.1%}")
    print("by category:", data["by_category"])
    print("ready_for_enforce:", data["ready_for_enforce"])
    if not data["ready_for_enforce"]:
        print("Gate failed: need >=30% access coverage AND personal_core touch > 0.")
        print("Keep collect for up to 14 days, or fix classification before enforce.")


if __name__ == "__main__":
    main()
