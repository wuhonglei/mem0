#!/usr/bin/env python3
"""Same-store A/B for Memory Decay.

Runs each query twice against /search with decay_override true/false.
Does not write access logs (override is set).

Usage:
  python server/scripts/decay_ab.py --user-id alice --queries-file queries.txt
  python server/scripts/decay_ab.py --user-id alice --query "我对什么过敏"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _post(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ids(result: Dict[str, Any]) -> List[str]:
    rows = result.get("results") or result.get("memories") or []
    return [str(row.get("id")) for row in rows if row.get("id")]


def rank_of(ids: List[str], memory_id: str) -> Optional[int]:
    try:
        return ids.index(memory_id) + 1
    except ValueError:
        return None


def movement_rate(off_ids: List[str], on_ids: List[str]) -> float:
    n = max(len(off_ids), 1)
    moved = 0
    for idx, mem_id in enumerate(off_ids):
        if idx >= len(on_ids) or on_ids[idx] != mem_id:
            moved += 1
    return moved / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Decay A/B against /search")
    parser.add_argument(
        "--base-url", default=os.environ.get("MEM0_BASE_URL", "http://localhost:8888"))
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--queries-file")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--core-ids", default="",
                        help="comma-separated personal_core ids")
    parser.add_argument("--stale-ids", default="",
                        help="comma-separated outdated probe ids")
    parser.add_argument(
        "--api-key", default=os.environ.get("ADMIN_API_KEY") or os.environ.get("MEM0_API_KEY"))
    args = parser.parse_args()

    queries = list(args.queries or [])
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as handle:
            queries.extend(line.strip() for line in handle if line.strip())
    if not queries:
        raise SystemExit("pass --query and/or --queries-file")

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    url = args.base_url.rstrip("/") + "/search"
    core_ids = {item for item in args.core_ids.split(",") if item}
    stale_ids = {item for item in args.stale_ids.split(",") if item}

    movements = []
    core_deltas = []
    stale_deltas = []
    stale_in_top5_on = 0
    stale_in_top5_off = 0

    for query in queries:
        payload = {"query": query, "filters": {
            "user_id": args.user_id}, "top_k": args.top_k}
        try:
            off = _post(url, {**payload, "decay_override": False}, headers)
            on = _post(url, {**payload, "decay_override": True}, headers)
        except (HTTPError, URLError) as exc:
            raise SystemExit(f"search failed for {query!r}: {exc}") from exc
        off_ids = _ids(off)
        on_ids = _ids(on)
        movements.append(movement_rate(off_ids, on_ids))
        print(f"\nQ: {query}")
        print("  off", off_ids)
        print("  on ", on_ids)
        print(f"  movement {movements[-1]:.0%}")

        for mem_id in core_ids:
            off_rank = rank_of(off_ids, mem_id)
            on_rank = rank_of(on_ids, mem_id)
            if off_rank and on_rank:
                core_deltas.append(off_rank - on_rank)

        for mem_id in stale_ids:
            off_rank = rank_of(off_ids, mem_id)
            on_rank = rank_of(on_ids, mem_id)
            if off_rank and on_rank:
                stale_deltas.append(on_rank - off_rank)
            if off_rank and off_rank <= 5:
                stale_in_top5_off += 1
            if on_rank and on_rank <= 5:
                stale_in_top5_on += 1

    print("\n=== summary ===")
    mean_move = statistics.mean(movements) if movements else 0.0
    print(f"perturbation (position move rate) {mean_move:.1%}  target 10-30%")
    if core_deltas:
        print(
            f"personal_core rank gain (off-on, + is better) mean={statistics.mean(core_deltas):.2f}")
        if min(core_deltas) < 0:
            print("FAIL life-line: at least one personal_core dropped")
        else:
            print("PASS life-line: personal_core did not drop")
    if stale_deltas:
        print(
            f"stale rank drop mean={statistics.mean(stale_deltas):.2f}  target >=3")
        print(
            f"stale in top-5  off={stale_in_top5_off} on={stale_in_top5_on}  target on ~0")


if __name__ == "__main__":
    main()
