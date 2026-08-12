#!/usr/bin/env python3
"""
Memory Cleanup: Remove semantic duplicates (sim > 0.95).
Strategy: keep the longest text per cluster, delete the rest.
Backs up deleted records to memories_cleanup_archive before removing.
"""

import json
import sys
from collections import defaultdict
from psycopg import connect

DB_URL = "postgresql://wuhonglei:ugbzVrahrP9hG_9@postgres:5432/postgres"
USER_ID = "c7d40833-6b26-4696-828f-a94b9de5b47d"
REPORT_PATH = "/tmp/semantic_dup_report.json"
SIM_THRESHOLD = 0.95


def build_clusters(pairs):
    """Union-Find: group overlapping pairs into clusters."""
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p["id_1"], p["id_2"])

    clusters = defaultdict(set)
    for node in parent:
        clusters[find(node)].add(node)

    return clusters


def main():
    dry_run = "--execute" not in sys.argv

    print("=" * 60)
    print("  Memory Cleanup — Semantic Duplicates (sim > 0.95)")
    print("=" * 60)
    print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"  Threshold: {SIM_THRESHOLD}")
    print()

    # 1. Load report
    print("[1/5] Loading detection report...")
    with open(REPORT_PATH) as f:
        report = json.load(f)

    all_pairs = report["pairs"]["exact_dup"] + report["pairs"]["high_dup"]
    print(f"  Found {len(all_pairs)} pairs with sim > {SIM_THRESHOLD}")

    # 2. Build clusters
    print("\n[2/5] Building duplicate clusters...")
    clusters = build_clusters(all_pairs)
    print(f"  {len(clusters)} clusters formed")

    # 3. Connect to DB, decide keep/delete
    print("\n[3/5] Analyzing clusters...")
    conn = connect(DB_URL)
    cur = conn.cursor()

    to_delete = []
    kept = []
    cluster_details = []

    for cid, members in clusters.items():
        # Fetch text lengths for all members
        ids = list(members)
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(f"""
            SELECT id, payload->>'data', LENGTH(payload->>'data')
            FROM memories
            WHERE id IN ({placeholders})
        """, ids)
        rows = cur.fetchall()
        rows.sort(key=lambda r: -(r[2] or 0))  # longest first

        keep_id = rows[0][0]
        keep_text = (rows[0][1] or "")[:60]
        kept.append(keep_id)

        cluster_info = {
            "keep": str(keep_id),
            "keep_text": keep_text,
            "delete": [],
        }

        for row in rows[1:]:
            del_id, del_text, del_len = row
            to_delete.append(del_id)
            cluster_info["delete"].append({
                "id": str(del_id),
                "text": (del_text or "")[:60],
                "length": del_len,
            })

        cluster_details.append(cluster_info)

    print(f"  Memories to KEEP:   {len(kept)}")
    print(f"  Memories to DELETE: {len(to_delete)}")

    # Show top clusters
    print(f"\n  Top 10 clusters (by delete count):")
    cluster_details.sort(key=lambda c: -len(c["delete"]))
    for i, c in enumerate(cluster_details[:10], 1):
        print(f"  [{i}] keep: {c['keep_text']}")
        for d in c["delete"]:
            print(f"      del:  {d['text']}")
        print()

    # 4. Archive and delete
    print("[4/5] Archiving and deleting...")

    if dry_run:
        print("  [DRY RUN] No changes made. Use --execute to apply.")
    else:
        # Create archive table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories_cleanup_archive (
                id uuid,
                vector vector(1024),
                payload jsonb,
                archived_at timestamptz DEFAULT NOW(),
                reason text
            )
        """)

        # Archive
        for del_id in to_delete:
            cur.execute("""
                INSERT INTO memories_cleanup_archive (id, vector, payload, reason)
                SELECT id, vector, payload, 'semantic_duplicate_sim_gt_0.95'
                FROM memories WHERE id = %s
            """, (del_id,))

        # Delete
        for del_id in to_delete:
            cur.execute("DELETE FROM memories WHERE id = %s", (del_id,))

        conn.commit()
        print(f"  Archived and deleted {len(to_delete)} memories")

        # Verify
        cur.execute("""
            SELECT COUNT(*) FROM memories
            WHERE payload->>'user_id' = %s
        """, (USER_ID,))
        remaining = cur.fetchone()[0]
        print(f"  Remaining memories: {remaining}")

    # 5. Summary
    print(f"\n[5/5] Summary")
    print(f"{'=' * 60}")
    print(f"  Clusters:         {len(clusters)}")
    print(f"  Kept:             {len(kept)}")
    print(f"  Deleted:          {len(to_delete)}")
    print(f"  Before:           4655")
    after = 4655 - len(to_delete)
    print(f"  After:            {after}")
    print(f"  Reduction:        {len(to_delete)/4655*100:.1f}%")
    print(f"  Archive table:    memories_cleanup_archive")
    print(f"  Rollback:         INSERT INTO memories SELECT id, vector, payload FROM memories_cleanup_archive WHERE reason = 'semantic_duplicate_sim_gt_0.95'")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"\n  ⚠️  This was a DRY RUN. Re-run with --execute to apply changes.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
