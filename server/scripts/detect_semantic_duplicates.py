#!/usr/bin/env python3
"""Semantic duplicate detection using embedding vectors."""

import sys
import json
import numpy as np
from psycopg import connect

DB_URL = "postgresql://wuhonglei:ugbzVrahrP9hG_9@postgres:5432/postgres"
USER_ID = "c7d40833-6b26-4696-828f-a94b9de5b47d"
THRESHOLDS = [0.98, 0.95, 0.92, 0.90, 0.85, 0.80]

def parse_vector(vec_str: str) -> np.ndarray:
    """Parse pgvector string '[0.1,0.2,...]' to numpy array."""
    return np.array([float(x) for x in vec_str.strip('[]').split(',')], dtype=np.float32)

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize rows
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = vectors / norms
    # Cosine similarity = dot product of normalized vectors
    return normalized @ normalized.T

def main():
    print("=" * 70)
    print("  Semantic Duplicate Detection — Embedding Cosine Similarity")
    print("=" * 70)

    # 1. Fetch all memories with vectors
    print("\n[1/4] Fetching memories and vectors from database...")
    conn = connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, payload->>'data', vector::text
        FROM memories
        WHERE payload->>'user_id' = %s
        ORDER BY (payload->>'created_at')::timestamp
    """, (USER_ID,))

    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vectors = np.array([parse_vector(r[2]) for r in rows], dtype=np.float32)

    print(f"  Loaded {len(ids)} memories, vector dim = {vectors.shape[1]}")

    # 2. Compute full cosine similarity matrix
    print("\n[2/4] Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity_matrix(vectors)
    # Zero out diagonal (self-similarity = 1.0)
    np.fill_diagonal(sim_matrix, 0.0)
    print(f"  Matrix shape: {sim_matrix.shape}")

    # 3. Find pairs above each threshold
    print("\n[3/4] Scanning thresholds...")
    i_upper, j_upper = np.triu_indices(len(ids), k=1)

    for threshold in THRESHOLDS:
        mask = sim_matrix[i_upper, j_upper] > threshold
        count = mask.sum()
        print(f"  similarity > {threshold:.2f}: {count} pairs")

    # 4. Detailed report for threshold 0.90
    print(f"\n[4/4] Detailed pairs with similarity > 0.90:")
    print("=" * 70)

    threshold = 0.90
    mask = sim_matrix[i_upper, j_upper] > threshold
    pair_indices = list(zip(i_upper[mask], j_upper[mask]))

    # Sort by similarity descending
    pair_sims = [(i, j, sim_matrix[i, j]) for i, j in pair_indices]
    pair_sims.sort(key=lambda x: -x[2])

    # Group by severity
    exact_dup = []      # > 0.98
    high_dup = []       # 0.95 - 0.98
    medium_dup = []     # 0.92 - 0.95
    low_dup = []        # 0.90 - 0.92

    for i, j, sim in pair_sims:
        entry = {
            "id_1": str(ids[i]),
            "id_2": str(ids[j]),
            "text_1": texts[i][:100] if texts[i] else "",
            "text_2": texts[j][:100] if texts[j] else "",
            "similarity": round(float(sim), 4),
        }
        if sim > 0.98:
            exact_dup.append(entry)
        elif sim > 0.95:
            high_dup.append(entry)
        elif sim > 0.92:
            medium_dup.append(entry)
        else:
            low_dup.append(entry)

    # Print results
    def print_group(name, entries, show_all=False):
        print(f"\n{'─' * 70}")
        print(f"  {name} ({len(entries)} pairs)")
        print(f"{'─' * 70}")
        display = entries if show_all else entries[:20]
        for idx, e in enumerate(display, 1):
            print(f"\n  [{idx}] similarity = {e['similarity']}")
            print(f"      A: {e['text_1']}")
            print(f"      B: {e['text_2']}")
            print(f"      id_a: {e['id_1'][:12]}...")
            print(f"      id_b: {e['id_2'][:12]}...")
        if not show_all and len(entries) > 20:
            print(f"\n  ... and {len(entries) - 20} more pairs")

    print_group("🔴 几乎完全相同 (similarity > 0.98)", exact_dup, show_all=True)
    print_group("🟠 高度重复 (0.95 < similarity ≤ 0.98)", high_dup, show_all=True)
    print_group("🟡 中度重复 (0.92 < similarity ≤ 0.95)", medium_dup)
    print_group("🟢 轻度相似 (0.90 < similarity ≤ 0.92)", low_dup)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  Total memories:          {len(ids)}")
    print(f"  🔴 Almost identical (>0.98): {len(exact_dup)} pairs")
    print(f"  🟠 High duplicate  (>0.95):  {len(high_dup)} pairs")
    print(f"  🟡 Medium duplicate (>0.92): {len(medium_dup)} pairs")
    print(f"  🟢 Low similar     (>0.90):  {len(low_dup)} pairs")
    print(f"  ─────────────────────────────")
    print(f"  Total similar pairs (>0.90): {len(pair_sims)} pairs")

    # Estimate cleanup impact
    removable = set()
    for i, j, sim in pair_sims:
        if sim > 0.95:
            # Keep the one with longer text
            if len(texts[i] or "") >= len(texts[j] or ""):
                removable.add(ids[j])
            else:
                removable.add(ids[i])
    print(f"\n  Auto-removable (sim > 0.95, keep longer): {len(removable)} memories")
    print(f"  After cleanup: ~{len(ids) - len(removable)} memories")

    # Save JSON report
    report = {
        "user_id": USER_ID,
        "total_memories": len(ids),
        "thresholds": {str(t): int((sim_matrix[i_upper, j_upper] > t).sum()) for t in THRESHOLDS},
        "pairs": {
            "exact_dup": exact_dup,
            "high_dup": high_dup,
            "medium_dup": medium_dup,
            "low_dup": low_dup,
        },
        "auto_removable_count": len(removable),
        "auto_removable_ids": [str(r) for r in removable],
    }
    with open("/tmp/semantic_dup_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved to /tmp/semantic_dup_report.json")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
