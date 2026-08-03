#!/usr/bin/env python3
"""
Backfill text_lemmatized field for old memories.

This script updates records where text_lemmatized == data (unlemmatized)
by running spaCy lemmatization on the original text.

Usage:
  docker exec mem0-dev-mem0-1 python3 /app/scripts/backfill_text_lemmatized.py [--dry-run] [--batch-size N]
"""

import argparse
import json
import sys
import time

import psycopg
from psycopg.rows import dict_row


DB_DSN = "postgresql://postgres:postgres@postgres:5432/postgres"


def get_unlemmatized_count(conn):
    """Count records where text_lemmatized needs fixing."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as cnt FROM memories
            WHERE payload->>'data' IS NOT NULL
              AND (
                -- Never lemmatized
                payload->>'text_lemmatized' = payload->>'data'
                OR payload->>'text_lemmatized' IS NULL
                OR payload->>'text_lemmatized' = ''
                -- CJK text that was incorrectly lemmatized by old code
                OR (
                  payload->>'data' ~ '[\u4e00-\u9fff]'
                  AND payload->>'text_lemmatized' != payload->>'data'
                )
              )
        """)
        return cur.fetchone()["cnt"]


def fetch_unlemmatized_batch(conn, batch_size, offset):
    """Fetch a batch of unlemmatized records."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, payload->>'data' as data, payload->>'text_lemmatized' as text_lemmatized
            FROM memories
            WHERE payload->>'data' IS NOT NULL
              AND (
                -- Never lemmatized
                payload->>'text_lemmatized' = payload->>'data'
                OR payload->>'text_lemmatized' IS NULL
                OR payload->>'text_lemmatized' = ''
                -- CJK text that was incorrectly lemmatized by old code
                OR (
                  payload->>'data' ~ '[\u4e00-\u9fff]'
                  AND payload->>'text_lemmatized' != payload->>'data'
                )
              )
            ORDER BY id
            LIMIT %s OFFSET %s
        """, (batch_size, offset))
        return cur.fetchall()


def update_text_lemmatized(conn, record_id, lemmatized):
    """Update text_lemmatized in the payload JSONB."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE memories
            SET payload = jsonb_set(payload, '{text_lemmatized}', %s::jsonb)
            WHERE id = %s
        """, (json.dumps(lemmatized), record_id))


def main():
    parser = argparse.ArgumentParser(description="Backfill text_lemmatized field")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset (default: 0)")
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all, default: 0)")
    args = parser.parse_args()

    # Import lemmatization function
    try:
        from mem0.utils.lemmatization import lemmatize_for_bm25
        print("✓ spaCy lemmatization loaded")
    except Exception as e:
        print(f"✗ Failed to load lemmatization: {e}")
        sys.exit(1)

    # Connect to database
    print(f"Connecting to PostgreSQL...")
    conn = psycopg.connect(DB_DSN, autocommit=False)
    # psycopg3 uses row_factory for dict-like access
    conn.row_factory = dict_row

    try:
        # Count total unlemmatized records
        total = get_unlemmatized_count(conn)
        print(f"Found {total} unlemmatized records")

        if total == 0:
            print("Nothing to do.")
            return

        # Calculate processing range
        start = args.offset
        end = start + args.limit if args.limit > 0 else total
        to_process = min(end - start, total - start)

        print(f"Processing records {start} to {start + to_process} (batch size: {args.batch_size})")
        if args.dry_run:
            print("DRY RUN - no changes will be made")
        print()

        processed = 0
        updated = 0
        skipped = 0
        start_time = time.time()

        offset = start
        while offset < start + to_process:
            batch_size = min(args.batch_size, start + to_process - offset)
            rows = fetch_unlemmatized_batch(conn, batch_size, offset)

            if not rows:
                break

            for row in rows:
                record_id = row["id"]
                original_text = row["data"]
                processed += 1

                # Run lemmatization
                lemmatized = lemmatize_for_bm25(original_text)

                # Skip only if the result matches what's already stored
                current_lemma = row.get("text_lemmatized") or ""
                if lemmatized == current_lemma:
                    skipped += 1
                    continue

                if not args.dry_run:
                    update_text_lemmatized(conn, record_id, lemmatized)
                    updated += 1
                else:
                    # Show sample in dry-run mode
                    if updated < 5:
                        print(f"  [{record_id}] {original_text[:60]}...")
                        print(f"    → {lemmatized[:60]}...")
                        print()
                    updated += 1

                # Progress update every 100 records
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (to_process - processed) / rate if rate > 0 else 0
                    print(f"  Progress: {processed}/{to_process} ({processed*100//to_process}%) "
                          f"| Updated: {updated} | Skipped: {skipped} "
                          f"| Rate: {rate:.0f}/s | ETA: {eta:.0f}s")

            # Commit batch
            if not args.dry_run:
                conn.commit()

            offset += batch_size

        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"Completed in {elapsed:.1f}s")
        print(f"  Processed: {processed}")
        print(f"  Updated:   {updated}")
        print(f"  Skipped:   {skipped} (Chinese/already lemmatized)")
        print("=" * 60)

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
