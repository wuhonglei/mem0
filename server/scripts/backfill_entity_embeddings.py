#!/usr/bin/env python3
"""
重新生成 memories_entities 表中所有实体的 embedding 向量。

用法：
  docker exec mem0-dev-mem0-1 python3 /app/scripts/backfill_entity_embeddings.py [--batch-size N]
"""

import json
import os
import urllib.request

import psycopg
from psycopg.rows import dict_row


def _build_dsn() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "wuhonglei")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DB_DSN = _build_dsn()
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MEM0_EMBEDDER_MODEL", "text-embedding-v4")
BASE_URL = os.environ.get("MEM0_EMBEDDER_BASE_URL", "https://llm-3jh83tx6m32lqpob.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量生成 embedding"""
    data = json.dumps({
        "model": MODEL,
        "input": texts,
        "encoding_format": "float"
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/embeddings",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    # 按 index 排序确保顺序正确
    sorted_data = sorted(result["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重新生成实体 embedding")
    parser.add_argument("--batch-size", type=int, default=10, help="批量大小（API 限制最大 10）")
    args = parser.parse_args()

    print(f"连接数据库: {DB_DSN.split('@')[1]}")
    conn = psycopg.connect(DB_DSN, row_factory=dict_row)

    try:
        # 获取所有实体
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, payload->>'data' as text FROM memories_entities ORDER BY id")
            entities = cur.fetchall()

        print(f"共 {len(entities)} 个实体需要生成 embedding")

        total = len(entities)
        success = 0
        failed = 0

        for i in range(0, total, args.batch_size):
            batch = entities[i:i + args.batch_size]
            texts = [e["text"] for e in batch]
            ids = [e["id"] for e in batch]

            # 跳过已成功生成的实体（非零向量）
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT id FROM memories_entities WHERE id = %s AND vector::text NOT LIKE %s", (ids[0], '[0,0,0%'))
                if cur.fetchone():
                    continue

            try:
                vectors = embed_batch(texts)

                # 更新数据库
                with conn.cursor() as cur:
                    for entity_id, vector in zip(ids, vectors):
                        cur.execute("""
                            UPDATE memories_entities
                            SET vector = %s::vector
                            WHERE id = %s
                        """, (str(vector), entity_id))
                conn.commit()
                success += len(batch)
                print(f"  [{min(i + args.batch_size, total)}/{total}] 已更新 {len(batch)} 条")

                # 限流
                import time
                time.sleep(0.1)

            except Exception as e:
                failed += len(batch)
                print(f"  [ERROR] 批次 {i+1}-{min(i+args.batch_size, total)} 失败: {e}")
                conn.rollback()
                # 如果连续失败，暂停一下
                import time
                time.sleep(1)

        print(f"\n完成: 成功 {success}, 失败 {failed}, 总计 {total}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
