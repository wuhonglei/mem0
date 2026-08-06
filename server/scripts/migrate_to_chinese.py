#!/usr/bin/env python3
"""
迁移脚本：将英文记忆翻译为中文，并重建 entities。

功能：
1. 翻译 memories 表中英文 data 为中文
2. 更新 text_lemmatized（中文分词）
3. 重建 memories_entities（spaCy 实体提取）

用法：
  docker exec mem0-dev-mem0-1 python3 /app/scripts/migrate_to_chinese.py [--dry-run] [--batch-size N]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row


# ─── 配置 ─────────────────────────────────────────────────────────────────────

def _build_dsn() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "wuhonglei")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    db = os.environ.get("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DB_DSN = _build_dsn()

LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
OPENAI_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# CJK 字符检测
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')

# 批量翻译大小
DEFAULT_BATCH_SIZE = 20


def has_cjk(text: str) -> bool:
    """检测文本是否包含中文字符"""
    return bool(_CJK_RE.search(text))


# ─── LLM 翻译 ────────────────────────────────────────────────────────────────

def translate_batch(texts: list[str], max_retries: int = 3) -> list[str]:
    """批量翻译英文为中文"""
    if not texts:
        return []

    # 构建批量翻译 prompt
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"""将以下英文翻译为中文。每行一条，保持序号格式，只返回翻译结果。

{numbered}"""

    for attempt in range(max_retries):
        try:
            data = json.dumps({
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "extra_body": {"thinking": {"type": "disabled"}},
            }).encode()

            req = urllib.request.Request(
                f"{LLM_BASE_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                },
            )

            resp = urllib.request.urlopen(req, timeout=120)
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()

            # 解析翻译结果
            lines = content.strip().split("\n")
            translated = []
            for line in lines:
                # 去掉序号前缀
                line = re.sub(r'^\d+\.\s*', '', line.strip())
                if line:
                    translated.append(line)

            # 数量匹配检查
            if len(translated) == len(texts):
                return translated
            else:
                print(f"  [WARN] 翻译数量不匹配: 输入 {len(texts)}, 输出 {len(translated)}, 重试...")
                time.sleep(2)

        except Exception as e:
            print(f"  [ERROR] 翻译失败 (attempt {attempt+1}): {e}")
            time.sleep(3)

    # 全部失败，返回原文
    print(f"  [FATAL] 翻译失败，返回原文")
    return texts


# ─── lemmatization ────────────────────────────────────────────────────────────

def lemmatize_chinese(text: str) -> str:
    """中文分词（复用 mem0 的 lemmatization 逻辑）"""
    try:
        from mem0.utils.lemmatization import lemmatize_for_bm25
        return lemmatize_for_bm25(text)
    except ImportError:
        # fallback: 简单按字符分
        return " ".join(text)


# ─── 主逻辑 ────────────────────────────────────────────────────────────────────

def get_english_memories(conn) -> list[dict]:
    """获取所有英文记忆"""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT id, payload
            FROM memories
            WHERE payload->>'data' !~ '[\u4e00-\u9fff]'
            ORDER BY (payload->>'created_at')
        """)
        return cur.fetchall()


def update_memory(conn, memory_id: str, new_data: str, new_lemmatized: str):
    """更新记忆的 data 和 text_lemmatized"""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE memories
            SET payload = jsonb_set(
                jsonb_set(payload, '{data}', %s::jsonb),
                '{text_lemmatized}', %s::jsonb
            )
            WHERE id = %s
        """, (json.dumps(new_data), json.dumps(new_lemmatized), memory_id))


def extract_entities_from_text(text: str) -> list[tuple[str, str]]:
    """使用 spaCy 提取实体"""
    try:
        from mem0.utils.entity_extraction import extract_entities
        return extract_entities(text)
    except Exception as e:
        print(f"  [WARN] 实体提取失败: {e}")
        return []


def rebuild_entities(conn):
    """重建 memories_entities 表"""
    print("\n=== 重建 memories_entities ===")

    # 1. 清空 entities 表
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories_entities")
    print("已清空 memories_entities")

    # 2. 获取所有记忆
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, payload FROM memories")
        all_memories = cur.fetchall()

    print(f"共 {len(all_memories)} 条记忆需要处理")

    # 3. 按 user_id 分组处理
    entities_map = {}  # (user_id, normalized_text) -> {entity_type, text, memory_ids}

    for mem in all_memories:
        memory_id = str(mem["id"])
        payload = mem["payload"]
        text = payload.get("data", "")
        user_id = payload.get("user_id", "")

        if not text:
            continue

        # 提取实体
        entities = extract_entities_from_text(text)

        for entity_type, entity_text in entities:
            normalized = entity_text.strip().lower()
            if not normalized:
                continue

            key = (user_id, normalized)
            if key in entities_map:
                entities_map[key]["memory_ids"].add(memory_id)
            else:
                entities_map[key] = {
                    "entity_type": entity_type,
                    "text": entity_text.strip(),
                    "user_id": user_id,
                    "memory_ids": {memory_id},
                }

    print(f"提取到 {len(entities_map)} 个唯一实体")

    # 4. 批量插入实体（使用 mem0 embedding model 生成向量）
    import uuid
    from mem0.utils.factory import EmbedderFactory

    # 获取 embedding model
    embedder_config = {
        "provider": "openai",
        "config": {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "model": os.environ.get("MEM0_EMBEDDER_MODEL", "text-embedding-v4"),
            "embedding_dims": int(os.environ.get("MEM0_EMBEDDER_DIMENSION", "1024")),
        }
    }
    if os.environ.get("MEM0_EMBEDDER_BASE_URL"):
        embedder_config["config"]["http_client_providers"] = None
        embedder_config["config"]["base_url"] = os.environ["MEM0_EMBEDDER_BASE_URL"]

    try:
        embedder = EmbedderFactory.create(embedder_config["provider"], embedder_config["config"])
    except Exception as e:
        print(f"  [WARN] 无法创建 embedding model: {e}，使用空向量")
        embedder = None

    with conn.cursor() as cur:
        entity_list = list(entities_map.values())
        total_entities = len(entity_list)

        for i in range(0, total_entities, 50):
            batch = entity_list[i:i+50]
            texts = [e["text"] for e in batch]

            # 生成向量
            if embedder:
                try:
                    vectors = embedder.embed_batch(texts, "add")
                except Exception as e:
                    print(f"  [WARN] 批量 embedding 失败: {e}，使用空向量")
                    vectors = [[0.0] * 1024 for _ in texts]
            else:
                vectors = [[0.0] * 1024 for _ in texts]

            for entity, vector in zip(batch, vectors):
                entity_id = str(uuid.uuid4())
                payload = {
                    "data": entity["text"],
                    "user_id": entity["user_id"],
                    "entity_type": entity["entity_type"],
                    "linked_memory_ids": list(entity["memory_ids"]),
                }
                cur.execute("""
                    INSERT INTO memories_entities (id, vector, payload)
                    VALUES (%s, %s::vector, %s::jsonb)
                """, (entity_id, str(vector), json.dumps(payload)))

            conn.commit()
            print(f"  [{min(i+50, total_entities)}/{total_entities}] 已插入")

    print(f"已插入 {len(entities_map)} 个实体")


def main():
    parser = argparse.ArgumentParser(description="迁移英文记忆为中文")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不实际修改")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批量翻译大小")
    parser.add_argument("--skip-entities", action="store_true", help="跳过实体重建")
    parser.add_argument("--only-entities", action="store_true", help="只重建实体，跳过翻译")
    args = parser.parse_args()

    print(f"连接数据库: {DB_DSN.split('@')[1]}")
    conn = psycopg.connect(DB_DSN, row_factory=dict_row)

    try:
        if not args.only_entities:
            # ─── 步骤 1：翻译英文记忆 ─────────────────────────────────────
            print("\n=== 步骤 1：翻译英文记忆 ===")
            english_memories = get_english_memories(conn)
            print(f"找到 {len(english_memories)} 条英文记忆")

            if args.dry_run:
                print("[DRY RUN] 以下记忆将被翻译：")
                for mem in english_memories[:5]:
                    print(f"  - {mem['payload'].get('data', '')[:60]}...")
                print(f"  ... 共 {len(english_memories)} 条")
            else:
                total = len(english_memories)
                translated_count = 0

                for i in range(0, total, args.batch_size):
                    batch = english_memories[i:i + args.batch_size]
                    texts = [m["payload"].get("data", "") for m in batch]

                    print(f"\n[{i+1}-{min(i+args.batch_size, total)}/{total}] 翻译中...")

                    # 批量翻译
                    translated = translate_batch(texts)

                    # 更新数据库
                    for mem, new_data in zip(batch, translated):
                        new_lemmatized = lemmatize_chinese(new_data)
                        update_memory(conn, str(mem["id"]), new_data, new_lemmatized)
                        translated_count += 1

                    conn.commit()
                    print(f"  已更新 {len(batch)} 条")

                    # 限流
                    time.sleep(0.5)

                print(f"\n翻译完成: {translated_count}/{total}")

        if not args.skip_entities:
            # ─── 步骤 2：重建实体 ───────────────────────────────────────
            if args.dry_run:
                print("\n[DRY RUN] 跳过实体重建")
            else:
                rebuild_entities(conn)

        print("\n=== 迁移完成 ===")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
