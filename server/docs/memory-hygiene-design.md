# Memory Hygiene Pipeline — 设计方案

> 记忆数据定期清理与质量保障方案

## 1. 背景

mem0 的 `memories` 表随着使用不断增长。以 `user_id = c7d40833-...` 为例，当前已有 **4655 条**记忆，其中存在：

- **16 组完全重复**（相同内容出现 2 次）
- **语义重复**（同一事实以不同措辞多次记录，如"用户喜欢 Python"出现 3 次）
- **事实冲突**（如儿子名字"思锦"与"思瑾"混用、足球偏好 C 罗 vs 梅西未标注时间演变）
- **信息碎片化**（同一主题拆成多条，如 Traveloka 相关 8 条、Zalando 相关 174 条）

这些问题会降低检索精度、浪费上下文窗口，并可能导致 LLM 引用过时或矛盾信息。

## 2. 数据库结构

```sql
-- memories 表结构
CREATE TABLE memories (
    id      uuid PRIMARY KEY,
    vector  vector(1024),       -- DashScope text-embedding-v4 向量
    payload jsonb               -- 记忆元数据
);

-- payload 结构
{
    "data": "记忆内容文本",
    "user_id": "c7d40833-6b26-4696-828f-a94b9de5b47d",
    "created_at": "2026-08-12T06:46:07.242346+00:00",
    "updated_at": "2026-08-12T06:46:07.242346+00:00",
    "attributed_to": "user | assistant | (empty)",
    "hash": "...",
    "text_lemmatized": "..."
}
```

已有索引：

- `memories_pkey` — 主键 (id)
- `memories_hnsw_idx` — HNSW 向量索引 (vector_cosine_ops)
- `memories_text_lemmatized_idx` — GIN 全文索引 (text_lemmatized)

## 3. 清理流水线架构

```
┌──────────────────────────────────────────────────────────────┐
│                 Memory Hygiene Pipeline                      │
│                                                              │
│  Phase 1            Phase 2             Phase 3              │
│  去重检测       ──▶  冲突检测       ──▶  执行清理            │
│  (Dedup)            (Conflict)          (Cleanup)            │
│                                                              │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ 1.1 完全   │   │ 2.1 规则匹配 │   │ 自动: 删重复      │   │
│  │     重复   │   │     (快/浅)  │   │ 半自动: 合并语义  │   │
│  ├────────────┤   ├──────────────┤   │ 人工: 解决冲突    │   │
│  │ 1.2 近似   │   │ 2.2 LLM 判断 │   └──────────────────┘   │
│  │     重复   │   │     (慢/深)  │                           │
│  ├────────────┤   └──────────────┘                           │
│  │ 1.3 语义   │                                              │
│  │     重复   │                                              │
│  └────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

## 4. Phase 1: 去重检测

### 4.1 完全重复（Exact Duplicate）

**判定条件**：`payload->>'data'` 字段完全相同。

**SQL 检测**：

```sql
SELECT payload->>'data' AS data, COUNT(*) AS cnt
FROM memories
WHERE payload->>'user_id' = :user_id
GROUP BY payload->>'data'
HAVING COUNT(*) > 1
ORDER BY cnt DESC;
```

**SQL 清理**（保留最新一条，删除旧的）：

```sql
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY payload->>'data'
               ORDER BY (payload->>'created_at') DESC
           ) AS rn
    FROM memories
    WHERE payload->>'user_id' = :user_id
)
DELETE FROM memories
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
```

**风险等级**：零风险。内容完全相同，删除旧记录无信息损失。

### 4.2 近似重复（Near Duplicate — Hash 级别）

**判定条件**：对 `data` 做归一化（去除标点、空格、全角半角转换）后取 MD5，hash 相同即为近似重复。

**检测逻辑**：

```python
import hashlib
import re

def normalize(text: str) -> str:
    """归一化：去除标点、空格、统一全角半角"""
    text = text.strip()
    text = re.sub(r'[，。！？、；：""''（）《》【】\s,.!?;:\'"()\[\]{}]', '', text)
    # 全角转半角
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)

def content_hash(text: str) -> str:
    return hashlib.md5(normalize(text).encode()).hexdigest()
```

**清理策略**：同 4.1，保留最新。
**风险等级**：极低。仅标点/空格差异，语义完全一致。

### 4.3 语义重复（Semantic Duplicate — 向量级别）

**判定条件**：两条记忆的 embedding 向量余弦相似度 > 0.92。

**SQL 检测**：

```sql
-- 注意：全量两两比对代价为 O(n²)，4655 条约 1080 万对
-- 实际执行需分批或限制范围
SELECT a.id AS id_a,
       b.id AS id_b,
       a.payload->>'data' AS data_a,
       b.payload->>'data' AS data_b,
       1 - (a.vector <=> b.vector) AS similarity
FROM memories a
JOIN memories b ON a.id < b.id
WHERE a.payload->>'user_id' = :user_id
  AND b.payload->>'user_id' = :user_id
  AND 1 - (a.vector <=> b.vector) > 0.92
ORDER BY similarity DESC;
```

**优化策略**（避免 O(n²) 全表扫描）：

```sql
-- 方案A：按时间窗口限制，只检查最近 N 天新增的记忆与历史记忆的重复
-- 新增记忆通常 < 50 条/天，与 4655 条比对 ≈ 23 万对，可接受
SELECT a.id, b.id, 1 - (a.vector <=> b.vector) AS similarity
FROM memories a, memories b
WHERE a.id < b.id
  AND a.payload->>'user_id' = :user_id
  AND b.payload->>'user_id' = :user_id
  AND (a.payload->>'created_at')::timestamp > NOW() - INTERVAL '7 days'
  AND 1 - (a.vector <=> b.vector) > 0.92;

-- 方案B：对已有 HNSW 索引，用单条向量做近邻搜索
-- 对每条新记忆，检索 top-10 最近邻，相似度 > 0.92 的标记为重复
SELECT id, 1 - (vector <=> :query_vector) AS similarity
FROM memories
WHERE payload->>'user_id' = :user_id
ORDER BY vector <=> :query_vector
LIMIT 10;
```

**清理策略**：

- 相似度 > 0.95：自动删除较短的一条（信息量更少）
- 相似度 0.92~0.95：生成报告，人工确认或 LLM 辅助合并

**风险等级**：中等。需人工确认边界情况。

## 5. Phase 2: 冲突检测

### 5.1 规则匹配（Rule-based — 快速扫描）

**原理**：预定义正反义关键词对，对同一主题下的记忆做正负匹配。

```python
CONFLICT_RULES = [
    {
        "name": "偏好反转",
        "topic_keywords": ["喜欢", "爱好", "偏好"],
        "positive": ["喜欢", "爱吃", "最爱", "偏好"],
        "negative": ["不喜欢", "不爱吃", "讨厌", "不再"],
    },
    {
        "name": "事实否定",
        "topic_keywords": ["是", "在", "有"],
        "positive": ["是", "在", "有", "住在"],
        "negative": ["不是", "不在", "没有", "不住"],
    },
    {
        "name": "能力否定",
        "topic_keywords": ["支持", "可以", "能够"],
        "positive": ["支持", "可以", "能够"],
        "negative": ["不支持", "不可以", "无法"],
    },
    {
        "name": "数值矛盾",
        "topic_keywords": ["价格", "薪资", "年龄", "数量"],
        "positive": [],  # 需要提取数值做比较
        "negative": [],
    },
]

def detect_rule_conflicts(memories: list[dict]) -> list[tuple]:
    """规则冲突检测，返回冲突记忆对列表"""
    conflicts = []
    for i, m1 in enumerate(memories):
        t1 = m1.get("data", "")
        for j, m2 in enumerate(memories):
            if j <= i:
                continue
            t2 = m2.get("data", "")
            for rule in CONFLICT_RULES:
                # 检查是否同主题
                t1_topic = any(k in t1 for k in rule["topic_keywords"])
                t2_topic = any(k in t2 for k in rule["topic_keywords"])
                if not (t1_topic and t2_topic):
                    continue
                # 检查正负对立
                t1_pos = any(k in t1 for k in rule["positive"])
                t2_neg = any(k in t2 for k in rule["negative"])
                t1_neg = any(k in t1 for k in rule["negative"])
                t2_pos = any(k in t2 for k in rule["positive"])
                if (t1_pos and t2_neg) or (t1_neg and t2_pos):
                    conflicts.append((m1, m2, rule["name"]))
    return conflicts
```

**优点**：快（毫秒级），无额外成本。
**缺点**：只能覆盖预定义模式，无法发现隐性冲突（如"月薪4万" vs "月薪6万"）。

### 5.2 LLM 判断（Deep Analysis — 精确检测）

**原理**：对规则匹配无法判定的记忆对，用 LLM 做语义级别的冲突判断。

**关键优化**：不做全量两两比对（O(n²) 不现实），只对 Phase 1.3 中相似度 0.75~0.92 的"相关但不完全重复"记忆对做 LLM 检测。

```python
CONFLICT_PROMPT = """你是一个记忆冲突检测器。判断以下两条记忆是否存在事实冲突。

记忆A: {mem_a}
记忆B: {mem_b}

请严格按以下格式回答：
- CONFLICT: 存在事实矛盾。原因: <简述矛盾点>
- COMPATIBLE: 不矛盾。原因: <说明为何可共存>
- EVOLUTION: 是同一事实的时间演变。变化: <A→B的变化>

只输出以上三种判定之一，不要输出其他内容。"""
```

**批量执行策略**：

```python
async def batch_conflict_check(pairs: list[tuple], llm_client) -> list[dict]:
    """批量 LLM 冲突检测"""
    results = []
    # 控制并发，避免触发 rate limit
    semaphore = asyncio.Semaphore(5)

    async def check_one(pair):
        async with semaphore:
            mem_a, mem_b = pair
            prompt = CONFLICT_PROMPT.format(
                mem_a=mem_a["data"], mem_b=mem_b["data"]
            )
            response = await llm_client.chat(prompt)
            return {
                "id_a": mem_a["id"],
                "id_b": mem_b["id"],
                "data_a": mem_a["data"][:80],
                "data_b": mem_b["data"][:80],
                "judgment": response,
            }

    tasks = [check_one(p) for p in pairs]
    results = await asyncio.gather(*tasks)
    return results
```

**成本估算**（以 4655 条记忆为例）：

| 筛选阶段 | 记忆对数量 | 说明 |
|----------|-----------|------|
| 全量两两比对 | ~1080 万 | 不可行 |
| 限制同主题（关键词过滤） | ~5000 | 可接受 |
| 相似度 0.75~0.92 | ~200 | 最终 LLM 调用量 |

**优点**：能发现隐性冲突（数值、时间线、逻辑矛盾）。
**缺点**：有 API 调用成本，延迟较高。

## 6. Phase 3: 执行策略

### 6.1 清理动作矩阵

| 检测类型 | 自动/人工 | 清理动作 | 回滚方式 |
|----------|----------|---------|---------|
| 完全重复 (1.1) | ✅ 自动 | 删除旧记录，保留最新 | 从备份恢复 |
| 近似重复 (1.2) | ✅ 自动 | 同上 | 从备份恢复 |
| 语义重复 >0.95 (1.3) | ✅ 自动 | 保留较长/较新的一条 | 从备份恢复 |
| 语义重复 0.92~0.95 (1.3) | 🔄 半自动 | 生成报告，人工确认 | N/A |
| 规则冲突 (2.1) | 🔄 半自动 | 生成报告，人工决定 | N/A |
| LLM 冲突 (2.2) | ⚠️ 人工 | 生成报告，人工决定 | N/A |

### 6.2 清理前备份

```sql
-- 备份即将删除的记忆到独立表
CREATE TABLE IF NOT EXISTS memories_cleanup_archive AS
SELECT *, NOW() AS archived_at FROM memories WHERE FALSE;

-- 在每次清理前插入待删除记录
INSERT INTO memories_cleanup_archive
SELECT *, NOW() AS archived_at
FROM memories
WHERE id IN (:ids_to_delete);
```

### 6.3 清理报告格式

```json
{
    "run_at": "2026-08-12T10:00:00Z",
    "user_id": "c7d40833-...",
    "summary": {
        "total_memories": 4655,
        "exact_duplicates_found": 16,
        "exact_duplicates_deleted": 16,
        "near_duplicates_found": 3,
        "near_duplicates_deleted": 3,
        "semantic_duplicates_found": 12,
        "semantic_auto_deleted": 8,
        "semantic_pending_review": 4,
        "rule_conflicts_found": 2,
        "llm_conflicts_found": 1
    },
    "pending_review": [
        {
            "type": "semantic_duplicate",
            "id_a": "...",
            "data_a": "用户最喜欢的编程语言是Python...",
            "id_b": "...",
            "data_b": "用户最喜爱的编程语言是 Python...",
            "similarity": 0.94,
            "action": "pending_review"
        }
    ],
    "conflicts": [
        {
            "type": "rule_conflict",
            "rule": "偏好反转",
            "id_a": "...",
            "data_a": "用户最喜欢的足球运动员是C罗",
            "id_b": "...",
            "data_b": "用户当前最喜欢的足球运动员是梅西",
            "judgment": "EVOLUTION"
        }
    ]
}
```

## 7. 执行频率

| 阶段 | 频率 | 预估耗时 | 预估 API 成本 |
|------|------|---------|-------------|
| 1.1 完全去重 | 每天 | < 1 秒 (纯 SQL) | ¥0 |
| 1.2 近似去重 | 每天 | < 1 秒 (纯 SQL) | ¥0 |
| 1.3 语义去重 | 每周 | ~10 秒 (向量查询) | ¥0 |
| 2.1 规则冲突 | 每周 | ~5 秒 (Python) | ¥0 |
| 2.2 LLM 冲突 | 每月 | ~5 分钟 (API) | ~¥0.5 |

## 8. 落地形式

### 方案 A：独立脚本（推荐起步方案）

```
server/scripts/memory_hygiene.py
├── --user-id <uuid>      指定用户（默认全部）
├── --phase <1|2|all>     执行阶段
├── --dry-run              只报告，不执行清理
├── --execute              执行清理（Phase 1 自动删除）
├── --output <json|text>   报告格式
└── --archive              清理前是否备份到 archive 表
```

**使用方式**：

```bash
# 仅检测，查看报告
python server/scripts/memory_hygiene.py --user-id c7d40833-... --phase all --dry-run

# 执行 Phase 1 自动清理（Phase 2 只报告）
python server/scripts/memory_hygiene.py --user-id c7d40833-... --phase 1 --execute --archive

# 执行全部（Phase 1 自动，Phase 2 报告供人工审核）
python server/scripts/memory_hygiene.py --phase all --execute --archive
```

### 方案 B：Hermes Cron Job（自动化调度）

```yaml
# 每周一 09:00 执行 Phase 1 自动清理 + Phase 2 报告
name: memory-hygiene-weekly
schedule: "0 9 * * 1"
script: server/scripts/memory_hygiene.py
args: --phase all --execute --archive --output text
deliver: telegram  # 推送报告到 Telegram
```

### 方案 C：集成到 mem0 server（长期方案）

在 `server/main.py` 中添加后台任务：

```python
# server/routers/hygiene.py
from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/hygiene", tags=["hygiene"])

@router.post("/run")
async def run_hygiene(background_tasks: BackgroundTasks, user_id: str = None):
    """触发记忆清理（后台执行）"""
    background_tasks.add_task(run_hygiene_pipeline, user_id)
    return {"status": "started"}

@router.get("/report")
async def get_report(user_id: str):
    """获取最新清理报告"""
    return get_latest_report(user_id)
```

### 推荐路径

```
起步 (Week 1)          稳定 (Week 2-4)        长期 (Month 2+)
    │                       │                       │
    ▼                       ▼                       ▼
 方案 A                方案 A + B              方案 A + B + C
 独立脚本              加上定时调度            集成到 server
 手动执行 dry-run       自动 Phase 1           API 端点
 验证结果              Phase 2 报告推送        Dashboard 可视化
```

## 9. 阈值参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EXACT_DUP` | (自动检测) | data 字段完全相同 |
| `NEAR_DUP_HASH` | (归一化后 MD5) | 去除标点空格后相同 |
| `SEMANTIC_DUP_AUTO` | 0.95 | 超过此阈值自动删除 |
| `SEMANTIC_DUP_REVIEW` | 0.92 | 超过此阈值标记待审核 |
| `SEMANTIC_CONFLICT_WINDOW` | 0.75 ~ 0.92 | 此范围内的对送入 LLM 检测冲突 |
| `MAX_LLM_PAIRS` | 500 | 每次运行最多送入 LLM 的对数 |
| `LLM_CONCURRENCY` | 5 | LLM 并发调用数 |

## 10. 已知问题与当前数据快照

以下为 2026-08-12 对 `user_id = c7d40833-...` 的分析结果：

### 10.1 完全重复（16 组）

每组 2 条，共 32 条记录，应保留 16 条、删除 16 条。

| 内容摘要 | 重复次数 |
|----------|---------|
| 用户喜欢重口味的食物 | 2 |
| 用户喜欢吃荔枝，特别喜欢桂味品种 | 2 |
| 用户在2026年8月6日晚上吃了一碗红烧牛肉面... | 2 |
| 已建议用户，感冒发烧期间不建议吃咖喱牛肉... | 2 |
| 清晖园是广东四大名园之一... | 2 |
| 湖北荆州市荆州区有一座明月公园... | 2 |
| Shopee内部发布了一款面向开发者的助手工具"ShopeeBot"... | 2 |
| Prompt Injection的本质是LLM无法可靠区分... | 2 |
| agent-browser 相关（4 条重复） | 各 2 |
| camofox-browser 相关（2 条重复） | 各 2 |

### 10.2 事实冲突

| 冲突类型 | 记忆 A | 记忆 B | 建议处理 |
|----------|--------|--------|---------|
| 名字不一致 | 儿子名叫吴思**锦** | 儿子名叫吴思**瑾** | 核实后统一 |
| 偏好未标注演变 | 最喜欢的足球运动员是 C 罗 | 当前最喜欢的是梅西（此前是 C 罽） | 为 A 添加时间标记 |

### 10.3 语义重复（高频话题）

| 话题 | 语义相似条数 | 建议 |
|------|------------|------|
| "最喜欢的编程语言是 Python" | 3 条 | 合并为 1 条 |
| 咖喱牛肉（喜欢/变种/建议） | 5 条 | 合并为 1-2 条 |
| Traveloka 公司信息 | 8 条 | 合并为 2-3 条 |
| Zalando 求职准备 | 174 条 | 按子话题聚合，保留关键决策 |
| Shopee 工作相关 | 147 条 | 同上 |
| PEARL 论文研究 | 50 条 | 按 Agent 类型聚合 |

## 11. 后续演进

1. **Dashboard 可视化**：在 mem0-dashboard 中添加记忆健康度面板
2. **实时去重**：在 `add()` 时增加语义去重（已有 hash 去重，补充向量去重）
3. **冲突预防**：在 `add()` 时检测与已有记忆的冲突，提示 LLM 更新而非新增
4. **记忆衰减**：对长期未被检索引用的记忆降低权重或标记为"冷记忆"
5. **自动合并**：用 LLM 将语义重复的记忆合并为一条更完整的信息
