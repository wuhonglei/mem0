"""Shared memory category taxonomy.

`category` is a first-class payload field: how long a fact stays true, then topic.
Decay is one consumer (it maps each class to a curve). Filters, dream, and
analytics can reuse the same six values without importing decay.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

CATEGORY_PERSONAL_CORE = "personal_core"
CATEGORY_PREFERENCES = "preferences"
CATEGORY_INTERESTS = "interests"
CATEGORY_STATE = "state"
CATEGORY_KNOWLEDGE = "knowledge"
CATEGORY_MISC = "misc"

MEMORY_CATEGORIES = frozenset(
    {
        CATEGORY_PERSONAL_CORE,
        CATEGORY_PREFERENCES,
        CATEGORY_INTERESTS,
        CATEGORY_STATE,
        CATEGORY_KNOWLEDGE,
        CATEGORY_MISC,
    }
)

_STATE_RE = re.compile(r"正在|计划|等待|对比")
# personal_core means "an identity, family or health fact about the user", so the
# cue has to be anchored to the user. An unanchored 职业/健康 also matches
# advice and third-party facts ("双轨制职业发展路径", "评估业务健康状况",
# "李兰迪出生于…"), which is how knowledge memories ended up on the slowest
# decay curve — where a wrong label is amplified instead of averaging out.
_IDENTITY_CUES = (
    r"姓名|名字|出生于|生日|生肖|星座|籍贯|老家|住在|居住|搬家|"
    r"配偶|妻子|丈夫|结婚|孩子|女儿|儿子|龙凤胎|双胞胎|父亲|母亲|家人|家庭|"
    r"过敏|病史|体检|身高|体重|血型"
)
_CORE_RE = re.compile(rf"(?:用户|我|本人)[^。；，、]{{0,12}}(?:{_IDENTITY_CUES})")
# Query/answer records and assistant-authored advice: never a fact about the user.
_ASSISTANT_SHAPED_RE = re.compile(
    r"用户(?:询问|问了|查询|问到|获得了)|助手(?:介绍|解释|说明|向用户|建议|推荐|指出|确认)|被建议"
)
_INTERESTS_RE = re.compile(r"推荐|行程|美食|景点|攻略|旅游|旅行|骑行路线|海鲜|早茶")
_PROFILE_RE = re.compile(r"技术栈|个人优势|端到端|工程|经验|简历|面试|职业|项目.*能力")


def sanitize_category(
    category: Optional[str],
    text: Optional[str] = None,
    attributed_to: Optional[str] = None,
) -> Optional[str]:
    """Reject a category the content cannot support.

    Only ``personal_core`` needs guarding today: it carries the slowest decay
    curve, so a wrong label there is amplified. A fact about the user cannot be
    introduced by the assistant alone, and a query record is not an identity
    fact; both fall back to the class their attribution implies.
    """
    if category != CATEGORY_PERSONAL_CORE:
        return category
    if attributed_to == "assistant":
        return CATEGORY_KNOWLEDGE
    if text and _ASSISTANT_SHAPED_RE.search(text):
        return CATEGORY_MISC
    return category


def infer_category(
    category: Optional[str] = None,
    attributed_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a taxonomy value.

    A valid six-class value is kept. Missing or non-enum values fall back by
    attribution (user → preferences, assistant → knowledge) and otherwise misc.
    """
    if category in MEMORY_CATEGORIES:
        return category
    if metadata:
        meta_category = metadata.get("category")
        if meta_category in MEMORY_CATEGORIES:
            return meta_category
        if attributed_to is None:
            attributed_to = metadata.get("attributed_to")
    if attributed_to == "user":
        return CATEGORY_PREFERENCES
    if attributed_to == "assistant":
        return CATEGORY_KNOWLEDGE
    return CATEGORY_MISC


def resolve_category(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return CATEGORY_MISC
    category = payload.get("category")
    if category in MEMORY_CATEGORIES:
        return category
    return CATEGORY_MISC


def assign_inferred_category(extracted: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Write category onto an infer=True payload using LLM output plus attribution fallback."""
    extracted = extracted or {}
    attributed_to = extracted.get("attributed_to") or metadata.get("attributed_to")
    metadata["category"] = sanitize_category(
        infer_category(extracted.get("category"), attributed_to, metadata),
        text=metadata.get("data") or extracted.get("data"),
        attributed_to=attributed_to,
    )


def assign_direct_category(metadata: Dict[str, Any]) -> None:
    """Fill category on infer=False writes. Keep a caller-supplied value, including custom strings."""
    existing = metadata.get("category")
    if existing in MEMORY_CATEGORIES:
        return
    if existing:
        return
    metadata["category"] = CATEGORY_MISC


def prelabel_category(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Deterministic backfill guess. None means 'leave for the LLM'."""
    if not payload:
        return None
    existing = payload.get("category")
    if existing in MEMORY_CATEGORIES:
        return None
    if payload.get("memory_kind") == "pattern":
        return CATEGORY_INTERESTS
    text = payload.get("data") or ""
    if _STATE_RE.search(text):
        return CATEGORY_STATE
    if _CORE_RE.search(text):
        return sanitize_category(
            CATEGORY_PERSONAL_CORE, text=text, attributed_to=payload.get("attributed_to"))
    if _INTERESTS_RE.search(text):
        return CATEGORY_INTERESTS
    if _PROFILE_RE.search(text):
        return CATEGORY_PREFERENCES
    if payload.get("attributed_to") == "assistant":
        return CATEGORY_KNOWLEDGE
    return None
