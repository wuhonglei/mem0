"""Category guards: personal_core carries the slowest decay curve, so a wrong
label there is amplified rather than averaged out.

Measured on the dev store before this guard: 40 of the personal_core rows were
assistant-attributed (advice, product details, third-party facts) and 15 were
query records — several of them reached the top-10 of unrelated queries at
relevance 0.175-0.41 because the 0.9 floor pins them near 1.5x.
"""

from __future__ import annotations

import pytest

from mem0.memory.categories import (
    CATEGORY_KNOWLEDGE,
    CATEGORY_MISC,
    CATEGORY_PERSONAL_CORE,
    assign_inferred_category,
    prelabel_category,
    sanitize_category,
)


class TestSanitizeCategory:
    def test_assistant_authored_content_is_never_personal_core(self):
        assert sanitize_category(
            CATEGORY_PERSONAL_CORE, text="助手建议用户每天投递3-5份岗位",
            attributed_to="assistant") == CATEGORY_KNOWLEDGE

    def test_query_records_are_not_identity_facts(self):
        assert sanitize_category(
            CATEGORY_PERSONAL_CORE, text="用户询问Quora是否相当于国内的知乎，助手确认…",
            attributed_to="user") == CATEGORY_MISC

    def test_real_identity_facts_survive(self):
        for text, who in [
            ("用户的女儿名叫吴思然", "user"),
            ("用户有一对龙凤胎，出生于2023年3月2日", "user"),
            ("用户对芒果过敏", "user"),
            ("用户住在深圳", "user"),
            ("用户的双胞胎生肖属兔", "user"),
        ]:
            assert sanitize_category(CATEGORY_PERSONAL_CORE, text=text, attributed_to=who) == \
                CATEGORY_PERSONAL_CORE, text

    def test_other_categories_are_untouched(self):
        for category in (CATEGORY_KNOWLEDGE, CATEGORY_MISC, "state", "preferences"):
            assert sanitize_category(category, text="助手建议…", attributed_to="assistant") == category

    def test_missing_text_and_attribution_leave_the_label_alone(self):
        assert sanitize_category(CATEGORY_PERSONAL_CORE) == CATEGORY_PERSONAL_CORE


class TestAssignInferredCategoryGuard:
    def test_llm_answer_about_the_assistant_is_downgraded(self):
        metadata = {"data": "助手介绍了读副本负载均衡的三种方案", "attributed_to": "assistant"}
        assign_inferred_category({"category": CATEGORY_PERSONAL_CORE}, metadata)
        assert metadata["category"] == CATEGORY_KNOWLEDGE

    def test_llm_answer_about_the_user_is_downgraded_to_misc(self):
        metadata = {"data": "用户询问了多个读副本如何进行负载均衡，助手介绍了…", "attributed_to": "user"}
        assign_inferred_category({"category": CATEGORY_PERSONAL_CORE}, metadata)
        assert metadata["category"] == CATEGORY_MISC

    def test_identity_fact_keeps_its_label(self):
        metadata = {"data": "用户的双胞胎出生于2023年3月2日", "attributed_to": "user"}
        assign_inferred_category({"category": CATEGORY_PERSONAL_CORE}, metadata)
        assert metadata["category"] == CATEGORY_PERSONAL_CORE

    def test_extracted_attribution_wins_over_metadata(self):
        metadata = {"data": "助手建议用户…", "attributed_to": "user"}
        assign_inferred_category({"category": CATEGORY_PERSONAL_CORE, "attributed_to": "assistant"},
                                 metadata)
        assert metadata["category"] == CATEGORY_KNOWLEDGE


class TestPrelabelRegexAnchoring:
    """An unanchored domain word used to be enough to earn the slowest curve."""

    @pytest.mark.parametrize("text", [
        "Zalando为软件工程师提供清晰的双轨制职业发展路径：Senior Engineer之后可选择成为工程经理",
        "计算ARR有助于评估业务健康状况和融资需求",
        "李兰迪出生于1999年9月2日，籍贯和出生地均为北京市",
        "助手建议用户在家人适应期适当降低求职强度",
    ])
    def test_unanchored_domain_words_do_not_earn_personal_core(self, text):
        assert prelabel_category({"data": text, "attributed_to": "assistant"}) != CATEGORY_PERSONAL_CORE

    @pytest.mark.parametrize("text", [
        "用户的女儿名叫吴思然",
        "用户对芒果过敏",
        "我的老家在潮汕",
    ])
    def test_user_anchored_identity_facts_still_match(self, text):
        assert prelabel_category({"data": text, "attributed_to": "user"}) == CATEGORY_PERSONAL_CORE

    def test_state_still_wins_over_identity(self):
        """A cue ordering guarantee: in-progress plans must stay on the state curve."""
        assert prelabel_category({"data": "用户正在计划搬家到上海", "attributed_to": "user"}) == "state"
