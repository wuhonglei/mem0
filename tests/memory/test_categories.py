from mem0.memory.categories import (
    CATEGORY_INTERESTS,
    CATEGORY_KNOWLEDGE,
    CATEGORY_MISC,
    CATEGORY_PERSONAL_CORE,
    CATEGORY_PREFERENCES,
    CATEGORY_STATE,
    MEMORY_CATEGORIES,
    assign_direct_category,
    assign_inferred_category,
    infer_category,
    prelabel_category,
    resolve_category,
)


def test_six_class_taxonomy():
    assert MEMORY_CATEGORIES == {
        CATEGORY_PERSONAL_CORE,
        CATEGORY_PREFERENCES,
        CATEGORY_INTERESTS,
        CATEGORY_STATE,
        CATEGORY_KNOWLEDGE,
        CATEGORY_MISC,
    }


class TestInferCategory:
    def test_keeps_valid_enum(self):
        assert infer_category(CATEGORY_STATE) == CATEGORY_STATE

    def test_user_attribution_falls_back_to_preferences(self):
        assert infer_category(attributed_to="user") == CATEGORY_PREFERENCES

    def test_assistant_attribution_falls_back_to_knowledge(self):
        assert infer_category(attributed_to="assistant") == CATEGORY_KNOWLEDGE

    def test_unknown_and_missing_fall_back_to_misc(self):
        assert infer_category("movies") == CATEGORY_MISC
        assert infer_category() == CATEGORY_MISC

    def test_reads_valid_category_from_metadata(self):
        assert infer_category(metadata={"category": CATEGORY_INTERESTS}) == CATEGORY_INTERESTS

    def test_reads_attribution_from_metadata(self):
        assert infer_category(metadata={"attributed_to": "user"}) == CATEGORY_PREFERENCES


class TestResolveCategory:
    def test_unknown_is_misc(self):
        assert resolve_category({"category": "movies"}) == CATEGORY_MISC
        assert resolve_category({}) == CATEGORY_MISC


class TestPrelabel:
    def test_skips_existing_enum(self):
        assert prelabel_category({"category": CATEGORY_MISC, "data": "正在计划"}) is None

    def test_pattern_is_interests(self):
        assert prelabel_category({"memory_kind": "pattern", "data": "x"}) == CATEGORY_INTERESTS

    def test_state_and_core_keywords(self):
        assert prelabel_category({"data": "用户正在对比两款车"}) == CATEGORY_STATE
        assert prelabel_category({"data": "用户对花生过敏"}) == CATEGORY_PERSONAL_CORE

    def test_assistant_knowledge(self):
        assert prelabel_category({"attributed_to": "assistant", "data": "Postgres uses MVCC"}) == CATEGORY_KNOWLEDGE

    def test_unlabeled_needs_llm(self):
        assert prelabel_category({"attributed_to": "user", "data": "User likes jazz"}) is None


class TestAssignCategory:
    def test_direct_write_defaults_to_misc(self):
        metadata = {}
        assign_direct_category(metadata)
        assert metadata["category"] == CATEGORY_MISC

    def test_direct_write_keeps_enum_and_custom_values(self):
        kept = {"category": CATEGORY_INTERESTS}
        assign_direct_category(kept)
        assert kept["category"] == CATEGORY_INTERESTS
        custom = {"category": "sports"}
        assign_direct_category(custom)
        assert custom["category"] == "sports"

    def test_inferred_uses_llm_then_attribution(self):
        payload = {"attributed_to": "user"}
        assign_inferred_category({"category": CATEGORY_STATE}, payload)
        assert payload["category"] == CATEGORY_STATE
        fallback = {"attributed_to": "assistant"}
        assign_inferred_category({}, fallback)
        assert fallback["category"] == CATEGORY_KNOWLEDGE
