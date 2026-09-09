from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mem0.memory.decay import (
    CATEGORY_INTERESTS,
    CATEGORY_KNOWLEDGE,
    CATEGORY_MISC,
    CATEGORY_PERSONAL_CORE,
    CATEGORY_PREFERENCES,
    CATEGORY_STATE,
    DECAY_CATEGORIES,
    DECAY_CURVES,
    DECAY_MODE_COLLECT,
    DECAY_MODE_ENFORCE,
    DECAY_MODE_OFF,
    FRESH_SCALE,
    append_access_event,
    apply_decay_rerank,
    assign_direct_category,
    assign_inferred_category,
    decay_mode,
    decay_scaling,
    finalize_search_scores,
    infer_category,
    prelabel_category,
    record_access,
    resolve_category,
    search_rank_pool_size,
    should_apply_scaling,
    should_record_access,
)

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _payload(category, *, age_days=None, last_accessed=True, extra=None):
    payload = {"category": category, "data": "x"}
    if age_days is not None:
        stamp = (NOW - timedelta(days=age_days)).isoformat()
        if last_accessed:
            payload["last_accessed_at"] = stamp
        else:
            payload["updated_at"] = stamp
    if extra:
        payload.update(extra)
    return payload


class TestCurves:
    @pytest.mark.parametrize("category", sorted(DECAY_CATEGORIES))
    def test_age_zero_is_fresh_scale(self, category):
        assert decay_scaling(_payload(category, age_days=0), now=NOW) == pytest.approx(FRESH_SCALE)

    @pytest.mark.parametrize("category", sorted(DECAY_CATEGORIES))
    def test_converges_to_floor(self, category):
        floor, _ = DECAY_CURVES[category]
        scale = decay_scaling(_payload(category, age_days=20000), now=NOW)
        assert scale == pytest.approx(floor, abs=1e-6)

    @pytest.mark.parametrize("category", sorted(DECAY_CATEGORIES))
    def test_monotonic_decreasing(self, category):
        ages = [0, 7, 30, 90, 180, 365]
        scales = [decay_scaling(_payload(category, age_days=age), now=NOW) for age in ages]
        assert scales == sorted(scales, reverse=True)

    def test_personal_core_at_one_half_life_stays_above_floor(self):
        scale = decay_scaling(_payload(CATEGORY_PERSONAL_CORE, age_days=365), now=NOW)
        floor, _ = DECAY_CURVES[CATEGORY_PERSONAL_CORE]
        assert scale > 0.9
        assert scale > floor
        assert scale < FRESH_SCALE

    def test_knowledge_at_three_half_lives_near_floor(self):
        scale = decay_scaling(_payload(CATEGORY_KNOWLEDGE, age_days=90), now=NOW)
        floor, _ = DECAY_CURVES[CATEGORY_KNOWLEDGE]
        assert scale == pytest.approx(floor, abs=0.07)
        assert scale < 0.4


class TestFallback:
    def test_uses_updated_at_when_no_access_history(self):
        with_access = decay_scaling(_payload(CATEGORY_MISC, age_days=60), now=NOW)
        with_updated = decay_scaling(_payload(CATEGORY_MISC, age_days=60, last_accessed=False), now=NOW)
        assert with_access == pytest.approx(with_updated)

    def test_prefers_last_accessed_over_updated_at(self):
        payload = _payload(CATEGORY_KNOWLEDGE, age_days=90)
        payload["updated_at"] = NOW.isoformat()
        stale = decay_scaling(payload, now=NOW)
        fresh = decay_scaling(_payload(CATEGORY_KNOWLEDGE, age_days=0), now=NOW)
        assert stale < fresh

    def test_missing_category_uses_misc_curve(self):
        misc = decay_scaling(_payload(CATEGORY_MISC, age_days=60), now=NOW)
        missing = decay_scaling({"updated_at": (NOW - timedelta(days=60)).isoformat()}, now=NOW)
        assert missing == pytest.approx(misc)

    def test_unknown_user_category_uses_misc_curve(self):
        movies = decay_scaling({"category": "movies", "updated_at": (NOW - timedelta(days=60)).isoformat()}, now=NOW)
        misc = decay_scaling(_payload(CATEGORY_MISC, age_days=60, last_accessed=False), now=NOW)
        assert movies == pytest.approx(misc)
        assert resolve_category({"category": "movies"}) == CATEGORY_MISC


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


class TestModeAndOverride:
    def test_mode_default_off(self, monkeypatch):
        monkeypatch.delenv("MEM0_DECAY_MODE", raising=False)
        assert decay_mode() == DECAY_MODE_OFF
        assert should_apply_scaling() is False
        assert should_record_access() is False

    def test_invalid_mode_falls_back_to_off(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", "maybe")
        assert decay_mode() == DECAY_MODE_OFF

    def test_collect_records_but_does_not_scale(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_COLLECT)
        assert should_apply_scaling() is False
        assert should_record_access() is True

    def test_enforce_scales_and_records(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_ENFORCE)
        assert should_apply_scaling() is True
        assert should_record_access() is True

    def test_override_false_disables_scaling_in_enforce(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_ENFORCE)
        assert should_apply_scaling(False) is False
        assert should_record_access(False) is False

    def test_override_true_enables_scaling_even_in_collect(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_COLLECT)
        assert should_apply_scaling(True) is True
        assert should_record_access(True) is False


def _candidates():
    return [
        {
            "id": "fresh-knowledge",
            "score": 0.6,
            "payload": _payload(CATEGORY_KNOWLEDGE, age_days=0),
        },
        {
            "id": "stale-knowledge",
            "score": 0.7,
            "payload": _payload(CATEGORY_KNOWLEDGE, age_days=90),
        },
        {
            "id": "core",
            "score": 0.55,
            "payload": _payload(CATEGORY_PERSONAL_CORE, age_days=365),
        },
    ]


class TestRerank:
    def test_off_and_collect_do_not_change_order(self):
        original = [row["id"] for row in sorted(_candidates(), key=lambda r: r["score"], reverse=True)]
        assert original == ["stale-knowledge", "fresh-knowledge", "core"]

    def test_enforce_promotes_fresh_and_protects_core(self):
        ranked = apply_decay_rerank(deepcopy(_candidates()), now=NOW, limit=3)
        ids = [row["id"] for row in ranked]
        assert ids[0] == "fresh-knowledge"
        assert ids[-1] == "stale-knowledge"
        assert "core" in ids

    def test_public_score_clamped_to_one(self):
        ranked = apply_decay_rerank(
            [{"id": "a", "score": 0.9, "payload": _payload(CATEGORY_PERSONAL_CORE, age_days=0)}],
            now=NOW,
            limit=1,
        )
        assert ranked[0]["score"] <= 1.0

    def test_explain_includes_decay_scale(self):
        ranked = apply_decay_rerank(
            [{"id": "a", "score": 0.5, "payload": _payload(CATEGORY_MISC, age_days=0)}],
            now=NOW,
            limit=1,
            explain=True,
        )
        assert ranked[0]["score_details"]["decay_scale"] == pytest.approx(FRESH_SCALE)

    def test_override_false_keeps_original_order(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_ENFORCE)
        rows = deepcopy(_candidates())
        original = [row["id"] for row in sorted(rows, key=lambda r: r["score"], reverse=True)]
        if not should_apply_scaling(False):
            kept = [row["id"] for row in sorted(rows, key=lambda r: r["score"], reverse=True)]
        else:
            kept = [row["id"] for row in apply_decay_rerank(rows, now=NOW, limit=3)]
        assert kept == original


class TestAccessLog:
    def test_ring_buffer_truncates_to_twenty(self):
        payload = {"access_log": [f"t{i}" for i in range(19)], "access_count": 19}
        updated = append_access_event(payload, "t19", max_n=20)
        assert len(updated["access_log"]) == 20
        updated = append_access_event(updated, "t20", max_n=20)
        assert updated["access_log"] == [f"t{i}" for i in range(1, 21)]
        assert updated["access_count"] == 21
        assert updated["last_accessed_at"] == "t20"

    def test_does_not_mutate_input(self):
        payload = {"access_log": ["a"], "access_count": 1}
        append_access_event(payload, "b", max_n=20)
        assert payload["access_log"] == ["a"]
        assert payload["access_count"] == 1

    def test_record_access_writes_payload_only(self):
        existing = {
            "id": "m1",
            "payload": {
                "data": "keep me",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "governance_status": "active",
                "access_count": 2,
                "access_log": ["old"],
            },
        }
        store = SimpleNamespace(
            get=lambda vector_id: SimpleNamespace(payload=deepcopy(existing["payload"])),
            updates=[],
        )

        def update(*, vector_id, payload, vector=None):
            store.updates.append({"id": vector_id, "payload": payload, "vector": vector})

        store.update = update
        record_access(store, [{"id": "m1"}], now=NOW)
        assert len(store.updates) == 1
        written = store.updates[0]
        assert written["vector"] is None
        assert written["payload"]["data"] == "keep me"
        assert written["payload"]["governance_status"] == "active"
        assert written["payload"]["updated_at"] == "2026-01-01T00:00:00+00:00"
        assert written["payload"]["access_count"] == 3
        assert written["payload"]["last_accessed_at"] == NOW.isoformat()

    def test_record_access_failure_is_logged(self, caplog):
        class Boom:
            def get(self, vector_id):
                raise RuntimeError("store down")

        with caplog.at_level("ERROR"):
            record_access(Boom(), [{"id": "m1"}], now=NOW)
        assert "record_access failed" in caplog.text


class TestUpdateMergePreservesAccess:
    def test_governance_metadata_merge_keeps_access_fields(self):
        existing = {
            "data": "fact",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "last_accessed_at": "2026-09-01T00:00:00+00:00",
            "access_count": 4,
            "access_log": ["a", "b"],
            "category": CATEGORY_PERSONAL_CORE,
        }
        incoming = {"governance_status": "merged", "merged_into": "other"}
        merged = deepcopy(existing)
        merged.update(incoming)
        assert merged["last_accessed_at"] == existing["last_accessed_at"]
        assert merged["access_count"] == 4
        assert merged["access_log"] == ["a", "b"]
        assert merged["category"] == CATEGORY_PERSONAL_CORE
        assert merged["governance_status"] == "merged"


class TestGovernanceOrder:
    def test_decay_operates_on_survivors_only(self):
        from mem0.memory.governance_filters import should_include_memory

        rows = [
            {
                "id": "active",
                "score": 0.4,
                "payload": {**_payload(CATEGORY_KNOWLEDGE, age_days=0), "governance_status": "active"},
            },
            {
                "id": "merged",
                "score": 0.9,
                "payload": {**_payload(CATEGORY_KNOWLEDGE, age_days=0), "governance_status": "merged"},
            },
        ]
        survivors = [row for row in rows if should_include_memory(row["payload"])]
        ranked = apply_decay_rerank(survivors, now=NOW, limit=5)
        assert [row["id"] for row in ranked] == ["active"]


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


class TestFinalizeSearch:
    def test_collect_does_not_rerank(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_COLLECT)
        rows = sorted(deepcopy(_candidates()), key=lambda r: r["score"], reverse=True)
        original = [row["id"] for row in rows]
        store = SimpleNamespace(get=lambda **kwargs: None, update=lambda **kwargs: None)
        out = finalize_search_scores(rows, limit=3, decay_override=None, vector_store=store)
        assert [row["id"] for row in out] == original

    def test_enforce_uses_internal_pool(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_ENFORCE)
        assert search_rank_pool_size(10, 60, None) == 60
        monkeypatch.setenv("MEM0_DECAY_MODE", DECAY_MODE_OFF)
        assert search_rank_pool_size(10, 60, None) == 10
        assert search_rank_pool_size(10, 60, True) == 60
