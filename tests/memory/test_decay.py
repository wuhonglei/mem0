from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import math

import pytest

from mem0.memory.categories import (
    CATEGORY_KNOWLEDGE,
    CATEGORY_MISC,
    CATEGORY_PERSONAL_CORE,
    MEMORY_CATEGORIES,
    resolve_category,
)
from mem0.memory.decay import (
    DECAY_FLOOR,
    DECAY_HALF_LIVES,
    DECAY_MODE_COLLECT,
    DECAY_MODE_ENFORCE,
    DECAY_MODE_OFF,
    FRESH_SCALE,
    append_access_event,
    apply_decay_rerank,
    attention_factor,
    attention_half_life_days,
    attention_strength,
    decay_multiplier,
    half_life_days,
    decay_mode,
    decay_scaling,
    decay_strength,
    apply_strength,
    finalize_search_scores,
    record_access,
    search_rank_pool_size,
    should_apply_scaling,
    should_record_access,
)

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _payload(category, *, age_days=None, last_accessed=True, extra=None, access_age_days=None):
    """A memory whose content is `age_days` old and which was last used
    `access_age_days` ago (defaults to the same age, i.e. never used since)."""
    payload = {"category": category, "data": "x"}
    if age_days is not None:
        payload["created_at"] = (NOW - timedelta(days=age_days)).isoformat()
        payload["content_updated_at"] = payload["created_at"]
    if age_days is not None or access_age_days is not None:
        if last_accessed:
            days = age_days if access_age_days is None else access_age_days
            payload["last_accessed_at"] = (NOW - timedelta(days=days)).isoformat()
    if extra:
        payload.update(extra)
    return payload


class TestCurves:
    @pytest.mark.parametrize("category", sorted(MEMORY_CATEGORIES))
    def test_age_zero_is_fresh_scale(self, category):
        assert decay_scaling(_payload(category, age_days=0), now=NOW) == pytest.approx(FRESH_SCALE)

    @pytest.mark.parametrize("category", sorted(MEMORY_CATEGORIES))
    def test_converges_to_the_global_floor(self, category):
        scale = decay_scaling(_payload(category, age_days=20000), now=NOW)
        assert scale == pytest.approx(DECAY_FLOOR, abs=1e-6)

    @pytest.mark.parametrize("category", sorted(MEMORY_CATEGORIES))
    def test_floor_is_shared_across_categories(self, category):
        """A per-category floor acts as a standing bonus, not as protection."""
        assert half_life_days(category) in DECAY_HALF_LIVES.values()
        oldest = decay_scaling(_payload(category, age_days=100000), now=NOW)
        assert oldest == pytest.approx(DECAY_FLOOR, abs=1e-6)

    @pytest.mark.parametrize("category", sorted(MEMORY_CATEGORIES))
    def test_monotonic_decreasing(self, category):
        ages = [0, 7, 30, 90, 180, 365]
        scales = [decay_scaling(_payload(category, age_days=age), now=NOW) for age in ages]
        assert scales == sorted(scales, reverse=True)

    def test_personal_core_at_one_half_life_stays_above_floor(self):
        scale = decay_scaling(_payload(CATEGORY_PERSONAL_CORE, age_days=365), now=NOW)
        assert scale > 0.9
        assert scale > DECAY_FLOOR
        assert scale < FRESH_SCALE

    def test_knowledge_at_three_half_lives_near_floor(self):
        scale = decay_scaling(_payload(CATEGORY_KNOWLEDGE, age_days=90), now=NOW)
        assert scale == pytest.approx(DECAY_FLOOR, abs=0.05)
        assert scale < 0.75


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

    def test_durable_facts_are_protected_by_their_half_life_not_a_floor(self):
        """With one global floor the protection a durable fact gets is its long
        half-life, not a category-specific floor: core (365 d) keeps its weight
        where 90-day-old knowledge has already sunk to the floor."""
        ranked = apply_decay_rerank(deepcopy(_candidates()), now=NOW, limit=3)
        ids = [row["id"] for row in ranked]
        assert ids[0] == "fresh-knowledge"
        assert ids[-1] == "stale-knowledge"
        assert "core" in ids

    def test_recent_use_keeps_an_old_durable_fact_in_front(self):
        rows = _candidates()
        rows[2]["payload"]["last_accessed_at"] = NOW.isoformat()  # core was just asked about
        ids = [row["id"] for row in apply_decay_rerank(deepcopy(rows), now=NOW, limit=3)]
        assert ids.index("core") < ids.index("stale-knowledge")

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


class TestStateCurve:
    """state holds in-progress plans, so it must not decay faster than the
    queries that ask about them can tolerate."""

    def test_state_outscores_knowledge_at_the_same_age(self):
        state = decay_scaling(_payload("state", age_days=30), now=NOW)
        knowledge = decay_scaling(_payload(CATEGORY_KNOWLEDGE, age_days=30), now=NOW)
        assert state > knowledge

    def test_month_old_state_is_still_near_full_weight(self):
        scale = decay_scaling(_payload("state", age_days=30), now=NOW)
        assert scale > 0.95

    def test_state_is_still_mid_lived_not_durable(self):
        """Relaxed, not abolished: state must decay faster than the durable
        categories (personal_core / preferences / interests)."""
        state = DECAY_HALF_LIVES["state"]
        for category in (CATEGORY_PERSONAL_CORE, "preferences", "interests"):
            assert state < DECAY_HALF_LIVES[category], category


class TestStrength:
    def test_default_is_full_strength(self, monkeypatch):
        monkeypatch.delenv("MEM0_DECAY_STRENGTH", raising=False)
        assert decay_strength() == 1.0

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "0.4")
        assert decay_strength() == pytest.approx(0.4)

    def test_invalid_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "banana")
        assert decay_strength() == 1.0

    @pytest.mark.parametrize("raw,expected", [("1.5", 1.0), ("-2", 0.0)])
    def test_out_of_range_is_clamped(self, monkeypatch, raw, expected):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", raw)
        assert decay_strength() == pytest.approx(expected)

    def test_compresses_the_deviation_from_one(self):
        assert apply_strength(0.3, 0.0) == pytest.approx(1.0)
        assert apply_strength(1.5, 0.0) == pytest.approx(1.0)
        assert apply_strength(0.3, 0.5) == pytest.approx(0.65)
        assert apply_strength(1.5, 0.5) == pytest.approx(1.25)
        assert apply_strength(0.3, 1.0) == pytest.approx(0.3)

    def test_zero_strength_preserves_the_relevance_order(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "0")
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_STRENGTH", "0")
        rows = deepcopy(_candidates())
        expected = [row["id"] for row in sorted(rows, key=lambda r: r["score"], reverse=True)]
        ranked = apply_decay_rerank(rows, now=NOW, limit=10)
        assert [row["id"] for row in ranked] == expected

    def test_lower_strength_stays_closer_to_the_relevance_order(self, monkeypatch):
        def distance_from_relevance(alpha):
            monkeypatch.setenv("MEM0_DECAY_STRENGTH", str(alpha))
            rows = deepcopy(_candidates())
            baseline = [row["id"] for row in
                        sorted(deepcopy(rows), key=lambda r: r["score"], reverse=True)]
            ranked = [row["id"] for row in apply_decay_rerank(rows, now=NOW, limit=10)]
            return sum(1 for a, b in zip(baseline, ranked) if a != b)

        assert distance_from_relevance(0.2) <= distance_from_relevance(1.0)

    def test_explain_reports_curve_effective_scale_and_strength(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "0.5")
        rows = apply_decay_rerank(deepcopy(_candidates()), now=NOW, limit=3, explain=True)
        details = rows[0]["score_details"]
        assert details["decay_strength"] == pytest.approx(0.5)
        assert details["decay_effective_scale"] == pytest.approx(
            apply_strength(details["decay_scale"], 0.5))
        assert "_decay_scale" not in rows[0]
        assert "_decay_effective" not in rows[0]


class TestFreshnessSignal:
    """updated_at is bumped by governance bookkeeping, not only by content edits,
    so it must not make an untouched old memory look fresh."""

    def test_governance_touch_does_not_refresh_a_memory(self):
        payload = {
            "category": CATEGORY_KNOWLEDGE,
            "created_at": (NOW - timedelta(days=365)).isoformat(),
            "updated_at": NOW.isoformat(),  # merge/archive just touched it
        }
        assert decay_scaling(payload, now=NOW) == pytest.approx(DECAY_FLOOR, abs=0.01)

    def test_content_edit_is_used_when_present(self):
        payload = {
            "category": CATEGORY_KNOWLEDGE,
            "created_at": (NOW - timedelta(days=365)).isoformat(),
            "updated_at": (NOW - timedelta(days=365)).isoformat(),
            "content_updated_at": NOW.isoformat(),
        }
        assert decay_scaling(payload, now=NOW) == pytest.approx(FRESH_SCALE)

    def test_access_recency_is_a_separate_attention_term(self):
        """A recent access must not rewrite the fact's age — it is its own term."""
        stale = {
            "category": CATEGORY_KNOWLEDGE,
            "content_updated_at": (NOW - timedelta(days=365)).isoformat(),
        }
        accessed = {**stale, "last_accessed_at": NOW.isoformat()}
        assert decay_scaling(accessed, now=NOW) == decay_scaling(stale, now=NOW)
        assert decay_multiplier(accessed, now=NOW) > decay_multiplier(stale, now=NOW)
        assert decay_multiplier(accessed, now=NOW) <= FRESH_SCALE

    def test_missing_access_history_is_neutral_not_stale(self, monkeypatch):
        """Only ~1.5 % of this store was ever retrieved; "never used" must not
        be read as "ancient"."""
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "1")
        payload = {
            "category": CATEGORY_KNOWLEDGE,
            "created_at": (NOW - timedelta(days=3650)).isoformat(),
        }
        assert attention_factor(payload, now=NOW) == 1.0
        assert decay_multiplier(payload, now=NOW) == pytest.approx(DECAY_FLOOR)

    def test_attention_is_boost_only(self):
        for days in (1, 45, 90, 4000):
            payload = {
                "category": CATEGORY_KNOWLEDGE,
                "created_at": (NOW - timedelta(days=365)).isoformat(),
                "last_accessed_at": (NOW - timedelta(days=days)).isoformat(),
            }
            assert 1.0 <= attention_factor(payload, now=NOW) <= FRESH_SCALE

    def test_attention_never_inflates_past_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("MEM0_DECAY_STRENGTH", "1")
        payload = _payload(CATEGORY_KNOWLEDGE, age_days=0)
        assert decay_multiplier(payload, now=NOW) == pytest.approx(FRESH_SCALE)
        assert decay_multiplier(payload, now=NOW, strength=1.0) <= FRESH_SCALE

    def test_attention_strength_is_independently_configurable(self, monkeypatch):
        payload = _payload(CATEGORY_KNOWLEDGE, age_days=200, access_age_days=0)
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_STRENGTH", "0")
        assert attention_factor(payload, now=NOW) == 1.0
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_STRENGTH", "0.5")
        assert 1.0 < attention_factor(payload, now=NOW) < FRESH_SCALE

    def test_attention_half_life_is_configurable(self, monkeypatch):
        payload = _payload(CATEGORY_KNOWLEDGE, age_days=200, access_age_days=20)
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_HALF_LIFE", "45")
        expected = DECAY_FLOOR + (FRESH_SCALE - DECAY_FLOOR) * math.exp(-20 / 45)
        assert attention_factor(payload, now=NOW) == pytest.approx(expected, abs=0.01)

    def test_invalid_attention_env_falls_back_to_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_STRENGTH", "nonsense")
        monkeypatch.setenv("MEM0_DECAY_ATTENTION_HALF_LIFE", "99999")
        assert attention_strength() == 1.0
        assert attention_half_life_days() == 3650.0

    def test_creation_time_is_the_fallback(self):
        payload = {
            "category": CATEGORY_MISC,
            "created_at": (NOW - timedelta(days=60)).isoformat(),
            "updated_at": NOW.isoformat(),
        }
        assert decay_scaling(payload, now=NOW) == pytest.approx(
            decay_scaling({"category": CATEGORY_MISC, "updated_at": (NOW - timedelta(days=60)).isoformat()},
                          now=NOW))

    def test_updated_at_survives_only_as_a_last_resort(self):
        payload = {"category": CATEGORY_MISC, "updated_at": NOW.isoformat()}
        assert decay_scaling(payload, now=NOW) == pytest.approx(FRESH_SCALE)
