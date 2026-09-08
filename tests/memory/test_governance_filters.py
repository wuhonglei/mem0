import pytest

from mem0.memory.governance_filters import (
    CORE_AND_PROMOTED_KEYS,
    fetch_limit_for_filters,
    is_pattern_memory,
    memory_is_active,
    should_include_memory,
)


@pytest.mark.parametrize(
    "payload,expected",
    [
        (None, True),
        ({}, True),
        ({"governance_status": "active"}, True),
        ({"governance_status": "merged"}, False),
        ({"governance_status": "superseded"}, False),
    ],
)
def test_memory_is_active(payload, expected):
    assert memory_is_active(payload) is expected


@pytest.mark.parametrize(
    "status,latest_only,include_merged,expected",
    [
        (None, False, False, True),
        ("active", False, False, True),
        ("superseded", False, False, True),
        ("merged", False, False, False),
        ("active", True, False, True),
        ("superseded", True, False, False),
        ("merged", True, False, False),
        ("active", False, True, True),
        ("superseded", False, True, True),
        ("merged", False, True, True),
        ("merged", True, True, False),
    ],
)
def test_should_include_memory(status, latest_only, include_merged, expected):
    payload = {} if status is None else {"governance_status": status}
    assert should_include_memory(payload, latest_only=latest_only, include_merged=include_merged) is expected


def test_is_pattern_memory_top_level_and_nested():
    assert is_pattern_memory({"memory_kind": "pattern"}) is True
    assert is_pattern_memory({"metadata": {"memory_kind": "pattern"}}) is True
    assert is_pattern_memory({"memory_kind": "fact"}) is False
    assert is_pattern_memory({}) is False


def test_governance_keys_are_promoted_not_left_in_metadata():
    for key in (
        "governance_status",
        "merged_into",
        "superseded_by",
        "synthesized_from",
        "memory_kind",
    ):
        assert key in CORE_AND_PROMOTED_KEYS


def test_fetch_limit_overfetches_when_filtering():
    assert fetch_limit_for_filters(10, show_expired=False) == 60
    assert fetch_limit_for_filters(10, latest_only=True, include_merged=True, show_expired=True) == 60
    assert fetch_limit_for_filters(10, latest_only=False, include_merged=True, show_expired=True) == 10
