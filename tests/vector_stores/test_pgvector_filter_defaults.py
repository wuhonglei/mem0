"""Filter semantics for payload keys with an implicit default.

``governance_status`` is absent on memories the governance layer never touched,
and every read path (``Memory.get_all``, ``get_all(..., latest_only=True)``,
``governance_filters.should_include_memory``) treats that absence as ``active``.
A caller passing the same filter through the payload-filter builder must get the
same rows, otherwise it silently loses them.
"""

from __future__ import annotations

import pytest

from mem0.vector_stores.pgvector import _build_filter_conditions


def _sql(filters) -> str:
    conditions, _ = _build_filter_conditions(filters)
    return " AND ".join(conditions)


def test_scalar_filter_treats_missing_status_as_active():
    sql = _sql({"governance_status": "active"})
    assert "COALESCE(payload->>%s, 'active') = %s" in sql


def test_in_filter_treats_missing_status_as_active():
    sql = _sql({"governance_status": {"in": ["active", "superseded"]}})
    assert "COALESCE(payload->>%s, 'active') = ANY(%s)" in sql


def test_ne_filter_uses_the_same_default():
    sql = _sql({"governance_status": {"ne": "merged"}})
    assert "COALESCE(payload->>%s, 'active') != %s" in sql


def test_nin_filter_uses_the_same_default():
    sql = _sql({"governance_status": {"nin": ["merged", "archived"]}})
    assert "NOT (COALESCE(payload->>%s, 'active') = ANY(%s))" in sql


def test_contains_filter_keeps_the_escape_clause():
    sql = _sql({"governance_status": {"contains": "act"}})
    assert sql.startswith("COALESCE(payload->>%s, 'active') LIKE %s ESCAPE")


def test_param_order_is_unchanged_by_the_wrapper():
    _, params = _build_filter_conditions({"governance_status": "active", "user_id": "u1"})
    assert params == ["governance_status", "active", "user_id", "u1"]


def test_other_keys_are_untouched():
    sql = _sql({"user_id": "u1", "category": {"in": ["state"]}})
    assert "COALESCE" not in sql
    assert "payload->>%s = %s" in sql


def test_presence_test_is_not_wrapped():
    """`{"key": "*"}` asks whether the key exists — a default would defeat it."""
    sql = _sql({"governance_status": "*"})
    assert sql == "payload ? %s"


def test_nested_or_groups_know_their_own_keys():
    sql = _sql({"$or": [{"governance_status": "active"}, {"user_id": "u1"}]})
    assert "COALESCE(payload->>%s, 'active') = %s" in sql
    assert "payload->>%s = %s" in sql


@pytest.mark.parametrize("value", [True, False])
def test_booleans_still_json_encoded_outside_the_default_set(value):
    _, params = _build_filter_conditions({"immutable": value})
    assert params == ["immutable", "true" if value else "false"]
