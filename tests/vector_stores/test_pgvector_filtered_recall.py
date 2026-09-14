"""Regression tests for the filtered-search recall guard in the pgvector store.

An HNSW scan applies the payload filter after the index scan, so a filtered
search can return almost nothing (measured 0 rows for a 240-row request on a
13 %-selective filter). These tests pin the remedy the store picks for each
pgvector version, and that the guard stays inert without filters, without an
HNSW index, or when explicitly disabled.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from mem0.vector_stores.pgvector import ENV_FILTERED_RECALL, PGVector


class FakeCursor:
    """Records every statement and answers the probe queries the guard makes."""

    def __init__(self, extversion: str = "0.5.1", has_hnsw: bool = True):
        self.extversion = extversion
        self.has_hnsw = has_hnsw
        self.statements: List[str] = []

    def execute(self, sql: Any, params: Optional[Tuple] = None) -> None:
        text = sql if isinstance(sql, str) else str(sql.as_string(None) if hasattr(sql, "as_string") else sql)
        self.statements.append(text)
        if "extversion" in text:
            self._row = (self.extversion,)
        elif "pg_indexes" in text:
            self._row = (1,) if self.has_hnsw else None
        else:
            self._row = None

    def fetchone(self):
        return getattr(self, "_row", None)

    def set_issued(self) -> List[str]:
        return [s for s in self.statements if s.upper().startswith("SET ")]


def _store(**attrs) -> PGVector:
    """A store instance without a connection pool (the guard touches no pool)."""
    store = PGVector.__new__(PGVector)
    store.collection_name = "memories"
    store._filtered_recall_probe = None
    for key, value in attrs.items():
        setattr(store, key, value)
    return store


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)


def test_old_pgvector_falls_back_to_full_filtered_scan(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    cur = FakeCursor(extversion="0.5.1")
    assert _store()._apply_filtered_recall_settings(cur, 240) == "seqscan"
    issued = cur.set_issued()
    assert any("enable_indexscan = off" in s for s in issued), issued
    assert any("enable_bitmapscan = off" in s for s in issued), issued
    assert not any("iterative_scan" in s for s in issued), issued


def test_iterative_scan_used_when_available(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    cur = FakeCursor(extversion="0.8.0")
    assert _store()._apply_filtered_recall_settings(cur, 240) == "iterative"
    issued = cur.set_issued()
    assert any("hnsw.iterative_scan = relaxed_order" in s for s in issued), issued
    assert any("hnsw.ef_search = 240" in s for s in issued), issued
    assert not any("enable_indexscan" in s for s in issued), issued


def test_ef_search_has_a_floor_for_small_top_k(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    cur = FakeCursor(extversion="0.8.0")
    _store()._apply_filtered_recall_settings(cur, 5)
    assert any("hnsw.ef_search = 200" in s for s in cur.set_issued()), cur.set_issued()


def test_two_digit_minor_version_is_not_string_compared(monkeypatch):
    """'0.10.0' must count as newer than 0.8 — a naive string compare says otherwise."""
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    cur = FakeCursor(extversion="0.10.0")
    assert _store()._apply_filtered_recall_settings(cur, 10) == "iterative"


@pytest.mark.parametrize("mode", ["off", "OFF", " seqscan "])
def test_disabled_and_explicit_modes(monkeypatch, mode):
    monkeypatch.setenv(ENV_FILTERED_RECALL, mode)
    cur = FakeCursor(extversion="0.8.0")
    applied = _store()._apply_filtered_recall_settings(cur, 240)
    if mode.strip().lower() == "off":
        assert applied == "off"
        assert cur.set_issued() == []
    else:
        assert applied == "seqscan"


def test_unknown_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv(ENV_FILTERED_RECALL, "banana")
    cur = FakeCursor(extversion="0.5.1")
    assert _store()._apply_filtered_recall_settings(cur, 240) == "seqscan"


def test_guard_is_inert_without_an_hnsw_index(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    cur = FakeCursor(extversion="0.5.1", has_hnsw=False)
    assert _store()._apply_filtered_recall_settings(cur, 240) == "off"
    assert cur.set_issued() == []


def test_index_probe_is_cached(monkeypatch):
    monkeypatch.delenv(ENV_FILTERED_RECALL, raising=False)
    store = _store()
    first = FakeCursor(extversion="0.5.1")
    store._apply_filtered_recall_settings(first, 10)
    probes_first = [s for s in first.statements if "pg_indexes" in s]
    second = FakeCursor(extversion="0.5.1")
    store._apply_filtered_recall_settings(second, 10)
    probes_second = [s for s in second.statements if "pg_indexes" in s]
    assert len(probes_first) == 1 and probes_second == []


class _RecordingStore(PGVector):
    def __init__(self):
        self.collection_name = "memories"
        self._filtered_recall_probe = None
        self.calls: List[Optional[int]] = []

    def _apply_filtered_recall_settings(self, cur, top_k=None):
        self.calls.append(top_k)
        return "seqscan"


def test_search_applies_guard_only_when_filters_are_present(monkeypatch):
    """The call site: filtered search guards recall, unfiltered search does not."""
    store = _RecordingStore()
    monkeypatch.setattr(store, "_ensure_collection", lambda: None)

    class _Cur:
        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return []

    class _Ctx:
        def __enter__(self):
            return _Cur()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(store, "_get_cursor", lambda commit=False: _Ctx())

    store.search(query="q", vectors=[0.0], top_k=5, filters={"user_id": "u1"})
    assert store.calls == [5]

    store.search(query="q", vectors=[0.0], top_k=5, filters=None)
    assert store.calls == [5]
