"""Dream governance: on-add merge/supersede, synthesis eligibility, REST forwarding."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from governance.actions import apply_merge, apply_supersede, evidence_hash, pick_canonical
from governance.dedup import search_similar, split_auto_and_llm_candidates
from governance.engine import is_synthesis_eligible, remap_llm_action, run_on_add, run_synthesis

from mem0.memory.governance_filters import should_include_memory


def _hit(memory_id, text, score, **payload):
    return SimpleNamespace(
        id=memory_id,
        score=score,
        payload={"data": text, "created_at": "2026-01-01T00:00:00+00:00", **payload},
    )


def _memory_mock():
    memory = MagicMock()
    memory.embedding_model.embed.return_value = [0.1, 0.2]
    memory.vector_store.search.return_value = []
    memory.get_all.return_value = {"results": []}
    memory.llm.generate_response.return_value = '{"actions": []}'
    return memory


def test_pick_canonical_prefers_longer_text():
    short = {"id": "a", "memory": "User has a dog named Rex", "created_at": "2026-02-01"}
    long = {"id": "b", "memory": "User's dog Rex is a 3-year-old golden retriever", "created_at": "2026-01-01"}
    assert pick_canonical([short, long])["id"] == "b"


def test_apply_merge_tags_via_update_metadata_only():
    memory = MagicMock()
    apply_merge(memory, source_ids=["old", "new"], canonical_id="new", pass_id="pass-1", reason="dup")
    memory.update.assert_called_once()
    args, kwargs = memory.update.call_args
    assert args[0] == "old"
    assert "text" not in kwargs
    assert kwargs["metadata"]["governance_status"] == "merged"
    assert kwargs["metadata"]["merged_into"] == "new"
    memory.vector_store.update.assert_not_called()


def test_apply_supersede_tags_old_memory():
    memory = MagicMock()
    apply_supersede(
        memory,
        old_id="lisbon",
        new_id="berlin",
        pass_id="pass-1",
        reason="moved",
        old_content="User lives in Lisbon",
        new_content="User moved to Berlin",
    )
    args, kwargs = memory.update.call_args
    assert args[0] == "lisbon"
    assert kwargs["metadata"]["governance_status"] == "superseded"
    assert kwargs["metadata"]["superseded_by"] == "berlin"


def test_search_similar_reuses_stored_vector():
    memory = _memory_mock()
    stored = [0.5, 0.25, 0.1]
    memory.vector_store.get.return_value = SimpleNamespace(
        id="N", payload={"data": "hello"}, vector=stored
    )
    memory.vector_store.search.return_value = []
    search_similar(memory, "hello", {"user_id": "u1"}, vector_id="N")
    memory.embedding_model.embed.assert_not_called()
    _, kwargs = memory.vector_store.search.call_args
    assert kwargs["vectors"] == stored


def test_search_similar_embeds_when_stored_vector_missing():
    memory = _memory_mock()
    memory.vector_store.get.return_value = SimpleNamespace(id="N", payload={"data": "hello"}, vector=None)
    memory.vector_store.search.return_value = []
    search_similar(memory, "hello", {"user_id": "u1"}, vector_id="N")
    memory.embedding_model.embed.assert_called_once_with("hello", "search")


def test_split_auto_vs_llm_bands():
    auto, llm = split_auto_and_llm_candidates(
        [{"id": "a", "score": 0.99}, {"id": "b", "score": 0.80}, {"id": "c", "score": 0.4}],
        auto_threshold=0.95,
        llm_min=0.70,
    )
    assert [c["id"] for c in auto] == ["a"]
    assert [c["id"] for c in llm] == ["b"]


def test_on_add_auto_merge_high_similarity():
    memory = _memory_mock()
    memory.vector_store.search.return_value = [
        _hit("old-rex", "User has a dog named Rex", 0.99),
    ]
    report = run_on_add(
        memory,
        {"results": [{"id": "new-rex", "event": "ADD", "memory": "User's dog Rex is a 3-year-old golden retriever"}]},
        user_id="u1",
    )
    assert report is not None
    assert report["stats"]["merged"] == 1
    assert report["actions"][0]["type"] == "merge"
    assert report["actions"][0]["canonical_id"] == "new-rex"
    memory.update.assert_called()
    _, kwargs = memory.update.call_args
    assert kwargs["metadata"]["governance_status"] == "merged"
    memory.llm.generate_response.assert_not_called()


def test_on_add_auto_merge_picks_one_canonical_for_the_whole_cluster():
    memory = _memory_mock()
    memory.vector_store.search.return_value = [
        _hit("A", "User has a dog named Rex", 0.99),
        _hit("B", "My dog is named Rex", 0.97),
        _hit(
            "C",
            "User's dog Rex is a 3-year-old golden retriever who loves parks",
            0.96,
        ),
    ]
    report = run_on_add(
        memory,
        {
            "results": [
                {
                    "id": "N",
                    "event": "ADD",
                    "memory": "User's dog Rex is a 3-year-old golden retriever",
                }
            ]
        },
        user_id="u1",
    )
    assert report["stats"]["merged"] == 3
    assert len(report["actions"]) == 1
    action = report["actions"][0]
    assert action["canonical_id"] == "C"
    assert set(action["source_ids"]) == {"N", "A", "B", "C"}
    tagged = {call.args[0] for call in memory.update.call_args_list}
    assert tagged == {"N", "A", "B"}
    for call in memory.update.call_args_list:
        assert call.kwargs["metadata"]["merged_into"] == "C"
    memory.llm.generate_response.assert_not_called()


def test_on_add_llm_supersede_in_mid_band():
    memory = _memory_mock()
    memory.vector_store.search.return_value = [
        _hit("lisbon", "User lives in Lisbon", 0.82),
    ]
    memory.llm.generate_response.return_value = """
    {"actions": [{"type": "supersede", "old_id": 1, "new_id": 0, "reason": "moved"}]}
    """
    report = run_on_add(
        memory,
        {"results": [{"id": "berlin", "event": "ADD", "memory": "User moved to Berlin in August 2026"}]},
        user_id="u1",
    )
    assert report["stats"]["superseded"] == 1
    _, kwargs = memory.update.call_args
    assert kwargs["metadata"]["governance_status"] == "superseded"
    assert kwargs["metadata"]["superseded_by"] == "berlin"


def test_on_add_no_candidates_skips_llm():
    memory = _memory_mock()
    report = run_on_add(
        memory,
        {"results": [{"id": "m1", "event": "ADD", "memory": "User likes tea"}]},
        user_id="u1",
    )
    assert report is None
    memory.llm.generate_response.assert_not_called()
    memory.update.assert_not_called()


def test_remap_llm_action_restores_indexes_and_drops_unknown():
    mapping = {"0": "berlin", "1": "lisbon"}
    restored = remap_llm_action({"type": "supersede", "old_id": 1, "new_id": "0", "reason": "moved"}, mapping)
    assert restored["old_id"] == "lisbon"
    assert restored["new_id"] == "berlin"
    assert remap_llm_action({"type": "supersede", "old_id": "lisbon", "new_id": 0}, mapping) is None
    assert remap_llm_action({"type": "supersede", "old_id": 9, "new_id": 0}, mapping) is None
    merge = remap_llm_action({"type": "merge", "canonical_id": 0, "source_ids": [0, 1]}, mapping)
    assert merge["canonical_id"] == "berlin"
    assert merge["source_ids"] == ["berlin", "lisbon"]


def test_on_add_llm_drops_raw_memory_ids():
    memory = _memory_mock()
    memory.vector_store.search.return_value = [
        _hit("lisbon", "User lives in Lisbon", 0.82),
    ]
    memory.llm.generate_response.return_value = """
    {"actions": [{"type": "supersede", "old_id": "lisbon", "new_id": "berlin", "reason": "moved"}]}
    """
    report = run_on_add(
        memory,
        {"results": [{"id": "berlin", "event": "ADD", "memory": "User moved to Berlin in August 2026"}]},
        user_id="u1",
    )
    assert report is None
    memory.update.assert_not_called()


def test_synthesis_excludes_existing_patterns():
    memory = _memory_mock()
    eligible = [
        {"id": f"m{i}", "memory": f"User trains fact {i}", "user_id": "u1", "governance_status": "active"}
        for i in range(3)
    ]
    pattern = {
        "id": "pat-1",
        "memory": "User follows a structured fitness routine",
        "user_id": "u1",
        "governance_status": "active",
        "memory_kind": "pattern",
        "synthesized_from": ["other-1", "other-2"],
    }
    memory.llm.generate_response.return_value = """
    {"patterns": [{"text": "User is an endurance athlete", "evidence_ids": [0, 1, 99]}]}
    """
    stats = {"created": 0, "synthesized": 0}
    actions = run_synthesis(
        memory,
        eligible + [pattern],
        user_id="u1",
        pass_id="pass-1",
        stats=stats,
        force=True,
    )
    call = memory.llm.generate_response.call_args
    messages = call.kwargs.get("messages") if call.kwargs else None
    if not messages and call.args:
        messages = call.args[0]
    prompt = messages[1]["content"]
    assert "pat-1" not in prompt
    assert '"id": 0' in prompt
    assert '"id": 1' in prompt
    assert '"m0"' not in prompt
    assert actions[0]["source_ids"] == ["m0", "m1"]
    add_kwargs = memory.add.call_args.kwargs
    assert add_kwargs["infer"] is False
    assert add_kwargs["metadata"]["memory_kind"] == "pattern"
    assert "pat-1" not in add_kwargs["metadata"]["synthesized_from"]


def test_synthesis_idempotent_on_evidence_hash():
    memory = _memory_mock()
    items = [
        {"id": "m0", "memory": "runs 40km", "user_id": "u1"},
        {"id": "m1", "memory": "lifts three times a week", "user_id": "u1"},
        {
            "id": "pat-1",
            "memory": "existing pattern",
            "user_id": "u1",
            "memory_kind": "pattern",
            "synthesis_evidence_hash": evidence_hash(["m0", "m1"]),
        },
    ]
    memory.llm.generate_response.return_value = """
    {"patterns": [{"text": "dup pattern", "evidence_ids": [1, 0]}]}
    """
    stats = {"created": 0, "synthesized": 0}
    actions = run_synthesis(memory, items, user_id="u1", pass_id="pass-1", stats=stats, force=True)
    assert actions == []
    memory.add.assert_not_called()


def test_is_synthesis_eligible_rejects_patterns_and_scoped():
    assert is_synthesis_eligible({"id": "a", "user_id": "u", "governance_status": "active"}) is True
    assert is_synthesis_eligible({"id": "a", "user_id": "u", "memory_kind": "pattern"}) is False
    assert is_synthesis_eligible({"id": "a", "user_id": "u", "agent_id": "bot"}) is False
    assert is_synthesis_eligible({"id": "a", "user_id": "u", "governance_status": "merged"}) is False


def test_merged_hidden_by_default_filter():
    assert should_include_memory({"governance_status": "merged"}) is False
    assert should_include_memory({"governance_status": "superseded"}) is True
    assert should_include_memory({"governance_status": "superseded"}, latest_only=True) is False


pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def _mock_memory():
    mock_instance = MagicMock()
    mock_instance.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    mock_instance.search.return_value = {"results": []}
    mock_instance.get_all.return_value = {"results": []}
    mock_instance.update.return_value = {"message": "Memory updated"}
    mock_instance.embedding_model.embed.return_value = [0.1]
    mock_instance.vector_store.search.return_value = []
    mock_instance.llm.generate_response.return_value = '{"actions": []}'
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "", "AUTH_DISABLED": "true", "JWT_SECRET": "test-secret"}):
        with patch("mem0.Memory.from_config", return_value=mock_instance):
            yield mock_instance


@pytest.fixture
def client(_mock_memory):
    import server.main as server_main

    with patch.dict(os.environ, {"AUTH_DISABLED": "true", "ADMIN_API_KEY": "", "JWT_SECRET": "test-secret"}):
        importlib.reload(server_main)
    return TestClient(server_main.app)


def test_add_still_succeeds_when_on_add_governance_raises(client, _mock_memory):
    _mock_memory.vector_store.search.side_effect = RuntimeError("vector boom")
    resp = client.post("/memories", json={"messages": [{"role": "user", "content": "hi"}], "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["id"] == "mem-1"


def test_search_forwards_latest_only_and_include_merged(client, _mock_memory):
    resp = client.post(
        "/search",
        json={"query": "food", "user_id": "u1", "latest_only": True, "include_merged": True},
    )
    assert resp.status_code == 200
    _, kwargs = _mock_memory.search.call_args
    assert kwargs["latest_only"] is True
    assert kwargs["include_merged"] is True


def test_get_all_forwards_governance_flags(client, _mock_memory):
    resp = client.get("/memories", params={"user_id": "u1", "latest_only": True, "include_merged": False})
    assert resp.status_code == 200
    _, kwargs = _mock_memory.get_all.call_args
    assert kwargs["latest_only"] is True
    assert kwargs["include_merged"] is False


def test_dream_post_runs_and_returns_report(client, _mock_memory):
    _mock_memory.get_all.return_value = {"results": []}
    with patch("routers.dream.persist_report"):
        resp = client.post("/dream", json={"user_id": "u1", "synthesize": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "manual"
    assert "pass_id" in body
    assert "stats" in body
    _, kwargs = _mock_memory.get_all.call_args
    assert kwargs["include_merged"] is False
    assert kwargs["latest_only"] is False
