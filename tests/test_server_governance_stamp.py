"""POST /memories must store an explicit governance status.

Every read path treats a missing ``governance_status`` as active, so the write
path has to agree instead of leaving the field out (which made payload filters,
dashboards and raw SQL each re-implement the fallback).
"""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def mock_memory():
    mock_instance = MagicMock()
    mock_instance.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "ADMIN_API_KEY": "",
                                 "AUTH_DISABLED": "true", "JWT_SECRET": "test-secret"}):
        with patch("mem0.Memory.from_config", return_value=mock_instance):
            yield mock_instance


@pytest.fixture
def client(mock_memory):
    import server.main as server_main
    with patch.dict(os.environ, {"ADMIN_API_KEY": "", "AUTH_DISABLED": "true",
                                 "JWT_SECRET": "test-secret"}):
        importlib.reload(server_main)
    return TestClient(server_main.app)


def _add(client, **body):
    payload = {"messages": [{"role": "user", "content": "I prefer window seats"}],
               "user_id": "u1", **body}
    response = client.post("/memories", json=payload)
    assert response.status_code == 200, response.text
    return response


def test_add_stamps_active_status(client, mock_memory):
    _add(client)
    metadata = mock_memory.add.call_args.kwargs["metadata"]
    assert metadata["governance_status"] == "active"


def test_add_keeps_a_caller_supplied_status(client, mock_memory):
    _add(client, metadata={"governance_status": "archived", "topic": "x"})
    metadata = mock_memory.add.call_args.kwargs["metadata"]
    assert metadata["governance_status"] == "archived"
    assert metadata["topic"] == "x"


def test_add_does_not_mutate_other_metadata(client, mock_memory):
    _add(client, metadata={"topic": "x"})
    metadata = mock_memory.add.call_args.kwargs["metadata"]
    assert metadata == {"topic": "x", "governance_status": "active"}
