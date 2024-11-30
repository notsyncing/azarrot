from collections.abc import Generator
from typing import Any

import pytest

from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_no_backend_server
from tests.integration.utils import create_openai_client


@pytest.fixture(scope="module")
def no_backend_server() -> Generator[Server, Any, Any]:
    yield from make_no_backend_server()


@pytest.fixture(autouse=True)
def cleanup_database(no_backend_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(no_backend_server)


def test_create_asssistant(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    assistant = client.beta.assistants.create(
        instructions="You are an HR bot",
        name="HR Helper",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": ["vs_123"]}},
        model="dummy_model",
    )

    assert assistant.name == "HR Helper"
    assert assistant.instructions == "You are an HR bot"
    assert assistant.model == "dummy_model"
    assert len(assistant.tools) == 1
    assert assistant.tools[0].type == "file_search"
    assert assistant.tool_resources is not None
    assert assistant.tool_resources.file_search is not None
    assert assistant.tool_resources.file_search.vector_store_ids == ["vs_123"]


def test_list_assistants(no_backend_server: Server) -> None:
    no_backend_server.agent_manager.create("dummy_model", "agent1", "test agent 1", "you are agent 1")
    no_backend_server.agent_manager.create("dummy_model", "agent2", "test agent 2", "you are agent 2")
    no_backend_server.agent_manager.create("dummy_model", "agent3", "test agent 3", "you are agent 3")

    client = create_openai_client(no_backend_server)

    assistants = client.beta.assistants.list(order="asc")
    assert len(assistants.data) == 3
    assert assistants.data[0].name == "agent1"
    assert assistants.data[1].name == "agent2"
    assert assistants.data[2].name == "agent3"
    assert assistants.model_dump()["has_more"] is False

    assistants = client.beta.assistants.list(order="asc", limit=2)
    assert len(assistants.data) == 2
    assert assistants.data[0].name == "agent1"
    assert assistants.data[1].name == "agent2"
    assert assistants.model_dump()["has_more"] is True

    assistants = assistants.get_next_page()
    assert len(assistants.data) == 1
    assert assistants.data[0].name == "agent3"
    assert assistants.model_dump()["has_more"] is False

    assistants = client.beta.assistants.list(order="desc")
    assert assistants.data[0].name == "agent3"
    assert assistants.data[1].name == "agent2"
    assert assistants.data[2].name == "agent1"


def test_retrieve_assistant(no_backend_server: Server) -> None:
    agent = no_backend_server.agent_manager.create("dummy_model", "agent1", "test agent 1", "you are agent 1")

    client = create_openai_client(no_backend_server)

    assistant = client.beta.assistants.retrieve(agent.id)

    assert assistant.id == agent.id
    assert assistant.name == agent.name
    assert assistant.description == agent.description
    assert assistant.model == agent.model_id
    assert assistant.instructions == agent.model_instruction


def test_modify_assistant(no_backend_server: Server) -> None:
    agent = no_backend_server.agent_manager.create("dummy_model", "agent1", "test agent 1", "you are agent 1")

    client = create_openai_client(no_backend_server)

    new_agent = client.beta.assistants.update(agent.id, name="agent1-new", instructions="you are new agent 1")

    assert new_agent.name == "agent1-new"
    assert new_agent.description == "test agent 1"
    assert new_agent.instructions == "you are new agent 1"


def test_delete_assistant(no_backend_server: Server) -> None:
    agent = no_backend_server.agent_manager.create("dummy_model", "agent1", "test agent 1", "you are agent 1")

    client = create_openai_client(no_backend_server)

    deleted_assistant = client.beta.assistants.delete(agent.id)

    assert deleted_assistant.deleted is True

    info_new = no_backend_server.agent_manager.get(agent.id)
    assert info_new is None
