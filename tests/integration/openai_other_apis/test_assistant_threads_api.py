import uuid
from collections.abc import Generator
from typing import Any

import dataclass_wizard
import pytest
from openai.types.beta.thread_update_params import ToolResources, ToolResourcesCodeInterpreter

from azarrot.chats.common_data import ChatMessageContentTextPart
from azarrot.chats.thread_manager import ChatMessageListPagedQuery, ChatThreadAgentToolResource
from azarrot.server import Server
from azarrot.tools.internal import INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.internal.tool_rag_search import RagSearchToolResources
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_no_backend_server
from tests.integration.utils import create_openai_client


@pytest.fixture(scope="module")
def no_backend_server() -> Generator[Server, Any, Any]:
    yield from make_no_backend_server()


@pytest.fixture(autouse=True)
def cleanup_database(no_backend_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(no_backend_server)


def test_create_empty_thread(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    empty_thread = client.beta.threads.create()

    assert empty_thread is not None
    assert empty_thread.id is not None
    assert empty_thread.created_at is not None
    assert empty_thread.object == "thread"

    thread_info = no_backend_server.chat_thread_manager.get(empty_thread.id)
    assert thread_info is not None


def test_create_message_thread(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    message_thread = client.beta.threads.create(
        messages=[
            {"role": "user", "content": "Hello, what is AI?"},
            {"role": "user", "content": "How does AI work? Explain it in simple terms."},
        ]
    )

    thread_info = no_backend_server.chat_thread_manager.get(message_thread.id)
    assert thread_info is not None

    page = no_backend_server.chat_thread_manager.get_messages(
        ChatMessageListPagedQuery(thread_id=thread_info.id, create_time_desc_order=False, page_size=10)
    )

    messages = page.data
    assert len(messages) == 2

    msg1 = messages[0]
    assert msg1.role == "user"
    assert isinstance(msg1.contents[0], ChatMessageContentTextPart)
    assert msg1.contents[0].to_persist_content() == "Hello, what is AI?"

    msg2 = messages[1]
    assert msg2.role == "user"
    assert isinstance(msg2.contents[0], ChatMessageContentTextPart)
    assert msg2.contents[0].to_persist_content() == "How does AI work? Explain it in simple terms."


def test_retrieve_thread(no_backend_server: Server) -> None:
    vs_id1 = uuid.uuid4()
    vs_id2 = uuid.uuid4()

    thread_info = no_backend_server.chat_thread_manager.create(
        additional_data={"a": 1},
        tool_resources=[
            ChatThreadAgentToolResource(
                tool_name=INTERNAL_TOOL_RAG_SEARCH,
                tool_resources=dataclass_wizard.asdict(
                    RagSearchToolResources(vector_stores=[str(vs_id1), str(vs_id2)])
                ),
            )
        ],
    )

    client = create_openai_client(no_backend_server)

    t = client.beta.threads.retrieve(thread_info.id)

    assert t.id == thread_info.id
    assert t.object == "thread"
    assert t.created_at > 0
    assert isinstance(t.metadata, dict)
    assert t.metadata["a"] == 1
    assert t.tool_resources is not None
    assert t.tool_resources.file_search is not None
    assert t.tool_resources.file_search.vector_store_ids is not None
    assert len(t.tool_resources.file_search.vector_store_ids) == 2
    assert t.tool_resources.file_search.vector_store_ids[0] == str(vs_id1)
    assert t.tool_resources.file_search.vector_store_ids[1] == str(vs_id2)


def test_modify_thread(no_backend_server: Server) -> None:
    vs_id1 = uuid.uuid4()
    vs_id2 = uuid.uuid4()

    thread_info = no_backend_server.chat_thread_manager.create(
        additional_data={"a": 1},
        tool_resources=[
            ChatThreadAgentToolResource(
                tool_name=INTERNAL_TOOL_RAG_SEARCH,
                tool_resources=dataclass_wizard.asdict(
                    RagSearchToolResources(vector_stores=[str(vs_id1), str(vs_id2)])
                ),
            )
        ],
    )

    client = create_openai_client(no_backend_server)

    t = client.beta.threads.update(
        str(thread_info.id),
        metadata={"modified": True, "user": "abc123"},
        tool_resources=ToolResources(
            code_interpreter=ToolResourcesCodeInterpreter(file_ids=[str(vs_id1), str(vs_id2)])
        ),
    )

    assert isinstance(t.metadata, dict)
    assert t.metadata["modified"] is True
    assert t.metadata["user"] == "abc123"
    assert t.metadata.get("a") is None
    assert t.tool_resources is not None
    assert t.tool_resources.file_search is None
    assert t.tool_resources.code_interpreter is not None
    assert t.tool_resources.code_interpreter.file_ids is not None
    assert len(t.tool_resources.code_interpreter.file_ids) == 2
    assert t.tool_resources.code_interpreter.file_ids[0] == str(vs_id1)
    assert t.tool_resources.code_interpreter.file_ids[1] == str(vs_id2)


def test_delete_thread(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    client = create_openai_client(no_backend_server)

    t = client.beta.threads.delete(thread_info.id)

    assert t.id == thread_info.id
    assert t.object == "thread.deleted"
    assert t.deleted is True

    info_new = no_backend_server.chat_thread_manager.get(thread_info.id)
    assert info_new is None
