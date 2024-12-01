from collections.abc import Generator
from typing import Any

import pytest

from azarrot.chats.common_data import ChatMessageContentTextPart, ChatMessageInputItem
from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_no_backend_server
from tests.integration.utils import create_openai_client


@pytest.fixture(scope="module")
def no_backend_server() -> Generator[Server, Any, Any]:
    yield from make_no_backend_server()


@pytest.fixture(autouse=True)
def cleanup_database(no_backend_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(no_backend_server)


def test_create_message(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    client = create_openai_client(no_backend_server)

    openai_msg = client.beta.threads.messages.create(
        thread_info.id,
        role="user",
        content="How does AI work? Explain it in simple terms.",
    )

    assert openai_msg.role == "user"
    assert openai_msg.created_at > 0
    assert openai_msg.thread_id == thread_info.id
    assert len(openai_msg.content) == 1
    assert openai_msg.content[0].type == "text"
    assert openai_msg.content[0].text.value == "How does AI work? Explain it in simple terms."

    message = no_backend_server.chat_thread_manager.get_message(openai_msg.id)
    assert message is not None

    assert message.id == openai_msg.id
    assert message.role == "user"
    assert message.thread_id == thread_info.id
    assert len(message.contents) == 1
    assert isinstance(message.contents[0], ChatMessageContentTextPart)
    assert message.contents[0].text == "How does AI work? Explain it in simple terms."


def test_list_messages(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    no_backend_server.chat_thread_manager.add_messages(
        thread_info.id,
        [
            ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 1")]),
            ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 2")]),
            ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 3")]),
        ],
    )

    client = create_openai_client(no_backend_server)

    messages = client.beta.threads.messages.list(thread_info.id, order="asc")
    assert len(messages.data) == 3
    assert messages.data[0].content[0].type == "text"
    assert messages.data[0].content[0].text.value == "message 1"
    assert messages.data[1].content[0].type == "text"
    assert messages.data[1].content[0].text.value == "message 2"
    assert messages.data[2].content[0].type == "text"
    assert messages.data[2].content[0].text.value == "message 3"
    assert messages.model_dump()["has_more"] is False

    messages = client.beta.threads.messages.list(thread_info.id, order="asc", limit=2)
    assert len(messages.data) == 2
    assert messages.data[0].content[0].type == "text"
    assert messages.data[0].content[0].text.value == "message 1"
    assert messages.data[1].content[0].type == "text"
    assert messages.data[1].content[0].text.value == "message 2"
    assert messages.model_dump()["has_more"] is True

    messages = messages.get_next_page()
    assert len(messages.data) == 1
    assert messages.data[0].content[0].type == "text"
    assert messages.data[0].content[0].text.value == "message 3"
    assert messages.model_dump()["has_more"] is False

    messages = client.beta.threads.messages.list(thread_info.id, order="desc")
    assert messages.data[0].content[0].type == "text"
    assert messages.data[0].content[0].text.value == "message 3"
    assert messages.data[1].content[0].type == "text"
    assert messages.data[1].content[0].text.value == "message 2"
    assert messages.data[2].content[0].type == "text"
    assert messages.data[2].content[0].text.value == "message 1"


def test_retrieve_message(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    msg = no_backend_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 1")])
    )

    assert msg is not None

    client = create_openai_client(no_backend_server)

    m = client.beta.threads.messages.retrieve(msg.id, thread_id=thread_info.id)

    assert m.id == msg.id
    assert m.created_at > 0
    assert m.content[0].type == "text"
    assert m.content[0].text.value == "message 1"


def test_modify_message(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    msg = no_backend_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 1")])
    )

    assert msg is not None

    client = create_openai_client(no_backend_server)

    m = client.beta.threads.messages.update(message_id=msg.id, thread_id=thread_info.id, metadata={"a": 1})

    assert isinstance(m.metadata, dict)
    assert m.metadata["a"] == 1

    message = no_backend_server.chat_thread_manager.get_message(m.id)
    assert message is not None
    assert message.additional_data is not None
    assert message.additional_data["a"] == 1


def test_delete_message(no_backend_server: Server) -> None:
    thread_info = no_backend_server.chat_thread_manager.create()

    msg = no_backend_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart("message 1")])
    )

    assert msg is not None

    client = create_openai_client(no_backend_server)

    m = client.beta.threads.messages.delete(msg.id, thread_id=thread_info.id)

    assert m.id == msg.id
    assert m.deleted is True

    msg_new = no_backend_server.chat_thread_manager.get_message(msg.id)
    assert msg_new is None
