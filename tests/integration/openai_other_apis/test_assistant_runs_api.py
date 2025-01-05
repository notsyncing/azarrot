import json
import logging
import time
from collections.abc import Generator
from typing import Any

import dataclass_wizard
import pytest

from azarrot.agents.chat_task_manager import (
    AgentChatTaskCreationRequest,
    AgentChatTaskDetailsListPagedQuery,
)
from azarrot.agents.common_data import AgentToolRequest
from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.chats.common_data import ChatMessageContentTextPart, ChatMessageInputItem, ChatMessageToolRequestsPart
from azarrot.server import Server
from azarrot.tools.tool import LocalizedToolDescription, LocalizedToolParameter
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_openvino_server
from tests.integration.utils import create_openai_client

QWEN2_CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WAIT_TIMEOUT = 120

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def openvino_server() -> Generator[Server, Any, Any]:
    yield from make_openvino_server()


@pytest.fixture(autouse=True)
def load_model(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation", skip_if_loaded=True
    )


@pytest.fixture(autouse=True)
def cleanup_database(openvino_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(openvino_server)


def test_create_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    msg = openvino_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart(text="你好！")])
    )

    assert msg is not None

    client = create_openai_client(openvino_server)

    openai_run = client.beta.threads.runs.create(thread_id=thread_info.id, assistant_id=agent.id)

    assert openai_run.assistant_id == agent.id
    assert openai_run.thread_id == thread_info.id
    assert openai_run.status in ("queued", "in_progress", "completed")

    log.info("Run created, id %s, status %s, waiting for response", openai_run.id, openai_run.status)

    msgs = []
    counter = WAIT_TIMEOUT

    while len(msgs) < 2:
        time.sleep(1)
        counter = counter - 1
        msgs = openvino_server.chat_thread_manager.get_latest_messages(thread_info.id, count=2)

        if counter <= 0:
            raise ValueError("Timeout while waiting for run!")

    log.info("Result: %s", msgs)

    assert msgs[0].id == msg.id
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"

    text = msgs[1].contents[0].to_persist_content()
    assert ("帮助" in text) or ("帮您" in text) or ("帮忙" in text)


def test_create_thread_and_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    client = create_openai_client(openvino_server)

    openai_run = client.beta.threads.create_and_run(
        assistant_id=agent.id, thread={"messages": [{"role": "user", "content": "你好！"}]}
    )

    assert openai_run.assistant_id == agent.id
    assert openai_run.status in ("queued", "in_progress", "completed")

    log.info("Run created, id %s, status %s, waiting for response", openai_run.id, openai_run.status)

    msgs = []
    counter = WAIT_TIMEOUT

    while len(msgs) < 2:
        time.sleep(1)
        counter = counter - 1
        msgs = openvino_server.chat_thread_manager.get_latest_messages(openai_run.thread_id, count=2)

        if counter <= 0:
            raise ValueError("Timeout while waiting for run!")

    log.info("Result: %s", msgs)

    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    assert "帮助" in msgs[1].contents[0].to_persist_content()


def test_list_runs(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    task1 = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="cancelled")
    )

    time.sleep(0.1)

    task2 = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="failed")
    )

    time.sleep(0.1)

    task3 = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="requires_action")
    )

    client = create_openai_client(openvino_server)

    runs = client.beta.threads.runs.list(thread_info.id, order="asc")
    assert len(runs.data) == 3
    assert runs.data[0].id == task1.id
    assert runs.data[1].id == task2.id
    assert runs.data[2].id == task3.id
    assert runs.model_dump()["has_more"] is False

    runs = client.beta.threads.runs.list(thread_info.id, order="asc", limit=2)
    assert len(runs.data) == 2
    assert runs.data[0].id == task1.id
    assert runs.data[1].id == task2.id
    assert runs.model_dump()["has_more"] is True

    runs = runs.get_next_page()
    assert len(runs.data) == 1
    assert runs.data[0].id == task3.id
    assert runs.model_dump()["has_more"] is False

    runs = client.beta.threads.runs.list(thread_info.id, order="desc")
    assert runs.data[0].id == task3.id
    assert runs.data[1].id == task2.id
    assert runs.data[2].id == task1.id


def test_retrieve_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    task = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="cancelled")
    )

    client = create_openai_client(openvino_server)

    run = client.beta.threads.runs.retrieve(thread_id=thread_info.id, run_id=task.id)

    assert run.id == task.id
    assert run.thread_id == thread_info.id
    assert run.assistant_id == agent.id
    assert run.status == "cancelled"


def test_modify_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    task = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="cancelled")
    )

    client = create_openai_client(openvino_server)

    run = client.beta.threads.runs.update(thread_id=thread_info.id, run_id=task.id, metadata={"a": 1})

    assert run.id == task.id
    assert run.thread_id == thread_info.id
    assert run.assistant_id == agent.id
    assert json.dumps(run.metadata) == '{"a": 1}'

    task_new = openvino_server.chat_task_manager.get(task.id, thread_info.id)
    assert task_new is not None
    assert json.dumps(task_new.additional_data) == '{"a": 1}'


def test_submit_tool_outputs_to_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(
        model_id=QWEN2_CHAT_MODEL,
        tools=[
            AgentToolRequest(
                tool_name="rrr-calc",
                tool_preset_parameters=dataclass_wizard.asdict(
                    LocalizedToolDescription(
                        name="rrr-calc",
                        display_name=None,
                        description="用于把两个数进行RRR运算的工具",
                        parameters=[
                            LocalizedToolParameter(name="a", type="number", description="第一个数", required=True),
                            LocalizedToolParameter(name="b", type="number", description="第二个数", required=True),
                        ],
                    )
                ),
            )
        ],
    )

    thread_info = openvino_server.chat_thread_manager.create()

    msg = openvino_server.chat_thread_manager.add_message(
        thread_info.id,
        ChatMessageInputItem(
            role="user",
            contents=[
                ChatMessageContentTextPart(
                    text="193与27的RRR运算结果是多少？必须通过RRR运算工具得到结果，并且只使用一次工具。不要用自己认为的结果。"
                )
            ],
        ),
    )

    assert msg is not None

    client = create_openai_client(openvino_server)

    openai_run = client.beta.threads.runs.create(thread_id=thread_info.id, assistant_id=agent.id)

    assert openai_run.status in ("queued", "in_progress", "completed")

    log.info("Run %s is running...", openai_run.id)

    counter = WAIT_TIMEOUT

    while True:
        time.sleep(2)
        counter = counter - 2

        task_info = openvino_server.chat_task_manager.get(openai_run.id, thread_info.id)
        assert task_info is not None
        assert task_info.status != "completed"

        if task_info.status == "requires_action":
            break

        log.info("Run %s is %s, waiting...", task_info.id, task_info.status)

        if counter <= 0:
            raise ValueError("Timeout while waiting for run into requires_action status!")

    log.info("Run %s is waiting for tool output...", openai_run.id)

    msgs = openvino_server.chat_thread_manager.get_latest_messages(thread_info.id)
    tool_req_msg = msgs[0]
    assert tool_req_msg.role == "assistant"
    assert isinstance(tool_req_msg.contents[0], ChatMessageToolRequestsPart)
    assert tool_req_msg.contents[0].tool_requests[0].function_name == "rrr-calc"

    tool_req_id = tool_req_msg.contents[0].tool_requests[0].id

    openai_run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_info.id, run_id=openai_run.id, tool_outputs=[{"tool_call_id": tool_req_id, "output": "888"}]
    )

    msgs = []
    counter = WAIT_TIMEOUT

    while True:
        time.sleep(1)
        counter = counter - 1
        msgs = openvino_server.chat_thread_manager.get_latest_messages(openai_run.thread_id)

        if msgs[0].role == "assistant":
            break

        if counter <= 0:
            raise ValueError("Timeout while waiting for run!")

    answer_msg = msgs[0]
    log.info("Output: %s", answer_msg)

    assert "888" in answer_msg.contents[0].to_persist_content()


def test_cancel_run(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    task = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id, status="requires_action")
    )

    client = create_openai_client(openvino_server)

    run = client.beta.threads.runs.cancel(thread_id=thread_info.id, run_id=task.id)

    assert run.status == "cancelled"

    new_task = openvino_server.chat_task_manager.get(task.id)
    assert new_task is not None
    assert new_task.status == "cancelled"
    assert new_task.complete_time is not None


def test_list_run_steps(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    msg = openvino_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart(text="你好！")])
    )

    assert msg is not None

    task = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id)
    )

    log.info("Task created, id %s, status %s, waiting for response", task.id, task.status)

    msgs = []
    counter = WAIT_TIMEOUT

    while len(msgs) < 2:
        time.sleep(1)
        counter = counter - 1
        msgs = openvino_server.chat_thread_manager.get_latest_messages(thread_info.id, count=2)

        if counter <= 0:
            raise ValueError("Timeout while waiting for run!")

    log.info("Result: %s", msgs)

    client = create_openai_client(openvino_server)

    run_steps = client.beta.threads.runs.steps.list(thread_id=thread_info.id, run_id=task.id)

    log.info("Run steps: %s", run_steps)

    assert len(run_steps.data) == 1
    assert run_steps.data[0].status == "completed"
    assert run_steps.data[0].type == "message_creation"
    assert run_steps.data[0].completed_at is not None
    assert run_steps.data[0].usage is not None
    assert run_steps.data[0].usage.completion_tokens > 0


def test_retrieve_run_step(openvino_server: Server) -> None:
    agent = openvino_server.agent_manager.create(model_id=QWEN2_CHAT_MODEL)

    thread_info = openvino_server.chat_thread_manager.create()

    msg = openvino_server.chat_thread_manager.add_message(
        thread_info.id, ChatMessageInputItem(role="user", contents=[ChatMessageContentTextPart(text="你好！")])
    )

    assert msg is not None

    task = openvino_server.chat_task_manager.create_task(
        AgentChatTaskCreationRequest(agent_id=agent.id, thread_id=thread_info.id)
    )

    log.info("Task created, id %s, status %s, waiting for response", task.id, task.status)

    msgs = []
    counter = WAIT_TIMEOUT

    while len(msgs) < 2:
        time.sleep(1)
        counter = counter - 1
        msgs = openvino_server.chat_thread_manager.get_latest_messages(thread_info.id, count=2)

        if counter <= 0:
            raise ValueError("Timeout while waiting for run!")

    log.info("Result: %s", msgs)

    details = openvino_server.chat_task_manager.get_details(
        AgentChatTaskDetailsListPagedQuery(
            thread_id=thread_info.id,
            agent_chat_task_id=task.id,
        )
    )

    client = create_openai_client(openvino_server)

    run_step = client.beta.threads.runs.steps.retrieve(
        thread_id=thread_info.id, run_id=task.id, step_id=details.data[0].id
    )

    log.info("Run steps: %s", run_step)

    assert run_step.status == "completed"
    assert run_step.type == "message_creation"
    assert run_step.completed_at is not None
    assert run_step.usage is not None
    assert run_step.usage.completion_tokens > 0
