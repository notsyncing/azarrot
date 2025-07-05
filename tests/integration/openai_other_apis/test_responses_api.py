import logging
from collections.abc import Generator
from typing import Any, cast

import pytest
from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseStreamEvent,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput

from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import make_openvino_server

QWEN3_CHAT_MODEL = "Qwen/Qwen3-1.7B"

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def openvino_server() -> Generator[Server, Any, Any]:
    yield from make_openvino_server()


def test_create_response_with_hello(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input="你好！",
    )

    reasoning_text = "".join(["".join([s.text for s in r.summary]) for r in resp.output if r.type == "reasoning"])
    log.info("Reasoning: %s", reasoning_text)
    assert reasoning_text.find("用户") >= 0

    result = resp.output_text
    assert result is not None
    log.info("Output: %s", result)
    assert result.find("你好") >= 0 or result.find("您好") >= 0

    assert resp.usage is not None
    assert resp.usage.output_tokens > 0
    assert resp.usage.output_tokens_details.reasoning_tokens > 0


def test_create_response_with_hello_streaming(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input="你好！",
        stream=True,
    )

    events: list[ResponseStreamEvent] = []

    for event in resp:
        log.info("Output event: %s", event)
        events.append(event)

    assert len(events) > 2

    completed_event = events[-1]
    assert completed_event.type == "response.completed"

    reasoning_text = "".join(
        ["".join([s.text for s in r.summary]) for r in completed_event.response.output if r.type == "reasoning"]
    )

    log.info("Reasoning: %s", reasoning_text)
    assert reasoning_text.find("用户") >= 0

    result = completed_event.response.output_text
    assert result is not None
    log.info("Output: %s", result)
    assert result.find("你好") >= 0 or result.find("您好") >= 0


def test_create_response_with_complex_hello(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input=[EasyInputMessageParam(content="你好！", role="user", type="message")],
    )

    result = resp.output_text
    assert result is not None
    log.info("Output: %s", result)
    assert result.find("你好") >= 0 or result.find("您好") >= 0


def test_create_response_with_conversation(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input=[
            {"role": "system", "content": "你是一个乐于助人的智能助理。", "type": "message"},
            {"role": "user", "content": "请记住：红=1，绿=2", "type": "message"},
            {"role": "assistant", "content": "好的，我记住了。", "type": "message"},
            {"role": "user", "content": "请问绿=几？", "type": "message"},
        ],
    )

    result = resp.output_text
    assert result is not None
    log.info("Output: %s", result)
    assert result.find("绿") >= 0
    assert result.find("2") >= 0


def test_create_response_with_tool_calling(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    tools = [
        {
            "type": "function",
            "name": "RRR运算",
            "description": "用于把两个数进行RRR运算的工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
            "strict": False,
        }
    ]

    messages: ResponseInputParam = [
        EasyInputMessageParam(
            role="user",
            content="193与27的RRR运算结果是多少？请通过工具得到结果，并且只使用一次工具。不要用自己认为的结果。",
            type="message",
        ),
    ]

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input=messages,
        tools=tools,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    )

    log.info("Tool call: %s", resp)

    result = cast("ResponseFunctionToolCall", resp.output[-1])
    assert result is not None
    assert result.name == "RRR运算"
    assert result.arguments == '{"a": 193, "b": 27}'
    assert result.call_id == "0"

    messages.append(
        ResponseFunctionToolCallParam(
            name=result.name, arguments=result.arguments, call_id=result.call_id, type="function_call"
        )
    )

    messages.append(
        FunctionCallOutput(
            call_id="0",
            output="888",
            type="function_call_output",
        )
    )

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input=messages,
        tools=tools,  # type: ignore[arg-type]   # pyright: ignore[reportArgumentType]
    )

    result = resp.output_text
    assert result is not None
    log.info("Output: %s", result)
    assert result.find("888") >= 0


def test_create_response_with_tool_calling_streaming(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN3_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation-with-past", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/openai/v1", api_key="__TEST__"
    )

    tools = [
        {
            "type": "function",
            "name": "RRR运算",
            "description": "用于把两个数进行RRR运算的工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
            "strict": False,
        }
    ]

    messages: ResponseInputParam = [
        EasyInputMessageParam(
            role="user",
            content="193与27的RRR运算结果是多少？请通过工具得到结果，并且只使用一次工具。不要用自己认为的结果。",
            type="message",
        ),
    ]

    resp = client.responses.create(
        model=QWEN3_CHAT_MODEL,
        input=messages,
        tools=tools,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        stream=True,
    )

    events: list[ResponseStreamEvent] = []

    for event in resp:
        log.info("Output event: %s", event)
        events.append(event)

    assert len(events) > 2
    assert isinstance(events[-1], ResponseCompletedEvent)

    result = cast("ResponseFunctionToolCall", events[-1].response.output[-1])
    assert result is not None
    assert result.name == "RRR运算"
    assert result.arguments == '{"a": 193, "b": 27}'
    assert result.call_id == "0"
