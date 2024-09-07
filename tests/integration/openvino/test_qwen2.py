from typing import Any

import pytest
from openai import OpenAI

from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.models.model_manager import DEFAULT_MODEL_PRESETS
from azarrot.server import Server
from azarrot.tools import GLOBAL_TOOL_MANAGER
from azarrot.tools.tool import Tool, ToolDescription, ToolParameter

QWEN2_CHAT_MODEL = "Qwen/Qwen2-7B-Instruct"


def test_qwen2_hello(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=[{"role": "system", "content": "你是一个乐于助人的智能助理。"}, {"role": "user", "content": "你好！"}],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("有什么我可以解答的问题") >= 0


def test_qwen2_conversation(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的智能助理。"},
            {"role": "user", "content": "请记住：红=1，绿=2"},
            {"role": "assistant", "content": "好的，我记住了。"},
            {"role": "user", "content": "请问绿=几？"},
        ],
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("绿是2") >= 0


@pytest.mark.skip(reason="Qwen2 not stable on OpenVINO yet")
def test_qwen2_tool_calling(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_OPENVINO, "text-generation", skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "description": "用于把两个数进行RRR运算的工具",
                "name": "rrr-calc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数"},
                        "b": {"type": "number", "description": "第二个数"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "193与27的RRR运算结果是多少？"}]

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=messages,  # pyright: ignore[reportArgumentType]
        tools=tools,  # pyright: ignore[reportArgumentType]
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "rrr-calc"
    assert tool_call.function.arguments == '{"a": 193, "b": 27}'

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [  # type: ignore[dict-item]    # pyright: ignore[reportArgumentType]
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
            ],
        }
    )

    messages.append({"role": "tool", "content": "888", "tool_call_id": tool_call.id})

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=messages,  # pyright: ignore[reportArgumentType]
        tools=tools,  # pyright: ignore[reportArgumentType]
        seed=100,
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("888") >= 0


class RRRTool(Tool):
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="rrr-calc",
            default_locale="zh-cn",
            display_name={"zh-cn": "RRR运算器"},
            description={"zh-cn": "用于把两个数进行RRR运算的工具"},
            parameters=[
                ToolParameter(name="a", type="number", description={"zh-cn": "第一个数"}, required=True),
                ToolParameter(name="b", type="number", description={"zh-cn": "第二个数"}, required=True),
            ],
        )

    def execute(self, **kwargs: Any) -> Any:
        return kwargs["a"] + kwargs["b"] - 200


def test_qwen2_internal_tool_calling(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL,
        BACKEND_ID_OPENVINO,
        "text-generation",
        skip_if_loaded=True,
        model_preset=DEFAULT_MODEL_PRESETS["qwen2"].with_enable_internal_tools(),
    )

    GLOBAL_TOOL_MANAGER.clear_registered_tools()
    GLOBAL_TOOL_MANAGER.register_tool(RRRTool())

    client = OpenAI(
        base_url=f"http://{openvino_server.config.host}:{openvino_server.config.port}/v1", api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL, messages=[{"role": "user", "content": "193与27的RRR运算结果是多少？"}], seed=100
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("结果是20") >= 0
