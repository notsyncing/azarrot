from openai import OpenAI

from azarrot.backends.ipex_llm_backend import BACKEND_ID_IPEX_LLM
from azarrot.server import Server

QWEN2_CHAT_MODEL = "Qwen/Qwen2-7B-Instruct"


def test_qwen2_hello(ipex_llm_server: Server) -> None:
    ipex_llm_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_IPEX_LLM, "text-generation",
        skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{ipex_llm_server.config.host}:{ipex_llm_server.config.port}/v1",
        api_key="__TEST__"
    )

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的智能助理。"},
            {"role": "user", "content": "你好！"}
        ],
        seed=100
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("你好！有什么问题我可以帮助你解答吗？") > 0


def test_qwen2_tool_calling(ipex_llm_server: Server) -> None:
    ipex_llm_server.model_manager.load_huggingface_model(
        QWEN2_CHAT_MODEL, BACKEND_ID_IPEX_LLM, "text-generation",
        skip_if_loaded=True
    )

    client = OpenAI(
        base_url=f"http://{ipex_llm_server.config.host}:{ipex_llm_server.config.port}/v1",
        api_key="__TEST__"
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
                        "a": {
                            "type": "number",
                            "description": "第一个数"
                        },
                        "b": {
                            "type": "number",
                            "description": "第二个数"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "193与27的RRR运算结果是多少？"}
    ]

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=messages,  # pyright: ignore[reportArgumentType]
        tools=tools,    # pyright: ignore[reportArgumentType]
        seed=100
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "rrr-calc"
    assert tool_call.function.arguments == '{"a": 193, "b": 27}'

    messages.append({
        "role": "assistant",
        "tool_calls": [     # pyright: ignore[reportArgumentType]
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
        ]
    })

    messages.append({
        "role": "tool",
        "content": "888",
        "tool_call_id": tool_call.id
    })

    completion = client.chat.completions.create(
        model=QWEN2_CHAT_MODEL,
        messages=messages,  # pyright: ignore[reportArgumentType]
        tools=tools,    # pyright: ignore[reportArgumentType]
        seed=100
    )

    result = completion.choices[0].message
    assert result is not None
    assert result.content is not None
    assert result.content.find("888") > 0
