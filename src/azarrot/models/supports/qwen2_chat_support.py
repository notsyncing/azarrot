# Chat templates are taken from https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/function_calling.py
# See licenses/LICENSE.qwen-agent for more details.

import json
import textwrap

from azarrot.common_data import (
    ModelQuirks,
    ModelToolCallConfig,
    ToolCallRequestMessageContent,
    ToolCallResponseMessageContent,
)

QWEN2_FUNCTION = "✿FUNCTION✿"
QWEN2_ARGS = "✿ARGS✿"
QWEN2_RESULT = "✿RESULT✿"
QWEN2_RETURN = "✿RETURN✿"


def parse_tool_calling_request_qwen2(raw_message: str) -> list[ToolCallRequestMessageContent]:
    requests = []
    last_start_index = 0
    counter = 0

    while True:
        f_start_index = raw_message.find(QWEN2_FUNCTION, last_start_index)

        if f_start_index < 0:
            break

        last_start_index = f_start_index + 1

        f_end_index = raw_message.find("\n", f_start_index)

        if f_end_index < 0 or f_end_index <= f_start_index + len(QWEN2_FUNCTION) + 1:
            continue

        function_name = raw_message[f_start_index + len(QWEN2_FUNCTION) + 1 : f_end_index].strip()

        a_start_index = raw_message.find(QWEN2_ARGS, f_end_index)

        if a_start_index < 0:
            continue

        a_end_index = raw_message.find("\n", a_start_index)

        if a_end_index < 0:
            a_end_index = len(raw_message)

        if a_end_index < 0 or a_end_index <= a_start_index + len(QWEN2_ARGS) + 1:
            continue

        function_args = raw_message[a_start_index + len(QWEN2_ARGS) + 1 : a_end_index].strip()

        if len(function_name) <= 0:
            continue

        requests.append(
            ToolCallRequestMessageContent(
                id=str(counter), function_name=function_name, function_arguments=json.loads(function_args)
            )
        )

        counter = counter + 1

    return requests


def format_tool_calling_request_qwen2(contents: list[ToolCallRequestMessageContent]) -> str:
    text = ""

    for content in contents:
        text += f"{QWEN2_FUNCTION}: {content.function_name}\n"
        text += f"{QWEN2_ARGS}: {json.dumps(content.function_arguments)}\n"

    return text


def format_tool_calling_response_qwen2(contents: list[ToolCallResponseMessageContent]) -> str:
    text = ""

    for content in contents:
        text += f"{QWEN2_RESULT}: {content.result}\n"

    text += f"{QWEN2_RETURN}: "
    return text


QWEN2_MODEL_TOOL_CALL_CONFIG = ModelToolCallConfig(
    prompts={
        "zh-cn": textwrap.dedent("""\
            # 工具

            ## 你拥有如下工具：

            {% for tool in tools %}
            ### {{tool.name}}

            {{tool.name}}: {{tool.description}} 输入参数：{{tool.parameters_json()}} 此工具的输入应为JSON对象。
            {% endfor %}
            {% if runtime_configs.enable_parallel_tool_calling %}
            ## 你可以在回复中插入以下命令以并行调用N个工具：

            ✿FUNCTION✿: 工具1的名称，必须是[{{tool_names}}]之一
            ✿ARGS✿: 工具1的输入
            ✿FUNCTION✿: 工具2的名称
            ✿ARGS✿: 工具2的输入
            ...
            ✿FUNCTION✿: 工具N的名称
            ✿ARGS✿: 工具N的输入
            ✿RESULT✿: 工具1的结果
            ✿RESULT✿: 工具2的结果
            ...
            ✿RESULT✿: 工具N的结果
            ✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来
            {% else %}
            ## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

            ✿FUNCTION✿: 工具名称，必须是[{{tool_names}}]之一。
            ✿ARGS✿: 工具输入
            ✿RESULT✿: 工具结果
            ✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来

            {% if force_use_any_tool %}
            ## 你必须在回复中使用上述的至少一种工具。
            {% endif %}

            {% if force_use_tool_name is not none %}
            ## 你必须在回复中使用工具{{force_use_tool_name}}。
            {% endif %}
            {% endif %}
        """)  # noqa: RUF001
    },
    indicators=[QWEN2_FUNCTION],
    request_parsing_method=parse_tool_calling_request_qwen2,
    request_formatting_method=format_tool_calling_request_qwen2,
    response_formatting_method=format_tool_calling_response_qwen2,
)

QWEN2_MODEL_QUIRKS = ModelQuirks(
    additional_stop_before_strings=[QWEN2_RESULT, QWEN2_RETURN], full_text_indicators=[QWEN2_FUNCTION]
)
