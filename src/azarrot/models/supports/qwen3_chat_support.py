import json
from typing import Any

from azarrot.common_data import (
    ModelQuirks,
    ModelToolCallConfig,
    ToolCallRequestMessageContent,
)

QWEN3_TOOL_CALL_BEGIN = "<tool_call>"
QWEN3_TOOL_CALL_END = "</tool_call>"
QWEN3_TOOL_RESP_BEGIN = "<tool_response>"
QWEN3_TOOL_RESP_END = "</tool_response>"


def parse_tool_calling_request_qwen3(raw_message: str) -> list[ToolCallRequestMessageContent]:
    requests = []
    last_start_index = 0
    counter = 0

    while True:
        f_start_index = raw_message.find(QWEN3_TOOL_CALL_BEGIN, last_start_index)

        if f_start_index < 0:
            break

        last_start_index = f_start_index + 1

        f_end_index = raw_message.find(QWEN3_TOOL_CALL_END, f_start_index)

        if f_end_index < 0 or f_end_index <= f_start_index + len(QWEN3_TOOL_CALL_BEGIN) + 1:
            continue

        f_data_text = raw_message[f_start_index + len(QWEN3_TOOL_CALL_BEGIN) + 1 : f_end_index].strip()

        if len(f_data_text) <= 0:
            continue

        f_data: dict[str, Any] = json.loads(f_data_text)

        requests.append(
            ToolCallRequestMessageContent(
                id=str(counter), function_name=f_data["name"], function_arguments=f_data["arguments"]
            )
        )

        counter = counter + 1

    return requests


QWEN3_MODEL_TOOL_CALL_CONFIG = ModelToolCallConfig(
    prompts=None,
    indicators=[QWEN3_TOOL_CALL_BEGIN],
    request_parsing_method=parse_tool_calling_request_qwen3,
    request_formatting_method=None,
    response_formatting_method=None,
)

QWEN3_MODEL_QUIRKS = ModelQuirks(
    additional_stop_before_strings=[QWEN3_TOOL_CALL_END], full_text_indicators=[QWEN3_TOOL_CALL_BEGIN]
)
