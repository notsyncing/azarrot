from typing import Any, Literal

from azarrot.common_data import (
    ModelQuirks,
    ModelToolCallConfig,
    ModelToolCallExtractedInfo,
)

DEFAULT_TOOL_CALL_BEGIN = "<tool_call>"
DEFAULT_TOOL_CALL_END = "</tool_call>"
DEFAULT_TOOL_RESP_BEGIN = "<tool_response>"
DEFAULT_TOOL_RESP_END = "</tool_response>"

DEFAULT_REASONING_BEGIN = "<think>"
DEFAULT_REASONING_END = "</think>"


def extract_tool_call_info_default(  # noqa: PLR0915
    config: ModelToolCallConfig, state: dict[str, Any], current_text: str
) -> ModelToolCallExtractedInfo:
    json_state: Literal["object", "key", "pre-value", "value", "unknown"] = state.get("json_state", "unknown")
    json_value_state: Literal["string", "number", "object", "boolean", "unknown"] = state.get(
        "json_value_state", "unknown"
    )
    json_object_stack: list[str] = state.get("json_object_stack", [])
    current_key: str = state.get("current_key", "")
    current_value_need_accumulation: bool = state.get("current_value_need_accumulation", False)
    current_value_need_output: bool = state.get("current_value_need_output", False)
    current_value_completed = False
    current_value = state.get("current_value", "") if current_value_need_accumulation else ""
    ignore_next_char = state.get("ignore_next_char", False)

    def is_empty_char(char: str) -> bool:
        return char in {" ", "\t", "\r", "\n"}

    current_tool_name = ""
    current_tool_arguments = ""

    for char in current_text:
        if ignore_next_char:
            ignore_next_char = False
            continue

        if char == "\\":
            ignore_next_char = True
            continue

        if json_state == "unknown":
            if char == "{":
                json_state = "object"
            else:
                continue
        elif json_state == "object":
            if is_empty_char(char):
                continue

            if char == '"':
                current_key = ""
                json_state = "key"
            elif char == ":":
                json_state = "pre-value"
            elif char == "}":
                json_state = "unknown"
            else:
                continue
        elif json_state == "key":
            if char == '"':
                json_state = "object"
            else:
                current_key += char
        elif json_state == "pre-value":
            if is_empty_char(char):
                continue

            if char == '"':
                json_state = "value"
                json_value_state = "string"
            elif char.isdigit():
                json_state = "value"
                json_value_state = "number"
            elif char in {"t", "f"}:
                json_state = "value"
                json_value_state = "boolean"
            elif char == "{":
                json_state = "value"
                json_value_state = "object"
                json_object_stack.append(char)
            else:
                continue
        elif json_state == "value":
            if (
                (json_value_state == "string" and char == '"')
                or (json_value_state == "number" and (not char.isdigit() and char != "."))
                or (json_value_state == "boolean" and not char.isalpha())
            ):
                current_value_completed = True
                json_value_state = "unknown"
                json_state = "object"
            elif json_value_state == "object" and char == "}":
                if len(json_object_stack) <= 0:
                    current_value_completed = True
                    json_value_state = "unknown"
                    json_state = "object"
                else:
                    json_object_stack.pop()
                    current_value += char
            else:
                current_value += char

        if current_key == "name":
            current_value_need_accumulation = True

            if current_value_completed:
                current_tool_name = current_value
                current_value_need_accumulation = False
                state["name_completed"] = True
            else:
                state["name_completed"] = False
        elif current_key == "arguments":
            if (config.tool_call_arguments_wrapped_by_string and json_state == "value") or (
                not config.tool_call_arguments_wrapped_by_string and json_state == "pre-value"
            ):
                current_value_need_accumulation = False
                current_value_need_output = True

        if json_state == "value" and current_value_need_output:
            current_tool_arguments += char

    state["json_state"] = json_state
    state["json_value_state"] = json_value_state
    state["json_object_stack"] = json_object_stack
    state["current_key"] = current_key
    state["current_value_need_accumulation"] = current_value_need_accumulation
    state["current_value_need_output"] = current_value_need_output
    state["ignore_next_char"] = ignore_next_char

    if current_value_need_accumulation:
        state["current_value"] = current_value
    elif "current_value" in state:
        del state["current_value"]

    return ModelToolCallExtractedInfo(
        name=current_tool_name, name_completed=state.get("name_completed", False), arguments=current_tool_arguments
    )


DEFAULT_MODEL_TOOL_CALL_CONFIG = ModelToolCallConfig(
    prompts=None,
    tool_call_start_indicator=DEFAULT_TOOL_CALL_BEGIN,
    tool_call_end_indicator=DEFAULT_TOOL_CALL_END,
    tool_call_info_extracting_method=extract_tool_call_info_default,
    tool_call_arguments_wrapped_by_string=False,
)

DEFAULT_MODEL_QUIRKS = ModelQuirks(
    additional_stop_before_strings=[DEFAULT_TOOL_CALL_END],
    reasoning_start_indicator=DEFAULT_REASONING_BEGIN,
    reasoning_end_indicator=DEFAULT_REASONING_END,
)
