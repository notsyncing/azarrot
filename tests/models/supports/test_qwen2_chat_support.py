import json

from azarrot.models.supports.qwen2_chat_support import parse_tool_calling_request_qwen2


def test_parse_qwen2_tool_calling_request() -> None:
    message = 'assistant\n✿FUNCTION✿: calculator\n✿ARGS✿: {"first_value": 3223, "second_value": 3442}'

    result = parse_tool_calling_request_qwen2(message)
    assert len(result) == 1
    assert result[0].function_name == "calculator"
    assert json.dumps(result[0].function_arguments) == '{"first_value": 3223, "second_value": 3442}'


def test_parse_qwen2_tool_calling_request_2() -> None:
    message = 'assistant\n✿FUNCTION✿: calculator\n✿ARGS✿: {"first_value": 3223, "second_value": 3442}\n✿RESULT✿:'

    result = parse_tool_calling_request_qwen2(message)
    assert len(result) == 1
    assert result[0].function_name == "calculator"
    assert json.dumps(result[0].function_arguments) == '{"first_value": 3223, "second_value": 3442}'


def test_parse_qwen2_tool_calling_request_3() -> None:
    message = 'label:\n\n✿FUNCTION✿: calculator\n✿ARGS✿: {"first_value": 102, "second_value": 33}\n responses'

    result = parse_tool_calling_request_qwen2(message)
    assert len(result) == 1
    assert result[0].function_name == "calculator"
    assert json.dumps(result[0].function_arguments) == '{"first_value": 102, "second_value": 33}'
