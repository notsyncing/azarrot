import logging

from azarrot.models.supports.default_chat_support import DEFAULT_MODEL_TOOL_CALL_CONFIG, extract_tool_call_info_default

log = logging.getLogger(__name__)


def test_extract_tool_call_info_default_1() -> None:
    data = ["\n\n", '{"name": ', '"rrr-calc", ', '"arguments": ', '{"a": ', "193, ", '"b": ', "27}}\n"]
    state = {}

    parts = [extract_tool_call_info_default(DEFAULT_MODEL_TOOL_CALL_CONFIG, state, item) for item in data]

    log.info("Result: %s", parts)

    assert len(parts) == len(data)

    assert parts[0].name == ""
    assert parts[0].name_completed == False
    assert parts[0].arguments == ""

    assert parts[1].name == ""
    assert parts[1].name_completed == False
    assert parts[1].arguments == ""

    assert parts[2].name == "rrr-calc"
    assert parts[2].name_completed == True
    assert parts[2].arguments == ""

    assert parts[3].name == ""
    assert parts[3].name_completed == True
    assert parts[3].arguments == ""

    assert parts[4].name == ""
    assert parts[4].name_completed == True
    assert parts[4].arguments == data[4]

    assert parts[5].name == ""
    assert parts[5].name_completed == True
    assert parts[5].arguments == data[5]

    assert parts[6].name == ""
    assert parts[6].name_completed == True
    assert parts[6].arguments == data[6]

    assert parts[7].name == ""
    assert parts[7].name_completed == True
    assert parts[7].arguments == "27}"
