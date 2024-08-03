from dataclasses import dataclass

import jinja2

from azarrot.common_data import (
    CallableToolsInfo,
    ModelPreset,
    ToolCallRequestMessageContent,
    ToolCallResponseMessageContent,
)
from azarrot.models.supports.qwen2_chat_support import QWEN2_MODEL_TOOL_CALL_CONFIG
from azarrot.tools.tool_manager import ToolManager

DEFAULT_LOCALE = "zh-cn"

DEFAULT_SYSTEM_PROMPT = {
    "zh-cn": "你是一个乐于助人的智能助手。",
    "en-us": "You are a helpful assistant."
}

BASE_SYSTEM_PROMPTS = {
    "internvl2": {
        "zh-cn": "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL，是一个有用无害的人工智能助手。" # noqa: RUF001, E501
    }
}

MODEL_TOOL_CALL_CONFIGS = {
    "qwen2": QWEN2_MODEL_TOOL_CALL_CONFIG
}


@dataclass
class ChatTemplateRuntimeConfigs:
    enable_parallel_tool_calling: bool = False


class ChatTemplateManager:
    _tool_manager: ToolManager
    _template_engine: jinja2.Environment

    def __init__(self, tool_manager: ToolManager) -> None:
        self._tool_manager = tool_manager
        self._template_engine = jinja2.Environment()    # noqa: S701

    def __generate_tools_prompt(
        self,
        generation_variant: str,
        tools_info: CallableToolsInfo | None,
        locale: str,
        runtime_configs: ChatTemplateRuntimeConfigs
    ) -> str:
        config = MODEL_TOOL_CALL_CONFIGS.get(generation_variant)

        if config is None:
            raise ValueError("No tool calling config found for %s", generation_variant)

        templates = config.prompts

        if templates is None:
            raise ValueError("No tool calling prompt template configured for %s", generation_variant)

        template = templates.get(locale)

        if template is None:
            raise ValueError("Locale %s not found in tool calling prompt templates of %s", locale, generation_variant)

        jinja_template = self._template_engine.from_string(template)

        tools = self._tool_manager.get_tool_list()
        tool_descriptions = [t.description().to_localized(locale) for t in tools]
        force_use_any_tool = False
        force_use_tool_name: str | None = None

        if tools_info is not None:
            tool_descriptions.extend(tools_info.tools)

            if tools_info.force_use_no_tool:
                return ""
            elif tools_info.force_use_any_tool:
                force_use_any_tool = True
            elif tools_info.force_use_tool_name is not None:
                force_use_tool_name = tools_info.force_use_tool_name

        return jinja_template.render({
            "tools": tool_descriptions,
            "tool_names": [t.name for t in tool_descriptions],
            "runtime_configs": runtime_configs,
            "force_use_any_tool": force_use_any_tool,
            "force_use_tool_name": force_use_tool_name
        })

    def get_system_prompt(
        self,
        generation_variant: str,
        model_preset: ModelPreset,
        runtime_configs: ChatTemplateRuntimeConfigs,
        tools_info: CallableToolsInfo | None
    ) -> str:
        base_sys_prompts = BASE_SYSTEM_PROMPTS.get(generation_variant, DEFAULT_SYSTEM_PROMPT)

        locale = DEFAULT_LOCALE

        if model_preset.preferred_locale is not None:
            locale = model_preset.preferred_locale

        base_sys_prompt = base_sys_prompts.get(locale, base_sys_prompts.get(DEFAULT_LOCALE))

        if base_sys_prompt is None:
            raise ValueError(f"Locale {locale} is not found for {generation_variant}")

        final_sys_prompt = base_sys_prompt

        if model_preset.supports_tool_calling:
            final_sys_prompt += "\n\n"
            final_sys_prompt += self.__generate_tools_prompt(generation_variant, tools_info, locale, runtime_configs)

        return final_sys_prompt

    def format_tool_calling_request(
        self,
        tool_calling_requests: list[ToolCallRequestMessageContent],
        generation_variant: str
    ) -> str:
        config = MODEL_TOOL_CALL_CONFIGS.get(generation_variant)

        if config is None:
            raise ValueError("No tool calling config found for %s", generation_variant)

        return config.request_formatting_method(tool_calling_requests)

    def parse_tool_calling_request(
        self,
        message: str,
        generation_variant: str,
        model_preset: ModelPreset
    ) -> tuple[bool, list[ToolCallRequestMessageContent] | None]:
        if not model_preset.supports_tool_calling:
            return False, None

        config = MODEL_TOOL_CALL_CONFIGS.get(generation_variant)

        if config is None:
            raise ValueError("No tool calling config found for %s", generation_variant)

        if config.indicators is None:
            return False, None

        for indicator in config.indicators:
            if message.find(indicator) >= 0:
                parsed_requests = config.request_parsing_method(message)
                return len(parsed_requests) > 0, parsed_requests

        return False, None

    def format_tool_calling_response(
        self,
        tool_calling_responses: list[ToolCallResponseMessageContent],
        generation_variant: str
    ) -> str:
        config = MODEL_TOOL_CALL_CONFIGS.get(generation_variant)

        if config is None:
            raise ValueError("No tool calling config found for %s", generation_variant)

        tool_calling_responses = sorted(tool_calling_responses, key=lambda c: int(c.to_id))
        return config.response_formatting_method(tool_calling_responses)
