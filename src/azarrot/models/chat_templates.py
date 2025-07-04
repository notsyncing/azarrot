from dataclasses import dataclass

import jinja2

from azarrot.common_data import (
    CallableToolsInfo,
    ModelPreset,
    ModelToolCallConfig,
)
from azarrot.models.supports.default_chat_support import DEFAULT_MODEL_TOOL_CALL_CONFIG
from azarrot.tools.tool_manager import ToolManager

DEFAULT_LOCALE = "zh-cn"

DEFAULT_SYSTEM_PROMPT = {"zh-cn": "你是一个乐于助人的智能助手。", "en-us": "You are a helpful assistant."}

BASE_SYSTEM_PROMPTS = {
    "internvl": {
        "zh-cn": "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL，是一个有用无害的人工智能助手。"  # noqa: RUF001, E501
    }
}

MODEL_TOOL_CALL_CONFIGS: dict[str, ModelToolCallConfig] = {}


@dataclass
class ChatTemplateRuntimeConfigs:
    enable_parallel_tool_calling: bool = False


class ChatTemplateManager:
    _tool_manager: ToolManager
    _template_engine: jinja2.Environment

    def __init__(self, tool_manager: ToolManager) -> None:
        self._tool_manager = tool_manager
        self._template_engine = jinja2.Environment()  # noqa: S701

    def __generate_tools_prompt(
        self,
        generation_variant: str,
        enable_internal_tools: bool,
        tools_info: CallableToolsInfo | None,
        locale: str,
        runtime_configs: ChatTemplateRuntimeConfigs,
    ) -> str | None:
        config = MODEL_TOOL_CALL_CONFIGS.get(generation_variant, DEFAULT_MODEL_TOOL_CALL_CONFIG)

        if config is None:
            raise ValueError("No tool calling config found for %s", generation_variant)

        templates = config.prompts

        if templates is None:
            return None

        template = templates.get(locale)

        if template is None:
            raise ValueError("Locale %s not found in tool calling prompt templates of %s", locale, generation_variant)

        jinja_template = self._template_engine.from_string(template)

        tool_descriptions = []
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
        elif enable_internal_tools:
            internal_tools = self._tool_manager.get_tool_list()
            tool_descriptions.extend([t.description().to_localized(locale) for t in internal_tools])

        if len(tool_descriptions) <= 0:
            return ""

        return jinja_template.render(
            {
                "tools": tool_descriptions,
                "tool_names": [t.name for t in tool_descriptions],
                "runtime_configs": runtime_configs,
                "force_use_any_tool": force_use_any_tool,
                "force_use_tool_name": force_use_tool_name,
            }
        )

    def determine_model_locale(self, model_preset: ModelPreset) -> str:
        locale = DEFAULT_LOCALE

        if model_preset.preferred_locale is not None:
            locale = model_preset.preferred_locale

        return locale

    def get_system_prompt(
        self,
        generation_variant: str,
        model_preset: ModelPreset,
        runtime_configs: ChatTemplateRuntimeConfigs,
        tools_info: CallableToolsInfo | None,
        *,
        base_sys_prompt: str | None = None,
        internal_tools_appended: bool = False,
    ) -> str:
        locale = self.determine_model_locale(model_preset)

        if base_sys_prompt is None:
            base_sys_prompts = BASE_SYSTEM_PROMPTS.get(generation_variant, DEFAULT_SYSTEM_PROMPT)
            base_sys_prompt = base_sys_prompts.get(locale, base_sys_prompts.get(DEFAULT_LOCALE))

        if base_sys_prompt is None:
            raise ValueError(f"Locale {locale} is not found for {generation_variant}")

        final_sys_prompt = base_sys_prompt

        if model_preset.supports_tool_calling:
            tool_calling_prompt = self.__generate_tools_prompt(
                generation_variant,
                not internal_tools_appended and model_preset.enable_internal_tools,
                tools_info,
                locale,
                runtime_configs,
            )

            if tool_calling_prompt is not None:
                final_sys_prompt += f"\n\n{tool_calling_prompt}"

        return final_sys_prompt
