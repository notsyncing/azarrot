import logging
from copy import copy, deepcopy
from typing import cast

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.common import (
    CompletionChunkStreamer,
)
from azarrot.common_data import (
    CallableToolsInfo,
    EmbeddingsGenerationRequest,
    GenerationMessage,
    GenerationStatistics,
    Model,
    RerankResultItem,
    ReranksGenerationRequest,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallRequestMessageContent,
    ToolCallResponseMessageContent,
)
from azarrot.models.chat_templates import (
    ChatTemplateManager,
    ChatTemplateRuntimeConfigs,
)
from azarrot.tools.tool_manager import ToolManager


class BackendPipe:
    _log = logging.getLogger(__name__)
    _backends: dict[str, BaseBackend]
    _chat_template_manager: ChatTemplateManager
    _tool_manager: ToolManager

    def __init__(
        self, backends: list[BaseBackend], chat_template_manager: ChatTemplateManager, tool_manager: ToolManager
    ) -> None:
        self._backends = {backend.id(): backend for backend in backends}
        self._chat_template_manager = chat_template_manager
        self._tool_manager = tool_manager

    def __append_internal_tools_to_request(self, model: Model, request: TextGenerationRequest) -> None:
        if model.preset.enable_internal_tools:
            locale = self._chat_template_manager.determine_model_locale(model.preset)
            internal_tools = self._tool_manager.get_tool_list()
            localized_internal_tools = [t.description().to_localized(locale) for t in internal_tools]

            if request.tools_info is None:
                request.tools_info = CallableToolsInfo(
                    tools=localized_internal_tools,
                    force_use_no_tool=False,
                    force_use_any_tool=False,
                    force_use_tool_name=None,
                )
            else:
                request.tools_info.tools.extend(localized_internal_tools)

    def generate(
        self, model: Model, request: TextGenerationRequest
    ) -> tuple[CompletionChunkStreamer, GenerationStatistics]:
        messages = []
        next_index = 0

        self.__append_internal_tools_to_request(model, request)

        if request.messages[0].role != "system":
            runtime_configs = ChatTemplateRuntimeConfigs(enable_parallel_tool_calling=request.parallel_tool_calling)

            system_prompt = self._chat_template_manager.get_system_prompt(
                generation_variant=model.generation_variant,
                model_preset=model.preset,
                runtime_configs=runtime_configs,
                tools_info=request.tools_info,
                internal_tools_appended=True,
            )

            messages.append(GenerationMessage("system", [TextGenerationMessageContent(system_prompt)]))
        else:
            system_msg = deepcopy(request.messages[0])

            if request.tools_info is not None:
                if not isinstance(system_msg.contents[0], TextGenerationMessageContent):
                    raise ValueError(f"Invalid system prompt message type {system_msg.contents[0]}")

                runtime_configs = ChatTemplateRuntimeConfigs(enable_parallel_tool_calling=request.parallel_tool_calling)

                system_msg.contents[0].text += self._chat_template_manager.get_system_prompt(
                    generation_variant=model.generation_variant,
                    model_preset=model.preset,
                    runtime_configs=runtime_configs,
                    tools_info=request.tools_info,
                    base_sys_prompt="",
                    internal_tools_appended=True,
                )

            messages.append(system_msg)
            next_index = 1

        for i in range(next_index, len(request.messages)):
            message = copy(request.messages[i])

            if isinstance(message.contents[0], ToolCallRequestMessageContent):
                tool_call_contents = message.contents

                tool_call_text = self._chat_template_manager.format_tool_calling_request(
                    cast("list[ToolCallRequestMessageContent]", tool_call_contents), model.generation_variant
                )

                if tool_call_text is not None:
                    message.contents = [TextGenerationMessageContent(text=tool_call_text)]

                messages.append(message)
            elif isinstance(message.contents[0], ToolCallResponseMessageContent):
                tool_call_responses = cast("list[ToolCallResponseMessageContent]", message.contents)

                text = self._chat_template_manager.format_tool_calling_response(
                    tool_call_responses, model.generation_variant
                )

                messages.append(GenerationMessage("tool", [TextGenerationMessageContent(text)]))
            else:
                messages.append(message)

        request.messages = messages

        bk = self._backends[model.backend]
        return bk.generate(request)

    def generate_embeddings(
        self, model: Model, request: EmbeddingsGenerationRequest
    ) -> tuple[list[list[float]], GenerationStatistics]:
        bk = self._backends[model.backend]
        return bk.generate_embeddings(request)

    def generate_reranks(
        self, model: Model, request: ReranksGenerationRequest
    ) -> tuple[list[RerankResultItem], GenerationStatistics]:
        bk = self._backends[model.backend]
        return bk.generate_reranks(request)
