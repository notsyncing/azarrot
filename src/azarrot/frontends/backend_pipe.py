from copy import copy
from typing import cast

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.common import CTIS_HAS_OBJECT, CustomTextIteratorStreamer, GenerationHandlers
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationMessage,
    GenerationStatistics,
    Model,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallRequestMessageContent,
    ToolCallRequestMessageContentList,
    ToolCallResponseMessageContent,
)
from azarrot.models.chat_templates import ChatTemplateManager, ChatTemplateRuntimeConfigs


class BackendPipe:
    _backends: dict[str, BaseBackend]
    _chat_template_manager: ChatTemplateManager

    def __init__(self, backends: list[BaseBackend], chat_template_manager: ChatTemplateManager) -> None:
        self._backends = {backend.id(): backend for backend in backends}
        self._chat_template_manager = chat_template_manager

    def __on_full_text_available(
        self, model: Model, streamer: CustomTextIteratorStreamer, full_text: str
    ) -> tuple[bool, str | None]:
        is_tool_calling_request, tool_calling_requests = self._chat_template_manager.parse_tool_calling_request(
            full_text, model.generation_variant, model.preset
        )

        if is_tool_calling_request and tool_calling_requests is not None:
            streamer.put_object(ToolCallRequestMessageContentList(tool_calling_requests))
            return True, CTIS_HAS_OBJECT

        return False, None

    def generate(
        self, model: Model, request: TextGenerationRequest
    ) -> tuple[CustomTextIteratorStreamer, GenerationStatistics]:
        messages = []
        next_index = 0

        if request.messages[0].role != "system":
            runtime_configs = ChatTemplateRuntimeConfigs(enable_parallel_tool_calling=request.parallel_tool_calling)

            system_prompt = self._chat_template_manager.get_system_prompt(
                generation_variant=model.generation_variant,
                model_preset=model.preset,
                runtime_configs=runtime_configs,
                tools_info=request.tools_info,
            )

            messages.append(GenerationMessage("system", [TextGenerationMessageContent(system_prompt)]))
        else:
            messages.append(request.messages[0])
            next_index = 1

        tool_call_responses: list[ToolCallResponseMessageContent] = []

        for i in range(next_index, len(request.messages)):
            message = copy(request.messages[i])

            if not isinstance(message.contents[0], ToolCallResponseMessageContent) and len(tool_call_responses) > 0:
                text = self._chat_template_manager.format_tool_calling_response(
                    tool_call_responses, model.generation_variant
                )

                messages.append(GenerationMessage("tool", [TextGenerationMessageContent(text)]))
                tool_call_responses = []

            if isinstance(message.contents[0], ToolCallRequestMessageContent):
                tool_call_contents = message.contents

                message.contents = [
                    TextGenerationMessageContent(
                        text=self._chat_template_manager.format_tool_calling_request(
                            cast(list[ToolCallRequestMessageContent], tool_call_contents), model.generation_variant
                        )
                    )
                ]

                messages.append(message)
            elif isinstance(message.contents[0], ToolCallResponseMessageContent):
                tool_call_responses.append(message.contents[0])
            else:
                messages.append(message)

        if len(tool_call_responses) > 0:
            text = self._chat_template_manager.format_tool_calling_response(
                tool_call_responses, model.generation_variant
            )

            messages.append(GenerationMessage("tool", [TextGenerationMessageContent(text)]))
            tool_call_responses = []

        request.messages = messages

        gen_handlers = GenerationHandlers(
            full_text_handler=lambda streamer, text: self.__on_full_text_available(model, streamer, text)
        )

        bk = self._backends[model.backend]
        return bk.generate(request, gen_handlers)

    def generate_embeddings(
        self, model: Model, request: EmbeddingsGenerationRequest
    ) -> tuple[list[float], GenerationStatistics]:
        bk = self._backends[model.backend]
        return bk.generate_embeddings(request)
