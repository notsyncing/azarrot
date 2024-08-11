import json
import logging
from copy import copy
from typing import Any, cast

from azarrot.backends.backend_base import BaseBackend
from azarrot.backends.common import (
    CTIS_DELEGATE_TO_NEXT,
    CTIS_HAS_OBJECT,
    CustomTextIteratorStreamer,
    GenerationHandlers,
)
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationMessage,
    GenerationMessageContent,
    GenerationStatistics,
    Model,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallRequestMessageContent,
    ToolCallRequestMessageContentList,
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

    def __delegate_internal_tool_calls(
        self,
        tool_calling_requests: list[ToolCallRequestMessageContent],
        model: Model,
        original_request: TextGenerationRequest,
    ) -> CustomTextIteratorStreamer:
        resp_map: dict[str, Any] = {}

        for req in tool_calling_requests:
            resp = self._tool_manager.execute_tool(req.function_name, req.function_arguments)
            resp_map[req.id] = resp

        tool_calling_responses = [
            ToolCallResponseMessageContent(to_id=tool_call_id, result=json.dumps(resp))
            for tool_call_id, resp in resp_map.items()
        ]

        new_request = copy(original_request)

        new_request.messages.append(
            GenerationMessage(role="tool", contents=cast(list[GenerationMessageContent], tool_calling_responses))
        )

        new_streamer, _ = self.generate(model, new_request)
        return new_streamer

    def __on_full_text_available(
        self,
        model: Model,
        streamer: CustomTextIteratorStreamer,
        original_request: TextGenerationRequest,
        full_text: str,
    ) -> tuple[bool, str | None]:
        is_tool_calling_request, tool_calling_requests = self._chat_template_manager.parse_tool_calling_request(
            full_text, model.generation_variant, model.preset
        )

        if is_tool_calling_request and tool_calling_requests is not None:
            internal_req_list = []
            external_req_list = []

            for req in tool_calling_requests:
                if self._tool_manager.is_internal_tool(req.function_name):
                    internal_req_list.append(req)
                else:
                    external_req_list.append(req)

            if len(internal_req_list) > 0 and len(external_req_list) <= 0:
                new_streamer = self.__delegate_internal_tool_calls(internal_req_list, model, original_request)
                streamer.set_next_streamer(new_streamer)
                return True, CTIS_DELEGATE_TO_NEXT
            elif len(external_req_list) > 0:
                if len(internal_req_list) > 0:
                    self._log.warn(
                        "The model called both internal and external tools. Internal tool calls will be ignored."
                    )

                streamer.put_object(ToolCallRequestMessageContentList(external_req_list))
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
                tool_call_responses.extend(cast(list[ToolCallResponseMessageContent], message.contents))
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
            full_text_handler=lambda streamer, text: self.__on_full_text_available(model, streamer, request, text)
        )

        bk = self._backends[model.backend]
        return bk.generate(request, gen_handlers)

    def generate_embeddings(
        self, model: Model, request: EmbeddingsGenerationRequest
    ) -> tuple[list[float], GenerationStatistics]:
        bk = self._backends[model.backend]
        return bk.generate_embeddings(request)
