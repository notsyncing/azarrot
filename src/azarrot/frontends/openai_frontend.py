import json
import logging
import os
import urllib.request
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI
from starlette.responses import StreamingResponse

from azarrot.backends.common import CTIS_HAS_OBJECT, CustomTextIteratorStreamer
from azarrot.common_data import (
    CallableToolsInfo,
    EmbeddingsGenerationRequest,
    GenerationMessage,
    GenerationMessageContent,
    GenerationStatistics,
    ImageGenerationMessageContent,
    Model,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallRequestMessageContent,
    ToolCallRequestMessageContentList,
    ToolCallResponseMessageContent,
    WorkingDirectories,
)
from azarrot.config import DEFAULT_MAX_TOKENS
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.openai_support.openai_data import (
    AssistantChatCompletionMessage,
    ChatCompletionRequest,
    CreateEmbeddingsRequest,
    SystemChatCompletionMessage,
    ToolChatCompletionMessage,
    ToolChoice,
    ToolInfo,
    UserChatCompletionMessage,
    UserChatImageContentItem,
    UserChatImageUrl,
    UserChatTextContentItem,
)
from azarrot.models.model_manager import ModelManager
from azarrot.tools.tool import LocalizedToolDescription, LocalizedToolParameter


class OpenAIFrontend:
    _log = logging.getLogger(__name__)
    _model_manager: ModelManager
    _backend_pipe: BackendPipe
    _working_dirs: WorkingDirectories

    def __init__(
        self, model_manager: ModelManager, backend_pipe: BackendPipe, api: FastAPI, working_dirs: WorkingDirectories
    ) -> None:
        self._model_manager = model_manager
        self._working_dirs = working_dirs
        self._backend_pipe = backend_pipe

        router = APIRouter()

        # Models API
        router.add_api_route("/v1/models", self.get_models, methods=["GET"])
        router.add_api_route("/v1/models/{model_id}", self.get_model, methods=["GET"])

        # Chat API
        router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"], response_model=None)

        # Embeddings API
        router.add_api_route("/v1/embeddings", self.create_embeddings, methods=["POST"])

        api.include_router(router)

    def __to_openai_model(self, model: Model) -> dict:
        return {"id": model.id, "object": "model", "created": int(model.create_time.timestamp()), "owned_by": "openai"}

    def get_models(self) -> dict:
        models = self._model_manager.get_models()
        data = [self.__to_openai_model(m) for m in models]

        return {"object": "list", "data": data}

    def get_model(self, model_id: str) -> dict:
        model = self._model_manager.get_model(model_id)

        if model is None:
            return {}

        return self.__to_openai_model(model)

    def __check_path(self, path: Path) -> None:
        prefix = Path(os.path.commonpath([path, self._working_dirs.uploaded_images]))

        if prefix != self._working_dirs.uploaded_images:
            raise ValueError("Target path %s is out of working directory %s", path, self._working_dirs.uploaded_images)

    def __store_uploaded_image(self, image_url: str) -> str:
        local_file: Path

        if image_url.startswith(("http://", "https://")):
            local_file = (self._working_dirs.uploaded_images / (str(uuid.uuid4()) + ".image")).resolve()
            self.__check_path(local_file)

            self._log.info("Downloading image from %s to %s", image_url, local_file)
            urllib.request.urlretrieve(image_url, local_file)  # noqa: S310
        else:
            local_file = (self._working_dirs.uploaded_images / image_url).resolve()
            self.__check_path(local_file)

        return str(local_file)

    def __to_backend_generation_messages(
        self,
        openai_messages: list[
            SystemChatCompletionMessage
            | UserChatCompletionMessage
            | AssistantChatCompletionMessage
            | ToolChatCompletionMessage
        ],
    ) -> list[GenerationMessage]:
        result: list[GenerationMessage] = []

        for m in openai_messages:
            content: list[GenerationMessageContent]

            if isinstance(m, UserChatCompletionMessage):
                if isinstance(m.content, str):
                    content = [TextGenerationMessageContent(m.content)]
                elif isinstance(m.content, list):
                    content = []

                    for c in m.content:
                        if isinstance(c, UserChatTextContentItem):
                            content.append(TextGenerationMessageContent(c.text))
                        elif isinstance(c, UserChatImageContentItem):
                            url: str

                            if isinstance(c.image_url, str):
                                url = c.image_url
                            elif isinstance(c.image_url, UserChatImageUrl):
                                url = c.image_url.url
                            else:
                                raise ValueError("Invalid image url %s", str(c.image_url))

                            image_path = self.__store_uploaded_image(url)
                            content.append(ImageGenerationMessageContent(image_path))
                        else:
                            raise ValueError("Invalid content %s", str(c))
                else:
                    raise ValueError("Invalid messsage %s", str(m))
            elif isinstance(m, AssistantChatCompletionMessage):
                if m.content is not None:
                    content = [TextGenerationMessageContent(m.content)]
                else:
                    if m.tool_calls is None:
                        raise ValueError("No content in assistant message %s, nor exists any tool calls!", m)

                    content = []

                    for tool_call in m.tool_calls:
                        content.append(
                            ToolCallRequestMessageContent(
                                id=tool_call.id,
                                function_name=tool_call.function.name,
                                function_arguments=json.loads(tool_call.function.arguments),
                            )
                        )
            elif isinstance(m, ToolChatCompletionMessage):
                content = [ToolCallResponseMessageContent(m.tool_call_id, m.content)]
            else:
                content = [TextGenerationMessageContent(m.content)]

            msg = GenerationMessage(role=m.role, contents=content)

            result.append(msg)

        return result

    def __to_backend_tool_parameters(
        self, tool_parameters: list[dict[str, Any]] | None
    ) -> list[LocalizedToolParameter]:
        if tool_parameters is None:
            return []

        return [
            LocalizedToolParameter(
                name=parameter["name"],
                description=parameter.get("description"),
                type=parameter["type"],
                required=parameter.get("required", False),
            )
            for parameter in tool_parameters
        ]

    def __to_backend_tools_info(
        self, tools_info: list[ToolInfo] | None, tools_choice: Literal["none", "auto", "required"] | ToolChoice | None
    ) -> CallableToolsInfo | None:
        if tools_info is None:
            return None

        tools = [
            LocalizedToolDescription(
                name=tool_info.function.name,
                display_name=None,
                description=tool_info.function.description,
                parameters=self.__to_backend_tool_parameters(tool_info.function.parameters),
            )
            for tool_info in tools_info
        ]

        return CallableToolsInfo(
            tools=tools,
            force_use_no_tool=tools_choice == "none",
            force_use_any_tool=tools_choice == "required",
            force_use_tool_name=tools_choice.function.name if isinstance(tools_choice, ToolChoice) else None,
        )

    def __to_openai_chat_completion_object(
        self,
        model: Model,
        content: Any | None,
        finish_reason: str | None = None,
        contains_usage_info: bool = False,
        usage_info: GenerationStatistics | None = None,
        is_delta: bool = False,
    ) -> dict:
        message: dict[str, Any]

        if isinstance(content, str):
            message = {"role": "assistant", "content": content}
        elif isinstance(content, ToolCallResponseMessageContent):
            message = {"role": "tool", "content": content.result, "tool_call_id": content.to_id}
        elif isinstance(content, ToolCallRequestMessageContentList):
            tool_calls = [
                {
                    "id": tool_call_req.id,
                    "type": "function",
                    "function": {
                        "name": tool_call_req.function_name,
                        "arguments": json.dumps(tool_call_req.function_arguments),
                    },
                }
                for tool_call_req in content
            ]

            message = {"role": "assistant", "tool_calls": tool_calls}

            is_delta = False
        else:
            message = {}

        resp = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk" if is_delta else "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model.id,
            "system_fingerprint": "azarrot",
            "choices": [
                {
                    "index": 0,
                    ("delta" if is_delta else "message"): message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if contains_usage_info and usage_info is not None:
            resp["usage"] = {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.completion_tokens,
                "total_tokens": usage_info.prompt_tokens + usage_info.completion_tokens,
            }

        return resp

    def __log_generation_statistics(self, generation_statistics: GenerationStatistics) -> None:
        time_delta = (generation_statistics.end_time - generation_statistics.start_time) / timedelta(milliseconds=1)
        ftt = (generation_statistics.first_token_time - generation_statistics.start_time) / timedelta(milliseconds=1)

        self._log.info(
            "Total tokens: %d (prompt %d, completion %d), first token latency: %d ms, cost %d ms, %.3f tok/s",
            generation_statistics.prompt_tokens + generation_statistics.completion_tokens,
            generation_statistics.prompt_tokens,
            generation_statistics.completion_tokens,
            ftt,
            time_delta,
            (generation_statistics.completion_tokens) / time_delta * 1000,
        )

    def __wrap_to_openai_chat_completion_stream(
        self,
        streamer: CustomTextIteratorStreamer,
        model: Model,
        generation_statistics: GenerationStatistics,
        contains_usage_info: bool = False,
    ) -> Generator[str, Any, None]:
        for text in streamer:
            if text == "":
                continue

            result = text

            if text == CTIS_HAS_OBJECT:
                result = streamer.fetch_object()

            yield (
                "data: "
                + json.dumps(
                    self.__to_openai_chat_completion_object(
                        model, result, finish_reason=None, contains_usage_info=False, is_delta=True
                    )
                )
                + "\n\n"
            )

        generation_statistics.end_time = datetime.now()
        self.__log_generation_statistics(generation_statistics)

        yield (
            "data: "
            + json.dumps(
                self.__to_openai_chat_completion_object(
                    model,
                    None,
                    finish_reason="stop",
                    contains_usage_info=contains_usage_info,
                    usage_info=generation_statistics,
                    is_delta=True,
                )
            )
            + "\n\n"
        )

    def __get_model(self, model_id: str) -> Model:
        model = self._model_manager.get_model(model_id)

        if model is None:
            raise ValueError(f"Requested model {model_id} is not loaded!")

        return model

    def chat_completions(self, request: ChatCompletionRequest) -> dict | StreamingResponse:
        generate_request = TextGenerationRequest(
            model_id=request.model,
            messages=self.__to_backend_generation_messages(request.messages),
            max_tokens=request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_TOKENS,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            tools_info=self.__to_backend_tools_info(request.tools, request.tool_choice),
            parallel_tool_calling=request.parallel_tool_calls,
        )

        model = self.__get_model(request.model)
        streamer, gen_stats = self._backend_pipe.generate(model, generate_request)

        if request.stream:
            return StreamingResponse(
                self.__wrap_to_openai_chat_completion_stream(
                    streamer, model, gen_stats, request.stream_options.include_usage
                ),
                media_type="text/event-stream",
            )

        result = None
        content = ""

        for text in streamer:
            if text == CTIS_HAS_OBJECT:
                result = streamer.fetch_object()
                break

            content += text

        if result is None:
            result = content

        gen_stats.end_time = datetime.now()
        self.__log_generation_statistics(gen_stats)

        return self.__to_openai_chat_completion_object(
            model, result, "stop", contains_usage_info=True, usage_info=gen_stats
        )

    def create_embeddings(self, request: CreateEmbeddingsRequest) -> dict:
        model = self.__get_model(request.model)
        gen_req = EmbeddingsGenerationRequest(request.model, request.input)
        data, gen_stats = self._backend_pipe.generate_embeddings(model, gen_req)

        self.__log_generation_statistics(gen_stats)

        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": data, "index": 0}],
            "model": request.model,
            "usage": {
                "prompt_tokens": gen_stats.prompt_tokens,
                "total_tokens": gen_stats.prompt_tokens + gen_stats.completion_tokens,
            },
        }
