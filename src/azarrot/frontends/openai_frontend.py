import json
import logging
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer

from azarrot.backends.backend_base import BaseBackend
from azarrot.common_data import EmbeddingsGenerationRequest, GenerationMessage, TextGenerationRequest, GenerationStatistics, Model
from azarrot.config import DEFAULT_MAX_TOKENS
from azarrot.models import ModelManager


class SystemChatCompletionMessage(BaseModel):
    content: str
    role: Literal["system"]
    name: str | None = None


class UserChatCompletionMessage(BaseModel):
    content: str
    role: Literal["user"]
    name: str | None = None


class AssistantChatCompletionMessage(BaseModel):
    content: str
    role: Literal["assistant"]
    name: str | None = None


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    messages: list[SystemChatCompletionMessage | UserChatCompletionMessage | AssistantChatCompletionMessage]
    model: str
    max_tokens: int | None = None
    stream: bool = False
    stream_options: ChatCompletionStreamOptions = Field(default=ChatCompletionStreamOptions())


class CreateEmbeddingsRequest(BaseModel):
    input: str
    model: str
    encoding_format: Literal["float", "base64"] = Field(default="float")
    dimensions: int | None = None
    user: str | None = None


class OpenAIFrontend:
    _log = logging.getLogger(__name__)
    _model_manager: ModelManager
    _backends: dict[str, BaseBackend]

    def __init__(self, model_manager: ModelManager, backends: list[BaseBackend], api: FastAPI) -> None:
        self._model_manager = model_manager

        self._backends = {}

        for backend in backends:
            self._backends[backend.id()] = backend

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

    def __to_backend_generation_messages(
        self,
        openai_messages: list[SystemChatCompletionMessage | UserChatCompletionMessage | AssistantChatCompletionMessage]
    ) -> list[GenerationMessage]:
        return [GenerationMessage(role=m.role, content=m.content) for m in openai_messages]

    def __to_openai_chat_completion_object(
        self,
        model: Model,
        content: str | None,
        finish_reason: str | None = None,
        contains_usage_info: bool = False,
        usage_info: GenerationStatistics | None = None,
        is_delta: bool = False,
    ) -> dict:
        resp = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk" if is_delta else "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model.id,
            "system_fingerprint": "azarrot",
            "choices": [
                {
                    "index": 0,
                    ("delta" if is_delta else "message"): {}
                    if content is None
                    else {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if contains_usage_info and usage_info is not None:
            resp["usage"] = {
                "prompt_tokens": usage_info.prompt_tokens,
                "completion_tokens": usage_info.total_tokens - usage_info.prompt_tokens,
                "total_tokens": usage_info.total_tokens,
            }

        return resp

    def __log_generation_statistics(self, generation_statistics: GenerationStatistics) -> None:
        time_delta = (generation_statistics.end_time - generation_statistics.start_time) / timedelta(milliseconds=1)

        self._log.info(
            "Total tokens: %d (prompt %d, completion %d), cost %d ms, %.3f tok/s",
            generation_statistics.total_tokens,
            generation_statistics.prompt_tokens,
            generation_statistics.total_tokens - generation_statistics.prompt_tokens,
            time_delta,
            (generation_statistics.total_tokens - generation_statistics.prompt_tokens) / time_delta * 1000,
        )

    def __wrap_to_openai_chat_completion_stream(
        self,
        streamer: TextIteratorStreamer,
        model: Model,
        generation_statistics: GenerationStatistics,
        contains_usage_info: bool = False,
    ) -> Generator[str, Any, None]:
        for text in streamer:
            if text == "":
                continue

            yield (
                "data: "
                + json.dumps(
                    self.__to_openai_chat_completion_object(
                        model, text, finish_reason=None, contains_usage_info=False, is_delta=True
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
        )

        model = self.__get_model(request.model)
        backend = self._backends[model.backend]
        streamer, gen_stats = backend.generate(generate_request)

        if request.stream:
            return StreamingResponse(
                self.__wrap_to_openai_chat_completion_stream(
                    streamer, model, gen_stats, request.stream_options.include_usage
                ),
                media_type="text/event-stream",
            )

        content = ""

        for text in streamer:
            content += text

        gen_stats.end_time = datetime.now()
        self.__log_generation_statistics(gen_stats)

        return self.__to_openai_chat_completion_object(
            model, content, "stop", contains_usage_info=True, usage_info=gen_stats
        )

    def create_embeddings(self, request: CreateEmbeddingsRequest) -> dict:
        model = self.__get_model(request.model)
        backend = self._backends[model.backend]

        gen_req = EmbeddingsGenerationRequest(request.model, request.input)
        data, gen_stats = backend.generate_embeddings(gen_req)

        self.__log_generation_statistics(gen_stats)

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": data,
                    "index": 0
                }
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": gen_stats.prompt_tokens,
                "total_tokens": gen_stats.total_tokens
            }
        }

