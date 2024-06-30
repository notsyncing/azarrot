import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from transformers import TextIteratorStreamer

from azarrot.backends.openvino_backend import GenerationMessage, GenerationRequest, OpenVINOBackend
from azarrot.common_data import Model
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


type ChatCompletionMessage = SystemChatCompletionMessage | UserChatCompletionMessage | AssistantChatCompletionMessage


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionMessage]
    model: str
    stream: bool = False
    stream_options: ChatCompletionStreamOptions = Field(default=ChatCompletionStreamOptions())


class OpenAIFrontend:
    _model_manager: ModelManager
    _backend: OpenVINOBackend

    def __init__(self, model_manager: ModelManager, backend: OpenVINOBackend, api: FastAPI) -> None:
        self._model_manager = model_manager
        self._backend = backend

        router = APIRouter()

        # Models API
        router.add_api_route("/v1/models", self.get_models, methods=["GET"])
        router.add_api_route("/v1/models/{model_id}", self.get_model, methods=["GET"])

        # Chat API
        router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"], response_model=None)

        api.include_router(router)


    def __to_openai_model(self, model: Model) -> dict:
        return {
            "id": model.id,
            "object": "model",
            "created": int(model.create_time.timestamp()),
            "owned_by": "openai"
        }


    def get_models(self) -> dict:
        models = self._model_manager.get_models()
        data = [self.__to_openai_model(m) for m in models]

        return {
            "object": "list",
            "data": data
        }


    def get_model(self, model_id: str) -> dict:
        model = self._model_manager.get_model(model_id)

        if model is None:
            return {}

        return self.__to_openai_model(model)


    def __to_backend_generation_messages(self, openai_messages: list[ChatCompletionMessage]) -> list[GenerationMessage]:
        return [
            GenerationMessage(
                role=m.role,
                content=m.content
            ) for m in openai_messages
        ]


    def __to_openai_chat_completion_object(self, model: Model, content: str | None, finish_reason: str | None = None,
                                           contains_usage_info: bool = False, is_delta: bool = False) -> dict:
        resp = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk" if is_delta else "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model.id,
            "system_fingerprint": "azarrot",
            "choices": [
                {
                    "index": 0,
                    ("delta" if is_delta else "message"): {} if content is None else {
                        "role": "assistant",
                        "content": content
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason
                }
            ]
        }

        if contains_usage_info:
            resp["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        return resp


    def __wrap_to_openai_chat_completion_stream(self, streamer: TextIteratorStreamer, model: Model,
                                                contains_usage_info: bool = False):
        for text in streamer:
            if text == "":
                continue

            yield "data: " + json.dumps(self.__to_openai_chat_completion_object(
                model, text, finish_reason=None, contains_usage_info=False, is_delta=True
            )) + "\n\n"

        yield "data: " + json.dumps(self.__to_openai_chat_completion_object(
            model, None, finish_reason="stop", contains_usage_info=contains_usage_info, is_delta=True
        )) + "\n\n"


    def chat_completions(self, request: ChatCompletionRequest) -> dict | StreamingResponse:
        generate_request = GenerationRequest(
            model_id=request.model,
            messages=self.__to_backend_generation_messages(request.messages),
        )

        model, streamer = self._backend.generate(generate_request)

        if request.stream:
            return StreamingResponse(
                self.__wrap_to_openai_chat_completion_stream(streamer, model, request.stream_options.include_usage),
                media_type="text/event-stream"
            )

        content = ""

        for text in streamer:
            content += text

        return self.__to_openai_chat_completion_object(model, content, "stop",
            contains_usage_info=True)
