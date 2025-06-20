import json
import logging
import uuid
from collections.abc import Generator
from datetime import datetime
from typing import Any, cast, override

import dataclass_wizard
import openai.types
from fastapi import APIRouter, FastAPI, HTTPException
from openai.pagination import SyncPage
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from azarrot.agents.chat_task_manager import AgentChatTaskManager
from azarrot.agents.manager import AgentManager
from azarrot.backends.common import CTIS_HAS_OBJECT, CustomTextIteratorStreamer
from azarrot.chats.thread_manager import ChatThreadManager
from azarrot.common_data import (
    EmbeddingsGenerationRequest,
    GenerationMessage,
    GenerationMessageContent,
    GenerationStatistics,
    ImageGenerationMessageContent,
    Model,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallRequestMessageContent,
    ToolCallRequestMessageContents,
    ToolCallResponseMessageContent,
    WorkingDirectories,
)
from azarrot.config import ServerConfig
from azarrot.file_store import FileStore
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.frontends.base import Frontend
from azarrot.frontends.openai_support.openai_assistant_messages import (
    OpenAIAssistantMessages,
)
from azarrot.frontends.openai_support.openai_assistant_runs import OpenAIAssistantRuns
from azarrot.frontends.openai_support.openai_assistant_threads import OpenAIAssistantThreads
from azarrot.frontends.openai_support.openai_assistants import OpenAIAssistants
from azarrot.frontends.openai_support.openai_data import (
    AssistantChatCompletionMessage,
    ChatCompletionRequest,
    SystemChatCompletionMessage,
    ToolChatCompletionMessage,
    UserChatCompletionMessage,
    UserChatImageContentItem,
    UserChatImageUrl,
    UserChatTextContentItem,
)
from azarrot.frontends.openai_support.openai_files import OpenAIFiles
from azarrot.frontends.openai_support.openai_vector_stores import OpenAIVectorStores
from azarrot.frontends.utils import (
    to_backend_tools_info,
    to_openai_embedding_token_usage,
    to_openai_token_usage,
    to_openai_tool_calls,
)
from azarrot.models.model_manager import ModelManager
from azarrot.utils.downloader import download_file_to_store
from azarrot.vector_store import VectorStoreManager


class OpenAIFrontend(Frontend):
    _log = logging.getLogger(__name__)
    _server_config: ServerConfig
    _model_manager: ModelManager
    _backend_pipe: BackendPipe
    _working_dirs: WorkingDirectories
    _openai_files: OpenAIFiles
    _assistants: OpenAIAssistants
    _threads: OpenAIAssistantThreads
    _messages: OpenAIAssistantMessages
    _vstores: OpenAIVectorStores
    _runs: OpenAIAssistantRuns
    _api: FastAPI

    def __init__(
        self,
        server_config: ServerConfig,
        model_manager: ModelManager,
        backend_pipe: BackendPipe,
        file_store: FileStore,
        agent_manager: AgentManager,
        chat_thread_manager: ChatThreadManager,
        agent_chat_task_manager: AgentChatTaskManager,
        vector_store: VectorStoreManager,
        api: FastAPI,
        working_dirs: WorkingDirectories,
    ) -> None:
        self._server_config = server_config
        self._model_manager = model_manager
        self._working_dirs = working_dirs
        self._backend_pipe = backend_pipe
        self._openai_files = OpenAIFiles(file_store)
        self._assistants = OpenAIAssistants(server_config.openai_configs, agent_manager, vector_store, model_manager)

        self._threads = OpenAIAssistantThreads(
            server_config.openai_configs, chat_thread_manager, vector_store, model_manager, file_store
        )

        self._messages = OpenAIAssistantMessages(file_store, chat_thread_manager, agent_chat_task_manager)
        self._vstores = OpenAIVectorStores(server_config.openai_configs, model_manager, vector_store)
        self._runs = OpenAIAssistantRuns(agent_chat_task_manager, chat_thread_manager, self._threads)
        self._api = api

        self.__init_routes()

    @override
    def id(self) -> str:
        return "OpenAI"

    def __init_routes(self) -> None:  # noqa: PLR0915
        router = APIRouter()

        # Models API
        router.add_api_route("/v1/models", self.get_models, methods=["GET"])
        router.add_api_route("/v1/models/{model_id}", self.get_model, methods=["GET"])

        # Chat API
        router.add_api_route("/v1/chat/completions", self.chat_completions, methods=["POST"], response_model=None)

        # Embeddings API
        router.add_api_route("/v1/embeddings", self.create_embeddings, methods=["POST"])

        # Files API
        router.add_api_route("/v1/files", self._openai_files.upload_file, methods=["POST"])
        router.add_api_route("/v1/files", self._openai_files.get_file_list, methods=["GET"])
        router.add_api_route("/v1/files/{file_id}", self._openai_files.get_file_info, methods=["GET"])
        router.add_api_route("/v1/files/{file_id}", self._openai_files.delete_file, methods=["DELETE"])
        router.add_api_route("/v1/files/{file_id}/content", self._openai_files.get_file_content, methods=["GET"])

        # Uploads API
        router.add_api_route("/v1/uploads", self._openai_files.create_upload, methods=["POST"])
        router.add_api_route("/v1/uploads/{upload_id}/parts", self._openai_files.add_upload_part, methods=["POST"])
        router.add_api_route("/v1/uploads/{upload_id}/complete", self._openai_files.complete_upload, methods=["POST"])
        router.add_api_route("/v1/uploads/{upload_id}/cancel", self._openai_files.cancel_upload, methods=["POST"])

        # Assistants - Assistants API
        router.add_api_route("/v1/assistants", self._assistants.create_assistant, methods=["POST"])
        router.add_api_route("/v1/assistants", self._assistants.get_assistant_list, methods=["GET"])
        router.add_api_route("/v1/assistants/{assistant_id}", self._assistants.get_assistant, methods=["GET"])
        router.add_api_route("/v1/assistants/{assistant_id}", self._assistants.update_assistant, methods=["POST"])
        router.add_api_route("/v1/assistants/{assistant_id}", self._assistants.delete_assistant, methods=["DELETE"])

        # Assistants - Threads API
        router.add_api_route("/v1/threads", self._threads.create_thread, methods=["POST"])
        router.add_api_route("/v1/threads/runs", self._runs.run_assistant_with_thread, methods=["POST"])
        router.add_api_route("/v1/threads/{thread_id}", self._threads.get_thread, methods=["GET"])
        router.add_api_route("/v1/threads/{thread_id}", self._threads.update_thread, methods=["POST"])
        router.add_api_route("/v1/threads/{thread_id}", self._threads.delete_thread, methods=["DELETE"])

        # Assistants - Messages API
        router.add_api_route("/v1/threads/{tid}/messages", self._messages.create_message, methods=["POST"])
        router.add_api_route("/v1/threads/{tid}/messages", self._messages.get_message_list, methods=["GET"])
        router.add_api_route("/v1/threads/{tid}/messages/{mid}", self._messages.get_message, methods=["GET"])
        router.add_api_route("/v1/threads/{tid}/messages/{mid}", self._messages.update_message, methods=["POST"])
        router.add_api_route("/v1/threads/{tid}/messages/{mid}", self._messages.delete_message, methods=["DELETE"])

        r_url = "/v1/threads/{tid}/runs"

        # Assistants - Runs API
        router.add_api_route(r_url, self._runs.run_assistant, methods=["POST"])
        router.add_api_route(r_url, self._runs.get_run_list, methods=["GET"])
        router.add_api_route(r_url + "/{rid}", self._runs.get_run, methods=["GET"])
        router.add_api_route(r_url + "/{rid}", self._runs.update_run, methods=["POST"])
        router.add_api_route(r_url + "/{rid}/submit_tool_outputs", self._runs.submit_tool_outputs, methods=["POST"])
        router.add_api_route(r_url + "/{rid}/cancel", self._runs.cancel_run, methods=["POST"])

        # Assistants - Run steps API
        router.add_api_route(r_url + "/{rid}/steps", self._runs.get_step_list, methods=["GET"])
        router.add_api_route(r_url + "/{rid}/steps/{sid}", self._runs.get_step, methods=["GET"])

        vs_url = "/v1/vector_stores"

        # Assistants - Vector stores API
        router.add_api_route(vs_url, self._vstores.create, methods=["POST"])
        router.add_api_route(vs_url, self._vstores.get_list, methods=["GET"])
        router.add_api_route(vs_url + "/{vector_store_id}", self._vstores.get, methods=["GET"])
        router.add_api_route(vs_url + "/{vector_store_id}", self._vstores.update, methods=["POST"])
        router.add_api_route(vs_url + "/{vstore_id}", self._vstores.delete, methods=["DELETE"])

        # Assistants - Vector store files API
        router.add_api_route(vs_url + "/{vid}/files", self._vstores.create_file, methods=["POST"])
        router.add_api_route(vs_url + "/{vid}/files", self._vstores.get_file_list, methods=["GET"])
        router.add_api_route(vs_url + "/{vid}/files/{fid}", self._vstores.get_file, methods=["GET"])
        router.add_api_route(vs_url + "/{vid}/files/{f}", self._vstores.delete_file, methods=["DELETE"])

        # Assistants - Vector store file batches API
        router.add_api_route(vs_url + "/{vid}/file_batches", self._vstores.create_batch, methods=["POST"])
        router.add_api_route(vs_url + "/{vid}/file_batches/{bid}", self._vstores.get_batch, methods=["GET"])
        router.add_api_route(vs_url + "/{vid}/file_batches/{bid}/cancel", self._vstores.cancel_batch, methods=["POST"])
        router.add_api_route(vs_url + "/{vid}/file_batches/{bid}/files", self._vstores.get_batch_files, methods=["GET"])

        self._api.include_router(router, prefix="/openai")

    def __to_openai_model(self, model: Model) -> openai.types.Model:
        return openai.types.Model(
            id=model.id,
            created=int(model.create_time.timestamp()),
            owned_by="openai",
            object="model"
        )

    def get_models(self) -> SyncPage[openai.types.Model]:
        models = self._model_manager.get_models()
        data = [self.__to_openai_model(m) for m in models]

        return SyncPage(data=data, object="list")

    def get_model(self, model_id: str) -> openai.types.Model:
        model = self._model_manager.get_model(model_id)

        if model is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        return self.__to_openai_model(model)

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

                            image_path = download_file_to_store(
                                url, self._working_dirs.uploaded_images, file_extension=".image"
                            )

                            content.append(ImageGenerationMessageContent(str(image_path)))
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
        elif isinstance(content, ToolCallRequestMessageContents):
            tool_calls = to_openai_tool_calls(content)
            message = {"role": "assistant", "tool_calls": [dataclass_wizard.asdict(tc) for tc in tool_calls]}
        else:
            message = {}

        resp: dict[str, Any] = {
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
            resp["usage"] = to_openai_token_usage(usage_info)

        return resp

    def __log_generation_statistics(self, generation_statistics: GenerationStatistics) -> None:
        self._log.info(generation_statistics.to_stats_text())

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
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            tools_info=to_backend_tools_info(request.tools, request.tool_choice),
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

        if self._server_config.log_generation_details:
            self._log.info("Generation response: %s", result)

        gen_stats.end_time = datetime.now()
        self.__log_generation_statistics(gen_stats)

        return self.__to_openai_chat_completion_object(
            model, result, "stop", contains_usage_info=True, usage_info=gen_stats
        )

    def create_embeddings(self, request: openai.types.EmbeddingCreateParams) -> openai.types.CreateEmbeddingResponse:
        text: str | list[str]

        if isinstance(request["input"], str):
            text = request["input"]
        elif isinstance(request["input"], list):
            if isinstance(request["input"][0], str):
                text = cast("list[str]", request["input"])
            else:
                raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported input type {type(request["input"])}")
        else:
            raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported input type {type(request["input"])}")

        model = self.__get_model(request["model"])
        gen_req = EmbeddingsGenerationRequest(request["model"], text)
        data_list, gen_stats = self._backend_pipe.generate_embeddings(model, gen_req)

        self.__log_generation_statistics(gen_stats)

        return openai.types.CreateEmbeddingResponse(
            data=[
                openai.types.Embedding(
                    embedding=data,
                    index=index,
                    object="embedding"
                )
                for index, data in enumerate(data_list)
            ],
            model=request["model"],
            object="list",
            usage=to_openai_embedding_token_usage(gen_stats)
        )
