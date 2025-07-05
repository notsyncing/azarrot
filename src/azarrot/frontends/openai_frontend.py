import json
import logging
import uuid
from collections.abc import Generator
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast, override

import openai.types
import openai.types.chat
import openai.types.chat.chat_completion_chunk
import openai.types.responses
from fastapi import APIRouter, FastAPI, HTTPException
from openai.pagination import SyncPage
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateParams,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningDeltaEvent,
    ResponseReasoningDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_reasoning_item import Summary
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from azarrot.agents.chat_task_manager import AgentChatTaskManager
from azarrot.agents.manager import AgentManager
from azarrot.backends.common import CompletionChunkStreamer
from azarrot.chats.thread_manager import ChatThreadManager
from azarrot.common_data import (
    DifferentChunkError,
    EmbeddingsGenerationRequest,
    EmptyMessageChunk,
    GeneratedMessageChunk,
    GenerationMessage,
    GenerationMessageContent,
    GenerationStatistics,
    ImageGenerationMessageContent,
    Model,
    ReasoningGeneratedMessageChunk,
    TextGeneratedMessageChunk,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallGeneratedMessageChunk,
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
    to_openai_responses_token_usage,
    to_openai_responses_tool_calls,
    to_openai_responses_tool_choice,
    to_openai_responses_tools,
    to_openai_token_usage2,
    to_openai_tool_calls2,
)
from azarrot.models.model_manager import ModelManager
from azarrot.utils.downloader import download_file_to_store
from azarrot.vector_store import VectorStoreManager

if TYPE_CHECKING:
    from pathlib import Path

    from openai.types.responses.response_input_item_param import FunctionCallOutput


class OpenAIExtendedChatCompletionMessage(openai.types.chat.ChatCompletionMessage):
    reasoning_content: str | None = None


class OpenAIExtendedChoiceDelta(openai.types.chat.chat_completion_chunk.ChoiceDelta):
    reasoning_content: str | None = None


@dataclass
class OpenAIResponseDeltaState:
    outputs: list[ResponseOutputItem]
    current_item: ResponseOutputItem | None
    seq_number: int

    def get_and_increase_seq_number(self) -> int:
        v = self.seq_number
        self.seq_number += 1
        return v


class OpenAIFrontend(Frontend):
    _log = logging.getLogger(__name__)
    _server_config: ServerConfig
    _model_manager: ModelManager
    _backend_pipe: BackendPipe
    _working_dirs: WorkingDirectories
    _file_store: FileStore
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
        self._file_store = file_store
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

        # Responses API
        router.add_api_route("/v1/responses", self.create_response, methods=["POST"], response_model=None)

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
            id=model.id, created=int(model.create_time.timestamp()), owned_by="openai", object="model"
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

    def __to_backend_generation_messages_for_responses(
        self,
        openai_inputs: str | ResponseInputParam,
    ) -> list[GenerationMessage]:
        result: list[GenerationMessage] = []

        if isinstance(openai_inputs, str):
            result = [GenerationMessage("user", [TextGenerationMessageContent(openai_inputs)])]
        else:
            for openai_input in openai_inputs:
                input_type = openai_input.get("type")

                if input_type is None:
                    raise ValueError(f"Unsupported input type {input_type} in {openai_inputs}")

                contents: list[GenerationMessageContent]

                if input_type == "message":
                    openai_input = cast("EasyInputMessageParam", openai_input)
                    contents = []

                    if isinstance(openai_input["content"], str):
                        contents.append(TextGenerationMessageContent(openai_input["content"]))
                    else:
                        for c in openai_input["content"]:
                            if c["type"] == "input_text":
                                contents.append(TextGenerationMessageContent(c["text"]))
                            elif c["type"] == "input_image":
                                image_path: Path

                                if "image_url" in c:
                                    assert c["image_url"] is not None
                                    url = c["image_url"]

                                    image_path = download_file_to_store(
                                        url, self._working_dirs.uploaded_images, file_extension=".image"
                                    )
                                elif "file_id" in c:
                                    assert c["file_id"] is not None
                                    image_path = self._file_store.make_store_file_path(c["file_id"])
                                else:
                                    raise ValueError(f"No image url or file id in input {openai_input}")

                                contents.append(ImageGenerationMessageContent(str(image_path)))
                            else:
                                raise ValueError(f"Unsupported content type {c['type']} in {c}")

                    result.append(GenerationMessage(openai_input["role"], contents))
                elif input_type == "function_call":
                    openai_input = cast("ResponseFunctionToolCallParam", openai_input)

                    contents = [
                        ToolCallRequestMessageContent(
                            id=openai_input["call_id"],
                            function_name=openai_input["name"],
                            function_arguments=json.loads(openai_input["arguments"]),
                        )
                    ]

                    result.append(GenerationMessage(role="assistant", contents=contents))
                elif input_type == "function_call_output":
                    openai_input = cast("FunctionCallOutput", openai_input)

                    contents = [
                        ToolCallResponseMessageContent(
                            to_id=openai_input["call_id"],
                            result=openai_input["output"],
                        )
                    ]

                    result.append(GenerationMessage(role="tool", contents=contents))
                else:
                    raise ValueError(f"Unsupported input type {input_type}")

        return result

    def __to_openai_chat_completion_object(
        self,
        model: Model,
        content: str
        | GenerationMessageContent
        | ToolCallRequestMessageContents
        | GeneratedMessageChunk
        | list[GeneratedMessageChunk]
        | None,
        completion_id: str,
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None,
        contains_usage_info: bool = False,
        usage_info: GenerationStatistics | None = None,
        is_delta: bool = False,
    ) -> openai.types.chat.ChatCompletionChunk | openai.types.chat.ChatCompletion:
        create_time = int(datetime.now().timestamp())

        if is_delta:
            choice = OpenAIExtendedChoiceDelta(
                role="assistant",
            )

            if content is None:
                choice.content = None
            elif isinstance(content, TextGeneratedMessageChunk):
                choice.content = content.content
            elif isinstance(content, ReasoningGeneratedMessageChunk):
                choice.reasoning_content = content.content
            elif isinstance(content, ToolCallGeneratedMessageChunk):
                choice.tool_calls = [
                    openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall(
                        index=content.index,
                        function=openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCallFunction(
                            name=content.name,
                            arguments=content.arguments,
                        ),
                    ),
                ]
            else:
                raise ValueError(f"Unsupported chunk type {content}")

            openai_chunk = openai.types.chat.ChatCompletionChunk(
                id=completion_id,
                created=create_time,
                model=model.id,
                object="chat.completion.chunk",
                choices=[
                    openai.types.chat.chat_completion_chunk.Choice(
                        index=0,
                        delta=choice,
                        finish_reason=finish_reason,
                    )
                ],
                system_fingerprint="azarrot",
            )

            if contains_usage_info and usage_info is not None:
                openai_chunk.usage = to_openai_token_usage2(usage_info)

            return openai_chunk
        else:
            assert finish_reason is not None

            message = OpenAIExtendedChatCompletionMessage(role="assistant")

            if isinstance(content, str):
                message.content = content
            elif isinstance(content, TextGenerationMessageContent):
                message.content = content.text
            elif isinstance(content, ToolCallGeneratedMessageChunk):
                tool_calls = to_openai_tool_calls2([content])
                message.tool_calls = tool_calls
            elif isinstance(content, list):
                text_contents = [c for c in content if isinstance(c, TextGeneratedMessageChunk)]
                reasoning_contents = [c for c in content if isinstance(c, ReasoningGeneratedMessageChunk)]
                tool_call_contents = [c for c in content if isinstance(c, ToolCallGeneratedMessageChunk)]
                tool_calls = to_openai_tool_calls2(tool_call_contents)
                message.content = "".join([c.content for c in text_contents])
                message.tool_calls = tool_calls if len(tool_calls) > 0 else None

                if len(reasoning_contents) > 0:
                    message.reasoning_content = "".join([c.content for c in reasoning_contents])
            else:
                raise ValueError(f"Unsupported content type {content}")

            openai_resp = openai.types.chat.ChatCompletion(
                id=completion_id,
                created=create_time,
                model=model.id,
                object="chat.completion",
                choices=[
                    openai.types.chat.chat_completion.Choice(finish_reason=finish_reason, index=0, message=message),
                ],
                system_fingerprint="azarrot",
            )

            if contains_usage_info and usage_info is not None:
                openai_resp.usage = to_openai_token_usage2(usage_info)

            return openai_resp

    def __generate_openai_response_item_done_events(
        self,
        delta_state: OpenAIResponseDeltaState,
        item: ResponseOutputItem,
        item_index: int,
    ) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []

        if item.type == "message":
            assert isinstance(item.content[-1], ResponseOutputText)
            item.status = "completed"

            events.append(
                ResponseTextDoneEvent(
                    content_index=len(item.content) - 1,
                    item_id=item.id,
                    output_index=item_index,
                    sequence_number=delta_state.get_and_increase_seq_number(),
                    text=item.content[-1].text,
                    type="response.output_text.done",
                )
            )

            events.append(
                ResponseContentPartDoneEvent(
                    content_index=len(item.content) - 1,
                    item_id=item.id,
                    output_index=item_index,
                    part=item.content[-1],
                    sequence_number=delta_state.get_and_increase_seq_number(),
                    type="response.content_part.done",
                )
            )
        elif item.type == "function_call":
            item.status = "completed"

            events.append(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=item.arguments,
                    item_id=item.id or "",
                    output_index=item_index,
                    sequence_number=delta_state.get_and_increase_seq_number(),
                    type="response.function_call_arguments.done",
                )
            )
        elif item.type == "reasoning":
            item.status = "completed"

            events.append(
                ResponseReasoningDoneEvent(
                    content_index=0,
                    item_id=item.id,
                    output_index=item_index,
                    sequence_number=delta_state.get_and_increase_seq_number(),
                    text=item.encrypted_content or "",
                    type="response.reasoning.done",
                )
            )

            events.append(
                ResponseReasoningSummaryDoneEvent(
                    item_id=item.id,
                    output_index=item_index,
                    sequence_number=delta_state.get_and_increase_seq_number(),
                    summary_index=0,
                    text=item.encrypted_content or "",
                    type="response.reasoning_summary.done",
                )
            )

        else:
            raise ValueError(f"Unsupported response output type {item.type} in item {item}")

        events.append(
            ResponseOutputItemDoneEvent(
                item=item,
                output_index=item_index,
                sequence_number=delta_state.get_and_increase_seq_number(),
                type="response.output_item.done",
            )
        )

        return events

    def __to_openai_response_object(  # noqa: PLR0915
        self,
        model: Model,
        content: str
        | GenerationMessageContent
        | ToolCallRequestMessageContents
        | GeneratedMessageChunk
        | list[GeneratedMessageChunk],
        completion_id: str,
        usage_info: GenerationStatistics,
        is_delta: bool = False,
        delta_state: OpenAIResponseDeltaState | None = None,
    ) -> list[ResponseStreamEvent] | openai.types.responses.Response:
        create_time = int(datetime.now().timestamp())

        if is_delta:
            events: list[ResponseStreamEvent] = []

            assert delta_state is not None

            if isinstance(content, TextGeneratedMessageChunk):
                if delta_state.current_item is None or delta_state.current_item.type != "message":
                    if delta_state.current_item is not None and delta_state.current_item.type != "message":
                        events.extend(
                            self.__generate_openai_response_item_done_events(
                                delta_state, delta_state.current_item, len(delta_state.outputs) - 1
                            )
                        )

                    output_item = ResponseOutputMessage(
                        id=str(uuid.uuid4()), content=[], role="assistant", status="in_progress", type="message"
                    )

                    delta_state.current_item = output_item
                    delta_state.outputs.append(output_item)

                    events.append(
                        ResponseOutputItemAddedEvent(
                            item=deepcopy(output_item),
                            output_index=len(delta_state.outputs) - 1,
                            sequence_number=delta_state.get_and_increase_seq_number(),
                            type="response.output_item.added",
                        )
                    )

                if len(delta_state.current_item.content) <= 0:
                    item_content = ResponseOutputText(
                        annotations=[],
                        text="",
                        type="output_text",
                    )

                    delta_state.current_item.content.append(item_content)

                    events.append(
                        ResponseContentPartAddedEvent(
                            content_index=0,
                            item_id=delta_state.current_item.id,
                            output_index=len(delta_state.outputs) - 1,
                            part=copy(item_content),
                            sequence_number=delta_state.get_and_increase_seq_number(),
                            type="response.content_part.added",
                        )
                    )

                assert isinstance(delta_state.current_item.content[0], ResponseOutputText)
                delta_state.current_item.content[0].text += content.content

                events.append(
                    ResponseTextDeltaEvent(
                        content_index=0,
                        delta=content.content,
                        item_id=delta_state.current_item.id,
                        output_index=len(delta_state.outputs) - 1,
                        sequence_number=delta_state.get_and_increase_seq_number(),
                        type="response.output_text.delta",
                    )
                )
            elif isinstance(content, ToolCallGeneratedMessageChunk):
                if delta_state.current_item is None or delta_state.current_item.type != "function_call":
                    if delta_state.current_item is not None and delta_state.current_item.type != "function_call":
                        events.extend(
                            self.__generate_openai_response_item_done_events(
                                delta_state, delta_state.current_item, len(delta_state.outputs) - 1
                            )
                        )

                    output_item = ResponseFunctionToolCall(
                        id=str(uuid.uuid4()),
                        name="",
                        arguments="",
                        call_id=str(content.index),
                        status="in_progress",
                        type="function_call",
                    )

                    delta_state.current_item = output_item
                    delta_state.outputs.append(output_item)

                delta_state.current_item.name += content.name or ""
                delta_state.current_item.arguments += content.arguments

                if content.name_completed:
                    events.append(
                        ResponseOutputItemAddedEvent(
                            item=copy(delta_state.current_item),
                            output_index=len(delta_state.outputs) - 1,
                            sequence_number=delta_state.get_and_increase_seq_number(),
                            type="response.output_item.added",
                        )
                    )

                if len(content.arguments) > 0:
                    assert delta_state.current_item.id is not None

                    events.append(
                        ResponseFunctionCallArgumentsDeltaEvent(
                            delta=content.arguments,
                            item_id=delta_state.current_item.id,
                            output_index=len(delta_state.outputs) - 1,
                            sequence_number=delta_state.get_and_increase_seq_number(),
                            type="response.function_call_arguments.delta",
                        )
                    )
            elif isinstance(content, ReasoningGeneratedMessageChunk):
                if delta_state.current_item is None or delta_state.current_item.type != "reasoning":
                    if delta_state.current_item is not None and delta_state.current_item.type != "reasoning":
                        events.extend(
                            self.__generate_openai_response_item_done_events(
                                delta_state, delta_state.current_item, len(delta_state.outputs) - 1
                            )
                        )

                    output_item = ResponseReasoningItem(
                        id=str(uuid.uuid4()),
                        summary=[Summary(text="", type="summary_text")],
                        type="reasoning",
                        encrypted_content="",
                        status="in_progress",
                    )

                    delta_state.current_item = output_item
                    delta_state.outputs.append(output_item)

                    events.append(
                        ResponseOutputItemAddedEvent(
                            item=deepcopy(output_item),
                            output_index=len(delta_state.outputs) - 1,
                            sequence_number=delta_state.get_and_increase_seq_number(),
                            type="response.output_item.added",
                        )
                    )

                delta_state.current_item.summary[0].text += content.content

                events.append(
                    ResponseReasoningDeltaEvent(
                        content_index=0,
                        delta={
                            "text": content.content,
                        },
                        item_id=delta_state.current_item.id,
                        output_index=len(delta_state.outputs) - 1,
                        sequence_number=delta_state.get_and_increase_seq_number(),
                        type="response.reasoning.delta",
                    )
                )
            else:
                raise ValueError(f"Unsupported chunk type {content}")

            return events
        else:
            outputs: list[ResponseOutputItem] = []
            real_contents = [content] if not isinstance(content, list) else content

            for real_content in real_contents:
                if isinstance(real_content, (str, TextGenerationMessageContent, TextGeneratedMessageChunk)):
                    text: str

                    if isinstance(real_content, str):
                        text = real_content
                    elif isinstance(real_content, TextGenerationMessageContent):
                        text = real_content.text
                    elif isinstance(real_content, TextGeneratedMessageChunk):
                        text = real_content.content
                    else:
                        raise ValueError(f"Unsupported text content {real_content}")

                    output = ResponseOutputMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=[ResponseOutputText(text=text, annotations=[], type="output_text")],
                        status="completed",
                        type="message",
                    )

                    outputs.append(output)
                elif isinstance(real_content, ReasoningGeneratedMessageChunk):
                    output = ResponseReasoningItem(
                        id=str(uuid.uuid4()),
                        summary=[Summary(text=real_content.content, type="summary_text")],
                        type="reasoning",
                    )

                    outputs.append(output)
                elif isinstance(real_content, ToolCallRequestMessageContents):
                    tool_calls = to_openai_responses_tool_calls(real_content)
                    outputs.extend(tool_calls)
                elif isinstance(real_content, ToolCallGeneratedMessageChunk):
                    tool_calls = to_openai_responses_tool_calls([real_content])
                    outputs.extend(tool_calls)
                else:
                    raise ValueError(f"Unsupported content type {real_content}")

            openai_resp = openai.types.responses.Response(
                id=completion_id,
                created_at=create_time,
                model=model.id,
                object="response",
                output=outputs,
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )

            openai_resp.usage = to_openai_responses_token_usage(usage_info)

            return openai_resp

    def __log_generation_statistics(self, generation_statistics: GenerationStatistics) -> None:
        self._log.info(generation_statistics.to_stats_text())

    def __wrap_to_openai_chat_completion_stream(
        self,
        streamer: CompletionChunkStreamer,
        model: Model,
        completion_id: str,
        generation_statistics: GenerationStatistics,
        contains_usage_info: bool = False,
    ) -> Generator[str, Any, None]:
        has_tool_calls = False

        for chunk in streamer:
            if isinstance(chunk, EmptyMessageChunk):
                continue

            if isinstance(chunk, ToolCallGeneratedMessageChunk):
                has_tool_calls = True

            openai_chunk = self.__to_openai_chat_completion_object(
                model, chunk, completion_id, finish_reason=None, contains_usage_info=False, is_delta=True
            )

            assert isinstance(openai_chunk, openai.types.chat.ChatCompletionChunk)

            yield f"data: {openai_chunk.model_dump_json()}\n\n"

        generation_statistics.end_time = datetime.now()
        self.__log_generation_statistics(generation_statistics)

        openai_chunk = self.__to_openai_chat_completion_object(
            model,
            None,
            completion_id,
            finish_reason="stop" if not has_tool_calls else "tool_calls",
            contains_usage_info=contains_usage_info,
            usage_info=generation_statistics,
            is_delta=True,
        )

        assert isinstance(openai_chunk, openai.types.chat.ChatCompletionChunk)

        yield f"data: {openai_chunk.model_dump_json()}\n\n"


    def __wrap_to_openai_response_stream(
        self,
        generate_request: TextGenerationRequest,
        streamer: CompletionChunkStreamer,
        model: Model,
        completion_id: str,
        generation_statistics: GenerationStatistics,
    ) -> Generator[str, Any, None]:
        delta_state = OpenAIResponseDeltaState(
            outputs=[],
            current_item=None,
            seq_number=0,
        )

        created_event = ResponseCreatedEvent(
            response=openai.types.responses.Response(
                id=completion_id,
                created_at=int(datetime.now().timestamp()),
                model=model.id,
                object="response",
                output=[],
                parallel_tool_calls=generate_request.parallel_tool_calling,
                tool_choice=to_openai_responses_tool_choice(generate_request.tools_info),
                tools=to_openai_responses_tools(generate_request.tools_info),
            ),
            sequence_number=delta_state.get_and_increase_seq_number(),
            type="response.created",
        )

        yield f"data: {created_event.model_dump_json()}\n\n"

        in_progress_event = ResponseInProgressEvent(
            response=created_event.response,
            sequence_number=delta_state.get_and_increase_seq_number(),
            type="response.in_progress",
        )

        yield f"data: {in_progress_event.model_dump_json()}\n\n"

        for chunk in streamer:
            if isinstance(chunk, EmptyMessageChunk):
                continue

            events: list[ResponseStreamEvent] = cast(
                "list[ResponseStreamEvent]",
                self.__to_openai_response_object(
                    model,
                    chunk,
                    completion_id,
                    usage_info=generation_statistics,
                    is_delta=True,
                    delta_state=delta_state,
                ),
            )

            for event in events:
                yield f"data: {event.model_dump_json()}\n\n"

        if delta_state.current_item is not None:
            events = self.__generate_openai_response_item_done_events(
                delta_state, delta_state.current_item, len(delta_state.outputs) - 1
            )

            for event in events:
                yield f"data: {event.model_dump_json()}\n\n"

        generation_statistics.end_time = datetime.now()
        self.__log_generation_statistics(generation_statistics)

        created_event.response.output = delta_state.outputs
        created_event.response.usage = to_openai_responses_token_usage(generation_statistics)

        completed_event = ResponseCompletedEvent(
            response=created_event.response,
            sequence_number=delta_state.get_and_increase_seq_number(),
            type="response.completed",
        )

        yield f"data: {completed_event.model_dump_json()}\n\n"

    def __get_model(self, model_id: str) -> Model:
        model = self._model_manager.get_model(model_id)

        if model is None:
            raise ValueError(f"Requested model {model_id} is not loaded!")

        return model

    def chat_completions(self, request: ChatCompletionRequest) -> openai.types.chat.ChatCompletion | StreamingResponse:
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
        completion_id = str(uuid.uuid4())

        if request.stream:
            return StreamingResponse(
                self.__wrap_to_openai_chat_completion_stream(
                    streamer,
                    model,
                    completion_id,
                    gen_stats,
                    request.stream_options.include_usage,
                ),
                media_type="text/event-stream",
            )

        contents: list[GeneratedMessageChunk] = []
        current_content: GeneratedMessageChunk | None = None

        for chunk in streamer:
            if current_content is None:
                current_content = chunk
            else:
                try:
                    current_content += chunk
                except DifferentChunkError:
                    contents.append(current_content)
                    current_content = chunk

        if current_content is not None:
            contents.append(current_content)

        if self._server_config.log_generation_details:
            self._log.info(f"Generation response: {contents}")

        gen_stats.end_time = datetime.now()
        self.__log_generation_statistics(gen_stats)

        r = self.__to_openai_chat_completion_object(
            model, contents, completion_id, "stop", contains_usage_info=True, usage_info=gen_stats
        )

        assert isinstance(r, openai.types.chat.ChatCompletion)
        return r

    def create_embeddings(self, request: openai.types.EmbeddingCreateParams) -> openai.types.CreateEmbeddingResponse:
        text: str | list[str]

        if isinstance(request["input"], str):
            text = request["input"]
        elif isinstance(request["input"], list):
            if isinstance(request["input"][0], str):
                text = cast("list[str]", request["input"])
            else:
                raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported input type {type(request['input'])}")
        else:
            raise HTTPException(HTTP_400_BAD_REQUEST, f"Unsupported input type {type(request['input'])}")

        model = self.__get_model(request["model"])
        gen_req = EmbeddingsGenerationRequest(request["model"], text)
        data_list, gen_stats = self._backend_pipe.generate_embeddings(model, gen_req)

        self.__log_generation_statistics(gen_stats)

        return openai.types.CreateEmbeddingResponse(
            data=[
                openai.types.Embedding(embedding=data, index=index, object="embedding")
                for index, data in enumerate(data_list)
            ],
            model=request["model"],
            object="list",
            usage=to_openai_embedding_token_usage(gen_stats),
        )

    def create_response(self, request: ResponseCreateParams) -> openai.types.responses.Response | StreamingResponse:
        generate_request = TextGenerationRequest(
            model_id=request["model"],
            messages=self.__to_backend_generation_messages_for_responses(request["input"]),
            max_tokens=request.get("max_output_tokens"),
            temperature=request.get("temperature") or 1.0,
            top_p=request.get("top_p") or 1.0,
            tools_info=to_backend_tools_info(list(request.get("tools", [])), request.get("tool_choice")),
            parallel_tool_calling=request.get("parallel_tool_calls") or False,
        )

        model = self.__get_model(request["model"])
        streamer, gen_stats = self._backend_pipe.generate(model, generate_request)
        completion_id = str(uuid.uuid4())

        if request.get("stream") or False:
            return StreamingResponse(
                self.__wrap_to_openai_response_stream(
                    generate_request,
                    streamer,
                    model,
                    completion_id,
                    gen_stats,
                ),
                media_type="text/event-stream",
            )

        contents: list[GeneratedMessageChunk] = []
        current_content: GeneratedMessageChunk | None = None

        for chunk in streamer:
            if current_content is None:
                current_content = chunk
            else:
                try:
                    current_content += chunk
                except DifferentChunkError:
                    contents.append(current_content)
                    current_content = chunk

        if current_content is not None:
            contents.append(current_content)

        if self._server_config.log_generation_details:
            self._log.info(f"Generation response: {contents}")

        gen_stats.end_time = datetime.now()
        self.__log_generation_statistics(gen_stats)

        r = self.__to_openai_response_object(model, contents, completion_id, gen_stats)

        assert isinstance(r, openai.types.responses.Response)
        return r
