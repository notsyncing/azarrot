from dataclasses import dataclass
from typing import Any

import dataclass_wizard
from fastapi import HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_404_NOT_FOUND

from azarrot.chats.thread_manager import ChatThreadAgentToolPresetParams, ChatThreadManager
from azarrot.config import OpenAIFrontendConfig
from azarrot.file_store import FileStore
from azarrot.frontends.openai_support.openai_assistant_messages import (
    OpenAIAssistantMessage,
    to_chat_message_input_items,
)
from azarrot.frontends.openai_support.openai_assistants import (
    OpenAICodeInterpreterToolResource,
    OpenAIFileSearchToolResource,
    OpenAIFileSearchToolVectorStoreCreationRequest,
    OpenAIToolResources,
    create_openai_requested_vector_stores,
    to_agent_code_interpreter_tool_options,
    to_agent_file_search_tool_options,
)
from azarrot.models.model_manager import ModelManager
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_FILE_SEARCH
from azarrot.tools.internal.tool_code_file_search import FileSearchToolConfigs
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterToolConfigs
from azarrot.vector_store.manager import VectorStoreManager


class OpenAIAssistantCreateThreadRequest(BaseModel):
    messages: list[OpenAIAssistantMessage] | None = None
    tool_resources: OpenAIToolResources | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OpenAIAssistantThread:
    id: str
    created_at: int
    tool_resources: OpenAIToolResources | None
    metadata: dict[str, Any] | None

    object: str = "thread"


class OpenAIAssistantUpdateThreadRequest(BaseModel):
    tool_resources: OpenAIToolResources | None = None
    metadata: dict[str, Any] | None = None


def to_chat_thread_tool_preset_params(
    openai_tool_resources: OpenAIToolResources | None, reranker_model_id: str | None = None
) -> tuple[list[ChatThreadAgentToolPresetParams] | None, list[OpenAIFileSearchToolVectorStoreCreationRequest] | None]:
    if openai_tool_resources is None:
        return None, None

    agent_tool_params = []
    new_vector_stores = None

    if openai_tool_resources.code_interpreter is not None:
        code_interpreter_params = ChatThreadAgentToolPresetParams(
            tool_name=INTERNAL_TOOL_CODE_INTERPRETER,
            tool_additional_preset_params=dataclass_wizard.asdict(
                to_agent_code_interpreter_tool_options(openai_tool_resources)
            ),
        )

        agent_tool_params.append(code_interpreter_params)

    if openai_tool_resources.file_search is not None:
        tool_params, new_vector_stores = to_agent_file_search_tool_options(
            None, openai_tool_resources, reranker_model_id=reranker_model_id
        )

        file_search_params = ChatThreadAgentToolPresetParams(
            tool_name=INTERNAL_TOOL_FILE_SEARCH, tool_additional_preset_params=dataclass_wizard.asdict(tool_params)
        )

        agent_tool_params.append(file_search_params)

    return agent_tool_params, new_vector_stores


class OpenAIAssistantThreads:
    _config: OpenAIFrontendConfig
    _chat_thread_manager: ChatThreadManager
    _vector_store: VectorStoreManager
    _model_manager: ModelManager
    _file_store: FileStore

    def __init__(
        self,
        config: OpenAIFrontendConfig,
        chat_thread_manager: ChatThreadManager,
        vector_stores: VectorStoreManager,
        model_manager: ModelManager,
        file_store: FileStore,
    ) -> None:
        self._config = config
        self._chat_thread_manager = chat_thread_manager
        self._vector_store = vector_stores
        self._model_manager = model_manager
        self._file_store = file_store

    def create_thread(self, request: OpenAIAssistantCreateThreadRequest) -> OpenAIAssistantThread:
        tool_params, new_vector_stores = to_chat_thread_tool_preset_params(
            request.tool_resources, reranker_model_id=self._config.assistant_file_search_reranker_default_model_id
        )

        thread_info = self._chat_thread_manager.create(
            additional_data=request.metadata, additional_tool_preset_parameters=tool_params
        )

        if request.messages is not None:
            message_items, image_upload_requests = to_chat_message_input_items(request.messages)

            self._chat_thread_manager.add_messages(thread_info.id, message_items)

            for req in image_upload_requests:
                self._file_store.download_file(req.image_url, to_file_id=req.to_file_id)

        if new_vector_stores is not None:
            create_openai_requested_vector_stores(
                self._config, self._model_manager, self._vector_store, new_vector_stores
            )

        return OpenAIAssistantThread(
            id=thread_info.id,
            created_at=int(thread_info.create_time.timestamp()),
            tool_resources=request.tool_resources,
            metadata=request.metadata,
        )

    def __to_openai_tool_resources(
        self, thread_tool_preset_params: list[ChatThreadAgentToolPresetParams]
    ) -> OpenAIToolResources | None:
        if len(thread_tool_preset_params) <= 0:
            return None

        res = OpenAIToolResources()

        for params in thread_tool_preset_params:
            if params.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
                code_interpreter_params = dataclass_wizard.fromdict(
                    CodeInterpreterToolConfigs, params.tool_additional_preset_params
                )

                res.code_interpreter = OpenAICodeInterpreterToolResource(
                    file_ids=[str(f) for f in code_interpreter_params.exposed_files]
                    if code_interpreter_params.exposed_files is not None
                    else []
                )
            elif params.tool_name == INTERNAL_TOOL_FILE_SEARCH:
                file_search_params = dataclass_wizard.fromdict(
                    FileSearchToolConfigs, params.tool_additional_preset_params
                )

                res.file_search = OpenAIFileSearchToolResource(
                    vector_store_ids=[str(f) for f in file_search_params.vector_stores]
                    if file_search_params.vector_stores is not None
                    else []
                )

        return res

    def get_thread(self, thread_id: str) -> OpenAIAssistantThread:
        thread_info = self._chat_thread_manager.get(thread_id)

        if thread_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        thread_tool_preset_params = self._chat_thread_manager.get_thread_tool_preset_params(thread_id)

        return OpenAIAssistantThread(
            id=thread_info.id,
            created_at=int(thread_info.create_time.timestamp()),
            tool_resources=self.__to_openai_tool_resources(thread_tool_preset_params),
            metadata=thread_info.additional_data,
        )

    def update_thread(self, thread_id: str, request: OpenAIAssistantUpdateThreadRequest) -> OpenAIAssistantThread:
        params, new_vector_stores = to_chat_thread_tool_preset_params(request.tool_resources)

        thread_info = self._chat_thread_manager.update(
            thread_id, new_metadata=request.metadata, new_tool_preset_params=params
        )

        if thread_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        if new_vector_stores is not None:
            create_openai_requested_vector_stores(
                self._config, self._model_manager, self._vector_store, new_vector_stores
            )

        thread_tool_preset_params = self._chat_thread_manager.get_thread_tool_preset_params(thread_id)

        return OpenAIAssistantThread(
            id=thread_info.id,
            created_at=int(thread_info.create_time.timestamp()),
            tool_resources=self.__to_openai_tool_resources(thread_tool_preset_params),
            metadata=thread_info.additional_data,
        )

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        r = self._chat_thread_manager.delete(thread_id)

        return {"id": thread_id, "object": "thread.deleted", "deleted": r}
