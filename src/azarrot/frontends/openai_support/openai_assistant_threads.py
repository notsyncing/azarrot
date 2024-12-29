from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_404_NOT_FOUND

from azarrot.agents.common_data import AgentToolResourceRequest
from azarrot.chats.thread_manager import ChatThreadAgentToolResource, ChatThreadManager
from azarrot.config import OpenAIFrontendConfig
from azarrot.file_store import FileStore
from azarrot.frontends.openai_support.openai_assistant_messages import (
    OpenAIAssistantMessageRequest,
    to_chat_message_input_items,
)
from azarrot.frontends.openai_support.openai_assistants import (
    OpenAIFileSearchToolVectorStoreCreationRequest,
    OpenAIToolResources,
    OpenAIUpdateToolResources,
    create_openai_requested_vector_stores,
    to_agent_tool_resource_requests,
)
from azarrot.frontends.utils import to_openai_assistant_tool_resources
from azarrot.models.model_manager import ModelManager
from azarrot.vector_store.manager import VectorStoreManager


class OpenAIAssistantCreateThreadRequest(BaseModel):
    messages: list[OpenAIAssistantMessageRequest] | None = None
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


def to_chat_thread_tool_resources(
    openai_tool_resources: OpenAIToolResources | OpenAIUpdateToolResources | None,
) -> tuple[list[ChatThreadAgentToolResource] | None, list[OpenAIFileSearchToolVectorStoreCreationRequest] | None]:
    agent_tool_res, new_vs = to_agent_tool_resource_requests(openai_tool_resources)

    if agent_tool_res is None:
        return None, None

    chat_tool_res = [
        ChatThreadAgentToolResource(tool_name=a.tool_name, tool_resources=a.tool_resources) for a in agent_tool_res
    ]

    return chat_tool_res, new_vs


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
        tool_res, new_vector_stores = to_chat_thread_tool_resources(request.tool_resources)

        thread_info = self._chat_thread_manager.create(additional_data=request.metadata, tool_resources=tool_res)

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
        self, thread_tool_resources: list[ChatThreadAgentToolResource]
    ) -> OpenAIToolResources | None:
        tool_resources = [
            AgentToolResourceRequest(tool_name=t.tool_name, tool_resources=t.tool_resources or {})
            for t in thread_tool_resources
        ]

        return to_openai_assistant_tool_resources(tool_resources)

    def get_thread(self, thread_id: str) -> OpenAIAssistantThread:
        thread_info = self._chat_thread_manager.get(thread_id)

        if thread_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        thread_tool_res = self._chat_thread_manager.get_thread_tool_resources(thread_id)

        return OpenAIAssistantThread(
            id=thread_info.id,
            created_at=int(thread_info.create_time.timestamp()),
            tool_resources=self.__to_openai_tool_resources(thread_tool_res),
            metadata=thread_info.additional_data,
        )

    def update_thread(self, thread_id: str, request: OpenAIAssistantUpdateThreadRequest) -> OpenAIAssistantThread:
        params, new_vector_stores = to_chat_thread_tool_resources(request.tool_resources)

        thread_info = self._chat_thread_manager.update(
            thread_id, new_metadata=request.metadata, new_tool_resources=params
        )

        if thread_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        if new_vector_stores is not None:
            create_openai_requested_vector_stores(
                self._config, self._model_manager, self._vector_store, new_vector_stores
            )

        thread_tool_res = self._chat_thread_manager.get_thread_tool_resources(thread_id)

        return OpenAIAssistantThread(
            id=thread_info.id,
            created_at=int(thread_info.create_time.timestamp()),
            tool_resources=self.__to_openai_tool_resources(thread_tool_res),
            metadata=thread_info.additional_data,
        )

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        r = self._chat_thread_manager.delete(thread_id)

        return {"id": thread_id, "object": "thread.deleted", "deleted": r}
