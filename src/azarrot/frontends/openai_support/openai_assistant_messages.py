import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from starlette.status import HTTP_404_NOT_FOUND

from azarrot.agents.chat_task_manager import AgentChatTaskInfo, AgentChatTaskManager
from azarrot.chats.common_data import (
    ChatMessageAttachmentItem,
    ChatMessageContentImagePart,
    ChatMessageContentPart,
    ChatMessageContentTextPart,
    ChatMessageInputItem,
    ChatMessageItem,
    ChatMessageToolOutputsPart,
    ChatMessageToolRequestsPart,
)
from azarrot.chats.thread_manager import ChatMessageListPagedQuery, ChatThreadManager
from azarrot.file_store import FileStore
from azarrot.frontends.openai_support.openai_assistants import OpenAIAssistantToolType
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_RAG_SEARCH

OpenAIAssistantMessageRole = Literal["assistant", "user"]


class OpenAIAssistantMessagePartImageFileInfo(BaseModel):
    file_id: str
    detail: str = "auto"


class OpenAIAssistantMessagePartImageUrlInfo(BaseModel):
    url: str
    detail: str = "auto"


class OpenAIAssistantMessagePartImageFile(BaseModel):
    type: Literal["image_file"] = "image_file"
    image_file: OpenAIAssistantMessagePartImageFileInfo


class OpenAIAssistantMessagePartImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIAssistantMessagePartImageUrlInfo


class OpenAIAssistantMessagePartTextFileCitation(BaseModel):
    file_id: str


class OpenAIAssistantMessagePartTextFilePath(BaseModel):
    file_id: str


class OpenAIAssistantMessagePartTextFileCitationAnnotation(BaseModel):
    type: Literal["file_citation"] = "file_citation"
    text: str
    file_citation: OpenAIAssistantMessagePartTextFileCitation


class OpenAIAssistantMessagePartTextFilePathAnnotation(BaseModel):
    type: Literal["file_path"] = "file_path"
    text: str
    file_path: OpenAIAssistantMessagePartTextFilePath


OpenAIAssistantMessagePartTextAnnotation = (
    OpenAIAssistantMessagePartTextFileCitationAnnotation | OpenAIAssistantMessagePartTextFilePathAnnotation
)


class OpenAIAssistantMessagePartTextAnnotated(BaseModel):
    value: str
    annotations: list[Annotated[OpenAIAssistantMessagePartTextAnnotation, Field(discriminator="type")]]


class OpenAIAssistantMessagePartText(BaseModel):
    type: Literal["text"] = "text"
    text: str | OpenAIAssistantMessagePartTextAnnotated


OpenAIAssistantMessagePart = (
    OpenAIAssistantMessagePartImageFile | OpenAIAssistantMessagePartImageUrl | OpenAIAssistantMessagePartText
)


class OpenAIAssistantMessageAttachmentTool(BaseModel):
    type: OpenAIAssistantToolType


class OpenAIAssistantMessageAttachment(BaseModel):
    file_id: str
    tools: list[OpenAIAssistantMessageAttachmentTool] | None = None


class OpenAIAssistantMessageRequest(BaseModel):
    role: OpenAIAssistantMessageRole
    content: str | list[Annotated[OpenAIAssistantMessagePart, Field(discriminator="type")]]
    attachments: list[OpenAIAssistantMessageAttachment] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OpenAIAssistantMessageContentImageUploadRequest:
    image_url: str
    to_file_id: uuid.UUID


OpenAIAssistantMessageStatus = Literal["in_progress", "incomplete", "completed"]


@dataclass
class OpenAIAssistantMessageIncompleteDetails:
    reason: str


@dataclass
class OpenAIAssistantMessageObject:
    id: str
    created_at: int
    thread_id: str
    status: OpenAIAssistantMessageStatus | None
    incomplete_details: OpenAIAssistantMessageIncompleteDetails | None
    completed_at: int | None
    incomplete_at: int | None
    role: str
    content: list[OpenAIAssistantMessagePart]
    assistant_id: str | None
    run_id: str | None
    attachments: list[OpenAIAssistantMessageAttachment] | None
    metadata: dict[str, Any] | None

    object: str = "thread.message"


class OpenAIAssistantMessageUpdateRequest(BaseModel):
    metadata: dict[str, Any] | None = None


def to_chat_message_input_item(
    openai_assistant_message: OpenAIAssistantMessageRequest,
) -> tuple[ChatMessageInputItem, list[OpenAIAssistantMessageContentImageUploadRequest] | None]:
    openai_msg_contents = openai_assistant_message.content

    image_upload_requests = []
    contents: list[ChatMessageContentPart]

    if isinstance(openai_msg_contents, str):
        contents = [ChatMessageContentTextPart(text=openai_msg_contents, extra_content=None)]
    else:
        contents = []

        for openai_msg_content in openai_msg_contents:
            part: ChatMessageContentPart

            if isinstance(openai_msg_content, OpenAIAssistantMessagePartText):
                if not isinstance(openai_msg_content.text, str):
                    raise ValueError(f"Input message content {openai_msg_content.text} is not plain text!")

                part = ChatMessageContentTextPart(text=openai_msg_content.text, extra_content=None)
            elif isinstance(openai_msg_content, OpenAIAssistantMessagePartImageUrl):
                to_file_id = uuid.uuid4()

                image_upload_request = OpenAIAssistantMessageContentImageUploadRequest(
                    image_url=openai_msg_content.image_url.url, to_file_id=to_file_id
                )

                image_upload_requests.append(image_upload_request)

                part = ChatMessageContentImagePart(to_file_id)
            elif isinstance(openai_msg_content, OpenAIAssistantMessagePartImageFile):
                part = ChatMessageContentImagePart(uuid.UUID(openai_msg_content.image_file.file_id))
            else:
                raise ValueError(f"Unsupported openai message content {openai_msg_content}")

            contents.append(part)

    attachments = []

    if openai_assistant_message.attachments is not None:
        for openai_attachment in openai_assistant_message.attachments:
            exposed_to_tools = []

            if openai_attachment.tools is not None:
                for tool in openai_attachment.tools:
                    if tool.type == "code_interpreter":
                        exposed_to_tools.append(INTERNAL_TOOL_CODE_INTERPRETER)
                    elif tool.type == "file_search":
                        exposed_to_tools.append(INTERNAL_TOOL_RAG_SEARCH)

            attachment = ChatMessageAttachmentItem(
                file_id=uuid.UUID(openai_attachment.file_id), exposed_to_tools=exposed_to_tools
            )

            attachments.append(attachment)

    return ChatMessageInputItem(
        role=openai_assistant_message.role,
        contents=contents,
        attachments=attachments,
        additional_data=openai_assistant_message.metadata,
    ), image_upload_requests


def to_chat_message_input_items(
    openai_assistant_messages: list[OpenAIAssistantMessageRequest],
) -> tuple[list[ChatMessageInputItem], list[OpenAIAssistantMessageContentImageUploadRequest]]:
    message_items = []
    upload_reqs = []

    for m in openai_assistant_messages:
        item, req_list = to_chat_message_input_item(m)
        message_items.append(item)

        if req_list is not None:
            upload_reqs.extend(req_list)

    return message_items, upload_reqs


class OpenAIAssistantMessages:
    _file_store: FileStore
    _chat_thread_manager: ChatThreadManager
    _agent_chat_task_manager: AgentChatTaskManager

    def __init__(
        self,
        file_store: FileStore,
        chat_thread_manager: ChatThreadManager,
        agent_chat_task_manager: AgentChatTaskManager,
    ) -> None:
        self._file_store = file_store
        self._chat_thread_manager = chat_thread_manager
        self._agent_chat_task_manager = agent_chat_task_manager

    def __to_openai_message_contents(
        self, msg_contents: Sequence[ChatMessageContentPart]
    ) -> list[OpenAIAssistantMessagePart]:
        results: list[OpenAIAssistantMessagePart] = []

        for msg_content in msg_contents:
            result: OpenAIAssistantMessagePart

            if isinstance(msg_content, ChatMessageContentTextPart):
                openai_annotations: list[OpenAIAssistantMessagePartTextAnnotation] = []

                if msg_content.extra_content is not None:
                    if msg_content.extra_content.file_citations is not None:
                        openai_annotations.extend(
                            [
                                OpenAIAssistantMessagePartTextFileCitationAnnotation(
                                    text=file_citation.placeholder,
                                    file_citation=OpenAIAssistantMessagePartTextFileCitation(
                                        file_id=file_citation.source_file_id
                                    ),
                                )
                                for file_citation in msg_content.extra_content.file_citations
                            ]
                        )

                    if msg_content.extra_content.file_paths is not None:
                        openai_annotations.extend(
                            [
                                OpenAIAssistantMessagePartTextFilePathAnnotation(
                                    text=file_path.placeholder,
                                    file_path=OpenAIAssistantMessagePartTextFilePath(file_id=file_path.target_file_id),
                                )
                                for file_path in msg_content.extra_content.file_paths
                            ]
                        )

                result = OpenAIAssistantMessagePartText(
                    text=OpenAIAssistantMessagePartTextAnnotated(value=msg_content.text, annotations=openai_annotations)
                )
            elif isinstance(msg_content, ChatMessageContentImagePart):
                result = OpenAIAssistantMessagePartImageFile(
                    image_file=OpenAIAssistantMessagePartImageFileInfo(file_id=str(msg_content.image_file_id))
                )
            elif isinstance(msg_content, ChatMessageToolRequestsPart | ChatMessageToolOutputsPart):
                continue
            else:
                raise ValueError(f"Unsupported chat message content type {msg_content}")

            results.append(result)

        return results

    def __to_openai_message_attachments(
        self, msg_attachments: list[ChatMessageAttachmentItem]
    ) -> list[OpenAIAssistantMessageAttachment]:
        results = []

        for msg_attachment in msg_attachments:
            if msg_attachment.exposed_to_tools is not None:
                tools = []

                for exposed_tool in msg_attachment.exposed_to_tools:
                    if exposed_tool == INTERNAL_TOOL_CODE_INTERPRETER:
                        tool = OpenAIAssistantMessageAttachmentTool(type="code_interpreter")
                    elif exposed_tool == INTERNAL_TOOL_RAG_SEARCH:
                        tool = OpenAIAssistantMessageAttachmentTool(type="file_search")
                    else:
                        raise ValueError(f"Unsupported chat message exposed tool type {exposed_tool}")

                    tools.append(tool)
            else:
                tools = None

            result = OpenAIAssistantMessageAttachment(file_id=str(msg_attachment.file_id), tools=tools)

            results.append(result)

        return results

    def __to_openai_message_object(
        self, msg_info: ChatMessageItem, agent_chat_task_info: AgentChatTaskInfo | None
    ) -> OpenAIAssistantMessageObject:
        status: OpenAIAssistantMessageStatus | None = None
        incomplete_details: OpenAIAssistantMessageIncompleteDetails | None = None
        incomplete_at: datetime | None = None
        assistant_id: str | None = None
        run_id: str | None = None

        if agent_chat_task_info is not None:
            if agent_chat_task_info.status in ("queued", "in_progress", "requires_action"):
                status = "in_progress"
            elif agent_chat_task_info.status == "completed":
                status = "completed"
            else:
                status = "incomplete"

                incomplete_details = OpenAIAssistantMessageIncompleteDetails(
                    reason=agent_chat_task_info.error_message or ""
                )

                incomplete_at = agent_chat_task_info.complete_time or datetime.now()

            assistant_id = agent_chat_task_info.agent_id
            run_id = agent_chat_task_info.id

        return OpenAIAssistantMessageObject(
            id=msg_info.id,
            created_at=int(msg_info.create_time.timestamp()),
            thread_id=msg_info.thread_id,
            status=status,
            incomplete_details=incomplete_details,
            completed_at=int(msg_info.create_time.timestamp()),
            incomplete_at=int(incomplete_at.timestamp()) if incomplete_at is not None else None,
            role=msg_info.role,
            content=self.__to_openai_message_contents(msg_info.contents),
            assistant_id=assistant_id,
            run_id=run_id,
            attachments=self.__to_openai_message_attachments(msg_info.attachments),
            metadata=msg_info.additional_data,
        )

    def create_message(self, tid: str, request: OpenAIAssistantMessageRequest) -> OpenAIAssistantMessageObject:
        msg_input, upload_req_list = to_chat_message_input_item(request)

        msg_info = self._chat_thread_manager.add_message(tid, msg_input)

        if msg_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        if upload_req_list is not None:
            for upload_req in upload_req_list:
                self._file_store.download_file(upload_req.image_url, to_file_id=upload_req.to_file_id)

        return self.__to_openai_message_object(msg_info, None)

    def get_message_list(
        self,
        tid: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        page = self._chat_thread_manager.get_messages(
            ChatMessageListPagedQuery(
                thread_id=tid,
                agent_chat_task_id=run_id,
                create_time_desc_order=order == "desc",
                page_size=limit,
                before_id=before,
                after_id=after,
            )
        )

        message_id_list = [i.id for i in page.data]
        agent_chat_tasks = self._agent_chat_task_manager.get_current_tasks_by_messages(message_id_list)

        openai_list = [self.__to_openai_message_object(d, agent_chat_tasks.get(d.id)) for d in page.data]

        return {
            "object": "list",
            "data": openai_list,
            "first_id": openai_list[0].id if len(openai_list) > 0 else None,
            "last_id": openai_list[-1].id if len(openai_list) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get_message(self, tid: str, mid: str) -> OpenAIAssistantMessageObject:
        msg_info = self._chat_thread_manager.get_message(mid, tid)

        if msg_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        agent_chat_tasks = self._agent_chat_task_manager.get_current_tasks_by_messages([mid])

        return self.__to_openai_message_object(msg_info, agent_chat_tasks.get(mid))

    def update_message(
        self, tid: str, mid: str, request: OpenAIAssistantMessageUpdateRequest
    ) -> OpenAIAssistantMessageObject:
        updated = self._chat_thread_manager.update_message(mid, tid, request.metadata)

        if not updated:
            raise HTTPException(HTTP_404_NOT_FOUND)

        new_msg_info = self._chat_thread_manager.get_message(mid, tid)

        if new_msg_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        agent_chat_tasks = self._agent_chat_task_manager.get_current_tasks_by_messages([mid])

        return self.__to_openai_message_object(new_msg_info, agent_chat_tasks.get(mid))

    def delete_message(self, tid: str, mid: str) -> dict[str, Any]:
        deleted = self._chat_thread_manager.delete_message(mid, tid)

        return {"id": mid, "object": "thread.message.deleted", "deleted": deleted}
