import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from azarrot.chats.common_data import (
    ChatMessageAttachmentItem,
    ChatMessageContentImagePart,
    ChatMessageContentPart,
    ChatMessageContentTextPart,
    ChatMessageInputItem,
)
from azarrot.frontends.openai_support.openai_assistants import OpenAIAssistantToolType
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_FILE_SEARCH

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


class OpenAIAssistantMessagePartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


OpenAIAssistantMessageParts = (
    OpenAIAssistantMessagePartImageFile | OpenAIAssistantMessagePartImageUrl | OpenAIAssistantMessagePartText
)


class OpenAIAssistantMessageAttachmentTool(BaseModel):
    type: OpenAIAssistantToolType


class OpenAIAssistantMessageAttachment(BaseModel):
    file_id: str
    tools: list[OpenAIAssistantMessageAttachmentTool] | None = None


class OpenAIAssistantMessage(BaseModel):
    role: OpenAIAssistantMessageRole
    content: str | list[Annotated[OpenAIAssistantMessageParts, Field(discriminator="type")]]
    attachments: list[OpenAIAssistantMessageAttachment] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OpenAIAssistantMessageContentImageUploadRequest:
    image_url: str
    to_file_id: uuid.UUID


def to_chat_message_input_item(
    openai_assistant_message: OpenAIAssistantMessage,
) -> tuple[ChatMessageInputItem, OpenAIAssistantMessageContentImageUploadRequest | None]:
    openai_msg_contents = openai_assistant_message.content

    image_upload_request = None
    contents: list[ChatMessageContentPart]

    if isinstance(openai_msg_contents, str):
        contents = [ChatMessageContentTextPart(openai_msg_contents)]
    else:
        contents = []

        for openai_msg_content in openai_msg_contents:
            part: ChatMessageContentPart

            if isinstance(openai_msg_content, OpenAIAssistantMessagePartText):
                part = ChatMessageContentTextPart(openai_msg_content.text)
            elif isinstance(openai_msg_content, OpenAIAssistantMessagePartImageUrl):
                to_file_id = uuid.uuid4()

                image_upload_request = OpenAIAssistantMessageContentImageUploadRequest(
                    image_url=openai_msg_content.image_url.url, to_file_id=to_file_id
                )

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
                        exposed_to_tools.append(INTERNAL_TOOL_FILE_SEARCH)

            attachment = ChatMessageAttachmentItem(
                file_id=uuid.UUID(openai_attachment.file_id), exposed_to_tools=exposed_to_tools
            )

            attachments.append(attachment)

    return ChatMessageInputItem(
        role=openai_assistant_message.role,
        contents=contents,
        attachments=attachments,
        additional_data=openai_assistant_message.metadata,
    ), image_upload_request


def to_chat_message_input_items(
    openai_assistant_messages: list[OpenAIAssistantMessage],
) -> tuple[list[ChatMessageInputItem], list[OpenAIAssistantMessageContentImageUploadRequest]]:
    message_items = []
    upload_reqs = []

    for m in openai_assistant_messages:
        item, req = to_chat_message_input_item(m)
        message_items.append(item)

        if req is not None:
            upload_reqs.append(req)

    return message_items, upload_reqs
