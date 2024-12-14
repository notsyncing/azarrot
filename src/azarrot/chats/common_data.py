import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import dataclass_wizard
from typing_extensions import override

from azarrot.database_schemas import (
    ChatMessage,
    ChatMessageAttachment,
    ChatMessageAttachmentToolExposure,
    ChatMessageContent,
)


class ChatMessageContentPart(ABC):
    @abstractmethod
    def to_persist_content(self) -> str:
        pass


@dataclass
class ChatMessageTextFileCitation:
    placeholder: str
    source_file_id: str


@dataclass
class ChatMessageTextFilePath:
    placeholder: str
    target_file_id: str


@dataclass
class ChatMessageTextExtraContent:
    file_citations: list[ChatMessageTextFileCitation] | None = None
    file_paths: list[ChatMessageTextFilePath] | None = None


@dataclass
class ChatMessageContentTextPart(ChatMessageContentPart):
    text: str
    extra_content: ChatMessageTextExtraContent | None = None

    @override
    def to_persist_content(self) -> str:
        return self.text


@dataclass
class ChatMessageContentImagePart(ChatMessageContentPart):
    image_file_id: uuid.UUID

    @override
    def to_persist_content(self) -> str:
        return str(self.image_file_id)


@dataclass
class ChatMessageToolOutputItem:
    tool_call_id: str
    output: str


@dataclass
class ChatMessageToolOutputsPart(ChatMessageContentPart):
    tool_outputs: list[ChatMessageToolOutputItem]

    @override
    def to_persist_content(self) -> str:
        return json.dumps(dataclass_wizard.asdict(self.tool_outputs))


@dataclass
class ChatMessageAttachmentItem:
    file_id: uuid.UUID
    exposed_to_tools: list[str] | None = None

    @staticmethod
    def from_db(
        dbo: ChatMessageAttachment, db_tool_exposures: Sequence[ChatMessageAttachmentToolExposure]
    ) -> "ChatMessageAttachmentItem":
        tools = [
            db_tool_exposure.tool_name
            for db_tool_exposure in db_tool_exposures
            if db_tool_exposure.attachment_id == dbo.id
        ]

        return ChatMessageAttachmentItem(file_id=dbo.file_id, exposed_to_tools=tools)


@dataclass
class ChatMessageInputItem:
    role: str
    contents: Sequence[ChatMessageContentPart]
    attachments: list[ChatMessageAttachmentItem] | None = None
    additional_data: dict[str, Any] | None = None


@dataclass
class ChatMessageItem:
    id: str
    thread_id: str
    role: str
    contents: Sequence[ChatMessageContentPart]
    attachments: list[ChatMessageAttachmentItem]
    create_time: datetime
    additional_data: dict[str, Any] | None = None

    @staticmethod
    def from_db(
        dbo: ChatMessage,
        db_contents: Sequence[ChatMessageContent],
        db_attachments: Sequence[ChatMessageAttachment],
        db_attachment_tool_exposures: Sequence[ChatMessageAttachmentToolExposure],
    ) -> "ChatMessageItem":
        contents = []

        for db_content in db_contents:
            content: ChatMessageContentPart

            if db_content.type == "text":
                content = ChatMessageContentTextPart(
                    text=db_content.content if db_content.content is not None else "",
                    extra_content=dataclass_wizard.fromdict(
                        ChatMessageTextExtraContent, json.loads(db_content.extra_content)
                    )
                    if db_content.extra_content is not None
                    else None,
                )
            elif db_content.type == "image_file":
                content = ChatMessageContentImagePart(uuid.UUID(db_content.content))
            else:
                raise ValueError(f"Unsupported chat message content type {db_content.type} on message id {dbo.id}")

            contents.append(content)

        return ChatMessageItem(
            id=str(dbo.id),
            thread_id=str(dbo.thread_id),
            role=dbo.role,
            contents=contents,
            attachments=[ChatMessageAttachmentItem.from_db(a, db_attachment_tool_exposures) for a in db_attachments],
            create_time=dbo.create_time,
            additional_data=json.loads(dbo.additional_data) if dbo.additional_data is not None else None,
        )
