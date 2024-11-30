import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, and_, delete, select, update
from sqlalchemy.orm import Session

from azarrot.chats.common_data import (
    ChatMessageContentImagePart,
    ChatMessageContentPart,
    ChatMessageContentTextPart,
    ChatMessageInputItem,
    ChatMessageItem,
)
from azarrot.common_types import MessageContentType
from azarrot.database_schemas import (
    ChatMessage,
    ChatMessageAttachment,
    ChatMessageAttachmentToolExposure,
    ChatMessageContent,
    ChatThread,
    ChatThreadToolPresetParams,
)
from azarrot.utils import sanitize_uuid


@dataclass
class ChatThreadAgentToolPresetParams:
    tool_name: str
    tool_additional_preset_params: dict[str, Any]

    @staticmethod
    def from_db(dbo: ChatThreadToolPresetParams) -> "ChatThreadAgentToolPresetParams":
        return ChatThreadAgentToolPresetParams(
            tool_name=dbo.tool_name, tool_additional_preset_params=json.loads(dbo.tool_preset_parameters)
        )


@dataclass
class ChatThreadInfo:
    id: str
    additional_data: dict[str, Any] | None
    create_time: datetime

    @staticmethod
    def from_db(dbo: ChatThread) -> "ChatThreadInfo":
        return ChatThreadInfo(
            id=str(dbo.id),
            additional_data=json.loads(dbo.additional_data) if dbo.additional_data is not None else None,
            create_time=dbo.create_time,
        )


class ChatThreadManager:
    _database: Engine

    def __init__(self, database: Engine) -> None:
        self._database = database

    def __insert_tool_preset_params(
        self,
        db: Session,
        thread_id: uuid.UUID,
        now: datetime,
        additional_tool_preset_parameters: list[ChatThreadAgentToolPresetParams] | None,
    ) -> None:
        if additional_tool_preset_parameters is None:
            return

        for params in additional_tool_preset_parameters:
            tool = ChatThreadToolPresetParams(
                thread_id=thread_id,
                tool_name=params.tool_name,
                tool_preset_parameters=json.dumps(params.tool_additional_preset_params),
                create_time=now,
                update_time=now,
            )

            db.add(tool)

    def create(
        self,
        additional_data: dict[str, Any] | None = None,
        additional_tool_preset_parameters: list[ChatThreadAgentToolPresetParams] | None = None,
    ) -> ChatThreadInfo:
        thread_id = uuid.uuid4()
        now = datetime.now()

        with Session(self._database) as db:
            thread = ChatThread(
                id=thread_id,
                additional_data=json.dumps(additional_data) if additional_data is not None else None,
                deleted=False,
                create_time=now,
                update_time=now,
            )

            db.add(thread)

            self.__insert_tool_preset_params(db, thread_id, now, additional_tool_preset_parameters)

            db.commit()

            return ChatThreadInfo.from_db(thread)

    def __is_thread_not_exist(self, db: Session, thread_id: uuid.UUID) -> bool:
        thread_deleted = db.execute(select(ChatThread.deleted).where(ChatThread.id == thread_id)).scalar_one_or_none()

        return thread_deleted is True or thread_deleted is None

    def get(self, thread_id: str | uuid.UUID) -> ChatThreadInfo | None:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            thread = db.execute(
                select(ChatThread).where(
                    and_(
                        ChatThread.id == thread_id,
                        ChatThread.deleted == False,  # noqa: E712
                    )
                )
            ).scalar_one_or_none()

            if thread is None:
                return None

            return ChatThreadInfo.from_db(thread)

    def get_thread_tool_preset_params(self, thread_id: str | uuid.UUID) -> list[ChatThreadAgentToolPresetParams]:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            if self.__is_thread_not_exist(db, thread_id):
                return []

            params = (
                db.execute(select(ChatThreadToolPresetParams).where(ChatThreadToolPresetParams.thread_id == thread_id))
                .scalars()
                .all()
            )

            return [ChatThreadAgentToolPresetParams.from_db(p) for p in params]

    def update(
        self,
        thread_id: str | uuid.UUID,
        new_metadata: dict[str, Any] | None = None,
        new_tool_preset_params: list[ChatThreadAgentToolPresetParams] | None = None,
    ) -> ChatThreadInfo | None:
        thread_id = sanitize_uuid(thread_id)
        now = datetime.now()

        with Session(self._database) as db:
            thread = db.execute(
                select(ChatThread).where(
                    and_(
                        ChatThread.id == thread_id,
                        ChatThread.deleted == False,  # noqa: E712
                    )
                )
            ).scalar_one_or_none()

            if thread is None:
                return None

            if thread.deleted:
                return None

            updated = False

            if new_metadata is not None:
                thread.additional_data = json.dumps(new_metadata)
                updated = True

            if new_tool_preset_params is not None:
                db.execute(delete(ChatThreadToolPresetParams).where(ChatThreadToolPresetParams.thread_id == thread_id))

                self.__insert_tool_preset_params(db, thread_id, now, new_tool_preset_params)

            if updated:
                thread.update_time = now

            db.commit()

            return ChatThreadInfo.from_db(thread)

    def delete(self, thread_id: str | uuid.UUID) -> bool:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            r = db.execute(
                update(ChatThread)
                .values(deleted=True)
                .where(
                    and_(
                        ChatThread.id == thread_id,
                        ChatThread.deleted == False,  # noqa: E712
                    )
                )
            )

            if r.rowcount <= 0:
                return False

            db.commit()
            return True

    def __to_msg_content_db_type(self, msg_content: ChatMessageContentPart) -> MessageContentType:
        if isinstance(msg_content, ChatMessageContentTextPart):
            return "text"
        elif isinstance(msg_content, ChatMessageContentImagePart):
            return "image_file"
        else:
            raise ValueError(f"Unsupported message content part type {type(msg_content)}")

    def add_messages(self, thread_id: str | uuid.UUID, messages: list[ChatMessageInputItem]) -> None:
        thread_id = sanitize_uuid(thread_id)
        now = datetime.now()

        with Session(self._database) as db:
            if self.__is_thread_not_exist(db, thread_id):
                raise ValueError(f"Thread {thread_id} does not exist!")

            for i in range(len(messages)):
                message = messages[i]
                message_id = uuid.uuid4()

                db_msg = ChatMessage(
                    id=message_id,
                    thread_id=thread_id,
                    role=message.role,
                    order=i,
                    additional_data=json.dumps(message.additional_data)
                    if message.additional_data is not None
                    else None,
                    create_time=now,
                    update_time=now,
                )

                db.add(db_msg)

                for j in range(len(message.contents)):
                    content = message.contents[j]

                    db_msg_content = ChatMessageContent(
                        id=uuid.uuid4(),
                        message_id=message_id,
                        type=self.__to_msg_content_db_type(content),
                        content=content.to_persist_content(),
                        extra_content=None,
                        order=j,
                        create_time=now,
                        update_time=now,
                    )

                    db.add(db_msg_content)

                for attachment in message.attachments:
                    attachment_id = uuid.uuid4()

                    db_attachment = ChatMessageAttachment(
                        id=attachment_id, message_id=message_id, file_id=attachment.file_id, create_time=now
                    )

                    db.add(db_attachment)

                    if attachment.exposed_to_tools is not None:
                        for tool_name in attachment.exposed_to_tools:
                            db_aet = ChatMessageAttachmentToolExposure(
                                attachment_id=attachment_id, tool_name=tool_name, create_time=now
                            )

                            db.add(db_aet)

            db.commit()

    def add_message(self, thread_id: str | uuid.UUID, message: ChatMessageInputItem) -> None:
        self.add_messages(thread_id, [message])

    def get_messages(self, thread_id: str | uuid.UUID) -> list[ChatMessageItem]:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            if self.__is_thread_not_exist(db, thread_id):
                return []

            db_msgs = (
                db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.thread_id == thread_id)
                    .order_by(ChatMessage.create_time, ChatMessage.order)
                )
                .scalars()
                .all()
            )

            result = []

            for db_msg in db_msgs:
                db_msg_contents = (
                    db.execute(select(ChatMessageContent).where(ChatMessageContent.message_id == db_msg.id))
                    .scalars()
                    .all()
                )

                db_msg_attachments = (
                    db.execute(select(ChatMessageAttachment).where(ChatMessageAttachment.message_id == db_msg.id))
                    .scalars()
                    .all()
                )

                db_msg_attachment_tools = (
                    db.execute(
                        select(ChatMessageAttachmentToolExposure).where(
                            ChatMessageAttachmentToolExposure.attachment_id.in_([a.id for a in db_msg_attachments])
                        )
                    )
                    .scalars()
                    .all()
                )

                result.append(
                    ChatMessageItem.from_db(db_msg, db_msg_contents, db_msg_attachments, db_msg_attachment_tools)
                )

            return result

    def clear_database(self) -> None:
        with Session(self._database) as db:
            db.execute(delete(ChatThread))
            db.execute(delete(ChatThreadToolPresetParams))
            db.execute(delete(ChatMessage))
            db.execute(delete(ChatMessageContent))
            db.execute(delete(ChatMessageAttachment))
            db.execute(delete(ChatMessageAttachmentToolExposure))
