import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from sqlakeyset import select_page
from sqlalchemy import Engine, and_, delete, select, update
from sqlalchemy.orm import Session

from azarrot.chats.common_data import (
    ChatMessageContentImagePart,
    ChatMessageContentPart,
    ChatMessageContentTextPart,
    ChatMessageInputItem,
    ChatMessageItem,
    ChatMessageToolOutputsPart,
    ChatMessageToolRequestsPart,
)
from azarrot.common_data import PageResult
from azarrot.common_types import MessageContentType
from azarrot.database_schemas import (
    AgentChatMessage,
    ChatMessage,
    ChatMessageAttachment,
    ChatMessageAttachmentToolExposure,
    ChatMessageContent,
    ChatThread,
    ChatThreadToolResources,
)
from azarrot.utils import sanitize_uuid


@dataclass
class ChatThreadAgentToolResource:
    tool_name: str
    tool_resources: dict[str, Any] | None

    @staticmethod
    def from_db(dbo: ChatThreadToolResources) -> "ChatThreadAgentToolResource":
        return ChatThreadAgentToolResource(tool_name=dbo.tool_name, tool_resources=json.loads(dbo.tool_resources))


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


@dataclass
class ChatMessageListPagedQuery:
    thread_id: str | uuid.UUID
    agent_chat_task_id: str | None = None
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None


class ChatThreadMessageListener(Protocol):
    def on_message_added(self, message: ChatMessageItem) -> None:
        pass


class ChatThreadManager:
    _database: Engine
    _message_listeners: list[ChatThreadMessageListener]

    def __init__(self, database: Engine) -> None:
        self._database = database
        self._message_listeners = []

    def register_message_listener(self, listener: ChatThreadMessageListener) -> None:
        self._message_listeners.append(listener)

    def __insert_tool_resources(
        self,
        db: Session,
        thread_id: uuid.UUID,
        now: datetime,
        tool_resources: list[ChatThreadAgentToolResource] | None,
    ) -> None:
        if tool_resources is None:
            return

        for params in tool_resources:
            tool = ChatThreadToolResources(
                thread_id=thread_id,
                tool_name=params.tool_name,
                tool_resources=json.dumps(params.tool_resources),
                create_time=now,
                update_time=now,
            )

            db.add(tool)

    def create(
        self,
        additional_data: dict[str, Any] | None = None,
        tool_resources: list[ChatThreadAgentToolResource] | None = None,
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

            self.__insert_tool_resources(db, thread_id, now, tool_resources)

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
                        ChatThread.deleted == False,
                    )
                )
            ).scalar_one_or_none()

            if thread is None:
                return None

            return ChatThreadInfo.from_db(thread)

    def get_thread_tool_resources(self, thread_id: str | uuid.UUID) -> list[ChatThreadAgentToolResource]:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            if self.__is_thread_not_exist(db, thread_id):
                return []

            params = (
                db.execute(select(ChatThreadToolResources).where(ChatThreadToolResources.thread_id == thread_id))
                .scalars()
                .all()
            )

            return [ChatThreadAgentToolResource.from_db(p) for p in params]

    def update(
        self,
        thread_id: str | uuid.UUID,
        new_metadata: dict[str, Any] | None = None,
        new_tool_resources: list[ChatThreadAgentToolResource] | None = None,
    ) -> ChatThreadInfo | None:
        thread_id = sanitize_uuid(thread_id)
        now = datetime.now()

        with Session(self._database) as db:
            thread = db.execute(
                select(ChatThread).where(
                    and_(
                        ChatThread.id == thread_id,
                        ChatThread.deleted == False,
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

            if new_tool_resources is not None:
                db.execute(delete(ChatThreadToolResources).where(ChatThreadToolResources.thread_id == thread_id))

                self.__insert_tool_resources(db, thread_id, now, new_tool_resources)

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
                        ChatThread.deleted == False,
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
        elif isinstance(msg_content, ChatMessageToolRequestsPart):
            return "tool_requests"
        elif isinstance(msg_content, ChatMessageToolOutputsPart):
            return "tool_outputs"
        else:
            raise ValueError(f"Unsupported message content part type {type(msg_content)}")

    def add_messages(
        self,
        thread_id: str | uuid.UUID,
        messages: list[ChatMessageInputItem],
        sender: ChatThreadMessageListener | None = None,
    ) -> list[ChatMessageItem]:
        thread_id = sanitize_uuid(thread_id)
        now = datetime.now()

        result = []

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
                    deleted=False,
                    additional_data=json.dumps(message.additional_data)
                    if message.additional_data is not None
                    else None,
                    create_time=now,
                    update_time=now,
                )

                db.add(db_msg)

                db_msg_contents = []

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
                    db_msg_contents.append(db_msg_content)

                db_msg_attachments = []
                db_msg_attachment_tools = []

                if message.attachments is not None:
                    for attachment in message.attachments:
                        attachment_id = uuid.uuid4()

                        db_attachment = ChatMessageAttachment(
                            id=attachment_id, message_id=message_id, file_id=attachment.file_id, create_time=now
                        )

                        db.add(db_attachment)
                        db_msg_attachments.append(db_attachment)

                        if attachment.exposed_to_tools is not None:
                            for tool_name in attachment.exposed_to_tools:
                                db_aet = ChatMessageAttachmentToolExposure(
                                    attachment_id=attachment_id, tool_name=tool_name, create_time=now
                                )

                                db.add(db_aet)
                                db_msg_attachment_tools.append(db_aet)

                result.append(
                    ChatMessageItem.from_db(db_msg, db_msg_contents, db_msg_attachments, db_msg_attachment_tools)
                )

            db.commit()

            for r in result:
                for listener in self._message_listeners:
                    if listener == sender:
                        continue

                    listener.on_message_added(r)

            return result

    def add_message(
        self, thread_id: str | uuid.UUID, message: ChatMessageInputItem, sender: ChatThreadMessageListener | None = None
    ) -> ChatMessageItem | None:
        msg_list = self.add_messages(thread_id, [message], sender)

        if len(msg_list) <= 0:
            return None

        return msg_list[0]

    def get_messages(self, query: ChatMessageListPagedQuery) -> PageResult[ChatMessageItem]:
        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        thread_id = sanitize_uuid(query.thread_id)

        with Session(self._database) as db:
            if self.__is_thread_not_exist(db, thread_id):
                return PageResult([], is_last_page=True)

            page_border_keyset = None

            if page_border_id is not None:
                page_border_conditions = db.execute(
                    select(ChatMessage.create_time, ChatMessage.order).where(
                        and_(ChatMessage.id == page_border_id, ChatMessage.deleted == False)
                    )
                ).first()

                if page_border_conditions is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                pb_create_time, pb_order = page_border_conditions._t  # noqa: SLF001

                page_border_keyset = (pb_create_time, pb_order, page_border_id)

            q = select(ChatMessage).where(and_(ChatMessage.thread_id == thread_id, ChatMessage.deleted == False))

            if query.agent_chat_task_id is not None:
                q = q.join(AgentChatMessage, AgentChatMessage.message_id == ChatMessage.id).where(
                    AgentChatMessage.agent_chat_task_id == query.agent_chat_task_id
                )

            if query.create_time_desc_order:
                q = q.order_by(ChatMessage.create_time.desc(), ChatMessage.order.desc(), ChatMessage.id.desc())
            else:
                q = q.order_by(ChatMessage.create_time, ChatMessage.order, ChatMessage.id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db, q, per_page=query.page_size, before=before, after=after)
            db_msgs: list[ChatMessage] = [r._tuple()[0] for r in data]  # noqa: SLF001

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

            return PageResult(
                data=result,
                is_last_page=not data.paging.has_next,
            )

    def get_message(
        self, message_id: str | uuid.UUID, thread_id: str | uuid.UUID | None = None
    ) -> ChatMessageItem | None:
        message_id = sanitize_uuid(message_id)

        if thread_id is not None:
            thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            query = select(ChatMessage).where(and_(ChatMessage.id == message_id, ChatMessage.deleted == False))

            if thread_id is not None:
                if self.__is_thread_not_exist(db, thread_id):
                    return None

                query = query.where(ChatMessage.thread_id == thread_id)

            db_msg = db.execute(query).scalar_one_or_none()

            if db_msg is None:
                return None

            db_msg_contents = (
                db.execute(select(ChatMessageContent).where(ChatMessageContent.message_id == db_msg.id)).scalars().all()
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

            return ChatMessageItem.from_db(db_msg, db_msg_contents, db_msg_attachments, db_msg_attachment_tools)

    def get_latest_messages(self, thread_id: str | uuid.UUID, count: int = 1) -> list[ChatMessageItem]:
        thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            latest_msg_id_list = (
                db.execute(
                    select(ChatMessage.id)
                    .where(and_(ChatMessage.thread_id == thread_id, ChatMessage.deleted == False))
                    .order_by(ChatMessage.create_time.desc(), ChatMessage.order.desc())
                    .limit(count)
                )
                .scalars()
                .all()
            )

        if latest_msg_id_list is None:
            return []

        results: list[ChatMessageItem] = []

        for msg_id in reversed(latest_msg_id_list):
            msg = self.get_message(msg_id, thread_id)

            if msg is None:
                continue

            results.append(msg)

        return results

    def update_message(
        self,
        message_id: str | uuid.UUID,
        thread_id: str | uuid.UUID | None = None,
        new_metadata: dict[str, Any] | None = None,
    ) -> bool:
        message_id = sanitize_uuid(message_id)

        if thread_id is not None:
            thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            query = select(ChatMessage).where(and_(ChatMessage.id == message_id, ChatMessage.deleted == False))

            if thread_id is not None:
                if self.__is_thread_not_exist(db, thread_id):
                    return False

                query = query.where(ChatMessage.thread_id == thread_id)

            db_msg = db.execute(query).scalar_one_or_none()

            if db_msg is None:
                return False

            if new_metadata is not None:
                db_msg.additional_data = json.dumps(new_metadata)

            db.commit()
            return True

    def delete_message(
        self,
        message_id: str | uuid.UUID,
        thread_id: str | uuid.UUID | None = None,
    ) -> bool:
        message_id = sanitize_uuid(message_id)

        if thread_id is not None:
            thread_id = sanitize_uuid(thread_id)

        with Session(self._database) as db:
            db_msg = db.execute(
                select(ChatMessage).where(and_(ChatMessage.id == message_id, ChatMessage.deleted == False))
            ).scalar_one_or_none()

            if db_msg is None:
                return False

            db_msg.deleted = True

            db.commit()
            return True

    def clear_database(self) -> None:
        with Session(self._database) as db:
            db.execute(delete(ChatThread))
            db.execute(delete(ChatThreadToolResources))
            db.execute(delete(ChatMessage))
            db.execute(delete(ChatMessageContent))
            db.execute(delete(ChatMessageAttachment))
            db.execute(delete(ChatMessageAttachmentToolExposure))
            db.commit()
