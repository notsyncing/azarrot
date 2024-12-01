import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Engine, and_, select
from sqlalchemy.orm import Session

from azarrot.common_types import AgentChatTaskStatus
from azarrot.database_schemas import AgentChatMessage, AgentChatTask
from azarrot.utils import sanitize_uuid


@dataclass
class AgentChatTaskInfo:
    id: str
    agent_id: str
    thread_id: str
    status: AgentChatTaskStatus
    complete_time: datetime | None
    error_message: str | None
    create_time: datetime

    @staticmethod
    def from_db(dbo: AgentChatTask) -> "AgentChatTaskInfo":
        return AgentChatTaskInfo(
            id=str(dbo.id),
            agent_id=str(dbo.agent_id),
            thread_id=str(dbo.thread_id),
            status=dbo.status,
            complete_time=dbo.complete_time,
            error_message=dbo.error_message,
            create_time=dbo.create_time,
        )


class AgentChatTaskManager:
    _database: Engine

    def __init__(self, database: Engine) -> None:
        self._database = database

    def get_current_tasks_by_messages(self, message_id_list: Sequence[str | uuid.UUID]) -> dict[str, AgentChatTaskInfo]:
        message_id_list = [sanitize_uuid(s) for s in message_id_list]

        with Session(self._database) as db:
            db_task_msgs = (
                db.execute(select(AgentChatMessage).where(and_(AgentChatMessage.message_id.in_(message_id_list))))
                .scalars()
                .all()
            )

            task_id_list = list(dict.fromkeys([m.agent_chat_task_id for m in db_task_msgs]))

            db_tasks = db.execute(select(AgentChatTask).where(AgentChatTask.id.in_(task_id_list))).scalars().all()

            db_task_map = {t.id: t for t in db_tasks}

            result = {}

            for task_msg in db_task_msgs:
                task_info = db_task_map.get(task_msg.agent_chat_task_id)

                if task_info is None:
                    continue

                result[str(task_msg.message_id)] = AgentChatTaskInfo.from_db(task_info)

            return result
