import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Any

import dataclass_wizard
from sqlakeyset import select_page
from sqlalchemy import Engine, and_, select
from sqlalchemy.orm import Session

from azarrot.agents.chat_task_executor import AgentChatTaskExecutor
from azarrot.agents.common_data import (
    AgentChatTaskAutoThreadHistoryStrategyParams,
    AgentChatTaskDetailsData,
    AgentChatTaskInfo,
    AgentChatTaskMessageDetailsData,
    AgentChatTaskThreadHistoryStrategyParams,
    AgentChatTaskToolCallDetailsData,
    AgentToolRequest,
)
from azarrot.agents.manager import AgentGenerationParameters
from azarrot.agents.utils import (
    convert_agent_chat_task_tool_request_to_database,
)
from azarrot.common_data import (
    CallableToolsInfo,
    GenerationStatistics,
    PageResult,
)
from azarrot.common_types import (
    AgentChatTaskDetailStatus,
    AgentChatTaskDetailType,
    AgentChatTaskThreadHistoryStrategy,
)
from azarrot.database_schemas import (
    Agent,
    AgentChatMessage,
    AgentChatTask,
    AgentChatTaskDetail,
    AgentChatTaskTool,
    ChatMessage,
    ChatThread,
)
from azarrot.utils import sanitize_uuid


@dataclass
class AgentChatTaskCreationRequest:
    agent_id: str | uuid.UUID
    thread_id: str | uuid.UUID
    model_id: str | None = None
    model_instruction: str | None = None
    generation_parameters: AgentGenerationParameters | None = None
    thread_history_strategy: AgentChatTaskThreadHistoryStrategy | None = None
    thread_history_strategy_params: AgentChatTaskThreadHistoryStrategyParams | None = None
    max_tokens: int | None = None
    tools: list[AgentToolRequest] | None = None
    tools_info: CallableToolsInfo | None = None
    parallel_tool_calling: bool = True
    additional_data: dict[str, Any] | None = None


@dataclass
class AgentChatTaskListPagedQuery:
    thread_id: str | uuid.UUID
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None


@dataclass
class AgentChatTaskDetailsListPagedQuery:
    thread_id: str | uuid.UUID
    agent_chat_task_id: str | uuid.UUID
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None
    include_file_search_contents: bool = False


@dataclass
class AgentChatTaskDetailItem:
    id: str
    agent_chat_task_id: str
    type: AgentChatTaskDetailType
    data: AgentChatTaskDetailsData
    status: AgentChatTaskDetailStatus
    complete_time: datetime | None
    error_message: str | None
    generation_statistics: GenerationStatistics | None
    create_time: datetime
    update_time: datetime

    @staticmethod
    def from_db(dbo: AgentChatTaskDetail) -> "AgentChatTaskDetailItem":
        if dbo.type == "message":
            data = dataclass_wizard.fromdict(AgentChatTaskMessageDetailsData, json.loads(dbo.data))
        elif dbo.type == "tool_call":
            data = dataclass_wizard.fromdict(AgentChatTaskToolCallDetailsData, json.loads(dbo.data))
        else:
            raise ValueError(f"Unsupported agent chat task detail type {dbo.type}")

        if dbo.generation_statistics is not None:
            gen_stats = dataclass_wizard.fromdict(GenerationStatistics, json.loads(dbo.generation_statistics))
        else:
            gen_stats = None

        return AgentChatTaskDetailItem(
            id=str(dbo.id),
            agent_chat_task_id=str(dbo.agent_chat_task_id),
            type=dbo.type,
            data=data,
            status=dbo.status,
            complete_time=dbo.complete_time,
            error_message=dbo.error_message,
            generation_statistics=gen_stats,
            create_time=dbo.create_time,
            update_time=dbo.update_time,
        )


class AgentChatTaskManager:
    _database: Engine
    _executor: AgentChatTaskExecutor

    def __init__(self, database: Engine, executor: AgentChatTaskExecutor) -> None:
        self._database = database
        self._executor = executor

    def create_task(self, request: AgentChatTaskCreationRequest) -> AgentChatTaskInfo:
        agent_id = sanitize_uuid(request.agent_id)
        thread_id = sanitize_uuid(request.thread_id)
        task_id = uuid.uuid4()
        now = datetime.now()

        with Session(self._database) as db:
            db_agent = db.query(Agent).where(Agent.id == agent_id).first()

            if db_agent is None:
                raise ValueError(f"Agent {agent_id} does not exist!")

            db_thread = (
                db.query(ChatThread).where(and_(ChatThread.id == thread_id, ChatThread.deleted == False)).first()
            )

            if db_thread is None:
                raise ValueError(f"Thread {thread_id} does not exist!")

            model_id = None
            model_instruction = None

            if request.model_id is not None:
                model_id = request.model_id

            if request.model_instruction is not None:
                model_instruction = request.model_instruction

            gen_params = None

            if request.generation_parameters is not None:
                gen_params = json.dumps(request.generation_parameters)

            thread_history_strategy = request.thread_history_strategy

            if thread_history_strategy is None:
                thread_history_strategy = "auto"

            if request.thread_history_strategy_params is not None:
                thread_history_strategy_params = json.dumps(request.thread_history_strategy_params)
            else:
                thread_history_strategy_params = json.dumps(AgentChatTaskAutoThreadHistoryStrategyParams())

            db_task = AgentChatTask(
                id=task_id,
                agent_id=agent_id,
                thread_id=thread_id,
                model_id=model_id,
                model_instruction=model_instruction,
                status="pending",
                current_required_action=None,
                current_required_action_data=None,
                start_time=None,
                complete_time=None,
                error_message=None,
                generation_parameters=gen_params,
                thread_history_strategy=thread_history_strategy,
                thread_history_strategy_params=thread_history_strategy_params,
                max_tokens=request.max_tokens if request.max_tokens is not None else -1,
                tools_info=json.dumps(request.tools_info) if request.tools_info is not None else None,
                parallel_tool_calling=request.parallel_tool_calling,
                additional_data=json.dumps(request.additional_data) if request.additional_data is not None else None,
                create_time=now,
                update_time=now,
            )

            db.add(db_task)

            if request.tools is not None:
                agent_tools = convert_agent_chat_task_tool_request_to_database(agent_id, request.tools, now, now)
                db.add_all(agent_tools)
            else:
                agent_tools = None

            db.commit()

            info = AgentChatTaskInfo.from_db(db_task, agent_tools)
            self._executor.add_task(info)
            return info

    def get_current_tasks_by_messages(self, message_id_list: Sequence[str | uuid.UUID]) -> dict[str, AgentChatTaskInfo]:
        message_id_list = [sanitize_uuid(s) for s in message_id_list]

        with Session(self._database) as db:
            db_task_msgs = (
                db.execute(
                    select(AgentChatMessage)
                    .join(ChatMessage, onclause=ChatMessage.id == AgentChatMessage.message_id)
                    .join(ChatThread, onclause=ChatThread.id == ChatMessage.thread_id)
                    .where(
                        and_(AgentChatMessage.message_id.in_(message_id_list)),
                        ChatMessage.deleted == False,
                        ChatThread.deleted == False,
                    )
                )
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

    def get_list(self, query: AgentChatTaskListPagedQuery) -> PageResult[AgentChatTaskInfo]:
        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        thread_id = sanitize_uuid(query.thread_id)

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._database) as db:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db.execute(
                    select(AgentChatTask.create_time).where(AgentChatTask.id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = (
                select(AgentChatTask)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(and_(AgentChatTask.thread_id == thread_id, ChatThread.deleted == False))
            )

            if query.create_time_desc_order:
                q = q.order_by(AgentChatTask.create_time.desc(), AgentChatTask.id.desc())
            else:
                q = q.order_by(AgentChatTask.create_time, AgentChatTask.id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db, q, per_page=query.page_size, before=before, after=after)
            db_agent_chat_tasks: list[AgentChatTask] = [r._tuple()[0] for r in data]  # noqa: SLF001

            task_id_list = [t.id for t in db_agent_chat_tasks]

            task_tools = (
                db.execute(select(AgentChatTaskTool).where(AgentChatTaskTool.agent_chat_task_id.in_(task_id_list)))
                .scalars()
                .all()
            )

            task_tool_map = dict(groupby(task_tools, lambda i: i.agent_chat_task_id))

            return PageResult(
                data=[AgentChatTaskInfo.from_db(t, list(task_tool_map.get(t.id, []))) for t in db_agent_chat_tasks],
                is_last_page=not data.paging.has_next,
            )

    def get(
        self, agent_chat_task_id: str | uuid.UUID, thread_id: str | uuid.UUID | None = None
    ) -> AgentChatTaskInfo | None:
        if thread_id is not None:
            thread_id = sanitize_uuid(thread_id)

        agent_chat_task_id = sanitize_uuid(agent_chat_task_id)

        with Session(self._database) as db:
            query = (
                select(AgentChatTask)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(and_(AgentChatTask.id == agent_chat_task_id, ChatThread.deleted == False))
            )

            if thread_id is not None:
                query = query.where(AgentChatTask.thread_id == thread_id)

            dbo = db.execute(query).scalar()

            if dbo is None:
                return None

            task_tools = (
                db.execute(select(AgentChatTaskTool).where(AgentChatTaskTool.agent_chat_task_id == dbo.id))
                .scalars()
                .all()
            )

            return AgentChatTaskInfo.from_db(dbo, task_tools)

    def update(
        self,
        thread_id: str | uuid.UUID,
        agent_chat_task_id: str | uuid.UUID,
        additional_data: dict[str, Any] | None = None,
    ) -> bool:
        thread_id = sanitize_uuid(thread_id)
        agent_chat_task_id = sanitize_uuid(agent_chat_task_id)

        with Session(self._database) as db:
            dbo = db.execute(
                select(AgentChatTask)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(
                    and_(
                        AgentChatTask.id == agent_chat_task_id,
                        AgentChatTask.thread_id == thread_id,
                        ChatThread.deleted == False,
                    )
                )
            ).scalar()

            if dbo is None:
                return False

            if additional_data is not None:
                dbo.additional_data = json.dumps(additional_data)

            db.commit()

            return True

    def cancel(self, thread_id: str | uuid.UUID, agent_chat_task_id: str | uuid.UUID) -> bool:
        thread_id = sanitize_uuid(thread_id)
        agent_chat_task_id = sanitize_uuid(agent_chat_task_id)

        with Session(self._database) as db:
            dbo = db.execute(
                select(AgentChatTask)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(
                    and_(
                        AgentChatTask.id == agent_chat_task_id,
                        AgentChatTask.thread_id == thread_id,
                        ChatThread.deleted == False,
                    )
                )
            ).scalar()

            if dbo is None:
                return False

            if dbo.status not in ("pending", "requires_action", "truncated"):
                raise ValueError(
                    f"Agent chat task {agent_chat_task_id} has status {dbo.status}, which cannot be cancelled!"
                )

            now = datetime.now()

            dbo.status = "cancelled"
            dbo.complete_time = now
            dbo.update_time = now

            db.commit()

            return True

    def get_details(self, query: AgentChatTaskDetailsListPagedQuery) -> PageResult[AgentChatTaskDetailItem]:
        thread_id = sanitize_uuid(query.thread_id)
        task_id = sanitize_uuid(query.agent_chat_task_id)

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._database) as db:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db.execute(
                    select(AgentChatTaskDetail.create_time).where(AgentChatTaskDetail.id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = (
                select(AgentChatTaskDetail)
                .join(AgentChatTask, onclause=AgentChatTask.id == AgentChatTaskDetail.agent_chat_task_id)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(
                    and_(AgentChatTask.thread_id == thread_id, AgentChatTask.id == task_id, ChatThread.deleted == False)
                )
            )

            if query.create_time_desc_order:
                q = q.order_by(AgentChatTaskDetail.create_time.desc(), AgentChatTaskDetail.id.desc())
            else:
                q = q.order_by(AgentChatTaskDetail.create_time, AgentChatTaskDetail.id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db, q, per_page=query.page_size, before=before, after=after)

            return PageResult(
                data=[AgentChatTaskDetailItem.from_db(r._tuple()[0]) for r in data],  # noqa: SLF001
                is_last_page=not data.paging.has_next,
            )

    def get_detail(
        self, thread_id: str | uuid.UUID, agent_chat_task_id: str | uuid.UUID, detail_id: str | uuid.UUID
    ) -> AgentChatTaskDetailItem | None:
        thread_id = sanitize_uuid(thread_id)
        task_id = sanitize_uuid(agent_chat_task_id)
        detail_id = sanitize_uuid(detail_id)

        with Session(self._database) as db:
            detail = db.execute(
                select(AgentChatTaskDetail)
                .join(AgentChatTask, onclause=AgentChatTask.id == AgentChatTaskDetail.agent_chat_task_id)
                .join(ChatThread, onclause=ChatThread.id == AgentChatTask.thread_id)
                .where(
                    and_(
                        AgentChatTask.thread_id == thread_id,
                        AgentChatTask.id == task_id,
                        AgentChatTaskDetail.id == detail_id,
                        ChatThread.deleted == False,
                    )
                )
            ).scalar_one_or_none()

            if detail is None:
                return None

            return AgentChatTaskDetailItem.from_db(detail)
