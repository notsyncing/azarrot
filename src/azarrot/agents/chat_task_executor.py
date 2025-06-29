import json
import logging
from datetime import datetime
from logging import Logger
from queue import Empty, Queue
from threading import Thread
from typing import Any, override
from uuid import UUID, uuid4

import dataclass_wizard
from sqlalchemy import Engine, and_, update
from sqlalchemy.orm import Session

from azarrot.agents.common_data import (
    AgentChatTaskAutoThreadHistoryStrategyParams,
    AgentChatTaskDetailToolCallItem,
    AgentChatTaskInfo,
    AgentChatTaskLastMessageThreadHistoryStrategyParams,
    AgentChatTaskMessageDetailsData,
    AgentChatTaskToolCallDetailsData,
    AgentToolRequest,
)
from azarrot.agents.manager import AgentManager
from azarrot.backends.common import CompletionChunkStreamer
from azarrot.chats.common_data import (
    ChatMessageContentImagePart,
    ChatMessageContentTextPart,
    ChatMessageInputItem,
    ChatMessageItem,
    ChatMessageToolOutputsPart,
    ChatMessageToolRequestItem,
    ChatMessageToolRequestsPart,
)
from azarrot.chats.thread_manager import ChatThreadManager, ChatThreadMessageListener
from azarrot.common_data import (
    CallableToolsInfo,
    DifferentChunkError,
    GeneratedMessageChunk,
    GenerationMessage,
    GenerationMessageContent,
    GenerationStatistics,
    ImageGenerationMessageContent,
    TextGeneratedMessageChunk,
    TextGenerationMessageContent,
    TextGenerationRequest,
    ToolCallGeneratedMessageChunk,
    ToolCallRequestMessageContent,
    ToolCallResponseMessageContent,
)
from azarrot.common_types import (
    AgentChatTaskDetailStatus,
    AgentChatTaskDetailType,
    AgentChatTaskRequiredAction,
    AgentChatTaskStatus,
)
from azarrot.config import ServerConfig
from azarrot.database_schemas import AgentChatTask, AgentChatTaskDetail, AgentChatTaskTool
from azarrot.file_store import FileStore
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.models.chat_templates import DEFAULT_LOCALE, ChatTemplateManager, ChatTemplateRuntimeConfigs
from azarrot.models.model_manager import ModelManager
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterTool
from azarrot.tools.internal.tool_rag_search import RagSearchTool
from azarrot.tools.tool import LocalizedToolDescription
from azarrot.utils import sanitize_uuid


class AgentChatTaskExecutor(ChatThreadMessageListener):
    _log: Logger = logging.getLogger(__name__)

    _database: Engine
    _server_config: ServerConfig
    _chat_template_manager: ChatTemplateManager
    _agent_manager: AgentManager
    _model_manager: ModelManager
    _file_store: FileStore
    _chat_thread_manager: ChatThreadManager
    _backend_pipe: BackendPipe
    _task_queue: Queue[AgentChatTaskInfo]
    _worker_thread: Thread | None = None
    _stop: bool = True

    def __init__(
        self,
        database: Engine,
        server_config: ServerConfig,
        chat_template_manager: ChatTemplateManager,
        agent_manager: AgentManager,
        model_manager: ModelManager,
        file_store: FileStore,
        chat_thread_manager: ChatThreadManager,
        backend_pipe: BackendPipe,
    ) -> None:
        self._database = database
        self._server_config = server_config
        self._chat_template_manager = chat_template_manager
        self._agent_manager = agent_manager
        self._model_manager = model_manager
        self._file_store = file_store
        self._chat_thread_manager = chat_thread_manager
        self._backend_pipe = backend_pipe
        self._task_queue = Queue()

        self._chat_thread_manager.register_message_listener(self)

    def start(self) -> None:
        if not self._stop:
            return

        self._stop = False
        self._worker_thread = Thread(target=self.__worker_loop)
        self._worker_thread.start()

        self._log.info("Agent chat task worker thread started.")

    def stop(self) -> None:
        if self._stop:
            return

        self._stop = True

        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None

        self._log.info("Agent chat task worker thread stopped.")

    def add_task(self, task: AgentChatTaskInfo) -> None:
        self._task_queue.put(task)

    @override
    def on_message_added(self, message: ChatMessageItem) -> None:
        thread_id = sanitize_uuid(message.thread_id)

        with Session(self._database) as db:
            db_tasks = db.query(AgentChatTask).where(AgentChatTask.thread_id == thread_id).all()

            for db_task in db_tasks:
                db_task_tools = (
                    db.query(AgentChatTaskTool).where(AgentChatTaskTool.agent_chat_task_id == db_task.id).all()
                )

                task = AgentChatTaskInfo.from_db(db_task, db_task_tools)
                self.add_task(task)

    def __update_task_status(
        self,
        task_id: UUID,
        from_status: AgentChatTaskStatus | list[AgentChatTaskStatus],
        to_status: AgentChatTaskStatus,
        *,
        current_required_action: AgentChatTaskRequiredAction | None = None,
        current_required_action_data: Any | None = None,
        error_message: str | None = None,
    ) -> bool:
        if not isinstance(from_status, list):
            from_status = [from_status]

        now = datetime.now()

        update_values: dict[str, Any] = {
            "status": to_status,
            "update_time": now,
        }

        if to_status in ("completed", "cancelled", "failed"):
            update_values["complete_time"] = now
            update_values["current_required_action"] = None
            update_values["current_required_action_data"] = None

        if to_status == "failed" and error_message is not None:
            update_values["error_message"] = error_message

        if "requires_action" in from_status and to_status != "requires_action":
            update_values["current_required_action"] = None
            update_values["current_required_action_data"] = None

        if to_status == "requires_action":
            if current_required_action is None:
                raise ValueError(
                    f"You wants to update task {task_id} to {to_status}, "
                    "but you did not specify current_required_action!"
                )

            update_values["current_required_action"] = current_required_action

            if current_required_action_data is not None:
                update_values["current_required_action_data"] = json.dumps(
                    dataclass_wizard.asdict(current_required_action_data)
                )
            else:
                update_values["current_required_action_data"] = None

        with Session(self._database) as db:
            r = db.execute(
                update(AgentChatTask)
                .values(**update_values)
                .where(and_(AgentChatTask.id == task_id, AgentChatTask.status.in_(from_status)))
            )

            c = r.rowcount

            db.commit()

            return c > 0

    def __worker_loop(self) -> None:
        while not self._stop:
            try:
                task = self._task_queue.get(timeout=1)
            except Empty:
                continue

            task_id = UUID(task.id)

            try:
                if not self.__update_task_status(task_id, ["pending", "requires_action"], "in_progress"):
                    self._log.warning("Failed to update status of task %s from pending to in_progress!")
                    continue

                try:
                    self.__execute_task(task)
                except Exception as e:
                    self._log.exception("Failed to execute agent chat task %s", task.id)

                    self.__update_task_status(task_id, "in_progress", "failed", error_message=str(e))
            except:
                self._log.exception("An exception occurred in agent chat task worker loop")

    def __to_backend_generation_messages(self, messages: list[ChatMessageItem]) -> list[GenerationMessage]:
        results: list[GenerationMessage] = []

        for message in messages:
            gmc: GenerationMessageContent
            gm_contents: list[GenerationMessageContent] = []

            for c in message.contents:
                if isinstance(c, ChatMessageContentTextPart):
                    gmc = TextGenerationMessageContent(text=c.text)

                    gm_contents.append(gmc)
                elif isinstance(c, ChatMessageContentImagePart):
                    gmc = ImageGenerationMessageContent(
                        image_file_path=str(self._file_store.make_store_file_path(c.image_file_id))
                    )

                    gm_contents.append(gmc)
                elif isinstance(c, ChatMessageToolRequestsPart):
                    for tool_request in c.tool_requests:
                        gmc = ToolCallRequestMessageContent(
                            id=tool_request.id,
                            function_name=tool_request.function_name,
                            function_arguments=tool_request.function_arguments,
                        )

                        gm_contents.append(gmc)
                elif isinstance(c, ChatMessageToolOutputsPart):
                    for tool_output in c.tool_outputs:
                        gmc = ToolCallResponseMessageContent(to_id=tool_output.tool_call_id, result=tool_output.output)

                        gm_contents.append(gmc)
                else:
                    raise ValueError(f"Unsupported message content type {c}")

            gm = GenerationMessage(role=message.role, contents=gm_contents)

            results.append(gm)

        return results

    def __determine_fetch_msg_count(self, task: AgentChatTaskInfo) -> int:
        fetch_msg_count = -1

        if task.thread_history_strategy == "auto":
            # TODO: Implement truncating with head_preserve_count
            assert isinstance(task.thread_history_strategy_params, AgentChatTaskAutoThreadHistoryStrategyParams)
            fetch_msg_count = task.thread_history_strategy_params.tail_preserve_count
        elif task.thread_history_strategy == "last_messages":
            assert isinstance(task.thread_history_strategy_params, AgentChatTaskLastMessageThreadHistoryStrategyParams)
            fetch_msg_count = task.thread_history_strategy_params.count

        return fetch_msg_count

    def __execute_task(self, task: AgentChatTaskInfo) -> None:
        latest_messages = self._chat_thread_manager.get_latest_messages(
            task.thread_id, count=self.__determine_fetch_msg_count(task)
        )

        if len(latest_messages) <= 0:
            raise ValueError(f"No message on thread {task.thread_id}")

        agent = self._agent_manager.get(task.agent_id)

        if agent is None:
            raise ValueError(f"Chat task {task.id} referenced a non-exist agent {task.agent_id}")

        model_id = agent.model_id

        if task.model_id is not None:
            model_id = task.model_id

        model = self._model_manager.get_model(model_id)

        if model is None:
            raise ValueError(f"Agent id {task.agent_id} wants non-exist model {agent.model_id}")

        model_instruction = task.model_instruction

        if model_instruction is None:
            model_instruction = agent.model_instruction

        if task.tools is not None and len(task.tools) > 0:
            available_tools = task.tools
        else:
            available_tools = [
                AgentToolRequest(
                    tool_name=t.tool_name,
                    tool_preset_parameters=t.tool_preset_parameters,
                    is_internal_tool=t.is_internal_tool,
                )
                for t in self._agent_manager.get_enabled_tools(agent.id)
            ]

        if task.tools_info is not None:
            tools_info = task.tools_info
            tools_info.tools = []
        else:
            tools_info = CallableToolsInfo(
                tools=[], force_use_no_tool=False, force_use_any_tool=False, force_use_tool_name=None
            )

        locale = DEFAULT_LOCALE

        if model.preset.preferred_locale is not None:
            locale = model.preset.preferred_locale

        for tool in available_tools:
            if tool.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
                tool_info = CodeInterpreterTool.description().to_localized(locale)
            elif tool.tool_name == INTERNAL_TOOL_RAG_SEARCH:
                tool_info = RagSearchTool.description().to_localized(locale)
            else:
                if tool.tool_preset_parameters is None:
                    raise ValueError(
                        f"Tool {tool.tool_name} in task {task.id} is not an internal tool, "
                        "but it has no preset parameters!"
                    )

                tool_info = dataclass_wizard.fromdict(LocalizedToolDescription, tool.tool_preset_parameters)

            tools_info.tools.append(tool_info)

        latest_messages.insert(
            0,
            ChatMessageItem(
                id="__SYSTEM__",
                thread_id=task.thread_id,
                role="system",
                contents=[
                    ChatMessageContentTextPart(
                        text=self._chat_template_manager.get_system_prompt(
                            generation_variant=model.generation_variant,
                            model_preset=model.preset,
                            runtime_configs=ChatTemplateRuntimeConfigs(
                                enable_parallel_tool_calling=task.parallel_tool_calling
                            ),
                            tools_info=tools_info,
                            base_sys_prompt=model_instruction,
                        )
                    )
                ],
                attachments=[],
                create_time=datetime.now(),
            ),
        )

        last_msg = latest_messages[-1]

        if isinstance(last_msg.contents[0], ChatMessageToolOutputsPart) and task.status == "requires_action":
            self.__update_task_latest_tool_call_details(UUID(task.id), last_msg.contents[0])

        gen_req = TextGenerationRequest(
            model_id=agent.model_id, messages=self.__to_backend_generation_messages(latest_messages)
        )

        streamer, gen_stats = self._backend_pipe.generate(model, gen_req)

        thread = Thread(target=self.__run_generation, args=[task, streamer, gen_stats], daemon=True)
        thread.start()

    def __run_generation(
        self, task: AgentChatTaskInfo, streamer: CompletionChunkStreamer, gen_stats: GenerationStatistics
    ) -> None:
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
        self._log.info(gen_stats.to_stats_text())

        self.__handle_generation_result(task, contents, gen_stats)

    def __handle_generation_result(self, task: AgentChatTaskInfo, result: Any, gen_stats: GenerationStatistics) -> None:
        task_id = UUID(task.id)

        if isinstance(result, TextGeneratedMessageChunk):
            message = ChatMessageInputItem(role="assistant", contents=[ChatMessageContentTextPart(result.content)])

            msg = self._chat_thread_manager.add_message(task.thread_id, message, self)

            if msg is None:
                raise ValueError(f"Unable to create message for task {task.id}, is thread {task.thread_id} exists?")

            self.__add_task_details(
                task_id=task_id,
                detail_type="message",
                detail_status="completed",
                data=AgentChatTaskMessageDetailsData(message_id=msg.id),
                generation_statistics=gen_stats,
            )

            self.__update_task_status(task_id, "in_progress", "completed")
        elif isinstance(result, (ToolCallGeneratedMessageChunk, list)):
            if isinstance(result, ToolCallGeneratedMessageChunk):
                result = [result]

            message = ChatMessageInputItem(
                role="assistant",
                contents=[
                    ChatMessageToolRequestsPart(
                        tool_requests=[
                            ChatMessageToolRequestItem(
                                id=str(c.index), function_name=c.name or "", function_arguments=json.loads(c.arguments)
                            )
                            for c in result
                            if isinstance(c, ToolCallGeneratedMessageChunk)
                        ]
                    )
                ],
            )

            self._chat_thread_manager.add_message(task.thread_id, message, self)

            self.__add_task_details(
                task_id=task_id,
                detail_type="tool_call",
                detail_status="in_progress",
                data=AgentChatTaskToolCallDetailsData(
                    tool_calls=[
                        AgentChatTaskDetailToolCallItem(
                            tool_call_id=str(c.index),
                            tool_name=c.name or "",
                            tool_input=c.arguments,
                            tool_output="",
                        )
                        for c in result
                        if isinstance(c, ToolCallGeneratedMessageChunk)
                    ]
                ),
                generation_statistics=gen_stats,
            )

            self.__update_task_required_action(task_id, "tool_call_request", result)

            self.__update_task_status(
                task_id,
                "in_progress",
                "requires_action",
                current_required_action="tool_call_request",
                current_required_action_data=result,
            )

            # TODO: Handle internal tools calling here, merge tool_resources from Assistant and Thread and pass to
            # corresponding tools, also tool preset params
            # agent_tool_res = self._agent_manager.get_enabled_tool_resources(agent.id)
            # thread_tool_res = self._chat_thread_manager.get_thread_tool_resources(task.thread_id)
        else:
            raise ValueError(f"Unsupported generation result {result}")

    def __add_task_details(
        self,
        task_id: UUID,
        detail_type: AgentChatTaskDetailType,
        detail_status: AgentChatTaskDetailStatus,
        data: Any,
        error_message: str | None = None,
        generation_statistics: GenerationStatistics | None = None,
    ) -> None:
        now = datetime.now()

        details = AgentChatTaskDetail()
        details.id = uuid4()
        details.agent_chat_task_id = task_id
        details.type = detail_type
        details.data = json.dumps(dataclass_wizard.asdict(data))
        details.status = detail_status

        if details.status in ("completed", "failed"):
            details.complete_time = now

        if error_message is not None:
            details.error_message = error_message

        if generation_statistics is not None:
            details.generation_statistics = json.dumps(dataclass_wizard.asdict(generation_statistics))

        details.create_time = now
        details.update_time = now

        with Session(self._database) as db:
            db.add(details)
            db.commit()

    def __update_task_latest_tool_call_details(
        self, task_id: UUID, tool_call_content: ChatMessageToolOutputsPart
    ) -> None:
        now = datetime.now()

        with Session(self._database) as db:
            latest_detail = (
                db.query(AgentChatTaskDetail)
                .where(
                    and_(
                        AgentChatTaskDetail.agent_chat_task_id == task_id,
                        AgentChatTaskDetail.type == "tool_call",
                        AgentChatTaskDetail.status == "in_progress",
                    )
                )
                .order_by(AgentChatTaskDetail.create_time.desc())
                .first()
            )

            if latest_detail is None:
                raise ValueError(
                    f"Agent chat task {task_id} has received a tool call response, "
                    "but no in progress tool request was found!"
                )

            latest_detail.status = "completed"
            latest_detail.complete_time = now
            latest_detail.update_time = now

            tool_request_data = dataclass_wizard.fromdict(
                AgentChatTaskToolCallDetailsData, json.loads(latest_detail.data)
            )

            for tool_request in tool_request_data.tool_calls:
                tool_output = next(
                    filter(lambda t: t.tool_call_id == tool_request.tool_call_id, tool_call_content.tool_outputs), None
                )

                if tool_output is None:
                    raise ValueError(
                        f"Agent chat task {task_id} expects tool call id {tool_request.tool_call_id}, "
                        "which is not contained in current request!"
                    )

                tool_request.tool_output = tool_output.output

            latest_detail.data = json.dumps(dataclass_wizard.asdict(tool_request_data))

            db.commit()

    def __update_task_required_action(
        self, task_id: UUID, action: AgentChatTaskRequiredAction, data: Any | None = None
    ) -> None:
        data_str = json.dumps(dataclass_wizard.asdict(data)) if data is not None else None

        with Session(self._database) as db:
            db.execute(
                update(AgentChatTask)
                .values(
                    current_required_action=action, current_required_action_data=data_str, update_time=datetime.now()
                )
                .where(AgentChatTask.id == task_id)
            )

            db.commit()
