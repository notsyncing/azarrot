import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import dataclass_wizard
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from starlette.status import HTTP_404_NOT_FOUND, HTTP_412_PRECONDITION_FAILED

from azarrot.agents.chat_task_manager import (
    AgentChatTaskAutoThreadHistoryStrategyParams,
    AgentChatTaskCreationRequest,
    AgentChatTaskDetailItem,
    AgentChatTaskDetailsListPagedQuery,
    AgentChatTaskDetailToolCallItem,
    AgentChatTaskInfo,
    AgentChatTaskLastMessageThreadHistoryStrategyParams,
    AgentChatTaskListPagedQuery,
    AgentChatTaskManager,
    AgentChatTaskMessageDetailsData,
    AgentChatTaskThreadHistoryStrategyParams,
    AgentChatTaskToolCallDetailsData,
)
from azarrot.agents.manager import AgentGenerationParameters
from azarrot.chats.common_data import ChatMessageInputItem, ChatMessageToolOutputItem, ChatMessageToolOutputsPart
from azarrot.chats.thread_manager import ChatThreadManager
from azarrot.common_types import AgentChatTaskDetailStatus, AgentChatTaskStatus, AgentChatTaskThreadHistoryStrategy
from azarrot.frontends.openai_support.openai_assistant_messages import (
    OpenAIAssistantMessageIncompleteDetails,
    OpenAIAssistantMessageRequest,
)
from azarrot.frontends.openai_support.openai_assistant_threads import (
    OpenAIAssistantCreateThreadRequest,
    OpenAIAssistantThreads,
)
from azarrot.frontends.openai_support.openai_assistants import (
    to_agent_tool_requests,
)
from azarrot.frontends.openai_support.openai_data import (
    OpenAIAssistantTool,
    OpenAIFileSearchToolRankingOptions,
    OpenAILastError,
    OpenAITokenUsage,
    OpenAIToolCallRequest,
    OpenAIToolChoiceConstant,
    ToolChoice,
)
from azarrot.frontends.utils import (
    to_backend_tools_info,
    to_openai_assistant_tools,
    to_openai_token_usage,
    to_openai_tool_calls,
    to_openai_tool_choice,
)
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.internal.tool_code_file_search import FileSearchOutputs
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterOutputs

OpenAIAssistantRunStatus = Literal[
    "queued",
    "in_progress",
    "requires_action",
    "cancelling",
    "cancelled",
    "failed",
    "completed",
    "incomplete",
    "expired",
]

OpenAIAssistantRunStepType = Literal["message_creation", "tool_calls"]
OpenAIAssistantRunStepStatus = Literal["in_progress", "cancelled", "failed", "completed", "expired"]
OpenAIAssistantToolCallsRunStepDetailType = Literal["code_interpreter", "file_search", "function"]
OpenAIAssistantToolCallsOutputType = Literal["logs", "image"]

OPENAI_ASSISTANT_RUN_STEPS_INCLUDE_FILE_SEARCH_CONTENTS = "step_details.tool_calls[*].file_search.results[*].content"

AGENT_CHAT_TASK_DETAIL_STATUS_MAP: dict[AgentChatTaskDetailStatus, OpenAIAssistantRunStepStatus] = {
    "in_progress": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "expired": "expired",
}


class OpenAIAssistantRunTruncationStrategy(BaseModel):
    type: Literal["auto", "last_messages"] = "auto"
    last_messages: int = 10


class OpenAIAssistantRunRequest(BaseModel):
    assistant_id: str
    thread: OpenAIAssistantCreateThreadRequest | None = None
    model: str | None = None
    instructions: str | None = None
    additional_instructions: str | None = None
    additional_messages: list[OpenAIAssistantMessageRequest] | None = None
    tools: list[Annotated[OpenAIAssistantTool, Field(discriminator="type")]] | None = None
    metadata: dict[str, Any] | None = None
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    stream: bool = False
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    truncation_strategy: OpenAIAssistantRunTruncationStrategy | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None
    parallel_tool_calls: bool = True


@dataclass
class OpenAIAssistantRunSubmitToolOutputsAction:
    tool_calls: list[OpenAIToolCallRequest] | None = None


@dataclass
class OpenAIAssistantRunRequiredAction:
    type: Literal["submit_tool_outputs"]
    submit_tool_outputs: OpenAIAssistantRunSubmitToolOutputsAction


@dataclass
class OpenAIAssistantRun:
    id: str
    created_at: int
    thread_id: str
    assistant_id: str
    status: OpenAIAssistantRunStatus
    required_action: OpenAIAssistantRunRequiredAction | None
    last_error: OpenAILastError | None
    expires_at: int | None
    started_at: int | None
    cancelled_at: int | None
    failed_at: int | None
    completed_at: int | None
    incomplete_details: OpenAIAssistantMessageIncompleteDetails | None
    model: str
    instructions: str
    tools: list[OpenAIAssistantTool]
    metadata: dict[str, Any]
    usage: OpenAITokenUsage
    temperature: float
    top_p: float
    max_prompt_tokens: int | None
    max_completion_tokens: int | None
    truncation_strategy: OpenAIAssistantRunTruncationStrategy | None
    tool_choice: OpenAIToolChoiceConstant | ToolChoice | None
    parallel_tool_calls: bool

    object: str = "thread.run"


class OpenAIAssistantUpdateRunRequest(BaseModel):
    metadata: dict[str, Any] | None = None


class OpenAIAssistantToolOutputRequest(BaseModel):
    tool_call_id: str
    output: str


class OpenAIAssistantSubmitToolOutputRequest(BaseModel):
    tool_outputs: list[OpenAIAssistantToolOutputRequest]
    stream: bool = Field(default=False)


@dataclass
class OpenAIAssistantMessageCreationRunStepDetails:
    message_id: str


@dataclass
class OpenAIAssistantMessageCreationRunStep:
    message_creation: OpenAIAssistantMessageCreationRunStepDetails
    type: OpenAIAssistantRunStepType = "message_creation"


@dataclass
class CodeInterpreterToolCallLogOutput:
    logs: str
    type: OpenAIAssistantToolCallsOutputType = "logs"


@dataclass
class CodeInterpreterToolCallImageOutputData:
    file_id: str


@dataclass
class CodeInterpreterToolCallImageOutput:
    image: CodeInterpreterToolCallImageOutputData
    type: OpenAIAssistantToolCallsOutputType = "image"


CodeInterpreterToolCallOutput = CodeInterpreterToolCallLogOutput | CodeInterpreterToolCallImageOutput


@dataclass
class CodeInterpreterToolCall:
    input: str
    outputs: list[CodeInterpreterToolCallOutput]


@dataclass
class OpenAIAssistantCodeInterpreterRunStepDetails:
    id: str
    code_interpreter: CodeInterpreterToolCall
    type: OpenAIAssistantToolCallsRunStepDetailType = "code_interpreter"


@dataclass
class FileSearchToolCallResultContent:
    type: Literal["text"]
    text: str


@dataclass
class FileSearchToolCallResult:
    file_id: str
    file_name: str
    score: float
    content: list[FileSearchToolCallResultContent]


@dataclass
class FileSearchToolCall:
    ranking_options: OpenAIFileSearchToolRankingOptions
    results: list[FileSearchToolCallResult]


@dataclass
class OpenAIAssistantFileSearchRunStepDetails:
    id: str
    file_search: FileSearchToolCall
    type: OpenAIAssistantToolCallsRunStepDetailType = "file_search"


@dataclass
class FunctionToolCall:
    name: str
    arguments: str
    output: str | None


@dataclass
class OpenAIAssistantFunctionRunStepDetails:
    id: str
    function: FunctionToolCall
    type: OpenAIAssistantToolCallsRunStepDetailType = "function"


OpenAIAssistantToolCallsRunStepDetails = (
    OpenAIAssistantCodeInterpreterRunStepDetails
    | OpenAIAssistantFileSearchRunStepDetails
    | OpenAIAssistantFunctionRunStepDetails
)


@dataclass
class OpenAIAssistantToolCallsRunStep:
    tool_calls: list[OpenAIAssistantToolCallsRunStepDetails]
    type: OpenAIAssistantRunStepType = "tool_calls"


OpenAIAssistantRunStepDetails = OpenAIAssistantMessageCreationRunStep | OpenAIAssistantToolCallsRunStep


@dataclass
class OpenAIAssistantRunStep(BaseModel):
    id: str
    created_at: int
    assistant_id: str
    thread_id: str
    run_id: str
    type: OpenAIAssistantRunStepType
    status: OpenAIAssistantRunStepStatus
    step_details: OpenAIAssistantRunStepDetails
    last_error: OpenAILastError | None
    expired_at: int | None
    cancelled_at: int | None
    failed_at: int | None
    completed_at: int | None
    usage: OpenAITokenUsage | None

    object: str = "thread.run.step"


class OpenAIAssistantRuns:
    _chat_task_manager: AgentChatTaskManager
    _chat_thread_manager: ChatThreadManager
    _openai_assistant_threads: OpenAIAssistantThreads

    def __init__(
        self,
        chat_task_manager: AgentChatTaskManager,
        chat_thread_manager: ChatThreadManager,
        openai_assistant_threads: OpenAIAssistantThreads,
    ) -> None:
        self._chat_task_manager = chat_task_manager
        self._chat_thread_manager = chat_thread_manager
        self._openai_assistant_threads = openai_assistant_threads

    def __combine_instructions(self, instructions: str | None, additional_instructions: str | None) -> str | None:
        if instructions is None and additional_instructions is None:
            return None

        result = instructions or ""

        if additional_instructions is not None:
            result = "\n" + additional_instructions

        return result

    def __to_thread_history_strategy(
        self, truncation_strategy: OpenAIAssistantRunTruncationStrategy | None
    ) -> tuple[AgentChatTaskThreadHistoryStrategy | None, AgentChatTaskThreadHistoryStrategyParams | None]:
        if truncation_strategy is None:
            return None, None

        if truncation_strategy.type == "auto":
            return "auto", AgentChatTaskAutoThreadHistoryStrategyParams()
        elif truncation_strategy.type == "last_messages":
            return "last_messages", AgentChatTaskLastMessageThreadHistoryStrategyParams(
                count=truncation_strategy.last_messages
            )
        else:
            raise ValueError(f"Unsupported OpenAI truncation strategy {truncation_strategy.type}")

    def __to_openai_truncation_strategy(
        self,
        thread_history_strategy: AgentChatTaskThreadHistoryStrategy,
        thread_history_strategy_params: AgentChatTaskThreadHistoryStrategyParams,
    ) -> OpenAIAssistantRunTruncationStrategy:
        if thread_history_strategy == "auto":
            return OpenAIAssistantRunTruncationStrategy(type="auto")
        elif thread_history_strategy == "last_messages":
            if not isinstance(thread_history_strategy_params, AgentChatTaskLastMessageThreadHistoryStrategyParams):
                raise ValueError(
                    f"Invalid parameter type for agent chat task thread history strategy {thread_history_strategy}"
                )

            return OpenAIAssistantRunTruncationStrategy(
                type="last_messages", last_messages=thread_history_strategy_params.count
            )
        else:
            raise ValueError(f"Unsupported agent chat task thread history strategy {thread_history_strategy}")

    def __to_openai_assistant_run_status(self, agent_chat_task_status: AgentChatTaskStatus) -> OpenAIAssistantRunStatus:
        if agent_chat_task_status == "cancelled":
            return "cancelled"
        elif agent_chat_task_status == "completed":
            return "completed"
        elif agent_chat_task_status == "expired":
            return "expired"
        elif agent_chat_task_status == "failed":
            return "failed"
        elif agent_chat_task_status == "in_progress":
            return "in_progress"
        elif agent_chat_task_status == "pending":
            return "queued"
        elif agent_chat_task_status == "requires_action":
            return "requires_action"
        else:
            raise ValueError(f"Unsupported agent chat task status {agent_chat_task_status}")

    def __to_openai_assistant_run_info(self, agent_chat_task_info: AgentChatTaskInfo) -> OpenAIAssistantRun:
        if agent_chat_task_info.current_required_action == "tool_call_request":
            if agent_chat_task_info.current_required_action_data is None:
                raise ValueError(
                    f"Current required action is {agent_chat_task_info.current_required_action}, "
                    "but no associated data is present!"
                )

            required_action = OpenAIAssistantRunRequiredAction(
                type="submit_tool_outputs",
                submit_tool_outputs=OpenAIAssistantRunSubmitToolOutputsAction(
                    tool_calls=to_openai_tool_calls(agent_chat_task_info.current_required_action_data)
                ),
            )
        elif agent_chat_task_info.current_required_action is None:
            required_action = None
        else:
            raise ValueError(
                f"Unsupported agent chat task required action {agent_chat_task_info.current_required_action}"
            )

        if agent_chat_task_info.start_time is not None:
            started_at = int(agent_chat_task_info.start_time.timestamp())
        else:
            started_at = None

        completed_at = None
        cancelled_at = None
        failed_at = None

        if agent_chat_task_info.complete_time is not None:
            completed_at = int(agent_chat_task_info.complete_time.timestamp())

            if agent_chat_task_info.status == "cancelled":
                cancelled_at = completed_at
            elif agent_chat_task_info.status == "failed":
                failed_at = completed_at

        tools, _ = to_openai_assistant_tools(agent_chat_task_info.agent_id, agent_chat_task_info.tools)

        return OpenAIAssistantRun(
            id=agent_chat_task_info.id,
            created_at=int(agent_chat_task_info.create_time.timestamp()),
            thread_id=agent_chat_task_info.thread_id,
            assistant_id=agent_chat_task_info.agent_id,
            status=self.__to_openai_assistant_run_status(agent_chat_task_info.status),
            required_action=required_action,
            last_error=OpenAILastError(code="server_error", message=agent_chat_task_info.error_message)
            if agent_chat_task_info.error_message is not None
            else None,
            expires_at=None,
            started_at=started_at,
            cancelled_at=cancelled_at,
            failed_at=failed_at,
            completed_at=completed_at,
            incomplete_details=OpenAIAssistantMessageIncompleteDetails(reason=agent_chat_task_info.error_message or "")
            if agent_chat_task_info.status == "truncated"
            else None,
            model=agent_chat_task_info.model_id,
            instructions=agent_chat_task_info.model_instruction or "",
            tools=tools or [],
            metadata=agent_chat_task_info.additional_data or {},
            usage=to_openai_token_usage(agent_chat_task_info.generation_statistics),
            temperature=agent_chat_task_info.generation_parameters.temperature,
            top_p=agent_chat_task_info.generation_parameters.top_p,
            max_prompt_tokens=None,
            max_completion_tokens=agent_chat_task_info.max_tokens,
            truncation_strategy=self.__to_openai_truncation_strategy(
                agent_chat_task_info.thread_history_strategy, agent_chat_task_info.thread_history_strategy_params
            ),
            tool_choice=to_openai_tool_choice(agent_chat_task_info.tools_info),
            parallel_tool_calls=agent_chat_task_info.parallel_tool_calling,
        )

    def run_assistant(self, tid: str, request: OpenAIAssistantRunRequest) -> OpenAIAssistantRun:
        if request.thread is not None:
            raise ValueError("You have specified to create a run, so you cannot specify new thread details!")

        ths, ths_params = self.__to_thread_history_strategy(request.truncation_strategy)
        tools, _ = to_agent_tool_requests(request.tools, None)

        req = AgentChatTaskCreationRequest(
            agent_id=request.assistant_id,
            thread_id=tid,
            model_id=request.model,
            model_instruction=self.__combine_instructions(request.instructions, request.additional_instructions),
            generation_parameters=AgentGenerationParameters(temperature=request.temperature, top_p=request.top_p),
            thread_history_strategy=ths,
            thread_history_strategy_params=ths_params,
            max_tokens=request.max_completion_tokens,
            tools=tools,
            tools_info=to_backend_tools_info([], request.tool_choice),
            parallel_tool_calling=request.parallel_tool_calls,
            additional_data=request.metadata,
        )

        info = self._chat_task_manager.create_task(req)
        return self.__to_openai_assistant_run_info(info)

    def run_assistant_with_thread(self, request: OpenAIAssistantRunRequest) -> OpenAIAssistantRun:
        if request.thread is None:
            raise ValueError("You have specified to create a new thread and run, but no new thread details specified!")

        thread_info = self._openai_assistant_threads.create_thread(request.thread)
        request.thread = None

        return self.run_assistant(thread_info.id, request)

    def get_run_list(
        self,
        tid: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        page = self._chat_task_manager.get_list(
            AgentChatTaskListPagedQuery(
                thread_id=tid, create_time_desc_order=order == "desc", page_size=limit, before_id=before, after_id=after
            )
        )

        openai_list = [self.__to_openai_assistant_run_info(d) for d in page.data]

        return {
            "object": "list",
            "data": openai_list,
            "first_id": openai_list[0].id if len(openai_list) > 0 else None,
            "last_id": openai_list[-1].id if len(openai_list) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get_run(self, tid: str, rid: str) -> OpenAIAssistantRun:
        agent_chat_task_info = self._chat_task_manager.get(rid, tid)

        if agent_chat_task_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND, f"Assistant run {rid} for thread {tid} does not exist!")

        return self.__to_openai_assistant_run_info(agent_chat_task_info)

    def update_run(self, tid: str, rid: str, request: OpenAIAssistantUpdateRunRequest) -> OpenAIAssistantRun:
        succeed = self._chat_task_manager.update(tid, rid, request.metadata)

        if not succeed:
            raise HTTPException(HTTP_404_NOT_FOUND, f"Assistant run {rid} for thread {tid} does not exist!")

        return self.get_run(tid, rid)

    def __to_backend_tool_output_messages(
        self, openai_tool_outputs: list[OpenAIAssistantToolOutputRequest]
    ) -> list[ChatMessageToolOutputItem]:
        return [
            ChatMessageToolOutputItem(tool_call_id=output.tool_call_id, output=output.output)
            for output in openai_tool_outputs
        ]

    def submit_tool_outputs(
        self, tid: str, rid: str, request: OpenAIAssistantSubmitToolOutputRequest
    ) -> OpenAIAssistantRun:
        run_info = self._chat_task_manager.get(rid, tid)

        if run_info is None:
            raise HTTPException(HTTP_404_NOT_FOUND, f"Assistant run {rid} for thread {tid} does not exist!")

        if run_info.current_required_action != "tool_call_request":
            raise HTTPException(
                HTTP_412_PRECONDITION_FAILED, f"Assistant run {rid} for thread {tid} does not need tool outputs!"
            )

        msg = ChatMessageInputItem(
            role="tool",
            contents=[
                ChatMessageToolOutputsPart(tool_outputs=self.__to_backend_tool_output_messages(request.tool_outputs))
            ],
        )

        self._chat_thread_manager.add_message(tid, msg)

        return self.get_run(tid, rid)

    def cancel_run(self, tid: str, rid: str) -> OpenAIAssistantRun:
        succeed = self._chat_task_manager.cancel(tid, rid)

        if not succeed:
            raise HTTPException(HTTP_404_NOT_FOUND, f"Assistant run {rid} for thread {tid} does not exist!")

        return self.get_run(tid, rid)

    def __to_openai_assistant_run_step_tool_call(
        self, tool_call: AgentChatTaskDetailToolCallItem
    ) -> OpenAIAssistantToolCallsRunStepDetails:
        if tool_call.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
            ci_outputs = dataclass_wizard.fromdict(CodeInterpreterOutputs, json.loads(tool_call.tool_output))

            openai_outputs: list[CodeInterpreterToolCallOutput] = []

            if ci_outputs.console is not None:
                openai_outputs.append(CodeInterpreterToolCallLogOutput(logs=ci_outputs.console))

            if ci_outputs.files is not None:
                openai_outputs.extend(
                    [
                        CodeInterpreterToolCallImageOutput(
                            image=CodeInterpreterToolCallImageOutputData(file_id=str(output_file_id))
                        )
                        for output_file_id in ci_outputs.files
                    ]
                )

            return OpenAIAssistantCodeInterpreterRunStepDetails(
                id=tool_call.tool_call_id,
                code_interpreter=CodeInterpreterToolCall(input=tool_call.tool_input, outputs=openai_outputs),
            )
        elif tool_call.tool_name == INTERNAL_TOOL_RAG_SEARCH:
            rs_outputs = dataclass_wizard.fromdict(FileSearchOutputs, json.loads(tool_call.tool_output))

            openai_results = [
                FileSearchToolCallResult(
                    file_id=sr.file_id,
                    file_name=sr.file_name or "",
                    score=sr.score,
                    content=[FileSearchToolCallResultContent(type="text", text=c) for c in sr.matched_chunks],
                )
                for sr in rs_outputs.search_results
            ]

            return OpenAIAssistantFileSearchRunStepDetails(
                id=tool_call.tool_call_id,
                file_search=FileSearchToolCall(
                    ranking_options=OpenAIFileSearchToolRankingOptions(
                        ranker=rs_outputs.current_configs.reranker_model_id,
                        score_threshold=rs_outputs.current_configs.reranker_score_threshold,
                    ),
                    results=openai_results,
                ),
            )
        else:
            return OpenAIAssistantFunctionRunStepDetails(
                id=tool_call.tool_call_id,
                function=FunctionToolCall(
                    name=tool_call.tool_name, arguments=tool_call.tool_input, output=tool_call.tool_output
                ),
            )

    def __to_openai_assistant_run_step(self, detail: AgentChatTaskDetailItem) -> OpenAIAssistantRunStep:
        chat_task = self._chat_task_manager.get(detail.agent_chat_task_id)

        if chat_task is None:
            raise ValueError(
                f"Agent chat task detail {detail.id} referenced non-exist chat task {detail.agent_chat_task_id}"
            )

        openai_type: OpenAIAssistantRunStepType
        openai_step_details: OpenAIAssistantRunStepDetails

        if detail.type == "message":
            assert isinstance(detail.data, AgentChatTaskMessageDetailsData)
            openai_type = "message_creation"

            openai_step_details = OpenAIAssistantMessageCreationRunStep(
                message_creation=OpenAIAssistantMessageCreationRunStepDetails(message_id=detail.data.message_id)
            )
        elif detail.type == "tool_call":
            assert isinstance(detail.data, AgentChatTaskToolCallDetailsData)
            openai_type = "tool_calls"

            openai_step_details = OpenAIAssistantToolCallsRunStep(
                tool_calls=[self.__to_openai_assistant_run_step_tool_call(t) for t in detail.data.tool_calls]
            )
        else:
            raise ValueError(f"Unsupported agent chat task detail type {detail.type}")

        return OpenAIAssistantRunStep(
            id=detail.id,
            created_at=int(detail.create_time.timestamp()),
            assistant_id=chat_task.agent_id,
            thread_id=chat_task.thread_id,
            run_id=detail.agent_chat_task_id,
            type=openai_type,
            status=AGENT_CHAT_TASK_DETAIL_STATUS_MAP[detail.status],
            step_details=openai_step_details,
            last_error=OpenAILastError(code="server_error", message=detail.error_message or "")
            if detail.status == "failed"
            else None,
            expired_at=None,
            cancelled_at=int(detail.complete_time.timestamp())
            if detail.status == "cancelled" and detail.complete_time is not None
            else None,
            failed_at=int(detail.complete_time.timestamp())
            if detail.status == "failed" and detail.complete_time is not None
            else None,
            completed_at=int(detail.complete_time.timestamp())
            if detail.status == "completed" and detail.complete_time is not None
            else None,
            usage=to_openai_token_usage(detail.generation_statistics)
            if detail.generation_statistics is not None
            else None,
        )

    def get_step_list(
        self,
        tid: str,
        rid: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        include_file_search_contents = (
            include is not None and OPENAI_ASSISTANT_RUN_STEPS_INCLUDE_FILE_SEARCH_CONTENTS in include
        )

        page = self._chat_task_manager.get_details(
            AgentChatTaskDetailsListPagedQuery(
                thread_id=tid,
                agent_chat_task_id=rid,
                create_time_desc_order=order == "desc",
                page_size=limit,
                before_id=before,
                after_id=after,
                include_file_search_contents=include_file_search_contents,
            )
        )

        openai_list = [self.__to_openai_assistant_run_step(d) for d in page.data]

        return {
            "object": "list",
            "data": openai_list,
            "first_id": openai_list[0].id if len(openai_list) > 0 else None,
            "last_id": openai_list[-1].id if len(openai_list) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get_step(self, tid: str, rid: str, sid: str) -> OpenAIAssistantRunStep:
        detail = self._chat_task_manager.get_detail(tid, rid, sid)

        if detail is None:
            raise HTTPException(HTTP_404_NOT_FOUND, f"Run step {sid} in run {rid} of thread {tid} does not exist!")

        return self.__to_openai_assistant_run_step(detail)
