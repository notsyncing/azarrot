import json
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import dataclass_wizard

from azarrot.common_data import (
    CallableToolsInfo,
    GenerationStatistics,
    ToolCallRequestMessageContents,
)
from azarrot.common_types import (
    AgentChatTaskDetailDataType,
    AgentChatTaskRequiredAction,
    AgentChatTaskStatus,
    AgentChatTaskThreadHistoryStrategy,
)
from azarrot.database_schemas import AgentChatTask, AgentChatTaskTool, AgentTool


@dataclass
class AgentToolRequest:
    tool_name: str
    tool_preset_parameters: dict[str, Any] | None = None
    is_internal_tool: bool = False


@dataclass
class AgentToolResourceRequest:
    tool_name: str
    tool_resources: dict[str, Any]


class AgentChatTaskDetailsData(ABC):
    type: AgentChatTaskDetailDataType


@dataclass
class AgentChatTaskMessageDetailsData(AgentChatTaskDetailsData):
    message_id: str
    type = "message"


@dataclass
class AgentChatTaskDetailToolCallItem:
    tool_call_id: str
    tool_name: str
    tool_input: str
    tool_output: str


@dataclass
class AgentChatTaskToolCallDetailsData(AgentChatTaskDetailsData):
    tool_calls: list[AgentChatTaskDetailToolCallItem]
    type = "tool_call"


class AgentChatTaskThreadHistoryStrategyParams:
    pass


@dataclass
class AgentChatTaskAutoThreadHistoryStrategyParams(AgentChatTaskThreadHistoryStrategyParams):
    head_preserve_count: int = 1
    tail_preserve_count: int = 5


@dataclass
class AgentChatTaskLastMessageThreadHistoryStrategyParams(AgentChatTaskThreadHistoryStrategyParams):
    count: int


@dataclass
class AgentGenerationParameters:
    temperature: float = 1
    top_p: float = 1


@dataclass
class AgentChatTaskInfo:
    id: str
    agent_id: str
    thread_id: str
    status: AgentChatTaskStatus
    current_required_action: AgentChatTaskRequiredAction | None
    current_required_action_data: ToolCallRequestMessageContents | None  # Add more data types as union type here
    start_time: datetime | None
    complete_time: datetime | None
    error_message: str | None
    model_id: str | None
    model_instruction: str | None
    tools: list[AgentToolRequest] | None
    tools_info: CallableToolsInfo | None
    parallel_tool_calling: bool
    additional_data: dict[str, Any] | None
    generation_statistics: GenerationStatistics
    generation_parameters: AgentGenerationParameters
    max_tokens: int
    thread_history_strategy: AgentChatTaskThreadHistoryStrategy
    thread_history_strategy_params: AgentChatTaskThreadHistoryStrategyParams
    create_time: datetime

    @staticmethod
    def from_db(dbo: AgentChatTask, db_tools: Sequence[AgentChatTaskTool] | None = None) -> "AgentChatTaskInfo":
        current_req_action = dbo.current_required_action

        if current_req_action == "tool_call_request" and dbo.current_required_action_data is not None:
            reqs = dataclass_wizard.fromdict(
                ToolCallRequestMessageContents, json.loads(dbo.current_required_action_data)
            )

            current_req_action_data = ToolCallRequestMessageContents(reqs.tool_requests)
        else:
            current_req_action_data = None

        if dbo.tools_info is not None:
            tools_info = dataclass_wizard.fromdict(CallableToolsInfo, json.loads(dbo.tools_info))
        else:
            tools_info = None

        if dbo.current_generation_statistics is not None:
            gen_stats = dataclass_wizard.fromdict(GenerationStatistics, json.loads(dbo.current_generation_statistics))
        else:
            gen_stats = GenerationStatistics(
                start_time=dbo.start_time or datetime.min,
                first_token_time=dbo.start_time or datetime.min,
                end_time=dbo.complete_time or datetime.min,
                prompt_tokens=0,
                completion_tokens=0,
            )

        if dbo.generation_parameters is not None:
            gen_params = dataclass_wizard.fromdict(AgentGenerationParameters, json.loads(dbo.generation_parameters))
        else:
            gen_params = AgentGenerationParameters()

        ths_params_type: type[AgentChatTaskThreadHistoryStrategyParams]

        if dbo.thread_history_strategy == "auto":
            ths_params_type = AgentChatTaskAutoThreadHistoryStrategyParams
        elif dbo.thread_history_strategy == "last_messages":
            ths_params_type = AgentChatTaskLastMessageThreadHistoryStrategyParams
        else:
            raise ValueError(f"Unsupported thread history strategy {dbo.thread_history_strategy}")

        ths_params = dataclass_wizard.fromdict(ths_params_type, json.loads(dbo.thread_history_strategy_params))

        return AgentChatTaskInfo(
            id=str(dbo.id),
            agent_id=str(dbo.agent_id),
            thread_id=str(dbo.thread_id),
            status=dbo.status,
            current_required_action=current_req_action,
            current_required_action_data=current_req_action_data,
            start_time=dbo.start_time,
            complete_time=dbo.complete_time,
            error_message=dbo.error_message,
            model_id=dbo.model_id,
            model_instruction=dbo.model_instruction,
            tools=convert_database_to_agent_tool_request(db_tools),
            tools_info=tools_info,
            parallel_tool_calling=dbo.parallel_tool_calling,
            additional_data=json.loads(dbo.additional_data) if dbo.additional_data is not None else None,
            generation_statistics=gen_stats,
            generation_parameters=gen_params,
            max_tokens=dbo.max_tokens,
            thread_history_strategy=dbo.thread_history_strategy,
            thread_history_strategy_params=ths_params,
            create_time=dbo.create_time,
        )


def convert_database_to_agent_tool_request(
    dbo: Sequence[AgentChatTaskTool] | Sequence[AgentTool] | None,
) -> list[AgentToolRequest] | None:
    if dbo is None:
        return None

    agent_tool_requests = []

    for item in dbo:
        tool_preset_parameters = json.loads(item.tool_preset_parameters) if item.tool_preset_parameters else None

        agent_tool_request = AgentToolRequest(
            tool_name=item.tool_name,
            is_internal_tool=item.is_internal_tool,
            tool_preset_parameters=tool_preset_parameters,
        )

        agent_tool_requests.append(agent_tool_request)

    return agent_tool_requests
