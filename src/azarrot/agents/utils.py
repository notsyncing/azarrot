import json
import uuid
from collections.abc import Sequence
from datetime import datetime

from azarrot.agents.common_data import AgentToolRequest
from azarrot.database_schemas import AgentChatTaskTool, AgentTool


def convert_agent_tool_request_to_database(
    agent_id: uuid.UUID, tool_requests: list[AgentToolRequest], create_time: datetime, update_time: datetime
) -> list[AgentTool]:
    agent_tools = []

    for tool in tool_requests:
        tool_param_text = None

        if tool.tool_preset_parameters is not None:
            tool_param_text = json.dumps(tool.tool_preset_parameters)

        agent_tool = AgentTool(
            agent_id=agent_id,
            tool_name=tool.tool_name,
            tool_preset_parameters=tool_param_text,
            is_internal_tool=tool.is_internal_tool,
            create_time=create_time,
            update_time=update_time,
        )

        agent_tools.append(agent_tool)

    return agent_tools


def convert_agent_chat_task_tool_request_to_database(
    agent_chat_task_id: uuid.UUID, tool_requests: list[AgentToolRequest], create_time: datetime, update_time: datetime
) -> list[AgentChatTaskTool]:
    agent_tools = []

    for tool in tool_requests:
        tool_param_text = None

        if tool.tool_preset_parameters is not None:
            tool_param_text = json.dumps(tool.tool_preset_parameters)

        agent_tool = AgentChatTaskTool(
            agent_chat_task_id=agent_chat_task_id,
            tool_name=tool.tool_name,
            tool_preset_parameters=tool_param_text,
            is_internal_tool=tool.is_internal_tool,
            create_time=create_time,
            update_time=update_time,
        )

        agent_tools.append(agent_tool)

    return agent_tools


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
