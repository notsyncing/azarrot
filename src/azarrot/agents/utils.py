import json
import uuid
from datetime import datetime

from azarrot.agents.common_data import AgentToolRequest, AgentToolResourceRequest
from azarrot.database_schemas import AgentChatTaskTool, AgentTool, AgentToolResource


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


def convert_agent_tool_resource_request_to_database(
    agent_id: uuid.UUID, tool_resources: list[AgentToolResourceRequest], create_time: datetime, update_time: datetime
) -> list[AgentToolResource]:
    results = []

    for tool_res in tool_resources:
        agent_tool_res = AgentToolResource(
            agent_id=agent_id,
            tool_name=tool_res.tool_name,
            tool_resources=json.dumps(tool_res.tool_resources),
            create_time=create_time,
            update_time=update_time,
        )

        results.append(agent_tool_res)

    return results


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
