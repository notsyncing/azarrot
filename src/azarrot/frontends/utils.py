import json
import uuid
from typing import Any, Literal

import dataclass_wizard
import openai.types
from openai.types.create_embedding_response import Usage

from azarrot.agents.common_data import AgentToolResourceRequest
from azarrot.agents.manager import AgentToolInfo, AgentToolRequest, AgentToolResourceInfo
from azarrot.common_data import CallableToolsInfo, GenerationStatistics, ToolCallRequestMessageContents
from azarrot.frontends.openai_support.openai_assistants import (
    OpenAIAssistantTool,
    OpenAICodeInterpreterTool,
    OpenAIFileSearchTool,
    OpenAIFileSearchToolOptions,
    OpenAIFunctionTool,
    OpenAIFunctionToolOptions,
    OpenAIToolResources,
)
from azarrot.frontends.openai_support.openai_data import (
    OpenAICodeInterpreterToolResource,
    OpenAIFileSearchToolRankingOptions,
    OpenAIFileSearchToolResource,
    OpenAITokenUsage,
    OpenAIToolCallFunction,
    OpenAIToolCallRequest,
    OpenAIToolChoiceConstant,
    ToolChoice,
    ToolChoiceFunction,
    ToolInfo,
)
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_RAG_SEARCH
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterToolResources
from azarrot.tools.internal.tool_rag_search import RagSearchToolConfigs, RagSearchToolResources
from azarrot.tools.tool import LocalizedToolDescription, LocalizedToolParameter


def to_backend_tool_parameters(tool_parameters: dict[str, Any] | None) -> list[LocalizedToolParameter]:
    if tool_parameters is None:
        return []

    param_type = tool_parameters["type"]

    if param_type != "object":
        raise ValueError(f"Unsupported tool parameter type {param_type}")

    required_params = tool_parameters.get("required", [])

    params = []

    if "properties" in tool_parameters:
        for k, v in tool_parameters["properties"].items():
            p = LocalizedToolParameter(
                name=k, description=v.get("description"), type=v.get("type"), required=k in required_params
            )

            params.append(p)

    return params


def to_backend_tools_info(
    tools_info: list[ToolInfo] | None, tools_choice: OpenAIToolChoiceConstant | ToolChoice | None
) -> CallableToolsInfo | None:
    if tools_info is None:
        return None

    tools = [
        LocalizedToolDescription(
            name=tool_info.function.name,
            display_name=None,
            description=tool_info.function.description,
            parameters=to_backend_tool_parameters(tool_info.function.parameters),
        )
        for tool_info in tools_info
    ]

    force_use_tool_name = None

    if isinstance(tools_choice, ToolChoice):
        if tools_choice.type == "function":
            if tools_choice.function is None:
                raise ValueError(f"OpenAI tool choice type is {tools_choice.type}, but no function name specified!")

            force_use_tool_name = tools_choice.function.name
        elif tools_choice.type == "file_search":
            force_use_tool_name = INTERNAL_TOOL_RAG_SEARCH
        else:
            raise ValueError(f"Unsupported OpenAI tool choice type {tools_choice.type}")

    return CallableToolsInfo(
        tools=tools,
        force_use_no_tool=tools_choice == "none",
        force_use_any_tool=tools_choice == "required",
        force_use_tool_name=force_use_tool_name,
    )


def to_openai_tool_choice(tools_info: CallableToolsInfo | None) -> OpenAIToolChoiceConstant | ToolChoice | None:
    if tools_info is None:
        return None

    if tools_info.force_use_no_tool:
        return "none"

    if tools_info.force_use_any_tool:
        return "required"

    if tools_info.force_use_tool_name is None:
        return "auto"
    else:
        function = None
        tool_type: Literal["file_search", "function"]

        if tools_info.force_use_tool_name == INTERNAL_TOOL_RAG_SEARCH:
            tool_type = "file_search"
        else:
            tool_type = "function"

            function = ToolChoiceFunction(name=tools_info.force_use_tool_name)

        return ToolChoice(type=tool_type, function=function)


def to_openai_tool_calls(content: ToolCallRequestMessageContents) -> list[OpenAIToolCallRequest]:
    return [
        OpenAIToolCallRequest(
            id=tool_call_req.id,
            type="function",
            function=OpenAIToolCallFunction(
                name=tool_call_req.function_name,
                arguments=json.dumps(tool_call_req.function_arguments),
            ),
        )
        for tool_call_req in content.tool_requests
    ]


def to_openai_assistant_tools(
    agent_id: str | uuid.UUID, agent_tools: list[AgentToolInfo] | list[AgentToolRequest] | None
) -> list[OpenAIAssistantTool] | None:
    if agent_tools is None:
        return None

    tools: list[OpenAIAssistantTool] = []

    for agent_tool in agent_tools:
        tool_params_dict = agent_tool.tool_preset_parameters
        tool: OpenAIAssistantTool

        if agent_tool.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
            tool = OpenAICodeInterpreterTool()
        elif agent_tool.tool_name == INTERNAL_TOOL_RAG_SEARCH:
            if tool_params_dict is None:
                raise ValueError(f"Agent {agent_id} enabled file-search tool, but it has no preset parameter!")

            rag_search_params = dataclass_wizard.fromdict(RagSearchToolConfigs, tool_params_dict)

            tool = OpenAIFileSearchTool(
                file_search=OpenAIFileSearchToolOptions(
                    max_num_results=rag_search_params.max_result_count,
                    ranking_options=OpenAIFileSearchToolRankingOptions(
                        ranker=rag_search_params.reranker_model_id,
                        score_threshold=rag_search_params.reranker_score_threshold,
                    ),
                )
            )
        else:
            if tool_params_dict is None:
                raise ValueError(
                    f"Agent {agent_id} enabled function tool {agent_tool.tool_name}, but it has no preset parameter!"
                )

            function_params = dataclass_wizard.fromdict(LocalizedToolDescription, tool_params_dict)

            tool = OpenAIFunctionTool(
                function=OpenAIFunctionToolOptions(
                    name=function_params.name,
                    description=function_params.description,
                    parameters=function_params.parameters_dict(),
                    strict=False,
                )
            )

        tools.append(tool)

    return tools


def to_openai_assistant_tool_resources(
    agent_tool_resources: list[AgentToolResourceInfo] | list[AgentToolResourceRequest] | None,
) -> OpenAIToolResources | None:
    if agent_tool_resources is None or len(agent_tool_resources) <= 0:
        return None

    tool_resources = OpenAIToolResources()

    for agent_tr in agent_tool_resources:
        tool_resource_dict = agent_tr.tool_resources

        if agent_tr.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
            configs = dataclass_wizard.fromdict(CodeInterpreterToolResources, tool_resource_dict)

            tool_resources.code_interpreter = OpenAICodeInterpreterToolResource(
                file_ids=[str(f) for f in configs.exposed_files] if configs.exposed_files is not None else []
            )
        elif agent_tr.tool_name == INTERNAL_TOOL_RAG_SEARCH:
            rag_search_params = dataclass_wizard.fromdict(RagSearchToolResources, tool_resource_dict)

            tool_resources.file_search = OpenAIFileSearchToolResource(
                vector_store_ids=[str(v) for v in rag_search_params.vector_stores],
            )
        else:
            continue

    return tool_resources


def to_openai_token_usage(gen_stats: GenerationStatistics) -> OpenAITokenUsage:
    return OpenAITokenUsage(
        prompt_tokens=gen_stats.prompt_tokens,
        completion_tokens=gen_stats.completion_tokens,
        total_tokens=gen_stats.prompt_tokens + gen_stats.completion_tokens,
    )


def to_openai_embedding_token_usage(gen_stats: GenerationStatistics) -> openai.types.create_embedding_response.Usage:
    return Usage(
        prompt_tokens=gen_stats.prompt_tokens,
        total_tokens=gen_stats.prompt_tokens,
    )
