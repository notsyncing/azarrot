import json
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import dataclass_wizard
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field
from starlette.status import HTTP_404_NOT_FOUND

from azarrot.agents.manager import (
    AgentGenerationParameters,
    AgentInfo,
    AgentListPagedQuery,
    AgentManager,
    AgentToolInfo,
    AgentToolRequest,
)
from azarrot.config import OpenAIFrontendConfig
from azarrot.frontends.openai_support.openai_vector_stores import OpenAIVectorStoreChunkingStrategy
from azarrot.frontends.utils import to_backend_tool_parameters
from azarrot.models.model_manager import ModelManager
from azarrot.tools.internal import INTERNAL_TOOL_CODE_INTERPRETER, INTERNAL_TOOL_FILE_SEARCH
from azarrot.tools.internal.tool_code_file_search import FileSearchToolConfigs
from azarrot.tools.internal.tool_code_interpreter import CodeInterpreterToolConfigs
from azarrot.tools.tool import LocalizedToolDescription
from azarrot.vector_store.manager import VectorStoreManager

OPENAI_TOOL_CODE_INTERPRETER: Literal["code_interpreter"] = "code_interpreter"
OPENAI_TOOL_FILE_SEARCH: Literal["file_search"] = "file_search"
OPENAI_TOOL_FUNCTION: Literal["function"] = "function"

OPENAI_FILE_SEARCH_MAX_RESULT_COUNT = 20
OPENAI_FILE_SEARCH_RERANKER_SCORE_THRESHOLD = 0.0

OpenAIAssistantToolType = Literal["code_interpreter", "file_search", "function"]


class OpenAICodeInterpreterTool(BaseModel):
    type: Literal["code_interpreter"] = OPENAI_TOOL_CODE_INTERPRETER


class OpenAIFileSearchToolRankingOptions(BaseModel):
    ranker: str | None = None
    score_threshold: float


class OpenAIFileSearchToolOptions(BaseModel):
    max_num_results: int | None = None
    ranking_options: OpenAIFileSearchToolRankingOptions | None = None


class OpenAIFileSearchTool(BaseModel):
    type: Literal["file_search"] = OPENAI_TOOL_FILE_SEARCH
    file_search: OpenAIFileSearchToolOptions | None = None


class OpenAIFunctionToolOptions(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class OpenAIFunctionTool(BaseModel):
    type: Literal["function"] = OPENAI_TOOL_FUNCTION
    function: OpenAIFunctionToolOptions


class OpenAICodeInterpreterToolResource(BaseModel):
    file_ids: list[str]


class OpenAIFileSearchToolVectorStoreCreationRequest(BaseModel):
    file_ids: list[str] | None = None
    chunking_strategy: Annotated[OpenAIVectorStoreChunkingStrategy, Field(discriminator="type")] | None = None
    metadata: dict[str, Any] | None = None
    vs_id: uuid.UUID | None = None


class OpenAIFileSearchToolResource(BaseModel):
    vector_store_ids: list[str] | None = None
    vector_stores: list[OpenAIFileSearchToolVectorStoreCreationRequest] | None = None


class OpenAIToolResources(BaseModel):
    code_interpreter: OpenAICodeInterpreterToolResource | None = None
    file_search: OpenAIFileSearchToolResource | None = None


OpenAIAssistantTool = OpenAICodeInterpreterTool | OpenAIFileSearchTool | OpenAIFunctionTool


class OpenAICreateAssistantRequest(BaseModel):
    model: str
    name: str | None = None
    description: str | None = None
    instructions: str | None = None

    tools: list[Annotated[OpenAIAssistantTool, Field(discriminator="type")]] | None = None

    tool_resources: OpenAIToolResources
    metadata: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None


class OpenAIUpdateCodeInterpreterToolResource(BaseModel):
    file_ids: list[str]


class OpenAIUpdateFileSearchToolResource(BaseModel):
    vector_store_ids: list[str] | None = None


class OpenAIUpdateToolResources(BaseModel):
    code_interpreter: OpenAIUpdateCodeInterpreterToolResource | None = None
    file_search: OpenAIUpdateFileSearchToolResource | None = None


class OpenAIUpdateAssistantRequest(BaseModel):
    model: str | None = None
    name: str | None = None
    description: str | None = None
    instructions: str | None = None

    tools: list[Annotated[OpenAIAssistantTool, Field(discriminator="type")]] | None = None

    tool_resources: OpenAIUpdateToolResources | None = None
    metadata: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None


def to_agent_code_interpreter_tool_options(
    openai_tool_resources: OpenAIToolResources | OpenAIUpdateToolResources | None,
) -> CodeInterpreterToolConfigs:
    options = CodeInterpreterToolConfigs()

    if openai_tool_resources is not None and openai_tool_resources.code_interpreter is not None:
        options.exposed_files = openai_tool_resources.code_interpreter.file_ids

    return options


def to_agent_file_search_tool_options(
    openai_tool_file_search: OpenAIFileSearchToolOptions | None,
    openai_tool_resources: OpenAIToolResources | OpenAIUpdateToolResources | None,
    reranker_model_id: str | None = None,
) -> tuple[FileSearchToolConfigs, list[OpenAIFileSearchToolVectorStoreCreationRequest] | None]:
    if openai_tool_resources is None:
        raise ValueError("You have added file search tool, but no resource was configured!")

    if openai_tool_resources.file_search is None:
        raise ValueError("You have added file search tool, but no file search resource was configured!")

    vector_stores = []

    exists_vector_stores = openai_tool_resources.file_search.vector_store_ids

    if exists_vector_stores is not None:
        vector_stores.extend(exists_vector_stores)

    new_vector_stores = None

    if isinstance(openai_tool_resources, OpenAIToolResources):
        new_vector_stores = openai_tool_resources.file_search.vector_stores

        if new_vector_stores is not None:
            for new_vector_store in new_vector_stores:
                vs_id = uuid.uuid4()
                new_vector_store.vs_id = vs_id
                vector_stores.append(str(vs_id))

    max_result_count = OPENAI_FILE_SEARCH_MAX_RESULT_COUNT
    reranker_score_threshold = OPENAI_FILE_SEARCH_RERANKER_SCORE_THRESHOLD

    if openai_tool_file_search is not None:
        if openai_tool_file_search.max_num_results is not None:
            max_result_count = openai_tool_file_search.max_num_results

        if openai_tool_file_search.ranking_options is not None:
            if openai_tool_file_search.ranking_options.ranker is not None:
                if openai_tool_file_search.ranking_options.ranker != "auto":
                    reranker_model_id = openai_tool_file_search.ranking_options.ranker

            reranker_score_threshold = openai_tool_file_search.ranking_options.score_threshold

    return FileSearchToolConfigs(
        vector_stores=vector_stores,
        max_result_count=max_result_count,
        reranker_model_id=reranker_model_id,
        reranker_score_threshold=reranker_score_threshold,
    ), new_vector_stores


def to_agent_function_call_options(openai_function: OpenAIFunctionToolOptions) -> LocalizedToolDescription:
    return LocalizedToolDescription(
        name=openai_function.name,
        display_name=None,
        description=openai_function.description,
        parameters=to_backend_tool_parameters(openai_function.parameters),
    )


def to_agent_tool_requests(
    openai_tools: list[OpenAIAssistantTool] | None,
    openai_tool_resources: OpenAIToolResources | OpenAIUpdateToolResources | None,
    reranker_model_id: str | None = None,
) -> tuple[list[AgentToolRequest] | None, list[OpenAIFileSearchToolVectorStoreCreationRequest] | None]:
    if openai_tools is None:
        return None, None

    agent_tools = []
    new_vector_stores = None

    for openai_tool in openai_tools:
        agent_tool: AgentToolRequest

        if isinstance(openai_tool, OpenAICodeInterpreterTool):
            agent_tool = AgentToolRequest(
                tool_name=INTERNAL_TOOL_CODE_INTERPRETER,
                tool_preset_parameters=dataclass_wizard.asdict(
                    to_agent_code_interpreter_tool_options(openai_tool_resources)
                ),
                is_internal_tool=True,
            )
        elif isinstance(openai_tool, OpenAIFileSearchTool):
            agent_tool_params, new_vector_stores = to_agent_file_search_tool_options(
                openai_tool.file_search, openai_tool_resources, reranker_model_id=reranker_model_id
            )

            agent_tool = AgentToolRequest(
                tool_name=INTERNAL_TOOL_FILE_SEARCH,
                tool_preset_parameters=dataclass_wizard.asdict(agent_tool_params),
                is_internal_tool=True,
            )
        elif isinstance(openai_tool, OpenAIFunctionTool):
            agent_tool = AgentToolRequest(
                tool_name=openai_tool.function.name,
                tool_preset_parameters=dataclass_wizard.asdict(to_agent_function_call_options(openai_tool.function)),
                is_internal_tool=False,
            )
        else:
            raise ValueError(f"Unsupported OpenAI tool type {openai_tool.type}")

        agent_tools.append(agent_tool)

    return agent_tools, new_vector_stores


@dataclass
class OpenAIAssistantInfo:
    id: str
    created_at: int
    name: str | None
    description: str | None
    model: str
    instructions: str | None
    tools: list[OpenAIAssistantTool] | None
    tool_resources: OpenAIToolResources | None
    metadata: dict[str, Any] | None
    temperature: float | None
    top_p: float | None

    object: str = "assistant"

    @staticmethod
    def from_agent_info(agent: AgentInfo, agent_tools: list[AgentToolInfo]) -> "OpenAIAssistantInfo":
        tools: list[OpenAIAssistantTool] = []

        tool_resources = OpenAIToolResources()

        for agent_tool in agent_tools:
            tool_params_dict = agent_tool.tool_preset_parameters
            tool: OpenAIAssistantTool

            if agent_tool.tool_name == INTERNAL_TOOL_CODE_INTERPRETER:
                tool = OpenAICodeInterpreterTool()

                if tool_params_dict is not None:
                    configs = dataclass_wizard.fromdict(CodeInterpreterToolConfigs, tool_params_dict)

                    tool_resources.code_interpreter = OpenAICodeInterpreterToolResource(
                        file_ids=[str(f) for f in configs.exposed_files] if configs.exposed_files is not None else []
                    )
            elif agent_tool.tool_name == INTERNAL_TOOL_FILE_SEARCH:
                if tool_params_dict is None:
                    raise ValueError(f"Agent {agent.id} enabled file-search tool, but it has no preset parameter!")

                rag_search_params = dataclass_wizard.fromdict(FileSearchToolConfigs, tool_params_dict)

                tool = OpenAIFileSearchTool(
                    file_search=OpenAIFileSearchToolOptions(
                        max_num_results=rag_search_params.max_result_count,
                        ranking_options=OpenAIFileSearchToolRankingOptions(
                            ranker=rag_search_params.reranker_model_id,
                            score_threshold=rag_search_params.reranker_score_threshold,
                        ),
                    )
                )

                tool_resources.file_search = OpenAIFileSearchToolResource(
                    vector_store_ids=[str(v) for v in rag_search_params.vector_stores],
                )
            else:
                if tool_params_dict is None:
                    raise ValueError(
                        f"Agent {agent.id} enabled function tool {agent_tool.tool_name}, "
                        "but it has no preset parameter!"
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

        agent_gen_params = {}

        if agent.default_generation_parameters is not None:
            agent_gen_params = json.loads(agent.default_generation_parameters)

        return OpenAIAssistantInfo(
            id=agent.id,
            created_at=int(agent.create_time.timestamp()),
            name=agent.name,
            description=agent.description,
            model=agent.model_id,
            instructions=agent.model_instruction,
            tools=tools,
            tool_resources=tool_resources,
            metadata=json.loads(agent.additional_data) if agent.additional_data is not None else None,
            temperature=agent_gen_params.get("temperature", None),
            top_p=agent_gen_params.get("top_p", None),
        )


def create_openai_requested_vector_stores(
    openai_config: OpenAIFrontendConfig,
    model_manager: ModelManager,
    vector_stores: VectorStoreManager,
    new_vector_stores: list[OpenAIFileSearchToolVectorStoreCreationRequest],
) -> None:
    if openai_config.vector_store_default_embedding_model_id is None:
        raise ValueError("OpenAI vector store default embedding model is not configured!")

    for new_vector_store in new_vector_stores:
        embedding_model = model_manager.get_model(openai_config.vector_store_default_embedding_model_id)

        if embedding_model is None:
            raise ValueError(
                "OpenAI vector store default embedding model "
                f"{openai_config.vector_store_default_embedding_model_id} does not exist!"
            )

        vs = vector_stores.create(
            name=None,
            store_id=new_vector_store.vs_id,
            embedding_model=embedding_model,
            additional_data=new_vector_store.metadata,
        )

        if new_vector_store.file_ids is not None and len(new_vector_store.file_ids) > 0:
            vector_stores.add_stored_files(vs.id, new_vector_store.file_ids)


class OpenAIAssistants:
    _openai_config: OpenAIFrontendConfig
    _agent_manager: AgentManager
    _vector_store: VectorStoreManager
    _model_manager: ModelManager

    def __init__(
        self,
        openai_config: OpenAIFrontendConfig,
        agent_manager: AgentManager,
        vector_store: VectorStoreManager,
        model_manager: ModelManager,
    ) -> None:
        self._openai_config = openai_config
        self._agent_manager = agent_manager
        self._vector_store = vector_store
        self._model_manager = model_manager

    def create_assistant(self, request: OpenAICreateAssistantRequest) -> OpenAIAssistantInfo:
        agent_tools_request, new_vector_stores = to_agent_tool_requests(
            request.tools,
            request.tool_resources,
            reranker_model_id=self._openai_config.assistant_file_search_reranker_default_model_id,
        )

        if new_vector_stores is not None:
            create_openai_requested_vector_stores(
                self._openai_config, self._model_manager, self._vector_store, new_vector_stores
            )

        agent_info = self._agent_manager.create(
            model_id=request.model,
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            default_generation_parameters=AgentGenerationParameters(
                temperature=request.temperature, top_p=request.top_p
            ),
            tools=agent_tools_request,
            additional_data=request.metadata,
        )

        agent_tools_info = self._agent_manager.get_enabled_tools(agent_info.id)

        return OpenAIAssistantInfo.from_agent_info(agent_info, agent_tools_info)

    def get_assistant_list(
        self,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        page = self._agent_manager.get_list(
            AgentListPagedQuery(
                create_time_desc_order=order == "desc", page_size=limit, before_id=before, after_id=after
            )
        )

        openai_list: list[OpenAIAssistantInfo] = []

        for agent in page.data:
            agent_tools = self._agent_manager.get_enabled_tools(agent.id)
            openai_store_info = OpenAIAssistantInfo.from_agent_info(agent, agent_tools)
            openai_list.append(openai_store_info)

        return {
            "object": "list",
            "data": openai_list,
            "first_id": openai_list[0].id if len(openai_list) > 0 else None,
            "last_id": openai_list[-1].id if len(openai_list) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get_assistant(self, assistant_id: str) -> OpenAIAssistantInfo:
        agent = self._agent_manager.get(assistant_id)

        if agent is None:
            raise HTTPException(HTTP_404_NOT_FOUND)

        agent_tools_info = self._agent_manager.get_enabled_tools(agent.id)

        return OpenAIAssistantInfo.from_agent_info(agent, agent_tools_info)

    def update_assistant(self, assistant_id: str, request: OpenAIUpdateAssistantRequest) -> OpenAIAssistantInfo:
        updated_agent = self._agent_manager.update(
            assistant_id,
            model_id=request.model,
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            default_generation_parameters=AgentGenerationParameters(
                temperature=request.temperature, top_p=request.top_p
            ),
            tools=to_agent_tool_requests(
                request.tools,
                request.tool_resources,
                reranker_model_id=self._openai_config.assistant_file_search_reranker_default_model_id,
            )[0],
            additional_data=request.metadata,
        )

        updated_agent_tools = self._agent_manager.get_enabled_tools(updated_agent.id)
        return OpenAIAssistantInfo.from_agent_info(updated_agent, updated_agent_tools)

    def delete_assistant(self, assistant_id: str) -> dict[str, Any]:
        self._agent_manager.delete(assistant_id)

        return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}
