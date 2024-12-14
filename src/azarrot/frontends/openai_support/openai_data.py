import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

OPENAI_TOOL_CODE_INTERPRETER: Literal["code_interpreter"] = "code_interpreter"
OPENAI_TOOL_FILE_SEARCH: Literal["file_search"] = "file_search"
OPENAI_TOOL_FUNCTION: Literal["function"] = "function"


@dataclass
class OpenAIList(Generic[T]):
    data: list[T]
    object: str = "list"


class UserChatImageUrl(BaseModel):
    url: str
    detail: Literal["low", "high", "auto"] = "auto"


class UserChatTextContentItem(BaseModel):
    type: Literal["text"]
    text: str


class UserChatImageContentItem(BaseModel):
    type: Literal["image_url"]
    image_url: str | UserChatImageUrl


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class AssistantToolCallRequest(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class SystemChatCompletionMessage(BaseModel):
    name: str | None = None
    content: str
    role: Literal["system"]


class UserChatCompletionMessage(BaseModel):
    name: str | None = None
    content: str | list[Annotated[UserChatTextContentItem | UserChatImageContentItem, Field(discriminator="type")]]
    role: Literal["user"]


class AssistantChatCompletionMessage(BaseModel):
    name: str | None = None
    content: str | None = None
    role: Literal["assistant"]
    tool_calls: list[AssistantToolCallRequest] | None = None


class ToolChatCompletionMessage(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class ChatCompletionStreamOptions(BaseModel):
    include_usage: bool = False


class ToolFunctionInfo(BaseModel):
    description: str | None = None
    name: str
    parameters: dict[str, Any] | None = None


class ToolInfo(BaseModel):
    type: Literal["function"]
    function: ToolFunctionInfo


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoice(BaseModel):
    type: Literal["function", "file_search"]
    function: ToolChoiceFunction | None


OpenAIToolChoiceConstant = Literal["none", "auto", "required"]


class ChatCompletionRequest(BaseModel):
    messages: list[
        Annotated[
            SystemChatCompletionMessage
            | UserChatCompletionMessage
            | AssistantChatCompletionMessage
            | ToolChatCompletionMessage,
            Field(discriminator="role"),
        ]
    ]

    model: str
    max_tokens: int | None = None
    stream: bool = False
    stream_options: ChatCompletionStreamOptions = Field(default=ChatCompletionStreamOptions())

    frequency_penalty: float = Field(default=0, ge=-2.0, le=2.0)
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    seed: int | None = None

    tools: list[ToolInfo] | None = None
    tool_choice: OpenAIToolChoiceConstant | ToolChoice | None = None
    parallel_tool_calls: bool = True


class CreateEmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = Field(default="float")
    dimensions: int | None = None
    user: str | None = None


class OpenAIVectorStoreAutoChunkingStrategy(BaseModel):
    type: Literal["auto"] = "auto"


class OpenAIVectorStoreStaticChunkingStrategyConfigs(BaseModel):
    max_chunk_size_tokens: int = Field(ge=100, le=4096)
    chunk_overlap_tokens: int


class OpenAIVectorStoreStaticChunkingStrategy(BaseModel):
    type: Literal["static"] = "static"
    static: OpenAIVectorStoreStaticChunkingStrategyConfigs


OpenAIVectorStoreChunkingStrategy = OpenAIVectorStoreAutoChunkingStrategy | OpenAIVectorStoreStaticChunkingStrategy


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


OpenAIAssistantTool = OpenAICodeInterpreterTool | OpenAIFileSearchTool | OpenAIFunctionTool


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


@dataclass
class OpenAILastError:
    code: Literal["server_error", "rate_limit_exceeded", "invalid_prompt"]
    message: str


@dataclass
class OpenAITokenUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class OpenAIToolCallFunction:
    name: str
    arguments: str


@dataclass
class OpenAIToolCallRequest:
    id: str
    type: Literal["function"]
    function: OpenAIToolCallFunction
