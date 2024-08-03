from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


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
    parameters: list[dict[str, Any]] | None = None


class ToolInfo(BaseModel):
    type: Literal["function"]
    function: ToolFunctionInfo


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: ToolChoiceFunction


class ChatCompletionRequest(BaseModel):
    messages: list[
        Annotated[
            SystemChatCompletionMessage | UserChatCompletionMessage | AssistantChatCompletionMessage
                | ToolChatCompletionMessage,
            Field(discriminator="role")
        ]
    ]

    model: str
    max_tokens: int | None = None
    stream: bool = False
    stream_options: ChatCompletionStreamOptions = Field(default=ChatCompletionStreamOptions())

    tools: list[ToolInfo] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None
    parallel_tool_calls: bool = True


class CreateEmbeddingsRequest(BaseModel):
    input: str
    model: str
    encoding_format: Literal["float", "base64"] = Field(default="float")
    dimensions: int | None = None
    user: str | None = None
