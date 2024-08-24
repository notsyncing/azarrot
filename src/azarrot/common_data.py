from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from azarrot.config import DEFAULT_MAX_TOKENS
from azarrot.tools.tool import LocalizedToolDescription


@dataclass
class WorkingDirectories:
    root: Path
    uploaded_images: Path


@dataclass
class IPEXLLMModelConfig:
    use_cache: bool


@dataclass
class ModelPreset:
    preferred_locale: Literal["zh-cn", "en-us"] | None
    supports_tool_calling: bool
    enable_internal_tools: bool


@dataclass
class ModelQuirks:
    output_buffering_length = 10
    additional_stop_before_strings: list[str] | None = None
    full_text_indicators: list[str] | None = None


@dataclass
class Model:
    # The following properties are from the content of model file

    id: str
    backend: str
    path: Path
    task: str

    generation_variant: Literal["normal", "internvl2", "qwen2"]
    preset: ModelPreset

    ipex_llm: IPEXLLMModelConfig | None

    # The following properties are computed at runtime

    create_time: datetime


@dataclass
class GenerationMessageContent:
    pass


@dataclass
class TextGenerationMessageContent(GenerationMessageContent):
    text: str


@dataclass
class ImageGenerationMessageContent(GenerationMessageContent):
    image_file_path: str


@dataclass
class ToolCallRequestMessageContent(GenerationMessageContent):
    id: str
    function_name: str
    function_arguments: dict[str, Any]


class ToolCallRequestMessageContentList(list[ToolCallRequestMessageContent]):
    pass


@dataclass
class ToolCallResponseMessageContent(GenerationMessageContent):
    to_id: str
    result: str


@dataclass
class GenerationMessage:
    role: str
    contents: list[GenerationMessageContent]


@dataclass
class CallableToolsInfo:
    tools: list[LocalizedToolDescription]
    force_use_no_tool: bool
    force_use_any_tool: bool
    force_use_tool_name: str | None


@dataclass
class TextGenerationRequest:
    model_id: str
    messages: list[GenerationMessage]
    max_tokens: int = DEFAULT_MAX_TOKENS

    repetition_penalty: float = 1
    temperature: float = 1
    top_p: float = 1

    tools_info: CallableToolsInfo | None = None
    parallel_tool_calling: bool = True


@dataclass
class EmbeddingsGenerationRequest:
    model_id: str
    text: str


@dataclass
class ToolCallResponse:
    order: int
    tool_result: str


@dataclass
class GenerationStatistics:
    start_time: datetime
    first_token_time: datetime
    end_time: datetime
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ModelToolCallConfig:
    prompts: dict[str, str]
    indicators: list[str]
    request_parsing_method: Callable[[str], list[ToolCallRequestMessageContent]]
    request_formatting_method: Callable[[list[ToolCallRequestMessageContent]], str]
    response_formatting_method: Callable[[list[ToolCallResponseMessageContent]], str]
