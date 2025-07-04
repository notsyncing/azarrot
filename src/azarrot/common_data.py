from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar, cast, override
from uu import Error

import dataclass_wizard

from azarrot.tools.tool import LocalizedToolDescription


@dataclass
class WorkingDirectories:
    root: Path
    uploaded_images: Path
    file_store: Path


@dataclass
class TransformersModelConfig:
    assistant_model: str | None = None


@dataclass
class OpenVINOQuantizationConfigs:
    bits: int
    sym: bool
    group_size: int | None
    ratio: float
    all_layers: bool | None
    quant_method: str
    weight_format: str | None


@dataclass
class OpenVINOModelConfig:
    quantization_configs: OpenVINOQuantizationConfigs | None


@dataclass
class PyTorchModelConfig:
    compile: bool
    compile_backend: str


@dataclass
class ModelPreset:
    preferred_locale: Literal["zh-cn", "en-us"] | None
    supports_tool_calling: bool
    enable_internal_tools: bool

    def with_enable_internal_tools(self) -> "ModelPreset":
        self.enable_internal_tools = True
        return self


@dataclass
class ModelQuirks:
    output_buffering_length: int = 0
    additional_stop_before_strings: list[str] | None = None
    does_not_support_batching: bool = False
    openvino_dont_patch_model_compile: bool = False

    reasoning_start_indicator: str | None = None
    reasoning_end_indicator: str | None = None

    def extend_with(self, **kwargs: Any) -> "ModelQuirks":
        current = cast("dict", dataclass_wizard.asdict(self))
        current.update(kwargs)
        return ModelQuirks(**current)


dataclass_wizard.DumpMeta(key_transform="NONE").bind_to(ModelQuirks)


class ModelInfo:
    pass


@dataclass
class EmbeddingModelInfo(ModelInfo):
    dimension: int


@dataclass
class Model:
    # The following properties are from the content of model file

    id: str
    backend: str
    path: Path
    task: str
    revision: str

    generation_variant: str
    preset: ModelPreset
    use_original_precision: bool
    is_for_raw_completion: bool
    is_reasoning_model: bool

    transformers: TransformersModelConfig | None

    openvino: OpenVINOModelConfig | None
    pytorch: PyTorchModelConfig | None

    # The following properties are computed at runtime

    device: str | None = None
    info: ModelInfo | None = None
    create_time: datetime = datetime.min


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


@dataclass
class ToolCallRequestMessageContents:
    tool_requests: list[ToolCallRequestMessageContent]


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
    max_tokens: int | None = None

    repetition_penalty: float = 1
    temperature: float = 1
    top_p: float = 1
    seed: int | None = None

    tools_info: CallableToolsInfo | None = None
    parallel_tool_calling: bool = True


@dataclass
class EmbeddingsGenerationRequest:
    model_id: str
    text: str | list[str]


@dataclass
class ToolCallResponse:
    order: int
    tool_result: str


class DifferentChunkError(Error):
    pass


class GeneratedMessageChunk(ABC):
    @abstractmethod
    def __add__(self, another: "GeneratedMessageChunk") -> "GeneratedMessageChunk":
        pass


class EmptyMessageChunk(GeneratedMessageChunk):
    @override
    def __add__(self, another: "GeneratedMessageChunk") -> "GeneratedMessageChunk":
        return another


@dataclass
class ReasoningGeneratedMessageChunk(GeneratedMessageChunk):
    content: str

    @override
    def __add__(self, another: GeneratedMessageChunk) -> "ReasoningGeneratedMessageChunk":
        if isinstance(another, EmptyMessageChunk):
            return self

        if not isinstance(another, ReasoningGeneratedMessageChunk):
            raise DifferentChunkError

        return ReasoningGeneratedMessageChunk(content=self.content + another.content)


@dataclass
class TextGeneratedMessageChunk(GeneratedMessageChunk):
    content: str

    @override
    def __add__(self, another: GeneratedMessageChunk) -> "TextGeneratedMessageChunk":
        if isinstance(another, EmptyMessageChunk):
            return self

        if not isinstance(another, TextGeneratedMessageChunk):
            raise DifferentChunkError

        return TextGeneratedMessageChunk(content=self.content + another.content)


@dataclass
class ToolCallGeneratedMessageChunk(GeneratedMessageChunk):
    index: int
    name: str | None
    name_completed: bool
    arguments: str

    @override
    def __add__(self, another: GeneratedMessageChunk) -> "ToolCallGeneratedMessageChunk":
        if isinstance(another, EmptyMessageChunk):
            return self

        if not isinstance(another, ToolCallGeneratedMessageChunk):
            raise DifferentChunkError

        if self.index != another.index:
            raise DifferentChunkError

        return ToolCallGeneratedMessageChunk(
            index=another.index,
            name=(self.name or "") + (another.name or ""),
            name_completed=another.name_completed,
            arguments=self.arguments + another.arguments,
        )


@dataclass
class GenerationStatistics:
    start_time: datetime
    first_token_time: datetime
    end_time: datetime
    prompt_tokens: int
    completion_tokens: int

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_stats_text(self) -> str:
        time_delta = (self.end_time - self.start_time) / timedelta(milliseconds=1)
        ftt = (self.first_token_time - self.start_time) / timedelta(milliseconds=1)

        total_tokens = self.prompt_tokens + self.completion_tokens
        speed = self.completion_tokens / time_delta * 1000
        prefill_speed = self.prompt_tokens / ftt * 1000
        decode_speed = self.completion_tokens / (time_delta - ftt) * 1000

        return (
            f"Total tokens: {total_tokens} (prompt {self.prompt_tokens}, completion {self.completion_tokens}), "
            f"first token latency: {ftt} ms, cost {time_delta} ms, {speed:.3f} tok/s (prefill {prefill_speed:.3f} "
            f"tok/s, decode {decode_speed:.3f} tok/s)"
        )


@dataclass
class ModelToolCallExtractedInfo:
    name: str | None = None
    name_completed: bool = False
    arguments: str | None = None


@dataclass
class ModelToolCallConfig:
    prompts: dict[str, str] | None = None

    tool_call_start_indicator: str | None = None
    tool_call_end_indicator: str | None = None
    tool_call_info_extracting_method: (
        Callable[["ModelToolCallConfig", dict[str, Any], str], ModelToolCallExtractedInfo] | None
    ) = None
    tool_call_arguments_wrapped_by_string: bool = False


PR_T = TypeVar("PR_T")


@dataclass
class PageResult[PR_T]:
    data: list[PR_T]
    is_last_page: bool


@dataclass
class ReranksGenerationRequest:
    model_id: str
    query: str
    documents: list[str]
    max_count: int | None = None
    max_tokens_per_document: int | None = None


@dataclass
class RerankResultItem:
    input_index: int
    score: float
