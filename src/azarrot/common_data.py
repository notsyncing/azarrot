from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from azarrot.config import DEFAULT_MAX_TOKENS


@dataclass
class WorkingDirectories:
    root: Path
    uploaded_images: Path


@dataclass
class IPEXLLMModelConfig:
    use_cache: bool
    generation_variant: Literal["normal", "internvl2"]


@dataclass
class Model:
    # The following properties are from the content of model file

    id: str
    backend: str
    path: Path
    task: str

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
class GenerationMessage:
    role: str
    content: list[GenerationMessageContent]


@dataclass
class TextGenerationRequest:
    model_id: str
    messages: list[GenerationMessage]
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class EmbeddingsGenerationRequest:
    model_id: str
    text: str


@dataclass
class GenerationStatistics:
    start_time: datetime
    first_token_time: datetime
    end_time: datetime
    prompt_tokens: int
    completion_tokens: int
