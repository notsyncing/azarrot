from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from azarrot.config import DEFAULT_MAX_TOKENS


@dataclass
class Model:
    # The following properties are from the content of model file

    id: str
    backend: str
    path: Path
    task: str

    # The following properties are computed at runtime

    create_time: datetime


@dataclass
class GenerationMessage:
    role: str
    content: str


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
    end_time: datetime
    prompt_tokens: int
    total_tokens: int
