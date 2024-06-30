from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

DEFAULT_MAX_TOKENS = 512


@dataclass
class ServerConfig:
    models_dir = Path("./models")
    working_dir = Path("./working")
    host = "127.0.0.1"
    port = 8080

    model_device_map: ClassVar[dict[str, str]] = {}
