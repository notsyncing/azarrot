from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_TOKENS = 512


@dataclass
class ServerConfig:
    models_dir = Path("./models")
    working_dir = Path("./working")
    host = "127.0.0.1"
    port = 8080

    model_device_map: dict[str, str] = field(default_factory=dict)
