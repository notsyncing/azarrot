from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_TOKENS = 512


@dataclass
class ServerConfig:
    models_dir: Path = Path("./models")
    working_dir: Path = Path("./working")
    host: str = "127.0.0.1"
    port: int = 8080

    model_device_map: dict[str, str] = field(default_factory=dict)
    single_token_generation_timeout: int = 60000
    auto_batch_threshold: int = 100
    auto_batch_max_size: int = 8

    partial_file_expire_time: int = 3600 * 1000
