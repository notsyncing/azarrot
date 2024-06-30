from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    models_dir: Path
    working_dir: Path
    host: str
    port: int

