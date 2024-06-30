from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Model:
    # The following properties are from the content of model file

    id: str
    path: Path
    task: str

    # The following properties are computed at runtime

    create_time: datetime
