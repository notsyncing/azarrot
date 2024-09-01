import logging
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from azarrot.backends.ipex_llm_backend import IPEXLLMBackend
from azarrot.config import ServerConfig
from azarrot.server import Server, create_server


@pytest.fixture(scope="module")
def ipex_llm_server() -> Generator[Server, Any, Any]:
    logging.basicConfig(level=logging.INFO)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name).absolute()

    server = create_server(
        config=ServerConfig(
            models_dir=tmp_path / "models",
            working_dir=tmp_path / "working"
        ),
        enable_backends=[IPEXLLMBackend]
    )

    thread = Thread(target=server.start, daemon=True)
    thread.start()

    time.sleep(5)

    yield server

    tmp_dir.cleanup()
