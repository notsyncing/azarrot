import logging
import tempfile
import time
from collections.abc import Generator
from os import environ
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from azarrot.backends.pytorch_backend import PyTorchBackend
from azarrot.config import ENV_AZARROT_TEST_MODE, ENV_AZARROT_TEST_RESOURCES_ROOT, ServerConfig
from azarrot.server import Server, create_server


@pytest.fixture(scope="module")
def pytorch_server() -> Generator[Server, Any, Any]:
    logging.basicConfig(level=logging.INFO)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name).absolute()

    config = ServerConfig(
        models_dir=tmp_path / "models", working_dir=tmp_path / "working", huggingface_download_to_home=True
    )

    server = create_server(
        config=config,
        enable_backends=[PyTorchBackend(config)],
        enable_schedule_thread=False,
    )

    environ[ENV_AZARROT_TEST_MODE] = "True"
    environ[ENV_AZARROT_TEST_RESOURCES_ROOT] = str(Path(__file__).resolve().parent / Path("../resources"))

    thread = Thread(target=server.start, daemon=True)
    thread.start()

    time.sleep(5)

    yield server

    server.stop()
    tmp_dir.cleanup()

    time.sleep(5)
