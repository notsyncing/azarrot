import logging
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from threading import Thread
from typing import Any

from azarrot.backends.openvino_backend import OpenVINOBackend
from azarrot.config import OpenAIFrontendConfig, ServerConfig, VectorStoreConfig
from azarrot.server import Server, create_server
from tests.integration.utils import get_file_store


def make_no_backend_server() -> Generator[Server, Any, Any]:
    logging.basicConfig(level=logging.INFO)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name).absolute()

    server = create_server(
        config=ServerConfig(
            models_dir=tmp_path / "models",
            working_dir=tmp_path / "working",
            openai_configs=OpenAIFrontendConfig(vector_store_default_embedding_model_id="dummy_model"),
        ),
        enable_backends=[],
    )

    server.frontends[0].set_test_mode(test_resources_root=Path(__file__).resolve().parent / Path("../resources"))

    thread = Thread(target=server.start, daemon=True)
    thread.start()

    time.sleep(5)

    yield server

    server.stop()
    tmp_dir.cleanup()

    time.sleep(5)


def make_openvino_server(
    worker_file_process_scan_interval: int = 1, worker_store_cleanup_scan_interval: int = 1
) -> Generator[Server, Any, Any]:
    logging.basicConfig(level=logging.INFO)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name).absolute()

    server = create_server(
        config=ServerConfig(
            models_dir=tmp_path / "models",
            working_dir=tmp_path / "working",
            openai_configs=OpenAIFrontendConfig(vector_store_default_embedding_model_id="BAAI/bge-m3"),
            vector_store_configs=VectorStoreConfig(
                worker_file_process_scan_interval=worker_file_process_scan_interval,
                worker_store_cleanup_scan_interval=worker_store_cleanup_scan_interval,
            ),
        ),
        enable_backends=[OpenVINOBackend],
    )

    server.frontends[0].set_test_mode(test_resources_root=Path(__file__).resolve().parent / Path("../resources"))

    thread = Thread(target=server.start, daemon=True)
    thread.start()

    time.sleep(5)

    yield server

    server.stop()
    tmp_dir.cleanup()

    time.sleep(5)


def do_clear_database(server: Server) -> Generator[None, Any, Any]:
    yield None

    file_store = get_file_store(server)
    file_store.clear_database()

    server.vector_store.clear_database()
    server.agent_manager.clear_database()
