from collections.abc import Generator
from typing import Any

import pytest

from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_openvino_server
from tests.integration.utils import create_openai_client

EMBEDDING_MODEL_ID = "BAAI/bge-m3"


@pytest.fixture(scope="module")
def openvino_server() -> Generator[Server, Any, Any]:
    yield from make_openvino_server()


@pytest.fixture(autouse=True)
def cleanup_database(openvino_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(openvino_server)


@pytest.fixture(autouse=True)
def prepare_model_and_store(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        EMBEDDING_MODEL_ID,
        BACKEND_ID_OPENVINO,
        "feature-extraction",
        skip_if_loaded=True,
        override_model_id="bge-m3"
    )


def test_get_models(openvino_server: Server) -> None:
    client = create_openai_client(openvino_server)
    page = client.models.list()
    assert len(page.data) == 1
    assert page.data[0].id == "bge-m3"
    assert page.data[0].created > 0


def test_get_model(openvino_server: Server) -> None:
    client = create_openai_client(openvino_server)
    model = client.models.retrieve("bge-m3")
    assert model is not None
    assert model.id == "bge-m3"
    assert model.created > 0
