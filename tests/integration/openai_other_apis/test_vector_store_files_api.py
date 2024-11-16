import time
import uuid
from collections.abc import Generator
from io import BytesIO
from typing import Any

import pytest

from azarrot.backends.openvino_backend import BACKEND_ID_OPENVINO
from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_openvino_server
from tests.integration.utils import create_openai_client

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
TEST_STORE_ID = uuid.uuid4()
TEST_FILE1_ID = uuid.uuid4()
TEST_FILE2_ID = uuid.uuid4()
TEST_FILE3_ID = uuid.uuid4()


@pytest.fixture(scope="module")
def openvino_server() -> Generator[Server, Any, Any]:
    yield from make_openvino_server()


@pytest.fixture(autouse=True)
def cleanup_database(openvino_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(openvino_server)


@pytest.fixture(autouse=True)
def prepare_model_and_store(openvino_server: Server) -> None:
    openvino_server.model_manager.load_huggingface_model(
        EMBEDDING_MODEL_ID, BACKEND_ID_OPENVINO, "feature-extraction", skip_if_loaded=True
    )

    model = openvino_server.model_manager.get_model(EMBEDDING_MODEL_ID)
    assert model is not None
    store = openvino_server.vector_store.create("test_store", model, store_id=TEST_STORE_ID)
    assert store is not None

    openvino_server.file_store.store_file(
        "test1.txt", None, None, BytesIO(b"A183743 is a type of aircraft."), file_id=TEST_FILE1_ID
    )

    openvino_server.file_store.store_file(
        "test2.txt", None, None, BytesIO(b"B637R is a type of car."), file_id=TEST_FILE2_ID
    )

    openvino_server.file_store.store_file(
        "test3.txt", None, None, BytesIO(b"C558 is a type of train."), file_id=TEST_FILE3_ID
    )


def test_create_vector_store_file(openvino_server: Server) -> None:
    client = create_openai_client(openvino_server)
    store = openvino_server.vector_store.get_store_info(TEST_STORE_ID)
    assert store is not None

    vector_store_file = client.beta.vector_stores.files.create(vector_store_id=store.id, file_id=str(TEST_FILE1_ID))

    file1_info = openvino_server.vector_store.get_stored_file_info(store.id, TEST_FILE1_ID)
    assert file1_info is not None

    assert vector_store_file.id == str(TEST_FILE1_ID)
    assert vector_store_file.created_at == int(file1_info.create_time.timestamp())
    assert vector_store_file.usage_bytes == 0
    assert vector_store_file.vector_store_id == str(TEST_STORE_ID)
    assert vector_store_file.status == "in_progress"
    assert vector_store_file.last_error is None

    completed = False
    counter = 0

    while not completed:
        time.sleep(5)
        info = openvino_server.vector_store.get_stored_file_info(TEST_STORE_ID, TEST_FILE1_ID)
        assert info is not None
        completed = info.state not in ("pending", "processing")
        counter = counter + 1

        if counter > 10:
            raise RuntimeError("Timeout waiting for vector store file process!")

    file1_info = openvino_server.vector_store.get_stored_file_info(store.id, TEST_FILE1_ID)
    assert file1_info is not None
    assert file1_info.state == "completed"
    assert file1_info.failed_message is None


def test_list_vector_store_files(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(
        TEST_STORE_ID, [TEST_FILE1_ID, TEST_FILE2_ID, TEST_FILE3_ID], state="completed"
    )

    client = create_openai_client(openvino_server)

    vector_store_files = client.beta.vector_stores.files.list(vector_store_id=str(TEST_STORE_ID), order="asc")

    assert len(vector_store_files.data) == 3
    assert any(d.id == str(TEST_FILE1_ID) for d in vector_store_files.data) is True
    assert any(d.id == str(TEST_FILE2_ID) for d in vector_store_files.data) is True
    assert any(d.id == str(TEST_FILE3_ID) for d in vector_store_files.data) is True
    assert vector_store_files.model_dump()["has_more"] is False


def test_retrieve_vector_store_file(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(TEST_STORE_ID, [TEST_FILE1_ID], state="completed")

    client = create_openai_client(openvino_server)

    vector_store_file = client.beta.vector_stores.files.retrieve(
        vector_store_id=str(TEST_STORE_ID), file_id=str(TEST_FILE1_ID)
    )

    file_info = openvino_server.vector_store.get_stored_file_info(TEST_STORE_ID, TEST_FILE1_ID)
    assert file_info is not None

    assert vector_store_file.id == str(TEST_FILE1_ID)
    assert vector_store_file.created_at == int(file_info.create_time.timestamp())
    assert vector_store_file.vector_store_id == str(TEST_STORE_ID)
    assert vector_store_file.status == "completed"
    assert vector_store_file.last_error is None


def test_delete_vector_store_file(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(TEST_STORE_ID, [TEST_FILE1_ID], state="completed")

    client = create_openai_client(openvino_server)

    deleted_vector_store_file = client.beta.vector_stores.files.delete(
        vector_store_id=str(TEST_STORE_ID), file_id=str(TEST_FILE1_ID)
    )

    assert deleted_vector_store_file.id == str(TEST_FILE1_ID)
    assert deleted_vector_store_file.deleted is True

    file_info = openvino_server.vector_store.get_stored_file_info(TEST_STORE_ID, TEST_FILE1_ID)
    assert file_info is None


def test_delete_vector_store_file_with_processing_file(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(TEST_STORE_ID, [TEST_FILE1_ID], state="processing")

    client = create_openai_client(openvino_server)

    deleted_vector_store_file = client.beta.vector_stores.files.delete(
        vector_store_id=str(TEST_STORE_ID), file_id=str(TEST_FILE1_ID)
    )

    assert deleted_vector_store_file.id == str(TEST_FILE1_ID)
    assert deleted_vector_store_file.deleted is False

    file_info = openvino_server.vector_store.get_stored_file_info(TEST_STORE_ID, TEST_FILE1_ID)
    assert file_info is not None
