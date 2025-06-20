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
    yield from make_openvino_server(worker_file_process_scan_interval=99999, worker_store_cleanup_scan_interval=99999)


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
    openvino_server.vector_store.create("test_store", model, store_id=TEST_STORE_ID)

    openvino_server.file_store.store_file(
        "test1.txt", None, None, BytesIO(b"A183743 is a type of aircraft."), file_id=TEST_FILE1_ID
    )

    openvino_server.file_store.store_file(
        "test2.txt", None, None, BytesIO(b"B637R is a type of car."), file_id=TEST_FILE2_ID
    )

    openvino_server.file_store.store_file(
        "test3.txt", None, None, BytesIO(b"C558 is a type of train."), file_id=TEST_FILE3_ID
    )


def test_create_vector_store_file_batch(openvino_server: Server) -> None:
    client = create_openai_client(openvino_server)
    store = openvino_server.vector_store.get_store_info(TEST_STORE_ID)
    assert store is not None

    batch = client.vector_stores.file_batches.create(
        vector_store_id=store.id, file_ids=[str(TEST_FILE1_ID), str(TEST_FILE2_ID), str(TEST_FILE3_ID)]
    )

    file1_info = openvino_server.vector_store.get_stored_file_info(store.id, TEST_FILE1_ID)
    assert file1_info is not None

    batch_info = openvino_server.vector_store.get_batch_status(TEST_STORE_ID, batch.id)
    assert batch_info is not None

    assert batch.id == batch_info.batch_id
    assert batch.created_at == int(file1_info.create_time.timestamp())
    assert batch.vector_store_id == str(TEST_STORE_ID)
    assert batch.status == "in_progress"
    assert batch.file_counts.total == 3
    assert batch.file_counts.in_progress == 3


def test_retrieve_vector_store_file_batch(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(
        TEST_STORE_ID, [TEST_FILE1_ID, TEST_FILE2_ID, TEST_FILE3_ID], state="completed", batch_id="batch1"
    )

    client = create_openai_client(openvino_server)

    batch = client.vector_stores.file_batches.retrieve(vector_store_id=str(TEST_STORE_ID), batch_id="batch1")

    file_info = openvino_server.vector_store.get_stored_file_info(TEST_STORE_ID, TEST_FILE1_ID)
    assert file_info is not None

    assert batch.id == "batch1"
    assert batch.created_at == int(file_info.create_time.timestamp())
    assert batch.vector_store_id == str(TEST_STORE_ID)
    assert batch.status == "completed"
    assert batch.file_counts.total == 3
    assert batch.file_counts.completed == 3


def test_cancel_vector_store_file_batch(openvino_server: Server) -> None:
    with openvino_server.vector_store_worker.pause():
        openvino_server.vector_store.add_stored_files(
            TEST_STORE_ID, [TEST_FILE1_ID], state="pending", batch_id="batch1"
        )

        client = create_openai_client(openvino_server)

        cancelled_batch = client.vector_stores.file_batches.cancel(
            vector_store_id=str(TEST_STORE_ID), batch_id="batch1"
        )

        assert cancelled_batch.id == "batch1"
        assert cancelled_batch.status == "cancelled"
        assert cancelled_batch.file_counts.total == 1
        assert cancelled_batch.file_counts.cancelled == 1


def test_list_vector_store_files(openvino_server: Server) -> None:
    openvino_server.vector_store.add_stored_files(
        TEST_STORE_ID, [TEST_FILE1_ID, TEST_FILE2_ID, TEST_FILE3_ID], state="completed", batch_id="batch1"
    )

    client = create_openai_client(openvino_server)

    batch_files = client.vector_stores.file_batches.list_files(
        vector_store_id=str(TEST_STORE_ID), batch_id="batch1", order="asc"
    )

    assert len(batch_files.data) == 3
    assert any(d.id == str(TEST_FILE1_ID) for d in batch_files.data) is True
    assert any(d.id == str(TEST_FILE2_ID) for d in batch_files.data) is True
    assert any(d.id == str(TEST_FILE3_ID) for d in batch_files.data) is True
    assert batch_files.model_dump()["has_more"] is False
