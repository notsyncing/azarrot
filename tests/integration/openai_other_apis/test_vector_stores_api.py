from collections.abc import Generator
from typing import Any

import pytest
from openai.types.beta.vector_store_create_params import ExpiresAfter

from azarrot.server import Server
from tests.integration.openai_other_apis.fixture_utils import do_clear_database, make_no_backend_server
from tests.integration.utils import create_fake_embedding_model, create_openai_client


@pytest.fixture(scope="module")
def no_backend_server() -> Generator[Server, Any, Any]:
    yield from make_no_backend_server()


@pytest.fixture(autouse=True)
def cleanup_database(no_backend_server: Server) -> Generator[None, Any, Any]:
    yield from do_clear_database(no_backend_server)


def test_create_vector_store(no_backend_server: Server) -> None:
    no_backend_server.model_manager.add_model(create_fake_embedding_model("dummy_model"))

    client = create_openai_client(no_backend_server)

    vector_store = client.beta.vector_stores.create(
        name="Support FAQ", expires_after=ExpiresAfter(anchor="last_active_at", days=10)
    )

    db_vector_store = no_backend_server.vector_store.get_store_info(vector_store.id)
    assert db_vector_store is not None
    assert db_vector_store.name == "Support FAQ"
    assert db_vector_store.embedding_model == "dummy_model"
    assert db_vector_store.embedding_dimension == 1024
    assert db_vector_store.expire_baseline == "access_time"
    assert db_vector_store.expire_interval == 10

    assert vector_store.name == "Support FAQ"
    assert int(db_vector_store.create_time.timestamp()) == vector_store.created_at
    assert int(db_vector_store.access_time.timestamp()) == vector_store.last_active_at
    assert vector_store.usage_bytes == 0
    assert vector_store.file_counts is not None
    assert vector_store.file_counts.in_progress == 0
    assert vector_store.file_counts.completed == 0
    assert vector_store.file_counts.failed == 0
    assert vector_store.file_counts.cancelled == 0
    assert vector_store.file_counts.total == 0


def test_list_vector_stores(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    vector_stores = client.beta.vector_stores.list()
    assert len(vector_stores.data) == 0
    assert vector_stores.model_dump()["has_more"] is False

    no_backend_server.vector_store.create("test1", create_fake_embedding_model("model1"))
    no_backend_server.vector_store.create("test2", create_fake_embedding_model("model2"))
    no_backend_server.vector_store.create("test3", create_fake_embedding_model("model3"))

    vector_stores = client.beta.vector_stores.list(order="asc")
    assert len(vector_stores.data) == 3
    assert vector_stores.data[0].name == "test1"
    assert vector_stores.data[1].name == "test2"
    assert vector_stores.data[2].name == "test3"
    assert vector_stores.model_dump()["has_more"] is False  # has_next_page() sucks!

    vector_stores = client.beta.vector_stores.list(order="asc", limit=2)
    assert len(vector_stores.data) == 2
    assert vector_stores.data[0].name == "test1"
    assert vector_stores.data[1].name == "test2"
    assert vector_stores.model_dump()["has_more"] is True

    vector_stores = vector_stores.get_next_page()
    assert len(vector_stores.data) == 1
    assert vector_stores.data[0].name == "test3"
    assert vector_stores.model_dump()["has_more"] is False

    vector_stores = client.beta.vector_stores.list(order="desc")
    assert vector_stores.data[0].name == "test3"
    assert vector_stores.data[1].name == "test2"
    assert vector_stores.data[2].name == "test1"


def test_retrieve_vector_store(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    info = no_backend_server.vector_store.create("test1", create_fake_embedding_model("model1"))

    vector_store = client.beta.vector_stores.retrieve(vector_store_id=info.id)

    assert vector_store.id == info.id
    assert vector_store.name == info.name


def test_modify_vector_store(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    info = no_backend_server.vector_store.create("test1", create_fake_embedding_model("model1"))

    vector_store = client.beta.vector_stores.update(vector_store_id=info.id, name="Support FAQ")

    info_new = no_backend_server.vector_store.get_store_info(info.id)
    assert info_new is not None
    assert vector_store.name == "Support FAQ"
    assert info_new.name == "Support FAQ"


def test_delete_vector_store(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    info = no_backend_server.vector_store.create("test1", create_fake_embedding_model("model1"))

    deleted_vector_store = client.beta.vector_stores.delete(vector_store_id=info.id)

    assert deleted_vector_store.deleted is True

    info_new = no_backend_server.vector_store.get_store_info(info.id)
    assert info_new is None
