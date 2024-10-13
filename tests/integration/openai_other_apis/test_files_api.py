from io import BytesIO
from pathlib import Path

from azarrot.server import Server
from tests.integration.utils import create_openai_client, create_temp_file, get_file_store


def test_upload_file(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    file_content = "test content"

    with create_temp_file(file_content) as f:
        fo = client.files.create(file=f.file, purpose="assistants")

    assert fo.id is not None
    assert len(fo.id) == 36
    assert fo.object == "file"
    assert fo.bytes == len(file_content)
    assert isinstance(fo.created_at, int)
    assert fo.filename == Path(f.name).name
    assert fo.purpose == "assistants"

    file_store = get_file_store(no_backend_server)
    stored_file_info = file_store.get_file_info(fo.id)
    stored_file_path = file_store._make_store_file_path(fo.id)

    assert stored_file_path is not None
    assert stored_file_path.read_text() == file_content
    assert stored_file_info is not None
    assert stored_file_info.id == fo.id
    assert stored_file_info.filename == fo.filename
    assert stored_file_info.size == fo.bytes
    assert stored_file_info.purpose == fo.purpose


def test_get_file_list(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    file_store.clear_database()

    file_content = "test content"
    file_content_data = file_content.encode("utf-8")

    file1 = file_store.store_file("file1.txt", "assistants", "text/plain", BytesIO(file_content_data))
    file2 = file_store.store_file("file2.txt", "assistants", "text/plain", BytesIO(file_content_data))
    file3 = file_store.store_file("file3.txt", "assistants", "text/plain", BytesIO(file_content_data))

    client = create_openai_client(no_backend_server)
    file_list = client.files.list()

    assert file_list is not None
    assert len(file_list.data) == 3
    assert file_list.data[0].id == file1.id
    assert file_list.data[0].filename == "file1.txt"
    assert file_list.data[1].id == file2.id
    assert file_list.data[1].filename == "file2.txt"
    assert file_list.data[2].id == file3.id
    assert file_list.data[2].filename == "file3.txt"


def test_get_file_info(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    file_store.clear_database()

    filename = "test.txt"
    file_content = b"Hello, World!"
    file_info = file_store.store_file(filename, "assistants", "text/plain", BytesIO(file_content))

    client = create_openai_client(no_backend_server)
    retrieved_file = client.files.retrieve(file_info.id)

    assert retrieved_file is not None
    assert retrieved_file.id == file_info.id
    assert retrieved_file.filename == filename
    assert retrieved_file.purpose == file_info.purpose
    assert retrieved_file.bytes == len(file_content)


def test_delete_file(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    file_store.clear_database()

    filename = "test.txt"
    file_content = b"Hello, World!"
    file_info = file_store.store_file(filename, "assistants", "text/plain", BytesIO(file_content))
    file_path = file_store._make_store_file_path(file_info.id)

    assert file_path is not None

    client = create_openai_client(no_backend_server)
    result = client.files.delete(file_info.id)

    assert result is not None
    assert result.id == file_info.id
    assert result.deleted is True
    assert file_path.exists() is False

    file_info_2 = file_store.get_file_info(file_info.id)
    assert file_info_2 is None


def test_get_file_content(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    file_store.clear_database()

    filename = "test.txt"
    file_content = b"Hello, World!"
    file_info = file_store.store_file(filename, "assistants", "text/plain", BytesIO(file_content))

    client = create_openai_client(no_backend_server)
    actual_content = client.files.content(file_info.id)

    assert actual_content is not None
    assert actual_content.content == file_content
