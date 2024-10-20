from io import BytesIO

from azarrot.server import Server
from tests.integration.utils import create_openai_client, create_temp_file, get_file_store


def test_create_upload(no_backend_server: Server) -> None:
    client = create_openai_client(no_backend_server)

    r = client.uploads.create(bytes=100, filename="test.txt", mime_type="text/plain", purpose="assistants")

    assert r is not None
    assert r.filename == "test.txt"
    assert r.bytes == 100
    assert r.purpose == "assistants"

    stored_partial_file_info = get_file_store(no_backend_server).get_partial_file_info(r.id)

    assert stored_partial_file_info is not None
    assert stored_partial_file_info.id == r.id
    assert stored_partial_file_info.filename == r.filename
    assert stored_partial_file_info.size == r.bytes
    assert stored_partial_file_info.purpose == r.purpose
    assert stored_partial_file_info.mime_type == "text/plain"
    assert int(stored_partial_file_info.create_time.timestamp()) == r.created_at
    assert int(stored_partial_file_info.expire_time.timestamp()) == r.expires_at


def test_add_upload_part(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    stored_partial_file_info = file_store.create_partial_file("test.txt", 100, "text/plain", "assistant")

    client = create_openai_client(no_backend_server)

    file_content = "test content"

    with create_temp_file(file_content) as f:
        r = client.uploads.parts.create(stored_partial_file_info.id, data=f.file)

    assert r is not None

    file_part = file_store.get_partial_file_part_info(r.upload_id, r.id)
    assert file_part is not None

    file = file_store._make_store_file_part_path(file_part.id)
    actual_content = file.read_text()
    assert actual_content == file_content


def test_complete_upload(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    stored_partial_file_info = file_store.create_partial_file("test.txt", 24, "text/plain", "assistant")
    part1_info = file_store.add_part_to_partial_file(stored_partial_file_info.id, BytesIO(b"content1"))
    part2_info = file_store.add_part_to_partial_file(stored_partial_file_info.id, BytesIO(b"content2"))
    part3_info = file_store.add_part_to_partial_file(stored_partial_file_info.id, BytesIO(b"content3"))

    client = create_openai_client(no_backend_server)

    r = client.uploads.complete(
        stored_partial_file_info.id, part_ids=[str(part2_info.id), str(part1_info.id), str(part3_info.id)]
    )

    assert r is not None

    merged_file_info = file_store.get_file_info(r.id)
    assert merged_file_info is not None
    assert r.filename == "test.txt"
    assert r.bytes == 24
    assert r.file is not None
    assert r.file.id == merged_file_info.id
    assert r.file.bytes == 24

    _, content = file_store.get_file_content(merged_file_info.id)
    assert content is not None

    with content:
        assert content.read() == b"content2content1content3"

    r2 = client.files.content(merged_file_info.id)
    assert r2.content == b"content2content1content3"


def test_cancel_upload(no_backend_server: Server) -> None:
    file_store = get_file_store(no_backend_server)
    stored_partial_file_info = file_store.create_partial_file("test.txt", 24, "text/plain", "assistant")
    part1_info = file_store.add_part_to_partial_file(stored_partial_file_info.id, BytesIO(b"content1"))

    client = create_openai_client(no_backend_server)

    r = client.uploads.cancel(stored_partial_file_info.id)

    assert r is not None
    assert r.id == stored_partial_file_info.id
    assert r.bytes == stored_partial_file_info.size
    assert r.filename == stored_partial_file_info.filename

    file_part = file_store.get_partial_file_part_info(stored_partial_file_info.id, r.id)
    assert file_part is None

    file = file_store._make_store_file_part_path(part1_info.id)
    assert file.exists() is False
