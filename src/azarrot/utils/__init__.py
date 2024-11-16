import hashlib
import uuid
from io import BytesIO
from typing import BinaryIO


def sanitize_uuid(value: str | uuid.UUID) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value

    return uuid.UUID(value)


def compute_sha256(data: BinaryIO | str) -> str:
    if isinstance(data, str):
        data = BytesIO(data.encode())

    hasher = hashlib.sha256()

    while True:
        chunk = data.read(4096)

        if not chunk:
            break

        hasher.update(chunk)

    data.seek(0)

    return hasher.hexdigest()


def compute_md5(data: BinaryIO | str) -> str:
    if isinstance(data, str):
        data = BytesIO(data.encode())

    hasher = hashlib.md5()  # noqa: S324

    while True:
        chunk = data.read(4096)

        if not chunk:
            break

        hasher.update(chunk)

    data.seek(0)

    return hasher.hexdigest()
