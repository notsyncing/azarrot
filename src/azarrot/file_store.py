import hashlib
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import Engine, and_, delete, select
from sqlalchemy.orm import Session

from azarrot.config import ServerConfig
from azarrot.database_schemas import File


@dataclass
class FileInfo:
    id: str
    size: int
    create_time: datetime
    filename: str | None
    purpose: str | None

    @staticmethod
    def from_db_file(db_file: File) -> "FileInfo":
        return FileInfo(
            id=str(db_file.id),
            size=db_file.size,
            create_time=db_file.create_time,
            filename=db_file.filename,
            purpose=db_file.purpose
        )


class FileStore:
    _log = logging.getLogger(__name__)
    _config: ServerConfig
    _store_path: Path
    _database: Engine

    def __init__(self, config: ServerConfig, store_path: Path, database: Engine) -> None:
        self._config = config
        self._store_path = store_path
        self._database = database

    def __make_store_file_path(self, file_id: str | uuid.UUID) -> Path:
        return self._store_path / f"{file_id}.file"

    def __compute_sha256(self, data: BinaryIO) -> str:
        hasher = hashlib.sha256()

        while True:
            chunk = data.read(4096)

            if not chunk:
                break

            hasher.update(chunk)

        data.seek(0)

        return hasher.hexdigest()

    def store_file(self, filename: str | None, purpose: str | None, data: BinaryIO) -> FileInfo:
        file_id = uuid.uuid4()
        create_time = datetime.now()

        checksum = self.__compute_sha256(data)

        with Session(self._database) as db_session:
            same_file = db_session.execute(
                select(File).where(
                    and_(
                        File.filename == filename,
                        File.checksum == checksum
                    )
                )
            ).scalar_one_or_none()

            if same_file is not None:
                self._log.warning(
                    "Uploaded file %s (checksum %s) already exists. Will return the existing file.",
                    filename, checksum
                )

                return FileInfo.from_db_file(same_file)

            store_file_path = self.__make_store_file_path(file_id)

            with store_file_path.open("wb") as store_file:
                shutil.copyfileobj(data, store_file)

            db_file = File()
            db_file.id = file_id
            db_file.filename = filename
            db_file.create_time = create_time
            db_file.checksum = checksum
            db_file.purpose = purpose
            db_file.size = store_file_path.stat().st_size

            db_session.add(db_file)
            db_session.commit()

            file_info = FileInfo.from_db_file(db_file)

        self._log.info(
            "Saved uploaded file %s size %d purpose %s to %s (checksum %s)",
            filename, file_info.size, purpose, store_file_path, checksum
        )

        return file_info

    def get_all_file_list(self) -> list[FileInfo]:
        with Session(self._database) as db_session:
            return [
                FileInfo.from_db_file(f)
                for f in db_session.execute(select(File)).scalars().all()
            ]

    def __sanitize_uuid(self, value: str | uuid.UUID) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value

        return uuid.UUID(value)

    def get_file_info(self, file_id: str | uuid.UUID) -> FileInfo | None:
        with Session(self._database) as db_session:
            f = db_session.execute(
                select(File).where(File.id == self.__sanitize_uuid(file_id))
            ).scalar_one_or_none()

            if f is not None:
                return FileInfo.from_db_file(f)
            else:
                return None

    def delete_file(self, file_id: str | uuid.UUID) -> None:
        with Session(self._database) as db_session:
            f = db_session.execute(
                select(File).where(File.id == self.__sanitize_uuid(file_id))
            ).scalar_one_or_none()

            if f is None:
                raise FileNotFoundError

            db_session.delete(f)

            file_path = self.__make_store_file_path(f.id)
            file_path.unlink(missing_ok=True)

            db_session.commit()

    def get_file_path(self, file_id: str | uuid.UUID) -> tuple[Path | None, FileInfo | None]:
        f = self.get_file_info(file_id)

        if f is None:
            return None, None

        return self.__make_store_file_path(f.id), f

    def clear_database(self) -> None:
        with Session(self._database) as db_session:
            db_session.execute(delete(File))
            db_session.commit()
