import errno
import hashlib
import logging
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from mimetypes import guess_type
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import Engine, and_, delete, select
from sqlalchemy.orm import Session

from azarrot.config import ServerConfig
from azarrot.database_schemas import File, PartialFile, PartialFilePart
from azarrot.utils.merged_file import MergedReadOnlyBinaryFile


@dataclass
class FileInfo:
    id: str
    size: int
    create_time: datetime
    filename: str | None
    purpose: str | None
    mime_type: str | None
    is_partial: bool

    @staticmethod
    def from_db_file(db_file: File) -> "FileInfo":
        return FileInfo(
            id=str(db_file.id),
            size=db_file.size,
            create_time=db_file.create_time,
            filename=db_file.filename,
            purpose=db_file.purpose,
            mime_type=db_file.mime_type,
            is_partial=db_file.is_partial,
        )


@dataclass
class PartialFileInfo:
    id: str
    size: int
    create_time: datetime
    expire_time: datetime
    filename: str
    mime_type: str
    purpose: str | None

    @staticmethod
    def from_db_partial_file(db_obj: PartialFile) -> "PartialFileInfo":
        return PartialFileInfo(
            id=str(db_obj.id),
            size=db_obj.size,
            create_time=db_obj.create_time,
            expire_time=db_obj.expire_time,
            filename=db_obj.filename,
            mime_type=db_obj.mime_type,
            purpose=db_obj.purpose,
        )


@dataclass
class PartialFilePartInfo:
    id: str
    partial_file_id: str
    size: int
    create_time: datetime

    @staticmethod
    def from_db_partial_file_part(db_obj: PartialFilePart) -> "PartialFilePartInfo":
        return PartialFilePartInfo(
            id=str(db_obj.id),
            partial_file_id=str(db_obj.partial_file_id),
            size=db_obj.size,
            create_time=db_obj.create_time,
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

    def _make_store_file_path(self, file_id: str | uuid.UUID) -> Path:
        return self._store_path / f"{file_id}.file"

    def _make_store_file_part_path(self, part_id: str | uuid.UUID) -> Path:
        return self._store_path / f"{part_id}.filepart"

    def __compute_sha256(self, data: BinaryIO) -> str:
        hasher = hashlib.sha256()

        while True:
            chunk = data.read(4096)

            if not chunk:
                break

            hasher.update(chunk)

        data.seek(0)

        return hasher.hexdigest()

    def __compute_md5(self, data: BinaryIO) -> str:
        hasher = hashlib.md5()  # noqa: S324

        while True:
            chunk = data.read(4096)

            if not chunk:
                break

            hasher.update(chunk)

        data.seek(0)

        return hasher.hexdigest()

    def __guess_file_mime_type(self, url: str | None) -> str | None:
        if url is None:
            return None

        t, e = guess_type(url)
        r = t

        if e is not None and r is not None:
            r = r + "; " + e

        return r

    def store_file(self, filename: str | None, purpose: str | None, mime_type: str | None, data: BinaryIO) -> FileInfo:
        file_id = uuid.uuid4()
        create_time = datetime.now()

        checksum = self.__compute_sha256(data)

        with Session(self._database) as db_session:
            same_file = db_session.execute(
                select(File).where(and_(File.filename == filename, File.checksum == checksum))
            ).scalar_one_or_none()

            if same_file is not None:
                self._log.warning(
                    "Uploaded file %s (checksum %s) already exists. Will return the existing file.", filename, checksum
                )

                return FileInfo.from_db_file(same_file)

            store_file_path = self._make_store_file_path(file_id)

            with store_file_path.open("wb") as store_file:
                shutil.copyfileobj(data, store_file)

            db_file = File()
            db_file.id = file_id
            db_file.filename = filename
            db_file.mime_type = mime_type if mime_type is not None else self.__guess_file_mime_type(filename)
            db_file.create_time = create_time
            db_file.checksum = checksum
            db_file.purpose = purpose
            db_file.is_partial = False
            db_file.size = store_file_path.stat().st_size

            db_session.add(db_file)
            db_session.commit()

            file_info = FileInfo.from_db_file(db_file)

        self._log.info(
            "Saved uploaded file %s type %s size %d purpose %s to %s (checksum %s)",
            filename,
            file_info.mime_type,
            file_info.size,
            purpose,
            store_file_path,
            checksum,
        )

        return file_info

    def get_all_file_list(self) -> list[FileInfo]:
        with Session(self._database) as db_session:
            return [FileInfo.from_db_file(f) for f in db_session.execute(select(File)).scalars().all()]

    def __sanitize_uuid(self, value: str | uuid.UUID) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value

        return uuid.UUID(value)

    def get_file_info(self, file_id: str | uuid.UUID) -> FileInfo | None:
        with Session(self._database) as db_session:
            f = db_session.execute(select(File).where(File.id == self.__sanitize_uuid(file_id))).scalar_one_or_none()

            if f is not None:
                return FileInfo.from_db_file(f)
            else:
                return None

    def delete_file(self, file_id: str | uuid.UUID) -> None:
        with Session(self._database) as db_session:
            f = db_session.execute(select(File).where(File.id == self.__sanitize_uuid(file_id))).scalar_one_or_none()

            if f is None:
                raise FileNotFoundError

            if f.is_partial:
                self.delete_partial_file(file_id, delete_merged_file=True)
                return

            db_session.delete(f)

            file_path = self._make_store_file_path(f.id)
            file_path.unlink(missing_ok=True)

            db_session.commit()

    def get_file_content(self, file_id: str | uuid.UUID) -> tuple[FileInfo | None, BinaryIO | None]:
        file_info = self.get_file_info(file_id)

        if file_info is None:
            return None, None

        if file_info.is_partial:
            file_parts = self.get_partial_file_all_parts(file_id)
            file_part_paths = [self._make_store_file_part_path(p.id) for p in file_parts]

            return file_info, MergedReadOnlyBinaryFile(file_part_paths)
        else:
            return file_info, self._make_store_file_path(file_info.id).open("rb")

    def create_partial_file(
        self, filename: str, final_size: int, mime_type: str, purpose: str | None
    ) -> PartialFileInfo:
        with Session(self._database) as db_session:
            db_file = PartialFile()
            db_file.id = uuid.uuid4()
            db_file.filename = filename
            db_file.size = final_size
            db_file.mime_type = mime_type
            db_file.purpose = purpose
            db_file.create_time = datetime.now()
            db_file.expire_time = db_file.create_time + timedelta(milliseconds=self._config.partial_file_expire_time)

            db_session.add(db_file)
            db_session.commit()

            self._log.info("Created partial file %s with id %s", filename, db_file.id)

        return PartialFileInfo.from_db_partial_file(db_file)

    def get_partial_file_info(self, partial_file_id: str | uuid.UUID) -> PartialFileInfo | None:
        with Session(self._database) as db_session:
            pf = db_session.execute(
                select(PartialFile).where(PartialFile.id == self.__sanitize_uuid(partial_file_id))
            ).scalar_one_or_none()

        return PartialFileInfo.from_db_partial_file(pf) if pf is not None else None

    def get_partial_file_all_parts(self, partial_file_id: str | uuid.UUID) -> list[PartialFilePartInfo]:
        with Session(self._database) as db_session:
            pf = db_session.execute(
                select(PartialFilePart)
                .where(PartialFilePart.partial_file_id == self.__sanitize_uuid(partial_file_id))
                .order_by(PartialFilePart.merged_order)
            ).scalars()

            return [PartialFilePartInfo.from_db_partial_file_part(p) for p in pf]

    def add_part_to_partial_file(self, partial_file_id: str | uuid.UUID, data: BinaryIO) -> PartialFilePartInfo:
        part_id = uuid.uuid4()
        create_time = datetime.now()

        if isinstance(partial_file_id, str):
            partial_file_id = uuid.UUID(partial_file_id)

        with Session(self._database) as db_session:
            partial_file = db_session.execute(
                select(PartialFile).where(PartialFile.id == partial_file_id)
            ).scalar_one_or_none()

            if partial_file is None:
                raise FileNotFoundError

        checksum = self.__compute_sha256(data)

        with Session(self._database) as db_session:
            same_part = db_session.execute(
                select(PartialFilePart).where(
                    and_(PartialFilePart.partial_file_id == partial_file_id, PartialFilePart.checksum == checksum)
                )
            ).scalar_one_or_none()

            if same_part is not None:
                self._log.warning(
                    "Uploaded part of partial file id %s (checksum %s) already exists. Will return the existing part.",
                    partial_file_id,
                    checksum,
                )

                return PartialFilePartInfo.from_db_partial_file_part(same_part)

            store_file_path = self._make_store_file_part_path(part_id)

            with store_file_path.open("wb") as store_file:
                shutil.copyfileobj(data, store_file)

            db_file_part = PartialFilePart()
            db_file_part.id = part_id
            db_file_part.partial_file_id = partial_file_id
            db_file_part.create_time = create_time
            db_file_part.checksum = checksum
            db_file_part.size = store_file_path.stat().st_size

            db_session.add(db_file_part)
            db_session.commit()

            part_info = PartialFilePartInfo.from_db_partial_file_part(db_file_part)

        self._log.info(
            "Saved uploaded file part id %s of partial file id %s size %d to %s (checksum %s)",
            part_id,
            partial_file_id,
            part_info.size,
            store_file_path,
            checksum,
        )

        return part_info

    def get_partial_file_part_info(
        self, partial_file_id: str | uuid.UUID, part_id: str | uuid.UUID
    ) -> PartialFilePartInfo | None:
        with Session(self._database) as db_session:
            part = db_session.scalar(
                select(PartialFilePart).where(
                    and_(
                        PartialFilePart.id == self.__sanitize_uuid(part_id),
                        PartialFilePart.partial_file_id == self.__sanitize_uuid(partial_file_id),
                    )
                )
            )

            return PartialFilePartInfo.from_db_partial_file_part(part) if part is not None else None

    def merge_partial_file(
        self,
        partial_file_id: str | uuid.UUID,
        ordered_file_parts: Sequence[str | uuid.UUID],
        expected_checksum_md5: str | None = None,
        expected_checksum_sha256: str | None = None,
    ) -> tuple[PartialFileInfo, FileInfo]:
        if len(ordered_file_parts) <= 0:
            raise ValueError("You must specify at least one file part to merge!")

        if isinstance(partial_file_id, str):
            partial_file_id = uuid.UUID(partial_file_id)

        with Session(self._database) as db_session:
            partial_file = db_session.execute(
                select(PartialFile).where(PartialFile.id == partial_file_id)
            ).scalar_one_or_none()

            if partial_file is None:
                raise FileNotFoundError(errno.ENOENT, "Partial file not found", str(partial_file_id))

            file_parts: list[PartialFilePart] = []
            total_part_size = 0
            file_part_id_list = [self.__sanitize_uuid(fid) for fid in ordered_file_parts]

            for index, file_part_id in enumerate(file_part_id_list):
                file_part = db_session.execute(
                    select(PartialFilePart).where(
                        and_(
                            PartialFilePart.id == file_part_id,
                            PartialFilePart.partial_file_id == partial_file_id,
                        )
                    )
                ).scalar_one_or_none()

                if file_part is None:
                    raise FileNotFoundError(
                        errno.ENOENT, f"Partial file {partial_file_id} does not contain part", str(file_part_id)
                    )

                file_part.merged_order = index

                file_parts.append(file_part)
                total_part_size = total_part_size + file_part.size

            if total_part_size != partial_file.size:
                raise ValueError(
                    "All parts total size %d, does not match expected file size %d", total_part_size, partial_file.size
                )

            file_part_paths = [self._make_store_file_part_path(p.id) for p in file_parts]
            merged_file = MergedReadOnlyBinaryFile(file_part_paths)
            checksum = self.__compute_sha256(merged_file)

            if expected_checksum_md5 is not None:
                md5_checksum = self.__compute_md5(merged_file)

                if md5_checksum != expected_checksum_md5:
                    raise ValueError(
                        "Merged file from partial file id %s has different md5 checksum (%s) as expected (%s)",
                        md5_checksum,
                        expected_checksum_md5,
                    )

            if expected_checksum_sha256 is not None:
                if checksum != expected_checksum_sha256:
                    raise ValueError(
                        "Merged file from partial file id %s has different sha256 checksum (%s) as expected (%s)",
                        checksum,
                        expected_checksum_sha256,
                    )

            merged_db_file = File()
            merged_db_file.id = partial_file.id
            merged_db_file.filename = partial_file.filename
            merged_db_file.create_time = partial_file.create_time
            merged_db_file.checksum = checksum
            merged_db_file.purpose = partial_file.purpose
            merged_db_file.mime_type = partial_file.mime_type
            merged_db_file.is_partial = True
            merged_db_file.size = partial_file.size

            db_session.add(merged_db_file)
            db_session.commit()

            self._log.info(
                "Merged file %s (id %s, checksum %s) from parts %s",
                merged_db_file.filename,
                merged_db_file.id,
                merged_db_file.checksum,
                ", ".join([str(fid) for fid in file_part_id_list]),
            )

            return PartialFileInfo.from_db_partial_file(partial_file), FileInfo.from_db_file(merged_db_file)

    def delete_partial_file(
        self, partial_file_id: str | uuid.UUID, delete_merged_file: bool = False
    ) -> PartialFileInfo:
        partial_file_id = self.__sanitize_uuid(partial_file_id)

        with Session(self._database) as db_session:
            partial_file = db_session.execute(
                select(PartialFile).where(PartialFile.id == partial_file_id)
            ).scalar_one_or_none()

            if partial_file is None:
                raise FileNotFoundError

            merged_file = db_session.execute(select(File).where(File.id == partial_file_id)).scalar_one_or_none()

            if merged_file is not None:
                if delete_merged_file:
                    db_session.delete(merged_file)
                else:
                    raise ValueError(f"Partial file {partial_file_id} is already merged, cannot delete!")

            file_parts = db_session.scalars(
                select(PartialFilePart).where(PartialFilePart.partial_file_id == partial_file_id)
            ).all()

            for info in file_parts:
                part_path = self._make_store_file_part_path(info.id)
                part_path.unlink(missing_ok=True)
                db_session.delete(info)

            db_session.delete(partial_file)
            db_session.commit()

        return PartialFileInfo.from_db_partial_file(partial_file)

    def clear_database(self) -> None:
        with Session(self._database) as db_session:
            db_session.execute(delete(File))
            db_session.execute(delete(PartialFile))
            db_session.execute(delete(PartialFilePart))
            db_session.commit()
