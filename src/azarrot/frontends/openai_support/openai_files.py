import logging
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from azarrot.file_store import FileInfo, FileStore, PartialFileInfo
from azarrot.frontends.openai_support.openai_data import OpenAIList


@dataclass
class OpenAIFile:
    id: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    object: str = "file"

    @staticmethod
    def from_store_file(store_file: FileInfo) -> "OpenAIFile":
        return OpenAIFile(
            id=store_file.id,
            bytes=store_file.size,
            created_at=int(store_file.create_time.timestamp()),
            filename=store_file.filename if store_file.filename is not None else "",
            purpose=store_file.purpose if store_file.purpose is not None else ""
        )


class OpenAIUploadRequest(BaseModel):
    filename: str
    purpose: str
    bytes: int
    mime_type: str


class OpenAICompleteUploadRequest(BaseModel):
    part_ids: list[str]
    md5: str | None = None


@dataclass
class OpenAIUpload:
    id: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str
    expires_at: int
    object: str = "upload"

    @staticmethod
    def from_store_partial_file(store_partial_file: PartialFileInfo, status: str) -> "OpenAIUpload":
        return OpenAIUpload(
            id=str(store_partial_file.id),
            bytes=store_partial_file.size,
            created_at=int(store_partial_file.create_time.timestamp()),
            filename=store_partial_file.filename,
            purpose=store_partial_file.purpose if store_partial_file.purpose is not None else "",
            status=status,
            expires_at=int(store_partial_file.expire_time.timestamp())
        )

class OpenAIFiles:
    _log = logging.getLogger(__name__)
    _file_store: FileStore

    def __init__(self, file_store: FileStore) -> None:
        self._file_store = file_store

    def upload_file(self, file: UploadFile, purpose: Annotated[str, Form()]) -> OpenAIFile:
        info = self._file_store.store_file(file.filename, purpose, file.content_type, file.file)
        return OpenAIFile.from_store_file(info)

    def get_file_list(self) -> OpenAIList[OpenAIFile]:
        files = self._file_store.get_all_file_list()

        return OpenAIList(
            data=[OpenAIFile.from_store_file(f) for f in files]
        )

    def get_file_info(self, file_id: str) -> OpenAIFile:
        info = self._file_store.get_file_info(file_id)

        if info is not None:
            return OpenAIFile.from_store_file(info)
        else:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"File id {file_id} does not exist!")

    def delete_file(self, file_id: str) -> dict[str, Any]:
        success: bool

        try:
            self._file_store.delete_file(file_id)
            success = True
        except:
            self._log.error("Failed to delete file %s", file_id, exc_info=True)
            success = False

        return {
            "id": file_id,
            "object": "file",
            "deleted": success
        }

    def get_file_content(self, file_id: str) -> StreamingResponse:
        file_info, file_handle = self._file_store.get_file_content(file_id)

        if file_info is None or file_handle is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"File id {file_id} does not exist!")

        headers = {}

        if file_info.filename is not None:
            headers["Content-Disposition"] = f"attachment; filename={file_info.filename}"

        return StreamingResponse(file_handle, media_type=file_info.mime_type, headers=headers)

    def create_upload(self, request: OpenAIUploadRequest) -> OpenAIUpload:
        partial_file = self._file_store.create_partial_file(
            request.filename, request.bytes, request.mime_type, request.purpose
        )

        return OpenAIUpload.from_store_partial_file(partial_file, "pending")

    def add_upload_part(self, upload_id: str, data: UploadFile) -> dict[str, Any]:
        part_info = self._file_store.add_part_to_partial_file(upload_id, data.file)

        return {
            "id": part_info.id,
            "object": "upload.part",
            "created_at": int(part_info.create_time.timestamp()),
            "upload_id": part_info.partial_file_id
        }

    def complete_upload(self, upload_id: str, request: OpenAICompleteUploadRequest) -> dict[str, Any]:
        partial_file_info, file_info = self._file_store.merge_partial_file(
            upload_id, request.part_ids, expected_checksum_md5=request.md5
        )

        return {
            "id": partial_file_info.id,
            "object": "upload",
            "bytes": partial_file_info.size,
            "created_at": int(partial_file_info.create_time.timestamp()),
            "filename": partial_file_info.filename,
            "purpose": partial_file_info.purpose,
            "status": "completed",
            "expires_at": int(partial_file_info.expire_time.timestamp()),
            "file": OpenAIFile.from_store_file(file_info)
        }

    def cancel_upload(self, upload_id: str) -> OpenAIUpload:
        partial_file = self._file_store.delete_partial_file(upload_id)
        return OpenAIUpload.from_store_partial_file(partial_file, "cancelled")
