import logging
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from azarrot.file_store import FileInfo, FileStore
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

class OpenAIFiles:
    _log = logging.getLogger(__name__)
    _file_store: FileStore

    def __init__(self, file_store: FileStore) -> None:
        self._file_store = file_store

    def upload_file(self, file: UploadFile, purpose: Annotated[str, Form()]) -> OpenAIFile:
        info = self._file_store.store_file(file.filename, purpose, file.file)
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

    def get_file_content(self, file_id: str) -> FileResponse:
        file_path, file_info = self._file_store.get_file_path(file_id)

        if file_path is None or file_info is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"File id {file_id} does not exist!")

        return FileResponse(file_path, filename=file_info.filename)
