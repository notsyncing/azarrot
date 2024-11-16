import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import fastapi
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from azarrot.common_types import VectorStoreExpireBaseline, VectorStoreFileState
from azarrot.config import OpenAIFrontendConfig
from azarrot.models.model_manager import ModelManager
from azarrot.vector_store import (
    VECTOR_STORE_DEFAULT_CHUNKING_CONFIG,
    VectorStoreChunkingConfig,
    VectorStoreFileBatchStatus,
    VectorStoreFileInfo,
    VectorStoreFileListPagedQuery,
    VectorStoreInfo,
    VectorStoreInfoUpdateRequest,
    VectorStoreListPagedQuery,
    VectorStoreManager,
    VectorStoreStatus,
)


@dataclass
class OpenAIVectorStoreFileCounts:
    in_progress: int
    completed: int
    failed: int
    cancelled: int
    total: int


@dataclass
class OpenAIVectorStoreExpirePolicy:
    anchor: Literal["last_active_at"]
    days: int

    def to_expire_baseline(self) -> VectorStoreExpireBaseline:
        if self.anchor == "last_active_at":
            return "access_time"

        raise ValueError(f"Unsupported VectorStoreExpireBaseline value {self.anchor}")

    @staticmethod
    def from_expire_baseline(
        expire_baseline: VectorStoreExpireBaseline | None, expire_interval: int
    ) -> "OpenAIVectorStoreExpirePolicy":
        if expire_baseline is None or expire_baseline == "access_time":
            return OpenAIVectorStoreExpirePolicy("last_active_at", expire_interval)

        raise ValueError(f"Unsupported VectorStoreExpireBaseline value {expire_baseline}")


OpenAIVectorStoreStatus = Literal["expired", "in_progress", "completed"]


@dataclass
class OpenAIVectorStoreInfo:
    id: str
    created_at: int
    name: str
    usage_bytes: int
    bytes: int  # Same as usage_bytes, may be compatiblity with demos in OpenAI API document
    file_counts: OpenAIVectorStoreFileCounts
    status: OpenAIVectorStoreStatus
    expires_after: OpenAIVectorStoreExpirePolicy | None
    expires_at: int | None
    last_active_at: int | None
    metadata: dict[str, Any]

    object: str = "vector_store"

    @staticmethod
    def from_vector_store(info: VectorStoreInfo, status: VectorStoreStatus) -> "OpenAIVectorStoreInfo":
        openai_vs_status: OpenAIVectorStoreStatus

        if info.expired:
            openai_vs_status = "expired"
        elif status.pending_file_count > 0 or status.processing_file_count > 0:
            openai_vs_status = "in_progress"
        else:
            openai_vs_status = "completed"

        return OpenAIVectorStoreInfo(
            id=info.id,
            created_at=int(info.create_time.timestamp()),
            name=info.name if info.name is not None else "",
            usage_bytes=status.estimated_size,
            bytes=status.estimated_size,
            file_counts=OpenAIVectorStoreFileCounts(
                in_progress=status.pending_file_count,
                completed=status.completed_file_count,
                failed=status.failed_file_count,
                cancelled=status.cancelled_file_count,
                total=status.total_file_count,
            ),
            status=openai_vs_status,
            expires_after=OpenAIVectorStoreExpirePolicy.from_expire_baseline(
                info.expire_baseline, info.expire_interval
            ),
            expires_at=int(status.will_expire_on.timestamp()),
            last_active_at=int(info.access_time.timestamp()),
            metadata=json.loads(info.additional_data) if info.additional_data is not None else {},
        )


class OpenAIVectorStoreAutoChunkingStrategy(BaseModel):
    type: Literal["auto"] = "auto"


class OpenAIVectorStoreStaticChunkingStrategyConfigs(BaseModel):
    max_chunk_size_tokens: int = Field(ge=100, le=4096)
    chunk_overlap_tokens: int


class OpenAIVectorStoreStaticChunkingStrategy(BaseModel):
    type: Literal["static"] = "static"
    static: OpenAIVectorStoreStaticChunkingStrategyConfigs


OpenAIVectorStoreChunkingStrategy = OpenAIVectorStoreAutoChunkingStrategy | OpenAIVectorStoreStaticChunkingStrategy


class OpenAIVectorStoreCreationRequest(BaseModel):
    file_ids: list[str] | None = None
    name: str | None = None
    expires_after: OpenAIVectorStoreExpirePolicy | None = None
    chunking_strategy: Annotated[OpenAIVectorStoreChunkingStrategy | None, Field(discriminator="type", default=None)]
    metadata: dict[str, Any] | None = None


class OpenAIVectorStoreUpdateRequest(BaseModel):
    name: str | None = None
    expires_after: OpenAIVectorStoreExpirePolicy | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OpenAIVectorFileErrorInfo:
    code: Literal["server_error", "rate_limited"]
    message: str


OpenAIVectorStoreFileStatus = Literal["in_progress", "completed", "cancelled", "failed"]


OPENAI_VECTOR_STORE_FILE_STATE_MAP: dict[VectorStoreFileState, OpenAIVectorStoreFileStatus] = {
    "pending": "in_progress",
    "processing": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}

OPENAI_VECTOR_STORE_FILE_STATE_MAP_REVERSED: dict[OpenAIVectorStoreFileStatus, VectorStoreFileState] = {
    v: k for k, v in OPENAI_VECTOR_STORE_FILE_STATE_MAP.items()
}


@dataclass
class OpenAIVectorStoreFileInfo:
    id: str
    usage_bytes: int
    created_at: int
    vector_store_id: str
    status: OpenAIVectorStoreFileStatus
    last_error: OpenAIVectorFileErrorInfo | None
    chunking_strategy: OpenAIVectorStoreChunkingStrategy

    object: str = "vector_store.file"

    @staticmethod
    def from_vector_store_file(file: VectorStoreFileInfo, store: VectorStoreInfo) -> "OpenAIVectorStoreFileInfo":
        last_error = None

        if file.state == "failed":
            last_error = OpenAIVectorFileErrorInfo(
                code="server_error", message=file.failed_message if file.failed_message is not None else ""
            )

        openai_chunking_strategy: OpenAIVectorStoreChunkingStrategy

        if file.chunking_strategy is None:
            openai_chunking_strategy = OpenAIVectorStoreAutoChunkingStrategy()
        else:
            openai_chunking_strategy = OpenAIVectorStoreStaticChunkingStrategy(
                static=OpenAIVectorStoreStaticChunkingStrategyConfigs(
                    max_chunk_size_tokens=file.chunking_strategy.max_chunk_size_tokens,
                    chunk_overlap_tokens=file.chunking_strategy.chunk_overlap_tokens,
                )
            )

        return OpenAIVectorStoreFileInfo(
            id=file.file_id,
            usage_bytes=file.vector_count * (store.embedding_dimension * 4),
            created_at=int(file.create_time.timestamp()),
            vector_store_id=file.vector_store_id,
            status=OPENAI_VECTOR_STORE_FILE_STATE_MAP[file.state],
            last_error=last_error,
            chunking_strategy=openai_chunking_strategy,
        )


class OpenAIVectorStoreCreateFileRequest(BaseModel):
    file_id: str
    chunking_strategy: OpenAIVectorStoreChunkingStrategy | None = None


class OpenAIVectorStoreCreateFileBatchRequest(BaseModel):
    file_ids: list[str]
    chunking_strategy: OpenAIVectorStoreChunkingStrategy | None = None


@dataclass
class OpenAIVectorStoreFileBatchInfo:
    id: str
    created_at: int
    vector_store_id: str
    status: OpenAIVectorStoreFileStatus
    file_counts: OpenAIVectorStoreFileCounts

    object: str = "vector_store.file_batch"

    @staticmethod
    def from_vector_store_file_batch_status(
        batch_status: VectorStoreFileBatchStatus,
    ) -> "OpenAIVectorStoreFileBatchInfo":
        return OpenAIVectorStoreFileBatchInfo(
            id=batch_status.batch_id,
            created_at=int(batch_status.create_time.timestamp()),
            vector_store_id=batch_status.vector_store_id,
            status=OPENAI_VECTOR_STORE_FILE_STATE_MAP[batch_status.state],
            file_counts=OpenAIVectorStoreFileCounts(
                in_progress=batch_status.pending_file_count + batch_status.processing_file_count,
                completed=batch_status.completed_file_count,
                failed=batch_status.failed_file_count,
                cancelled=batch_status.cancelled_file_count,
                total=batch_status.total_file_count,
            ),
        )


class OpenAIVectorStores:
    _openai_frontend_config: OpenAIFrontendConfig
    _model_manager: ModelManager
    _vector_store: VectorStoreManager

    def __init__(
        self,
        openai_frontend_config: OpenAIFrontendConfig,
        model_manager: ModelManager,
        vector_store: VectorStoreManager,
    ) -> None:
        self._openai_frontend_config = openai_frontend_config
        self._model_manager = model_manager
        self._vector_store = vector_store

    def __convert_chunking_strategy(
        self, openai_strategy: OpenAIVectorStoreChunkingStrategy | None
    ) -> VectorStoreChunkingConfig | None:
        if isinstance(openai_strategy, OpenAIVectorStoreAutoChunkingStrategy):
            return VECTOR_STORE_DEFAULT_CHUNKING_CONFIG
        elif isinstance(openai_strategy, OpenAIVectorStoreStaticChunkingStrategy):
            return VectorStoreChunkingConfig(
                max_chunk_size_tokens=openai_strategy.static.max_chunk_size_tokens,
                chunk_overlap_tokens=openai_strategy.static.chunk_overlap_tokens,
            )
        else:
            return None

    def create(self, request: OpenAIVectorStoreCreationRequest) -> OpenAIVectorStoreInfo:
        model_id = self._openai_frontend_config.vector_store_default_embedding_model_id

        if model_id is None:
            raise ValueError("No default OpenAI vector store embedding model specified!")

        embedding_model = self._model_manager.get_model(model_id)

        if embedding_model is None:
            raise ValueError(f"Specified embedding model {model_id} does not exist!")

        expire_baseline: VectorStoreExpireBaseline | None
        expire_interval: int

        if request.expires_after is not None:
            expire_baseline = request.expires_after.to_expire_baseline()
            expire_interval = request.expires_after.days
        else:
            expire_baseline = None
            expire_interval = -1

        info = self._vector_store.create(
            request.name, embedding_model, expire_baseline=expire_baseline, expire_interval=expire_interval
        )

        store_id = info.id

        if request.file_ids is not None and len(request.file_ids) > 0:
            self._vector_store.add_stored_files(
                info.id, request.file_ids, self.__convert_chunking_strategy(request.chunking_strategy)
            )

        current_info, status = self._vector_store.get_status(store_id)

        if current_info is None or status is None:
            raise ValueError(f"Vector store id {store_id} does not exist, which should not happen!")

        return OpenAIVectorStoreInfo.from_vector_store(current_info, status)

    def get_list(
        self,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        page = self._vector_store.get_list(
            VectorStoreListPagedQuery(
                create_time_desc_order=order == "desc", page_size=limit, before_id=before, after_id=after
            )
        )

        openai_list: list[OpenAIVectorStoreInfo] = []

        for store in page.data:
            _, store_status = self._vector_store.get_status(store.id)

            if store_status is None:
                raise ValueError(f"Vector store id {store.id} does not exist!")

            openai_store_info = OpenAIVectorStoreInfo.from_vector_store(store, store_status)
            openai_list.append(openai_store_info)

        return {
            "object": "list",
            "data": openai_list,
            "first_id": openai_list[0].id if len(openai_list) > 0 else None,
            "last_id": openai_list[-1].id if len(openai_list) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get(self, vector_store_id: str) -> OpenAIVectorStoreInfo:
        store, store_status = self._vector_store.get_status(vector_store_id)

        if store is None or store_status is None:
            raise HTTPException(fastapi.status.HTTP_404_NOT_FOUND)

        return OpenAIVectorStoreInfo.from_vector_store(store, store_status)

    def update(self, vector_store_id: str, request: OpenAIVectorStoreUpdateRequest) -> OpenAIVectorStoreInfo:
        expire_baseline = request.expires_after.to_expire_baseline() if request.expires_after is not None else None

        self._vector_store.update(
            vector_store_id,
            VectorStoreInfoUpdateRequest(
                name=request.name,
                expire_baseline=expire_baseline,
                expire_interval=request.expires_after.days if request.expires_after is not None else None,
                store_metadata=request.metadata,
            ),
        )

        new_store_info, new_store_status = self._vector_store.get_status(vector_store_id)

        if new_store_info is None or new_store_status is None:
            raise ValueError(f"Vector store id {vector_store_id} does not exist, which should not happen!")

        return OpenAIVectorStoreInfo.from_vector_store(new_store_info, new_store_status)

    def delete(self, vstore_id: str) -> dict[str, Any]:
        self._vector_store.delete(vstore_id)

        return {"id": vstore_id, "object": "vector_store.deleted", "deleted": True}

    def create_file(self, vid: str, request: OpenAIVectorStoreCreateFileRequest) -> OpenAIVectorStoreFileInfo:
        chunking_strategy = request.chunking_strategy

        if chunking_strategy is None:
            chunking_strategy = OpenAIVectorStoreAutoChunkingStrategy()

        store = self._vector_store.get_store_info(vid)

        if store is None:
            raise ValueError(f"Vector store {vid} does not exist!")

        files = self._vector_store.add_stored_files(
            vid, [request.file_id], chunking_strategy=self.__convert_chunking_strategy(chunking_strategy)
        )

        return OpenAIVectorStoreFileInfo.from_vector_store_file(files[0], store)

    def get_file_list(
        self,
        vid: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: OpenAIVectorStoreFileStatus | None = None,  # noqa: A002
    ) -> dict[str, Any]:
        store = self._vector_store.get_store_info(vid)

        if store is None:
            raise ValueError(f"Vector store {vid} does not exist!")

        states: list[VectorStoreFileState] | None = (
            [OPENAI_VECTOR_STORE_FILE_STATE_MAP_REVERSED[filter]] if filter is not None else None
        )

        page = self._vector_store.get_stored_file_list(
            vid,
            query=VectorStoreFileListPagedQuery(
                create_time_desc_order=order == "desc", page_size=limit, before_id=before, after_id=after, states=states
            ),
        )

        return {
            "object": "list",
            "data": [OpenAIVectorStoreFileInfo.from_vector_store_file(f, store) for f in page.data],
            "first_id": page.data[0].file_id if len(page.data) > 0 else None,
            "last_id": page.data[-1].file_id if len(page.data) > 0 else None,
            "has_more": not page.is_last_page,
        }

    def get_file(self, vid: str, fid: str) -> OpenAIVectorStoreFileInfo:
        store = self._vector_store.get_store_info(vid)

        if store is None:
            raise ValueError(f"Vector store {vid} does not exist!")

        store_file = self._vector_store.get_stored_file_info(vid, fid)

        if store_file is None:
            raise HTTPException(fastapi.status.HTTP_404_NOT_FOUND)

        return OpenAIVectorStoreFileInfo.from_vector_store_file(store_file, store)

    def delete_file(self, vid: str, f: str) -> dict[str, Any]:
        r = self._vector_store.delete_stored_file(vid, f)

        return {"id": f, "object": "vector_store.file.deleted", "deleted": r}

    def create_batch(
        self, vid: str, request: OpenAIVectorStoreCreateFileBatchRequest
    ) -> OpenAIVectorStoreFileBatchInfo:
        chunking_strategy = None

        if request.chunking_strategy is not None:
            chunking_strategy = self.__convert_chunking_strategy(request.chunking_strategy)

        added_files = self._vector_store.add_stored_files(vid, request.file_ids, chunking_strategy=chunking_strategy)

        f = added_files[0]

        batch_status = self._vector_store.get_batch_status(vid, f.batch_id)

        if batch_status is None:
            raise ValueError(f"Vector store id {vid} batch id {f.batch_id} does not exist, which should not happen!")

        return OpenAIVectorStoreFileBatchInfo.from_vector_store_file_batch_status(batch_status)

    def get_batch(self, vid: str, bid: str) -> OpenAIVectorStoreFileBatchInfo:
        batch_status = self._vector_store.get_batch_status(vid, bid)

        if batch_status is None:
            raise HTTPException(fastapi.status.HTTP_404_NOT_FOUND)

        return OpenAIVectorStoreFileBatchInfo.from_vector_store_file_batch_status(batch_status)

    def cancel_batch(self, vid: str, bid: str) -> OpenAIVectorStoreFileBatchInfo:
        self._vector_store.cancel_batch(vid, bid)

        batch_status = self._vector_store.get_batch_status(vid, bid)

        if batch_status is None:
            raise ValueError(f"Vector store id {vid} batch id {bid} does not exist, which should not happen!")

        return OpenAIVectorStoreFileBatchInfo.from_vector_store_file_batch_status(batch_status)

    def get_batch_files(
        self,
        vid: str,
        bid: str,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        order: Literal["asc", "desc"] = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: OpenAIVectorStoreFileStatus | None = None,  # noqa: A002
    ) -> dict[str, Any]:
        store = self._vector_store.get_store_info(vid)

        if store is None:
            raise ValueError(f"Vector store {vid} does not exist!")

        states: list[VectorStoreFileState] | None = (
            [OPENAI_VECTOR_STORE_FILE_STATE_MAP_REVERSED[filter]] if filter is not None else None
        )

        page = self._vector_store.get_batch_files(
            vid,
            bid,
            query=VectorStoreFileListPagedQuery(
                create_time_desc_order=order == "desc", page_size=limit, before_id=before, after_id=after, states=states
            ),
        )

        return {
            "object": "list",
            "data": [OpenAIVectorStoreFileInfo.from_vector_store_file(f, store) for f in page.data],
            "first_id": page.data[0].file_id if len(page.data) > 0 else None,
            "last_id": page.data[-1].file_id if len(page.data) > 0 else None,
            "has_more": not page.is_last_page,
        }
