import dataclasses
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from pymilvus import Collection, MilvusClient
from sqlakeyset import select_page
from sqlalchemy import Engine, and_, delete, func, select, update
from sqlalchemy.orm import Session

from azarrot.common_data import EmbeddingModelInfo, Model, PageResult
from azarrot.common_types import VectorStoreExpireBaseline, VectorStoreFileFailedReason, VectorStoreFileState
from azarrot.config import ServerConfig
from azarrot.database_schemas import VectorStore, VectorStoreFile
from azarrot.utils import sanitize_uuid
from azarrot.vector_store.utils import METADATA_KEY_FILE_ID, load_collection, make_collection_name


@dataclass
class VectorStoreInfo:
    id: str
    name: str | None
    embedding_model: str
    embedding_dimension: int
    expire_baseline: VectorStoreExpireBaseline | None
    expire_interval: int
    expired: bool
    additional_data: str | None
    create_time: datetime
    access_time: datetime
    update_time: datetime

    @staticmethod
    def from_db_vector_store(db_vector_store: VectorStore) -> "VectorStoreInfo":
        return VectorStoreInfo(
            id=str(db_vector_store.id),
            name=db_vector_store.name,
            embedding_model=db_vector_store.embedding_model,
            embedding_dimension=db_vector_store.embedding_dimension,
            expire_baseline=db_vector_store.expire_baseline,
            expire_interval=db_vector_store.expire_interval,
            expired=db_vector_store.expired,
            additional_data=db_vector_store.additional_data,
            create_time=db_vector_store.create_time,
            access_time=db_vector_store.access_time,
            update_time=db_vector_store.update_time,
        )


@dataclass
class VectorStoreChunkingConfig:
    max_chunk_size_tokens: int
    chunk_overlap_tokens: int


VECTOR_STORE_DEFAULT_CHUNKING_CONFIG = VectorStoreChunkingConfig(max_chunk_size_tokens=800, chunk_overlap_tokens=400)


@dataclass
class VectorStoreFileInfo:
    vector_store_id: str
    file_id: str
    batch_id: str
    chunking_strategy: VectorStoreChunkingConfig | None
    state: VectorStoreFileState
    vector_count: int
    failed_reason: VectorStoreFileFailedReason | None
    failed_message: str | None
    create_time: datetime
    update_time: datetime

    @staticmethod
    def from_db_vector_store_file(db_vector_store_file: VectorStoreFile) -> "VectorStoreFileInfo":
        chunking_strategy = None

        if db_vector_store_file.chunking_strategy is not None:
            cs = json.loads(db_vector_store_file.chunking_strategy)
            chunking_strategy = VectorStoreChunkingConfig(**cs)

        return VectorStoreFileInfo(
            vector_store_id=str(db_vector_store_file.vector_store_id),
            file_id=str(db_vector_store_file.file_id),
            batch_id=db_vector_store_file.batch_id,
            chunking_strategy=chunking_strategy,
            state=db_vector_store_file.state,
            vector_count=db_vector_store_file.vector_count,
            failed_reason=db_vector_store_file.failed_reason,
            failed_message=db_vector_store_file.failed_message,
            create_time=db_vector_store_file.create_time,
            update_time=db_vector_store_file.update_time,
        )


@dataclass
class VectorStoreStatus:
    estimated_size: int
    total_file_count: int
    pending_file_count: int
    processing_file_count: int
    completed_file_count: int
    failed_file_count: int
    cancelled_file_count: int
    will_expire_on: datetime


@dataclass
class VectorStoreListPagedQuery:
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None


@dataclass
class VectorStoreInfoUpdateRequest:
    name: str | None = None
    expire_baseline: VectorStoreExpireBaseline | None = None
    expire_interval: int | None = None
    store_metadata: dict[str, Any] | None = None


@dataclass
class VectorStoreFileListPagedQuery:
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None
    states: list[VectorStoreFileState] | None = None


@dataclass
class VectorStoreFileBatchStatus:
    batch_id: str
    vector_store_id: str
    create_time: datetime
    state: VectorStoreFileState
    total_file_count: int
    pending_file_count: int
    processing_file_count: int
    completed_file_count: int
    failed_file_count: int
    cancelled_file_count: int


class VectorStoreManager:
    _log = logging.getLogger(__name__)
    _server_config: ServerConfig
    _main_db: Engine
    _vector_db: MilvusClient

    def __init__(self, server_config: ServerConfig, main_db: Engine, vector_db: MilvusClient) -> None:
        self._server_config = server_config
        self._main_db = main_db
        self._vector_db = vector_db

    def create(
        self,
        name: str | None,
        embedding_model: Model,
        store_id: uuid.UUID | None = None,
        expire_baseline: VectorStoreExpireBaseline | None = None,
        expire_interval: int = 0,
        additional_data: dict[str, Any] | None = None,
    ) -> VectorStoreInfo:
        if not isinstance(embedding_model.info, EmbeddingModelInfo):
            raise ValueError(f"Specified model {embedding_model.id} is not an embedding model!")

        with Session(self._main_db) as db_session:
            vs_id = uuid.uuid4() if store_id is None else store_id

            vs = VectorStore()
            vs.id = vs_id
            vs.name = name
            vs.embedding_model = embedding_model.id
            vs.embedding_dimension = embedding_model.info.dimension
            vs.expire_baseline = expire_baseline
            vs.expire_interval = expire_interval
            vs.expired = False

            if additional_data is not None:
                vs.additional_data = json.dumps(additional_data)

            create_time = datetime.now()
            vs.create_time = create_time
            vs.update_time = create_time
            vs.access_time = create_time

            db_session.add(vs)
            db_session.commit()

            info = VectorStoreInfo.from_db_vector_store(vs)

        self._log.info("Created vector store id %s", vs_id)

        return info

    def get_store_info(self, store_id: str | uuid.UUID) -> VectorStoreInfo | None:
        store_id = sanitize_uuid(store_id)

        with Session(self._main_db) as db_session:
            store = db_session.execute(select(VectorStore).where(VectorStore.id == store_id)).scalar_one_or_none()

            if store is None:
                return None

            return VectorStoreInfo.from_db_vector_store(store)

    def add_stored_files(
        self,
        store_id: str | uuid.UUID,
        stored_file_id_list: Sequence[str | uuid.UUID],
        chunking_strategy: VectorStoreChunkingConfig | None = None,
        state: VectorStoreFileState | None = None,
        batch_id: str | None = None,
    ) -> list[VectorStoreFileInfo]:
        if chunking_strategy is None:
            chunking_strategy = VECTOR_STORE_DEFAULT_CHUNKING_CONFIG

        store_id = sanitize_uuid(store_id)
        now = datetime.now()

        with Session(self._main_db) as db_session:
            store = db_session.execute(select(VectorStore).where(VectorStore.id == store_id)).scalar_one_or_none()

            if store is None:
                raise ValueError(f"Vector store id {store_id} does not exist!")

            store.access_time = now

            db_files = []

            if batch_id is None:
                batch_id = str(uuid.uuid4())

            for f in stored_file_id_list:
                file_id = sanitize_uuid(f)

                file = VectorStoreFile()
                file.vector_store_id = store_id
                file.file_id = file_id
                file.batch_id = batch_id
                file.chunking_strategy = json.dumps(dataclasses.asdict(chunking_strategy))
                file.state = "pending" if state is None else state
                file.vector_count = 0
                file.create_time = now
                file.update_time = now

                db_files.append(file)

            db_session.add(store)
            db_session.add_all(db_files)
            db_session.commit()

            self._log.info("Added %d file(s) to vector store id %s", len(db_files), store_id)

            return [VectorStoreFileInfo.from_db_vector_store_file(f) for f in db_files]

    def get_status(self, store_id: str | uuid.UUID) -> tuple[VectorStoreInfo | None, VectorStoreStatus | None]:
        store = self.get_store_info(store_id)

        if store is None:
            return None, None

        collection_name = make_collection_name(store_id)
        estimated_size: int

        if self._vector_db.has_collection(collection_name):
            collection = Collection(collection_name)
            vector_count = collection.num_entities
            estimated_size = vector_count * (store.embedding_dimension * 4)
        else:
            estimated_size = 0

        with Session(self._main_db) as db_session:
            total_file_count = db_session.execute(select(func.count()).select_from(VectorStoreFile)).scalar_one()

            pending_file_count = db_session.execute(
                select(func.count()).select_from(VectorStoreFile).where(VectorStoreFile.state == "pending")
            ).scalar_one()

            processing_file_count = db_session.execute(
                select(func.count()).select_from(VectorStoreFile).where(VectorStoreFile.state == "processing")
            ).scalar_one()

            completed_file_count = db_session.execute(
                select(func.count()).select_from(VectorStoreFile).where(VectorStoreFile.state == "completed")
            ).scalar_one()

            failed_file_count = db_session.execute(
                select(func.count()).select_from(VectorStoreFile).where(VectorStoreFile.state == "failed")
            ).scalar_one()

            cancelled_file_count = db_session.execute(
                select(func.count()).select_from(VectorStoreFile).where(VectorStoreFile.state == "cancelled")
            ).scalar_one()

        will_expire_on: datetime

        if store.expire_baseline is None or store.expire_interval <= 0:
            will_expire_on = datetime(2999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)  # noqa: UP017
        else:
            time_base: datetime

            if store.expire_baseline == "access_time":
                time_base = store.access_time
            elif store.expire_baseline == "create_time":
                time_base = store.create_time
            elif store.expire_baseline == "update_time":
                time_base = store.update_time
            else:
                raise ValueError(f"Invalid expire baseline {store.expire_baseline} in vector store id {store.id}")

            will_expire_on = time_base + timedelta(days=store.expire_interval)

        return store, VectorStoreStatus(
            estimated_size=estimated_size,
            total_file_count=total_file_count,
            pending_file_count=pending_file_count,
            processing_file_count=processing_file_count,
            completed_file_count=completed_file_count,
            failed_file_count=failed_file_count,
            cancelled_file_count=cancelled_file_count,
            will_expire_on=will_expire_on,
        )

    def get_list(self, query: VectorStoreListPagedQuery) -> PageResult[VectorStoreInfo]:
        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._main_db) as db_session:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db_session.execute(
                    select(VectorStore.create_time).where(VectorStore.id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = select(VectorStore)

            if query.create_time_desc_order:
                q = q.order_by(VectorStore.create_time.desc(), VectorStore.id.desc())
            else:
                q = q.order_by(VectorStore.create_time, VectorStore.id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db_session, q, per_page=query.page_size, before=before, after=after)

            return PageResult(
                data=[VectorStoreInfo.from_db_vector_store(r._tuple()[0]) for r in data],  # noqa: SLF001
                is_last_page=not data.paging.has_next,
            )

    def update(self, store_id: str | uuid.UUID, update_fields: VectorStoreInfoUpdateRequest) -> None:
        store_id = sanitize_uuid(store_id)

        with Session(self._main_db) as db_session:
            store = db_session.execute(select(VectorStore).where(VectorStore.id == store_id)).scalar_one_or_none()

            if store is None:
                raise ValueError(f"Vector store id {store_id} does not exist!")

            if update_fields.name is not None:
                store.name = update_fields.name

            if update_fields.expire_baseline is not None:
                store.expire_baseline = update_fields.expire_baseline

            if update_fields.expire_interval is not None:
                store.expire_interval = update_fields.expire_interval

            if update_fields.store_metadata is not None:
                store.additional_data = json.dumps(update_fields.store_metadata)

            now = datetime.now()
            store.update_time = now
            store.access_time = now

            db_session.commit()

    def delete(self, store_id: str | uuid.UUID) -> None:
        store_id = sanitize_uuid(store_id)
        collection_name = make_collection_name(store_id)

        if self._vector_db.has_collection(collection_name):
            self._vector_db.drop_collection(collection_name)

        with Session(self._main_db) as db_session:
            db_session.execute(delete(VectorStore).where(VectorStore.id == store_id))

            db_session.execute(delete(VectorStoreFile).where(VectorStoreFile.vector_store_id == store_id))

            db_session.commit()

        self._log.info("Deleted vector store id %s", store_id)

    def get_stored_file_info(self, store_id: str | uuid.UUID, file_id: str | uuid.UUID) -> VectorStoreFileInfo | None:
        store_id = sanitize_uuid(store_id)
        file_id = sanitize_uuid(file_id)

        with Session(self._main_db) as db_session:
            file = db_session.execute(
                select(VectorStoreFile).where(
                    and_(VectorStoreFile.vector_store_id == store_id, VectorStoreFile.file_id == file_id)
                )
            ).scalar_one_or_none()

            if file is None:
                return None

            return VectorStoreFileInfo.from_db_vector_store_file(file)

    def get_stored_file_list(
        self, store_id: str | uuid.UUID, query: VectorStoreFileListPagedQuery
    ) -> PageResult[VectorStoreFileInfo]:
        store_id = sanitize_uuid(store_id)

        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._main_db) as db_session:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db_session.execute(
                    select(VectorStoreFile.create_time).where(VectorStoreFile.file_id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = select(VectorStoreFile)

            if query.states is not None:
                q = q.where(VectorStoreFile.state.in_(query.states))

            if query.create_time_desc_order:
                q = q.order_by(VectorStoreFile.create_time.desc(), VectorStoreFile.file_id.desc())
            else:
                q = q.order_by(VectorStoreFile.create_time, VectorStoreFile.file_id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db_session, q, per_page=query.page_size, before=before, after=after)

            return PageResult(
                data=[VectorStoreFileInfo.from_db_vector_store_file(r._tuple()[0]) for r in data],  # noqa: SLF001
                is_last_page=not data.paging.has_next,
            )

    def delete_stored_file(self, store_id: str | uuid.UUID, file_id: str | uuid.UUID) -> bool:
        store_id = sanitize_uuid(store_id)
        file_id = sanitize_uuid(file_id)
        collection_name = make_collection_name(store_id)

        if self._vector_db.has_collection(collection_name):
            load_collection(self._vector_db, collection_name)

            self._vector_db.delete(collection_name=collection_name, filter=f'{METADATA_KEY_FILE_ID} == "{file_id!s}"')

        with Session(self._main_db) as db_session:
            r = db_session.execute(
                delete(VectorStoreFile).where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.file_id == file_id,
                        VectorStoreFile.state.in_(["pending", "completed", "failed", "cancelled"]),
                    )
                )
            )

            c = r.rowcount

            db_session.execute(update(VectorStore).where(VectorStore.id == store_id).values(access_time=datetime.now()))

            db_session.commit()
            return c > 0

    def get_batch_status(self, store_id: str | uuid.UUID, batch_id: str) -> VectorStoreFileBatchStatus | None:
        store_id = sanitize_uuid(store_id)

        with Session(self._main_db) as db_session:
            any_file = db_session.execute(
                select(VectorStoreFile)
                .where(and_(VectorStoreFile.vector_store_id == store_id, VectorStoreFile.batch_id == batch_id))
                .limit(1)
            ).scalar_one_or_none()

            if any_file is None:
                return None

            total_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(and_(VectorStoreFile.vector_store_id == store_id, VectorStoreFile.batch_id == batch_id))
            ).scalar_one()

            pending_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.batch_id == batch_id,
                        VectorStoreFile.state == "pending",
                    )
                )
            ).scalar_one()

            processing_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.batch_id == batch_id,
                        VectorStoreFile.state == "processing",
                    )
                )
            ).scalar_one()

            completed_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.batch_id == batch_id,
                        VectorStoreFile.state == "completed",
                    )
                )
            ).scalar_one()

            failed_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.batch_id == batch_id,
                        VectorStoreFile.state == "failed",
                    )
                )
            ).scalar_one()

            cancelled_file_count = db_session.execute(
                select(func.count())
                .select_from(VectorStoreFile)
                .where(
                    and_(
                        VectorStoreFile.vector_store_id == store_id,
                        VectorStoreFile.batch_id == batch_id,
                        VectorStoreFile.state == "cancelled",
                    )
                )
            ).scalar_one()

            state: VectorStoreFileState

            if pending_file_count > 0 and processing_file_count == 0:
                state = "pending"
            elif pending_file_count > 0 and processing_file_count > 0:
                state = "processing"
            elif cancelled_file_count == total_file_count - failed_file_count:
                state = "cancelled"
            elif completed_file_count == total_file_count - failed_file_count:
                state = "completed"
            elif failed_file_count > 0:
                state = "failed"
            else:
                raise ValueError(f"Cannot determine state of vector store file batch id {batch_id}")

            return VectorStoreFileBatchStatus(
                batch_id=any_file.batch_id,
                vector_store_id=str(any_file.vector_store_id),
                create_time=any_file.create_time,
                state=state,
                total_file_count=total_file_count,
                pending_file_count=pending_file_count,
                processing_file_count=processing_file_count,
                completed_file_count=completed_file_count,
                failed_file_count=failed_file_count,
                cancelled_file_count=cancelled_file_count,
            )

    def cancel_batch(self, store_id: str | uuid.UUID, batch_id: str) -> None:
        store_id = sanitize_uuid(store_id)

        with Session(self._main_db) as db_session:
            r = db_session.execute(
                update(VectorStoreFile)
                .where(and_(VectorStoreFile.batch_id == batch_id, VectorStoreFile.state == "pending"))
                .values(state="cancelled", update_time=datetime.now())
            )

            db_session.execute(update(VectorStore).where(VectorStore.id == store_id).values(access_time=datetime.now()))

            c = r.rowcount
            db_session.commit()

            if c <= 0:
                raise RuntimeError(f"Failed to cancel batch {batch_id} of store {store_id}")

    def get_batch_files(
        self, store_id: str | uuid.UUID, batch_id: str, query: VectorStoreFileListPagedQuery
    ) -> PageResult[VectorStoreFileInfo]:
        store_id = sanitize_uuid(store_id)

        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._main_db) as db_session:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db_session.execute(
                    select(VectorStoreFile.create_time).where(VectorStoreFile.file_id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = select(VectorStoreFile).where(
                and_(VectorStoreFile.vector_store_id == store_id, VectorStoreFile.batch_id == batch_id)
            )

            if query.states is not None:
                q = q.where(VectorStoreFile.state.in_(query.states))

            if query.create_time_desc_order:
                q = q.order_by(VectorStoreFile.create_time.desc(), VectorStoreFile.file_id.desc())
            else:
                q = q.order_by(VectorStoreFile.create_time, VectorStoreFile.file_id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db_session, q, per_page=query.page_size, before=before, after=after)

            return PageResult(
                data=[VectorStoreFileInfo.from_db_vector_store_file(r._tuple()[0]) for r in data],  # noqa: SLF001
                is_last_page=not data.paging.has_next,
            )

    def clear_database(self) -> None:
        for collection_name in self._vector_db.list_collections():
            self._vector_db.drop_collection(collection_name)

        with Session(self._main_db) as db_session:
            db_session.execute(delete(VectorStore))
            db_session.execute(delete(VectorStoreFile))
            db_session.commit()
