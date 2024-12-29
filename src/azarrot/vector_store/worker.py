import dataclasses
import json
import logging
import uuid
from datetime import datetime

import schedule
from langchain_milvus import Milvus
from langchain_unstructured import UnstructuredLoader
from sqlalchemy import Engine, Interval, and_, case, func, select, update
from sqlalchemy.orm import Session

from azarrot.common_types import VectorStoreFileFailedReason, VectorStoreFileState
from azarrot.config import VectorStoreConfig
from azarrot.database_schemas import VectorStore, VectorStoreFile
from azarrot.file_store import FileStore
from azarrot.frontends.backend_pipe import BackendPipe
from azarrot.models.model_manager import ModelManager
from azarrot.vector_store.local_embeddings import LocalEmbeddings
from azarrot.vector_store.manager import (
    VECTOR_STORE_DEFAULT_CHUNKING_CONFIG,
    VectorStoreChunkingConfig,
    VectorStoreManager,
)
from azarrot.vector_store.utils import METADATA_KEY_FILE_ID, make_collection_name


class VectorStoreWorkerPauseHandle:
    _worker: "VectorStoreWorker"

    def __init__(self, worker: "VectorStoreWorker") -> None:
        self._worker = worker

    def __enter__(self) -> None:
        self._worker._paused = True  # noqa: SLF001

    def __exit__(self, *args: object) -> None:
        self._worker._paused = False  # noqa: SLF001


class VectorStoreWorker:
    _log = logging.getLogger(__name__)
    _config: VectorStoreConfig
    _vector_store: VectorStoreManager
    _model_manager: ModelManager
    _file_store: FileStore
    _backend_pipe: BackendPipe
    _main_db: Engine
    _vector_db_uri: str
    _running: bool = False
    _paused: bool = False
    _jobs: list[schedule.Job]

    def __init__(
        self,
        config: VectorStoreConfig,
        vector_store: VectorStoreManager,
        model_manager: ModelManager,
        file_store: FileStore,
        backend_pipe: BackendPipe,
        main_db: Engine,
        vector_db_uri: str,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._model_manager = model_manager
        self._file_store = file_store
        self._backend_pipe = backend_pipe
        self._main_db = main_db
        self._vector_db_uri = vector_db_uri
        self._jobs = []

    def start(self) -> None:
        if self._running:
            self._log.warning("Vector store worker is already started!")
            return

        self._running = True

        j1 = schedule.every(self._config.worker_file_process_scan_interval).seconds.do(self.__worker_process_files)

        self._jobs.append(j1)

        j2 = schedule.every(self._config.worker_store_cleanup_scan_interval).seconds.do(
            self.__worker_cleanup_expired_stores
        )

        self._jobs.append(j2)

        self._log.info("Vector store worker started, config %s", json.dumps(dataclasses.asdict(self._config)))

    def stop(self) -> None:
        if not self._running:
            self._log.warning("Vector store worker is already stopped!")
            return

        self._running = False
        self._log.info("Vector store worker is stopping...")

        for j in self._jobs:
            schedule.cancel_job(j)

        self._jobs = []

    def pause(self) -> VectorStoreWorkerPauseHandle:
        return VectorStoreWorkerPauseHandle(self)

    def __worker_process_files(self) -> type[schedule.CancelJob] | None:
        if not self._running:
            self._log.info("Stopping file processing job...")
            return schedule.CancelJob

        self.__process_pending_files()

        return None

    def __try_lock_file(self, db_session: Session, file_id: uuid.UUID) -> bool:
        r = db_session.execute(
            update(VectorStoreFile)
            .where(and_(VectorStoreFile.file_id == file_id, VectorStoreFile.state == "pending"))
            .values(state="processing", update_time=datetime.now())
        )

        success = r.rowcount > 0
        db_session.commit()
        return success

    def __unlock_file(
        self,
        db_session: Session,
        file_id: uuid.UUID,
        new_state: VectorStoreFileState,
        failed_reason: VectorStoreFileFailedReason | None = None,
        failed_message: str | None = None,
    ) -> None:
        db_session.execute(
            update(VectorStoreFile)
            .where(and_(VectorStoreFile.file_id == file_id, VectorStoreFile.state == "processing"))
            .values(
                state=new_state, update_time=datetime.now(), failed_reason=failed_reason, failed_message=failed_message
            )
        )

        db_session.commit()

    def __process_pending_files(self) -> None:
        with Session(self._main_db) as db_session:
            pending_files = (
                db_session.execute(
                    select(VectorStoreFile)
                    .where(VectorStoreFile.state == "pending")
                    .order_by(VectorStoreFile.create_time)
                    .limit(self._config.worker_max_file_count_per_scan)
                )
                .scalars()
                .all()
            )

            file_count = len(pending_files)

            if file_count <= 0:
                return

            self._log.info("Got %d files to process.", file_count)

            langchain_milvus_clients: dict[uuid.UUID, Milvus] = {}

            try:
                for file in pending_files:
                    self._log.info("Processing vector store file %s", file.file_id)
                    self.__process_pending_file(db_session, file, langchain_milvus_clients)
            finally:
                for c in langchain_milvus_clients.values():
                    c.client.close()

    def __process_pending_file(
        self, db_session: Session, file: VectorStoreFile, langchain_milvus_clients: dict[uuid.UUID, Milvus]
    ) -> None:
        langchain_milvus_client: Milvus

        if file.vector_store_id not in langchain_milvus_clients:
            store = db_session.execute(
                select(VectorStore).where(VectorStore.id == file.vector_store_id)
            ).scalar_one_or_none()

            if store is None:
                self._log.error("File %s references a non-exist vector store %s", file.file_id, file.vector_store_id)

                file.state = "failed"

                return

            model_id = store.embedding_model
            model = self._model_manager.get_model(model_id)

            if model is None:
                self._log.error("Vector store %s references a non-exist embedding model %s", file.file_id, model_id)

                return

            store.access_time = datetime.now()
            db_session.commit()

            langchain_milvus_client = Milvus(
                embedding_function=LocalEmbeddings(model, self._backend_pipe),
                collection_name=make_collection_name(store.id),
                connection_args={"uri": self._vector_db_uri},
                enable_dynamic_field=True,
                auto_id=True,
            )

            langchain_milvus_clients[file.vector_store_id] = langchain_milvus_client
        else:
            langchain_milvus_client = langchain_milvus_clients[file.vector_store_id]

        if not self.__try_lock_file(db_session, file.file_id):
            self._log.warning("Failed to lock vector file %s, will skip it!", file.file_id)
            return

        if langchain_milvus_client.client.has_collection(make_collection_name(file.vector_store_id)):
            langchain_milvus_client.delete(expr=f'{METADATA_KEY_FILE_ID} == "{file.file_id}"')

        successful = False
        failed_reason: VectorStoreFileFailedReason | None = None
        failed_message = ""

        chunking_strategy: VectorStoreChunkingConfig

        if file.chunking_strategy is not None:
            cs = json.loads(file.chunking_strategy)
            chunking_strategy = VectorStoreChunkingConfig(**cs)
        else:
            chunking_strategy = VECTOR_STORE_DEFAULT_CHUNKING_CONFIG

        try:
            process_start_time = datetime.now()

            file_loader = UnstructuredLoader(
                file_path=self._file_store.make_store_file_path(file.file_id),
                partition_via_api=False,
                chunking_strategy="basic",
                max_characters=chunking_strategy.max_chunk_size_tokens,
                overlap=chunking_strategy.chunk_overlap_tokens,
            )

            documents = file_loader.load()

            load_complete_time = datetime.now()

            for d in documents:
                d.metadata[METADATA_KEY_FILE_ID] = str(file.file_id)

            vectors = langchain_milvus_client.add_documents(documents)
            vector_count = len(vectors)
            successful = True

            process_complete_time = datetime.now()

            self._log.info(
                "Successfully processed file %s, vector count %d, total time %d s (load %d s, process %d s)",
                file.file_id,
                vector_count,
                (process_complete_time - process_start_time).total_seconds(),
                (load_complete_time - process_start_time).total_seconds(),
                (process_complete_time - load_complete_time).total_seconds(),
            )

            file.vector_count = vector_count
            db_session.commit()
        except Exception as e:
            self._log.error("Failed to process file %s", file.file_id, exc_info=True)
            failed_reason = "system_error"
            failed_message = str(e)
            successful = False
        finally:
            self.__unlock_file(
                db_session,
                file.file_id,
                "completed" if successful else "failed",
                failed_reason if not successful else None,
                failed_message if not successful else None,
            )

    def __worker_cleanup_expired_stores(self) -> type[schedule.CancelJob] | None:
        if not self._running:
            self._log.info("Stopping expired store cleanup job...")
            return schedule.CancelJob

        self.__cleanup_expired_stores()

        return None

    def __cleanup_expired_stores(self) -> None:
        with Session(self._main_db) as db_session:
            expired_stores = (
                db_session.execute(
                    select(VectorStore)
                    .where(
                        and_(
                            VectorStore.expire_baseline.is_not(None),
                            (func.now() - func.cast(VectorStore.expire_interval + " DAYS", Interval))
                            > (
                                case(
                                    (VectorStore.expire_baseline == "create_time", VectorStore.create_time),
                                    (VectorStore.expire_baseline == "access_time", VectorStore.access_time),
                                    (VectorStore.expire_baseline == "update_time", VectorStore.update_time),
                                    else_=VectorStore.access_time,
                                )
                            ),
                        )
                    )
                    .limit(10)
                )
                .scalars()
                .all()
            )

            if len(expired_stores) <= 0:
                return

            for store in expired_stores:
                self._log.info(
                    "Deleting expired vector store %s (created at %s, last accessed at %s, updated at %s)",
                    store.id,
                    store.create_time,
                    store.access_time,
                    store.update_time,
                )

                self._vector_store.delete(store.id)
