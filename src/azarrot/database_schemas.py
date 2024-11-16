import uuid
from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from azarrot.common_types import VectorStoreExpireBaseline, VectorStoreFileFailedReason, VectorStoreFileState


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    filename: Mapped[str | None]
    mime_type: Mapped[str | None]
    size: Mapped[int]
    checksum: Mapped[str]
    purpose: Mapped[str | None]
    is_partial: Mapped[bool]
    create_time: Mapped[datetime]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"filename={self.filename}, "
            f"mime_type={self.mime_type}, "
            f"size={self.size}, "
            f"checksum={self.checksum}, "
            f"purpose={self.purpose}, "
            f"is_partial={self.is_partial}, "
            f"create_time={self.create_time}"
            ")"
        )


class PartialFile(Base):
    __tablename__ = "partial_files"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    filename: Mapped[str]
    mime_type: Mapped[str]
    size: Mapped[int]
    purpose: Mapped[str | None]
    create_time: Mapped[datetime]
    expire_time: Mapped[datetime | None]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"filename={self.filename}, "
            f"mime_type={self.mime_type}, "
            f"size={self.size}, "
            f"purpose={self.purpose}, "
            f"create_time={self.create_time}, "
            f"expire_time={self.expire_time}"
            ")"
        )


class PartialFilePart(Base):
    __tablename__ = "partial_file_parts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    partial_file_id: Mapped[uuid.UUID]
    size: Mapped[int]
    checksum: Mapped[str]
    merged_order: Mapped[int]
    create_time: Mapped[datetime]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"partial_file_id={self.partial_file_id}, "
            f"size={self.size}, "
            f"checksum={self.checksum}, "
            f"merged_order={self.merged_order}, "
            f"create_time={self.create_time}"
            ")"
        )


class VectorStore(Base):
    __tablename__ = "vector_stores"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    embedding_model: Mapped[str]
    embedding_dimension: Mapped[int]
    expire_baseline: Mapped[VectorStoreExpireBaseline | None]
    expire_interval: Mapped[int]
    expired: Mapped[bool]
    additional_data: Mapped[str | None]
    create_time: Mapped[datetime]
    access_time: Mapped[datetime]
    update_time: Mapped[datetime]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"name={self.name}, "
            f"embedding_model={self.embedding_model}, "
            f"embedding_dimension={self.embedding_dimension}, "
            f"expire_baseline={self.expire_baseline}, "
            f"expire_interval={self.expire_interval}, "
            f"expired={self.expired}, "
            f"additional_data={self.additional_data}, "
            f"create_time={self.create_time}, "
            f"access_time={self.access_time}, "
            f"update_time={self.update_time}"
            ")"
        )


class VectorStoreFile(Base):
    __tablename__ = "vector_store_files"

    vector_store_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    file_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    batch_id: Mapped[str]
    chunking_strategy: Mapped[str | None]
    state: Mapped[VectorStoreFileState]
    vector_count: Mapped[int]
    failed_reason: Mapped[VectorStoreFileFailedReason | None]
    failed_message: Mapped[str | None]
    create_time: Mapped[datetime]
    update_time: Mapped[datetime]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vector_store_id={self.vector_store_id}, "
            f"file_id={self.file_id}, "
            f"batch_id={self.batch_id}, "
            f"chunking_strategy={self.chunking_strategy}, "
            f"state={self.state}, "
            f"vector_count={self.vector_count}, "
            f"failed_reason={self.failed_reason}, "
            f"failed_message={self.failed_message}, "
            f"create_time={self.create_time}, "
            f"update_time={self.update_time}"
            ")"
        )
