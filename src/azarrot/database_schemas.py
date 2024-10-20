import uuid
from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
