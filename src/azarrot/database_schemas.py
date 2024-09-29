import uuid
from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    filename: Mapped[str | None]
    size: Mapped[int]
    checksum: Mapped[str]
    purpose: Mapped[str | None]
    create_time: Mapped[datetime]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"filename={self.filename}, "
            f"size={self.size}, "
            f"checksum={self.checksum}, "
            f"purpose={self.purpose}, "
            f"create_time={self.create_time}"
            ")"
        )

