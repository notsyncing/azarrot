"""create file store tables

Revision ID: fd27ad131911
Revises:
Create Date: 2024-09-28 15:34:21.524777

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fd27ad131911"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "files",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(256)),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("mime_type", sa.String(512)),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("purpose", sa.String(128)),
        sa.Column("is_partial", sa.Boolean, nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "partial_files",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(256), nullable=False),
        sa.Column("mime_type", sa.String(512), nullable=False),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("purpose", sa.String(128)),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("expire_time", sa.DateTime),
    )

    op.create_table(
        "partial_file_parts",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("partial_file_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("size", sa.Integer, nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("merged_order", sa.Integer),
        sa.Column("create_time", sa.DateTime, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("files")
    op.drop_table("partial_files")
    op.drop_table("partial_file_parts")
