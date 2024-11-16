"""create vector store tables

Revision ID: 4fe055d6ee46
Revises: fd27ad131911
Create Date: 2024-10-26 21:12:52.444459

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4fe055d6ee46"
down_revision: str | None = "fd27ad131911"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vector_stores",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256)),
        sa.Column("embedding_model", sa.String(256), nullable=False),
        sa.Column("embedding_dimension", sa.Integer, nullable=False),
        sa.Column("expire_baseline", sa.String(64)),
        sa.Column("expire_interval", sa.Integer, nullable=False),  # Days, non-positive value means never expire
        sa.Column("expired", sa.Boolean, nullable=False),
        sa.Column("additional_data", sa.String),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("access_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),  # Won't consider files changed in this store
    )

    op.create_table(
        "vector_store_files",
        sa.Column("vector_store_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("file_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("batch_id", sa.String(64), nullable=False),
        sa.Column("chunking_strategy", sa.String),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("failed_reason", sa.String(32)),
        sa.Column("failed_message", sa.String),
        sa.Column("vector_count", sa.Integer, nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("vector_stores")
    op.drop_table("vector_store_files")
