"""create agent and chat tables

Revision ID: 783ebcfd29b1
Revises: 4fe055d6ee46
Create Date: 2024-11-23 17:38:33.081990

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "783ebcfd29b1"
down_revision: str | None = "4fe055d6ee46"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256)),
        sa.Column("description", sa.String(2048)),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("model_instruction", sa.Text),
        sa.Column("default_generation_parameters", sa.Text),
        sa.Column("additional_data", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "agent_tools",
        sa.Column("agent_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(256), primary_key=True),
        sa.Column("is_internal_tool", sa.Boolean, nullable=False),
        sa.Column("tool_preset_parameters", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agents")
    op.drop_table("agent_tools")
