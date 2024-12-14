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

    op.create_table(
        "agent_chat_tasks",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("agent_id", sa.Uuid(as_uuid=True), index=True),
        sa.Column("thread_id", sa.Uuid(as_uuid=True), index=True),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("model_instruction", sa.Text),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("current_required_action", sa.String(64)),
        sa.Column("current_required_action_data", sa.Text),
        sa.Column("start_time", sa.DateTime),
        sa.Column("complete_time", sa.DateTime),
        sa.Column("error_message", sa.Text),
        sa.Column("generation_parameters", sa.Text),
        sa.Column("thread_history_strategy", sa.String(32), nullable=False),
        sa.Column("thread_history_strategy_params", sa.Text, nullable=False),
        sa.Column("max_tokens", sa.Integer, nullable=False),
        sa.Column("tools_info", sa.Text),
        sa.Column("parallel_tool_calling", sa.Boolean, nullable=False),
        sa.Column("current_generation_statistics", sa.Text),
        sa.Column("additional_data", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "agent_chat_task_tools",
        sa.Column("agent_chat_task_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(256), primary_key=True),
        sa.Column("is_internal_tool", sa.Boolean, nullable=False),
        sa.Column("tool_preset_parameters", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "agent_chat_task_details",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("agent_chat_task_id", sa.Uuid(as_uuid=True), nullable=False, index=True),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("data", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("complete_time", sa.DateTime),
        sa.Column("error_message", sa.Text),
        sa.Column("generation_statistics", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "agent_chat_messages",
        sa.Column("message_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("agent_chat_task_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("create_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_threads",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("additional_data", sa.Text),
        sa.Column("deleted", sa.Boolean, nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_thread_tool_preset_params",
        sa.Column("thread_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(256), primary_key=True),
        sa.Column("tool_preset_parameters", sa.Text, nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("thread_id", sa.Uuid(as_uuid=True), index=True),
        sa.Column("role", sa.String(64), nullable=False),
        sa.Column("order", sa.Integer, nullable=False),
        sa.Column("deleted", sa.Boolean, nullable=False),
        sa.Column("additional_data", sa.Text),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_message_contents",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("message_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text),
        sa.Column("extra_content", sa.Text),
        sa.Column("order", sa.Integer, nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
        sa.Column("update_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_message_attachments",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("message_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("file_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("create_time", sa.DateTime, nullable=False),
    )

    op.create_table(
        "chat_message_attachment_tool_exposures",
        sa.Column("attachment_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("tool_name", sa.String(256), primary_key=True),
        sa.Column("create_time", sa.DateTime, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agents")
    op.drop_table("agent_tools")
    op.drop_table("agent_chat_tasks")
    op.drop_table("agent_chat_task_tools")
    op.drop_table("agent_chat_messages")
    op.drop_table("chat_threads")
    op.drop_table("chat_thread_tool_preset_params")
    op.drop_table("chat_messages")
    op.drop_table("chat_message_contents")
    op.drop_table("chat_message_attachments")
    op.drop_table("chat_message_attachment_tool_exposures")
