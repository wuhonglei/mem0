"""Create dream_passes and dream_actions audit tables.

Revision ID: 007
Revises: 006
Create Date: 2026-09-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dream_passes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("pass_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("agent_id", sa.String(255), nullable=True),
        sa.Column("run_id", sa.String(255), nullable=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="manual"),
        sa.Column("stats", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_dream_passes_pass_id", "dream_passes", ["pass_id"], unique=True)
    op.create_index("ix_dream_passes_user_id", "dream_passes", ["user_id"])

    op.create_table(
        "dream_actions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("pass_pk", sa.Uuid(), sa.ForeignKey("dream_passes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pass_id", sa.String(64), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("source_ids", JSONB(), nullable=True),
        sa.Column("canonical_id", sa.String(64), nullable=True),
        sa.Column("old_id", sa.String(64), nullable=True),
        sa.Column("new_id", sa.String(64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("old_content", sa.Text(), nullable=True),
        sa.Column("new_content", sa.Text(), nullable=True),
    )
    op.create_index("ix_dream_actions_pass_pk", "dream_actions", ["pass_pk"])
    op.create_index("ix_dream_actions_pass_id", "dream_actions", ["pass_id"])


def downgrade() -> None:
    op.drop_index("ix_dream_actions_pass_id", table_name="dream_actions")
    op.drop_index("ix_dream_actions_pass_pk", table_name="dream_actions")
    op.drop_table("dream_actions")
    op.drop_index("ix_dream_passes_user_id", table_name="dream_passes")
    op.drop_index("ix_dream_passes_pass_id", table_name="dream_passes")
    op.drop_table("dream_passes")
