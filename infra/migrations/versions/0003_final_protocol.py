"""同一正式评价协议只能受理一个任务；旧开发任务不受此唯一约束影响。"""

import sqlalchemy as sa
from alembic import op

revision = "0003_final_protocol"
down_revision = "0002_model_artifacts"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index("uq_final_evaluation_protocol", "experiment_tasks",
                    [sa.text("(spec ->> 'final_protocol_id')")], unique=True,
                    postgresql_where=sa.text("spec ->> 'purpose' = 'final_evaluation'"))


def downgrade():
    op.drop_index("uq_final_evaluation_protocol", table_name="experiment_tasks")
