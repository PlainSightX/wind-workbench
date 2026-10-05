"""初始化脚本必须先读到有效 secret，再执行 SQL；不连接任何数据库。"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(SH is None, reason="POSIX init script requires sh")


def run_init(tmp_path, *, password, read_exit):
    # 只替换外部命令，不改被测脚本；假密码绝不来自本机运行目录。
    commands = tmp_path / "commands"
    commands.mkdir()
    cat = commands / "cat"
    cat.write_text(
        '#!/bin/sh\n'
        'if [ "$TEST_SECRET_READ_EXIT" -ne 0 ]; then\n'
        '  printf "Permission denied\\n" >&2\n'
        '  exit "$TEST_SECRET_READ_EXIT"\n'
        'fi\n'
        'printf "%s" "$TEST_SECRET_VALUE"\n',
        encoding="utf-8",
    )
    psql = commands / "psql"
    psql.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_SQL_TRACE"\n',
        encoding="utf-8",
    )
    cat.chmod(0o700)
    psql.chmod(0o700)
    trace = tmp_path / "sql-trace.txt"
    env = dict(os.environ)
    env.update(
        POSTGRES_USER="fixture_admin",
        POSTGRES_DB="fixture_database",
        TEST_SECRET_READ_EXIT=str(read_exit),
        TEST_SECRET_VALUE=password,
        TEST_SQL_TRACE=trace.as_posix(),
    )
    result = subprocess.run(
        [SH, "-c", 'PATH="$PWD/commands:$PATH"; export PATH; . "$1"',
         "wind-init-test", (ROOT / "infra/init-database.sh").as_posix()],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10,
        check=False,
    )
    return result, trace


@pytest.mark.parametrize("read_exit,password", [(13, ""), (0, ""), (0, "\n")])
def test_invalid_secret_stops_before_any_sql(tmp_path, read_exit, password):
    result, trace = run_init(tmp_path, password=password, read_exit=read_exit)
    assert result.returncode != 0
    assert not trace.exists(), "Invalid secret must not create a passwordless role"


def test_readable_nonempty_secret_reaches_sql_without_printing_it(tmp_path):
    password = "fixture-password-not-a-credential"
    result, trace = run_init(tmp_path, password=password, read_exit=0)
    assert result.returncode == 0, result.stderr
    assert f"app_password={password}" in trace.read_text(encoding="utf-8")
    assert password not in result.stdout + result.stderr
