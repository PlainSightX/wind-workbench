"""检查真实开发数据库；重启探针仅创建自己命名的表，验证后清理。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".local/runtime/restart-probe.json"


class RollbackProbe(Exception):
    pass


def compose_connection() -> tuple[Path, int]:
    """读取实际 Compose 配置；新终端不依赖手工导出的另一套默认值。"""
    docker = shutil.which("docker")
    if docker is None:
        docker = str(Path.home() / "AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe")
    result = subprocess.run(
        [docker, "compose", "--profile", "app", "config", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    config = json.loads(result.stdout)
    if config["name"] == "wind-workbench":
        raise RuntimeError("Retired 8000 stack is not an ordinary diagnostic target")
    password_file = Path(config["secrets"]["app_password"]["file"])
    port = next(p["published"] for p in config["services"]["postgres"]["ports"] if p["target"] == 5432)
    return password_file, int(port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["check", "prepare-restart", "verify-restart"], default="check", nargs="?"
    )
    args = parser.parse_args()
    password_file, port = compose_connection()
    password = password_file.read_text(encoding="utf-8").strip()
    with psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="wind_workbench",
        user="wind_app",
        password=password,
        connect_timeout=5,
        autocommit=True,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_user, rolsuper, rolcreatedb, rolcreaterole "
                "FROM pg_roles WHERE rolname=current_user"
            )
            role = cursor.fetchone()
            if role != ("wind_app", False, False, False):
                raise RuntimeError("Application role has unexpected privileges")
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            extension = cursor.fetchone()
            if extension is None:
                raise RuntimeError("pgvector extension is missing")
            cursor.execute("CREATE TEMP TABLE rollback_probe (id integer PRIMARY KEY)")
            try:
                with connection.transaction():
                    cursor.execute("INSERT INTO rollback_probe VALUES (1)")
                    raise RollbackProbe()
            except RollbackProbe:
                pass
            cursor.execute("SELECT count(*) FROM rollback_probe")
            if cursor.fetchone()[0] != 0:
                raise RuntimeError("Transaction rollback did not remove the row")

            result = {"application_role": role[0], "pgvector": extension[0], "rollback": "passed"}
            if args.mode == "prepare-restart":
                if STATE.exists():
                    raise RuntimeError(
                        "A restart probe is pending; verify it before preparing another"
                    )
                token = uuid4()
                table = "infra_probe_" + token.hex
                # 唯一表名限定清理范围，不触碰真实应用表或其他任务的数据。
                STATE.write_text(json.dumps({"token": str(token)}), encoding="utf-8")
                with connection.transaction():
                    cursor.execute(
                        sql.SQL("CREATE TABLE {} (marker uuid PRIMARY KEY)").format(
                            sql.Identifier(table)
                        )
                    )
                    cursor.execute(
                        sql.SQL("INSERT INTO {} VALUES (%s)").format(sql.Identifier(table)),
                        (token,),
                    )
                result["restart_probe"] = "prepared; restart only this compose PostgreSQL service"
            elif args.mode == "verify-restart":
                token = UUID(json.loads(STATE.read_text(encoding="utf-8"))["token"])
                table = "infra_probe_" + token.hex
                with connection.transaction():
                    cursor.execute(sql.SQL("SELECT marker FROM {}").format(sql.Identifier(table)))
                    if cursor.fetchall() != [(token,)]:
                        raise RuntimeError("Committed probe data was not preserved")
                    cursor.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
                STATE.unlink()
                result["restart_probe"] = "passed and probe table removed"
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
