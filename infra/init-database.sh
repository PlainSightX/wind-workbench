#!/bin/sh
set -eu

# 只在空卷初始化执行：管理员安装扩展，应用账户只拥有本应用 schema 的建表权。
# 密码来自本机 secret 文件，不存进源码，也不把管理员账户交给应用服务。
# 命令参数中的 cat 失败不会让 psql 失败，必须先独立检查，避免创建无密码账户。
if ! app_password=$(cat /run/secrets/app_password); then
  printf 'Cannot read application password secret\n' >&2
  exit 1
fi
if [ -z "$app_password" ]; then
  printf 'Application password secret is empty\n' >&2
  exit 1
fi
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set ON_ERROR_STOP=1 --set app_password="$app_password" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE wind_app LOGIN PASSWORD :'app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE;
REVOKE CONNECT ON DATABASE wind_workbench FROM PUBLIC;
GRANT CONNECT ON DATABASE wind_workbench TO wind_app;
GRANT USAGE, CREATE ON SCHEMA public TO wind_app;
SQL
