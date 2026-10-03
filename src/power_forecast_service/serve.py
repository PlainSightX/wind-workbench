"""本机启动入口；Windows 的异步 PostgreSQL 连接使用 SelectorEventLoop。"""

import asyncio
import os
import selectors

import uvicorn


def make_event_loop():
    if os.name == "nt":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main():
    server = uvicorn.Server(
        uvicorn.Config(
            "power_forecast_service.main:app",
            host="127.0.0.1",
            port=int(os.getenv("WIND_API_PORT", "8000")),
        )
    )
    # 不改进程全局 policy；只为该服务创建与 psycopg 兼容的运行循环。
    asyncio.run(server.serve(), loop_factory=make_event_loop)


if __name__ == "__main__":
    main()
