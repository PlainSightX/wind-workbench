"""启动时读取 provider 配置；导入模块不读取凭据。"""

import os

from .contracts import AssistantError
from .retrieval import runtime_root


def provider_key():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        try:
            key = (runtime_root() / "provider-key.txt").read_text(encoding="utf-8").strip()
        except OSError:
            pass
    if not key:
        raise AssistantError("provider_not_configured")
    return key
