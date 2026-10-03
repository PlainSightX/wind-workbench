"""读取事前冻结的选型，不允许HTTP请求自由选择正式测试参数。"""

import hashlib
import json
from pathlib import Path

from .development_protocol import INPUT_SHA256, TEST_START
from .sequence_protocol import CONFIG, SEQUENCE_KEYS

FINAL_SELECTION = Path(__file__).with_name("final_selection.json")


def frozen_selection():
    if not FINAL_SELECTION.is_file():
        raise ValueError("final_protocol_not_frozen")
    record = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    if (record.get("sequence_key") not in SEQUENCE_KEYS or record.get("config") != CONFIG
            or record.get("input_sha256") != INPUT_SHA256 or record.get("test_start") != TEST_START
            or record.get("seed") != 42 or record.get("test_scored") is not False
            or not record.get("development_selection_sha256") or not record.get("selected_model")):
        raise ValueError("final_protocol_invalid")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return record, hashlib.sha256(canonical.encode()).hexdigest()
