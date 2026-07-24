from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest[:length]
