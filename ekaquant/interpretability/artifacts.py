from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable
from uuid import uuid4


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_run_dir(base_dir: str, prefix: str) -> str:
    ensure_dir(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid4().hex[:8]
    run_dir = os.path.join(base_dir, f"{prefix}_{stamp}_{run_id}")
    ensure_dir(run_dir)
    return run_dir


def write_json(path: str, payload: Dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_jsonl(path: str, rows: Iterable[Dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
