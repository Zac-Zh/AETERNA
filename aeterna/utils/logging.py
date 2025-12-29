import json
from pathlib import Path
from typing import Dict


def log_jsonl(path: str, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
