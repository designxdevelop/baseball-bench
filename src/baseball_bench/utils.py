from __future__ import annotations

import json
import random
import re
from hashlib import sha256
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(normalized.encode("utf-8")).hexdigest()


def seeded_random(seed: int | str) -> random.Random:
    if isinstance(seed, str):
        seed = int(sha256(seed.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed)


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_answer(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).strip()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "item"

