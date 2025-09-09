from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .logging import get_logger
from ..config import PERSONA_DIR, STATE_FILE


log = get_logger(__name__)


def persona_path(user_id: int) -> Path:
    return PERSONA_DIR / f"{user_id}.json"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        log.error("Failed to read %s: %s", path, e)
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.error("Failed to write %s: %s", path, e)


def list_personas() -> list[int]:
    ids: list[int] = []
    for p in PERSONA_DIR.glob("*.json"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def delete_persona(user_id: int) -> bool:
    p = persona_path(user_id)
    if p.exists():
        try:
            p.unlink()
            return True
        except Exception as e:
            log.error("Failed to delete persona %s: %s", user_id, e)
            return False
    return False


def get_active_persona_map() -> Dict[str, int]:
    data = read_json(STATE_FILE) or {}
    # map: channel_id(str) -> user_id(int)
    return {k: int(v) for k, v in data.items()}


def set_active_persona(channel_id: int, user_id: int) -> None:
    data = get_active_persona_map()
    data[str(channel_id)] = int(user_id)
    write_json(STATE_FILE, data)


def clear_active_persona(channel_id: int) -> None:
    data = get_active_persona_map()
    if str(channel_id) in data:
        del data[str(channel_id)]
        write_json(STATE_FILE, data)

