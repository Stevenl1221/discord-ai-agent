from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import DATA_DIR, cfg
from .logging import get_logger


log = get_logger(__name__)

_PATH: Path = DATA_DIR / "caption_cache.json"


def _load() -> Dict[str, Any]:
    try:
        if _PATH.exists():
            return json.loads(_PATH.read_text())
    except Exception as e:
        log.info("caption cache load failed: %s", e)
    return {}


def _save(data: Dict[str, Any]) -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(data))
    except Exception as e:
        log.info("caption cache save failed: %s", e)


def purge_expired(ttl_seconds: int | None = None) -> None:
    ttl = ttl_seconds or cfg.caption_ttl_seconds
    now = int(time.time())
    data = _load()
    removed = 0
    for k in list(data.keys()):
        try:
            if now - int(data[k].get("ts", 0)) > ttl:
                data.pop(k, None)
                removed += 1
        except Exception:
            data.pop(k, None)
            removed += 1
    if removed:
        _save(data)


def get(url: str) -> Optional[str]:
    try:
        rec = _load().get(url)
        if rec and isinstance(rec, dict):
            return str(rec.get("caption")) if rec.get("caption") else None
    except Exception:
        return None
    return None


def set(url: str, caption: str) -> None:
    try:
        data = _load()
        data[url] = {"caption": caption, "ts": int(time.time())}
        _save(data)
    except Exception as e:
        log.info("caption cache set failed: %s", e)

