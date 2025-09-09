from __future__ import annotations

import asyncio
from typing import Dict

from ..config import cfg


_speak_sem = asyncio.Semaphore(max(1, cfg.speak_global_concurrency))
_chan_locks: Dict[int, asyncio.Lock] = {}


def channel_lock(channel_id: int) -> asyncio.Lock:
    lock = _chan_locks.get(channel_id)
    if lock is None:
        lock = asyncio.Lock()
        _chan_locks[channel_id] = lock
    return lock


class SpeakGuard:
    def __init__(self, channel_id: int | None):
        self._chan_id = channel_id
        self._chan_lock = channel_lock(channel_id) if (cfg.speak_channel_exclusive and channel_id is not None) else None

    async def __aenter__(self):
        await _speak_sem.acquire()
        if self._chan_lock is not None:
            await self._chan_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._chan_lock is not None and self._chan_lock.locked():
            self._chan_lock.release()
        _speak_sem.release()

