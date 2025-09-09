from __future__ import annotations

from typing import List

import requests

from ..config import cfg
from ..utils.logging import get_logger


log = get_logger(__name__)


def fetch_docs_snippets(max_chars: int = 1200) -> List[str]:
    if not cfg.use_mcp_context:
        return []
    urls = [u for u in [cfg.mcp_context7_url, cfg.mcp_discord_docs_url] if u]
    snippets: List[str] = []
    for u in urls:
        try:
            r = requests.get(u, timeout=5)
            if r.ok:
                text = r.text.strip()
                if text:
                    snippets.append(text[:max_chars])
        except Exception as e:
            log.info("Skipping MCP docs fetch from %s: %s", u, e)
    return snippets

