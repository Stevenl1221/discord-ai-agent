from __future__ import annotations

from typing import List, Optional, Set, Tuple, Dict
from datetime import datetime

import discord
from ..utils.logging import get_logger


log = get_logger(__name__)


async def fetch_recent_messages_from_channel(
    channel: discord.abc.Messageable,
    user: Optional[discord.abc.User],
    limit: int = 200,
    user_ids: Optional[Set[int]] = None,
    include_non_text: bool = False,
    after: Optional[datetime] = None,
) -> List[str]:
    """
    Fetch recent messages from a channel filtered by author.

    - If user_ids is provided, include messages from authors whose IDs are in that set.
    - Else, if user is provided, include only that user's messages.
    - If include_non_text is True, include simple placeholders for attachment-only messages.
    Returns list oldest-first.
    """
    messages: List[str] = []
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            async for m in channel.history(limit=limit, after=after):
                allow = False
                if user_ids is not None:
                    allow = m.author and (m.author.id in user_ids)
                elif user is not None:
                    allow = m.author and (m.author.id == user.id)
                if not allow:
                    continue
                if m.content:
                    messages.append(m.content)
                elif include_non_text and getattr(m, "attachments", None):
                    try:
                        names = ", ".join(att.filename for att in m.attachments)
                        messages.append(f"[attachments: {names}]")
                    except Exception:
                        messages.append("[attachments]")
        else:
            log.info("Channel type %s not supported for history", type(channel))
    except discord.Forbidden:
        log.warning("Missing permissions to read history in #%s", getattr(channel, "name", channel.id))
    except discord.HTTPException as e:
        log.warning("HTTP error while fetching history: %s", e)
    return list(reversed(messages))  # oldest first


async def fetch_texts_and_image_urls_from_channel(
    channel: discord.abc.Messageable,
    user: discord.abc.User,
    limit: int = 400,
    after: Optional[datetime] = None,
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    images: List[str] = []
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            async for m in channel.history(limit=limit, after=after):
                if m.author.id != user.id:
                    continue
                if m.content:
                    texts.append(m.content)
                if getattr(m, "attachments", None):
                    for att in m.attachments:
                        try:
                            if (att.content_type and att.content_type.startswith("image/")) or str(att.filename).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                images.append(att.url)
                        except Exception:
                            continue
        else:
            log.info("Channel type %s not supported for history", type(channel))
    except discord.Forbidden:
        log.warning("Missing permissions to read history in #%s", getattr(channel, "name", channel.id))
    except discord.HTTPException as e:
        log.warning("HTTP error while fetching history: %s", e)
    return list(reversed(texts)), list(reversed(images))


async def fetch_texts_and_image_urls_multi(
    channel: discord.abc.Messageable,
    user_ids: Set[int],
    limit: int = 600,
    after: Optional[datetime] = None,
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    images: List[str] = []
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            async for m in channel.history(limit=limit, after=after):
                if not m.author or m.author.id not in user_ids:
                    continue
                if m.content:
                    texts.append(m.content)
                if getattr(m, "attachments", None):
                    for att in m.attachments:
                        try:
                            if (att.content_type and att.content_type.startswith("image/")) or str(att.filename).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                images.append(att.url)
                        except Exception:
                            continue
        else:
            log.info("Channel type %s not supported for history", type(channel))
    except discord.Forbidden:
        log.warning("Missing permissions to read history in #%s", getattr(channel, "name", channel.id))
    except discord.HTTPException as e:
        log.warning("HTTP error while fetching history: %s", e)
    return list(reversed(texts)), list(reversed(images))


async def fetch_image_items_from_channel(
    channel: discord.abc.Messageable,
    user: discord.abc.User,
    limit: int = 400,
    after: Optional[datetime] = None,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            async for m in channel.history(limit=limit, after=after):
                if m.author.id != user.id:
                    continue
                msg_text = m.content or ""
                if getattr(m, "attachments", None):
                    for att in m.attachments:
                        try:
                            if (att.content_type and att.content_type.startswith("image/")) or str(att.filename).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                items.append({
                                    "url": att.url,
                                    "text": msg_text,
                                    "filename": att.filename or "",
                                })
                        except Exception:
                            continue
        else:
            log.info("Channel type %s not supported for history", type(channel))
    except discord.Forbidden:
        log.warning("Missing permissions to read history in #%s", getattr(channel, "name", channel.id))
    except discord.HTTPException as e:
        log.warning("HTTP error while fetching history: %s", e)
    return list(reversed(items))


async def fetch_image_items_multi(
    channel: discord.abc.Messageable,
    user_ids: Set[int],
    limit: int = 600,
    after: Optional[datetime] = None,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            async for m in channel.history(limit=limit, after=after):
                if not m.author or m.author.id not in user_ids:
                    continue
                msg_text = m.content or ""
                if getattr(m, "attachments", None):
                    for att in m.attachments:
                        try:
                            if (att.content_type and att.content_type.startswith("image/")) or str(att.filename).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                items.append({
                                    "url": att.url,
                                    "text": msg_text,
                                    "filename": att.filename or "",
                                })
                        except Exception:
                            continue
        else:
            log.info("Channel type %s not supported for history", type(channel))
    except discord.Forbidden:
        log.warning("Missing permissions to read history in #%s", getattr(channel, "name", channel.id))
    except discord.HTTPException as e:
        log.warning("HTTP error while fetching history: %s", e)
    return list(reversed(items))
