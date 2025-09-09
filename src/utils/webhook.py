from __future__ import annotations

from typing import Dict, Optional

import discord

from .logging import get_logger


log = get_logger(__name__)


_cache: Dict[int, discord.Webhook] = {}


async def get_or_create_channel_webhook(channel: discord.TextChannel | discord.Thread) -> Optional[discord.Webhook]:
    try:
        ch_id = channel.id
        if ch_id in _cache:
            return _cache[ch_id]
        # Try to reuse an existing webhook created by the bot
        if isinstance(channel, discord.TextChannel):
            hooks = await channel.webhooks()
            for h in hooks:
                try:
                    if h.user and h.user.bot:  # created by a bot (likely us)
                        _cache[ch_id] = h
                        return h
                except Exception:
                    continue
        # Create new
        name = "Persona Bot"
        wh = await channel.create_webhook(name=name)
        _cache[ch_id] = wh
        return wh
    except discord.Forbidden:
        log.info("Missing Manage Webhooks permission in #%s", getattr(channel, "name", channel.id))
    except Exception as e:
        log.info("Webhook setup failed in %s: %s", channel, e)
    return None


async def send_via_webhook(
    channel: discord.abc.Messageable,
    content: str,
    *,
    username: Optional[str] = None,
    avatar_url: Optional[str] = None,
) -> Optional[discord.Message]:
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        return False
    wh = await get_or_create_channel_webhook(channel)
    if not wh:
        return False
    try:
        kwargs = {
            "allowed_mentions": discord.AllowedMentions.none(),
            "wait": True,
        }
        if username is not None:
            kwargs["username"] = username
        if avatar_url is not None:
            kwargs["avatar_url"] = avatar_url
        msg = await wh.send(content, **kwargs)
        return msg
    except Exception as e:
        log.info("Webhook send failed in %s: %s", channel, e)
        return None


async def ensure_channel_webhook_named(
    channel: discord.TextChannel | discord.Thread,
    name: str,
    avatar_url: Optional[str] = None,
) -> bool:
    """Ensure the channel webhook exists and is named as requested.
    Returns True on success (webhook present and named), False otherwise.
    """
    wh = await get_or_create_channel_webhook(channel)
    if not wh:
        return False
    try:
        await wh.edit(name=name, avatar=None)  # cannot set remote avatar URL directly here
        # Note: discord.py doesn't support setting webhook avatar via URL; requires bytes.
        # We keep the name updated; avatar will be provided per message when available.
        return True
    except Exception as e:
        log.info("Failed to rename webhook in %s: %s", channel, e)
        return False
