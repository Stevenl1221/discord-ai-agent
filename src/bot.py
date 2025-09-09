from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands

from .config import cfg
from .utils.logging import get_logger


log = get_logger(__name__)


class PersonaBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # must be enabled in Dev Portal too
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # Load persona cog
        from .commands import persona as persona_mod
        from .commands import mention_speak as mention_mod

        await persona_mod.setup(self)
        await mention_mod.setup(self)
        # Sync slash commands (guild-scoped only)
        if cfg.guild_id:
            guild = discord.Object(id=int(cfg.guild_id))
            await self.tree.sync(guild=guild)
            log.info("Synced commands to guild %s", cfg.guild_id)
        else:
            # Intentionally skipping global sync to keep commands guild-scoped only.
            log.warning("GUILD_ID not set; skipping command sync. Set DISCORD_GUILD_ID or GUILD_ID in .env.")

    async def on_ready(self):
        log.info("Bot ready as %s (%s)", self.user, getattr(self.user, 'id', ''))
        # Optional non-blocking prewarm to reduce first-token latency
        from .config import cfg as _cfg
        if _cfg.prewarm_llm:
            import asyncio
            from .llm.local_client import client as _llm

            async def _prewarm():
                try:
                    # Run in a thread to avoid blocking the event loop
                    await asyncio.to_thread(_llm.complete, "ok", max_tokens=8)
                    log.info("LLM prewarm completed")
                except Exception as e:
                    log.info("LLM prewarm skipped: %s", e)

            asyncio.create_task(_prewarm())


def main():
    if not cfg.discord_token:
        raise SystemExit("DISCORD_TOKEN not set in .env")
    bot = PersonaBot()
    bot.run(cfg.discord_token)


if __name__ == "__main__":
    main()
