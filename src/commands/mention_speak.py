from __future__ import annotations

import asyncio
from typing import Deque, Dict, List, Tuple
from collections import deque

import discord
from discord.ext import commands

from ..config import cfg, INDEX_DIR
from ..utils.logging import get_logger
from ..utils import persistence as pers
from ..rag.retriever import Retriever
from ..llm.local_client import client as llm
from ..llm.prompting import build_speak_prompt
from ..utils.webhook import send_via_webhook
from ..utils.concurrency import SpeakGuard


log = get_logger(__name__)


Turn = Tuple[str, str]  # (role, content) where role in {"user","assistant"}


class MentionSpeakCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.sessions: Dict[int, Deque[Turn]] = {}

    def _get_session(self, channel_id: int) -> Deque[Turn]:
        if channel_id not in self.sessions:
            self.sessions[channel_id] = deque(maxlen=cfg.session_max_turns)
        return self.sessions[channel_id]

    def _postprocess_by_traits(self, text: str, pdata: dict) -> List[str]:
        # Light-touch adjustments to reflect punctuation and burst habits.
        ts = pdata.get("text_style", {})
        conv = pdata.get("conversation", {})
        bursts = conv.get("message_bursts", "balanced")
        punct = (ts.get("punctuation") or "").lower()
        typos = (ts.get("typos") or "").lower()

        out = text.strip()
        # Punctuation emphasis
        if "!!!" in punct or "???" in punct or "frequent" in punct:
            if out.endswith("!"):
                out = out + "!"
            elif out.endswith("?"):
                out = out + "?"

        # Occasional elongated words if indicated
        if "elongated" in typos or "typos" in typos:
            import re, random

            def elongate(m):
                ch = m.group(1)
                return ch * random.randint(2, 4)

            if random.random() < 0.35:
                out = re.sub(r"([aeiouAEIOU])", elongate, out, count=1)

        if bursts.startswith("often"):
            # Split on sentences into up to 2-3 bursts
            parts = [p.strip() for p in out.replace("!?", ".").split(".") if p.strip()]
            if len(parts) >= 2:
                return [parts[0] + ".", " ".join(parts[1:])]
        return [out]

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not cfg.enable_mention_speak:
            return
        if message.author.bot:
            return
        if not message.guild:
            return
        # Only respond when bot is mentioned
        if not any(u.id == self.bot.user.id for u in message.mentions):  # type: ignore
            return

        # Require active persona for the channel
        active = pers.get_active_persona_map()
        uid = active.get(str(message.channel.id))
        if not uid:
            return await message.reply("No active persona here. Use /persona switch @user first.")
        pdata = pers.read_json(pers.persona_path(uid)) or {}
        username = pdata.get("username", str(uid))

        # Indicate typing early to show progress
        try:
            await message.channel.trigger_typing()
        except Exception:
            pass

        # Session context
        sess = self._get_session(message.channel.id)
        sess.append(("user", message.clean_content))

        # Retrieval (offload to thread to avoid blocking event loop)
        index_path = INDEX_DIR / f"{uid}.idx"
        retr = Retriever(index_path, embed_fn=llm.embed)
        k = max(1, cfg.rag_k)
        if retr.is_ready():
            try:
                retrieved = await asyncio.to_thread(
                    lambda: [t[: cfg.rag_snippet_max_chars] for t, _ in retr.query(message.clean_content, k=k)]
                )
            except Exception:
                retrieved = []
        else:
            retrieved = []

        # Build style (reuse persona speak path)
        style_prompt = pdata.get("style_prompt", "")
        from ..llm.prompting import rich_traits_to_style

        rich_block = rich_traits_to_style(
            username,
            {
                "text_style": pdata.get("text_style", {}),
                "personality": pdata.get("personality", {}),
                "conversation": pdata.get("conversation", {}),
                "topics": pdata.get("topics", {}),
                "media": pdata.get("media", {}),
            },
        )
        combined_style = (style_prompt or "").strip()
        if rich_block.strip():
            combined_style = (combined_style + "\n\n" + rich_block).strip()
        if len(combined_style) > cfg.style_max_chars:
            combined_style = combined_style[: cfg.style_max_chars]

        # Simple context window rendering
        context_lines = []
        for role, content in list(sess):
            tag = "User" if role == "user" else username
            context_lines.append(f"{tag}: {content}")
        context_block = "\n".join(context_lines[- (cfg.session_max_turns - 1) :])
        if context_block:
            context_block = "\n\n[Conversation so far]\n" + context_block

        sprompt = build_speak_prompt(username, combined_style + context_block, message.clean_content, retrieved)
        # Concurrency guard to limit parallel LLM calls; keep UI responsive
        async with SpeakGuard(message.channel.id):
            async with message.channel.typing():
                reply = await asyncio.to_thread(
                    llm.complete,
                    sprompt,
                    max_tokens=cfg.speak_max_tokens,
                    temperature=cfg.speak_temperature,
                    num_ctx=cfg.speak_num_ctx,
                )
        sess.append(("assistant", reply))

        # Punctuation/burst-aware sending
        parts = self._postprocess_by_traits(reply, pdata)
        # Try webhook with persona-styled display name; fallback to channel.send
        avatar_url = None
        try:
            mem = message.guild.get_member(uid) if message.guild else None
            if not mem and message.guild:
                mem = await message.guild.fetch_member(uid)
            if mem and getattr(mem, "display_avatar", None):
                avatar_url = mem.display_avatar.url  # type: ignore[attr-defined]
        except Exception:
            pass
        for i, part in enumerate(parts):
            if i > 0:
                await asyncio.sleep(cfg.burst_send_delay_ms / 1000.0)
            # Use webhook with name set at /persona switch; do not override per message
            msg = await send_via_webhook(
                message.channel,  # type: ignore[arg-type]
                part,
                username=None,
                avatar_url=avatar_url,
            )
            if msg is None:
                tag = f"Persona Bot (@{username})"
                await message.channel.send(f"{tag} {part}")


async def setup(bot: commands.Bot):
    cog = MentionSpeakCog(bot)
    await bot.add_cog(cog)
