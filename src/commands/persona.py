from __future__ import annotations

from typing import Optional, List

import discord
from discord import app_commands
from discord.ext import commands

from ..config import cfg, INDEX_DIR
from ..utils.logging import get_logger
from ..utils import persistence as pers
from ..ingest.discord_fetch import (
    fetch_recent_messages_from_channel,
    fetch_texts_and_image_urls_from_channel,
    fetch_texts_and_image_urls_multi,
    fetch_image_items_from_channel,
    fetch_image_items_multi,
)
from ..ingest.preprocess import clean_texts, extract_basic_traits, extract_rich_traits
from ..llm.local_client import client as llm
from ..llm.prompting import (
    build_persona_creation_prompt,
    build_speak_prompt,
    build_summarize_prompt,
    style_from_traits,
)
from ..rag.retriever import Retriever
from ..utils.webhook import send_via_webhook, ensure_channel_webhook_named
from ..utils.progress import bar as progress_bar
from ..utils import caption_cache as capcache


log = get_logger(__name__)


class PersonaCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    persona = app_commands.Group(name="persona", description="Manage AI personas")

    @persona.command(name="create", description="Create a persona from recent messages of a user")
    async def persona_create(self, interaction: discord.Interaction, user: discord.Member):
        await interaction.response.defer(thinking=True)
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return await interaction.followup.send("Please run in a text channel or thread.")
        if not interaction.client.intents.message_content:
            return await interaction.followup.send("Message Content intent is disabled. Cannot read history.")

        # Quick path: compute and save initial persona, then finalize in background
        import asyncio as _asyncio, time as _time
        start_ts = _time.monotonic()

        texts, image_urls = await fetch_texts_and_image_urls_from_channel(channel, user, limit=cfg.create_msg_fetch_limit)
        if not texts and not image_urls:
            return await interaction.followup.send("No recent messages found from that user in this channel.")
        texts = clean_texts(texts)
        traits = extract_basic_traits(texts)
        rich_traits = extract_rich_traits(texts, media_captions=[])
        base_style = style_from_traits(user.display_name, traits)
        doc = {
            "user_id": user.id,
            "username": user.display_name,
            "version": 1,
            "style_prompt": base_style[:1200],
            "traits": traits,
            "text_style": rich_traits.get("text_style", {}),
            "personality": rich_traits.get("personality", {}),
            "conversation": rich_traits.get("conversation", {}),
            "topics": rich_traits.get("topics", {}),
            "beliefs": rich_traits.get("beliefs", {}),
            "culture": rich_traits.get("culture", {}),
            "media": {},
            "examples": [],
        }
        pers.write_json(pers.persona_path(user.id), doc)

        # Post immediate progress message in channel
        prog_msg = None
        try:
            prog_msg = await channel.send(
                f"Persona init for @{user.display_name} saved. Enriching style & indexing in background…"
            )
        except Exception:
            prog_msg = None

        async def _finalize():
            try:
                # Optionally skip images entirely for speed
                captions: List[str] = []
                if cfg.create_include_images:
                    try:
                        items = await fetch_image_items_from_channel(channel, user, limit=cfg.create_msg_fetch_limit)
                    except Exception:
                        items = [{"url": u, "text": "", "filename": ""} for u in image_urls]
                    items = items[: max(0, cfg.create_image_captions)]
                    sem = _asyncio.Semaphore(max(1, cfg.create_caption_concurrency))

                    async def _cap(it: dict):
                        async with sem:
                            try:
                                c = await _asyncio.to_thread(
                                    llm.vision_describe,
                                    it.get("url", ""),
                                    hint=(it.get("text") or None),
                                    filename=(it.get("filename") or None),
                                )
                                if c and cfg.caption_refine:
                                    rp = (
                                        "Refine this image caption to be strictly factual and grounded in any context text."
                                        " Remove any guesses. If unsure, say 'uncertain'.\n"
                                        f"Caption: {c}\nContext: {it.get('text','')}"
                                    )
                                    c = await _asyncio.to_thread(
                                        llm.complete,
                                        rp,
                                        max_tokens=cfg.caption_refine_max_tokens,
                                        temperature=0.2,
                                        top_p=0.5,
                                    )
                                if c:
                                    captions.append(c)
                            except Exception:
                                return

                    await _asyncio.gather(*(_cap(it) for it in items))

                # Refine style via LLM (hierarchical)
                from ..llm.prompting import build_merge_style_prompt
                refined_style = None
                try:
                    guides: List[str] = []
                    # Enforce time budget for style build
                    now = _time.monotonic()
                    remaining = max(0.0, cfg.create_time_budget_seconds - (now - start_ts))
                    do_hier = cfg.create_hierarchical and remaining > 30
                    if do_hier:
                        # Use up to create_style_msgs * chunks recent messages to build chunk guides
                        desired_chunks = cfg.create_chunk_count if remaining > 60 else max(2, min(cfg.create_chunk_count, 2))
                        recent_for_style = texts[-(cfg.create_style_msgs * max(1, desired_chunks)) :]
                        chunks = max(1, desired_chunks)
                        size = max(1, (len(recent_for_style) + chunks - 1) // chunks)
                        chunk_lists = [recent_for_style[i : i + size] for i in range(0, len(recent_for_style), size)]
                        for idx, cl in enumerate(chunk_lists, start=1):
                            sp = build_persona_creation_prompt(user.display_name, cl)
                            g = await _asyncio.to_thread(
                                llm.complete,
                                sp,
                                max_tokens=min((64 if remaining < 60 else cfg.create_chunk_max_tokens), cfg.create_max_tokens),
                                temperature=cfg.create_temperature,
                                num_ctx=cfg.create_num_ctx,
                                top_p=cfg.create_top_p,
                                model=cfg.create_model_name or cfg.text_model_name,
                            )
                            guides.append(g)
                            if prog_msg:
                                try:
                                    pct = 20 + int(idx * 30 / max(1, len(chunk_lists)))
                                    await prog_msg.edit(content=f"Persona style building for @{user.display_name}: {progress_bar(pct)}")
                                except Exception:
                                    pass
                        media_keywords = []
                        try:
                            from ..ingest.preprocess import extract_rich_traits as _ert

                            mk = _ert(texts, media_captions=captions).get("media", {}).get("keywords", [])
                            media_keywords = mk if isinstance(mk, list) else []
                        except Exception:
                            media_keywords = []
                        mp = build_merge_style_prompt(user.display_name, guides, media_keywords=media_keywords)
                        refined_style = await _asyncio.to_thread(
                            llm.complete,
                            mp,
                            max_tokens=cfg.create_max_tokens,
                            temperature=cfg.create_temperature,
                            num_ctx=cfg.create_num_ctx,
                            top_p=cfg.create_top_p,
                            model=cfg.create_model_name or cfg.text_model_name,
                        )
                    else:
                        sprompt = build_persona_creation_prompt(user.display_name, texts[-cfg.create_style_msgs:])
                        refined_style = await _asyncio.to_thread(
                            llm.complete,
                            sprompt,
                            max_tokens=cfg.create_max_tokens,
                            temperature=cfg.create_temperature,
                            num_ctx=cfg.create_num_ctx,
                            top_p=cfg.create_top_p,
                            model=cfg.create_model_name or cfg.text_model_name,
                        )
                except Exception:
                    refined_style = None
                doc_update = pers.read_json(pers.persona_path(user.id)) or {}
                if refined_style and not refined_style.strip().startswith("[stubbed LLM]"):
                    doc_update["style_prompt"] = refined_style[: min(cfg.style_max_chars, 1200)]
                doc_update["media"] = {"captions": captions[:20]}
                pers.write_json(pers.persona_path(user.id), doc_update)
                if prog_msg:
                    try:
                        await prog_msg.edit(content=f"Persona style enriched for @{user.display_name}. Indexing…")
                    except Exception:
                        pass

                # Index with progress
                index_path = INDEX_DIR / f"{user.id}.idx"
                reps = texts[-cfg.create_index_snippets:]
                # Cap index size based on remaining time budget estimate
                now2 = _time.monotonic()
                remaining2 = max(0.0, cfg.create_time_budget_seconds - (now2 - start_ts))
                est_ms = max(10, cfg.create_embed_time_est_ms)
                allowed = max(20, int((remaining2 * 1000) / est_ms))
                if len(reps) > allowed:
                    reps = reps[-allowed:]
                retr = Retriever(index_path, embed_fn=llm.embed)
                total = len(reps)
                batch_size = max(8, min(64, cfg.embed_concurrency * 8))
                for i in range(0, total, batch_size):
                    batch = reps[i : i + batch_size]
                    await _asyncio.to_thread(retr.add_texts, batch)
                    pct = int(min(total, i + len(batch)) * 100 / max(1, total))
                    if prog_msg:
                        try:
                            await prog_msg.edit(content=f"Indexing progress for @{user.display_name}: {progress_bar(pct)} ({min(total, i+len(batch))}/{total})")
                        except Exception:
                            pass
                if prog_msg:
                    try:
                        await prog_msg.edit(content=f"Indexing complete for @{user.display_name}: {progress_bar(100)} ({total}/{total})")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    await channel.send(f"Persona finalize failed: {e}")
                except Exception:
                    pass

        _asyncio.create_task(_finalize())

        elapsed = _time.monotonic() - start_ts
        await interaction.followup.send(
            f"Persona created for @{user.display_name} with initial style. Continuing enrichment and indexing in background (took {elapsed:.1f}s)."
        )

    @persona.command(name="switch", description="Set the active persona for this channel")
    async def persona_switch(self, interaction: discord.Interaction, user: discord.Member):
        if not pers.read_json(pers.persona_path(user.id)):
            return await interaction.response.send_message("Persona not found. Run /persona create first.", ephemeral=True)
        pers.set_active_persona(interaction.channel_id, user.id)
        # Try to rename (or create) the channel webhook to reflect the active persona
        try:
            if isinstance(interaction.channel, (discord.TextChannel, discord.Thread)):
                await ensure_channel_webhook_named(
                    interaction.channel,
                    name=f"Persona Bot (@{user.display_name})",
                )
        except Exception:
            pass
        await interaction.response.send_message(f"Active persona set to @{user.display_name} for this channel.")

    @persona.command(name="speak", description="Speak using the current active persona")
    async def persona_speak(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(thinking=True)
        active_map = pers.get_active_persona_map()
        uid = active_map.get(str(interaction.channel_id))
        if not uid:
            return await interaction.followup.send("No active persona. Use /persona switch @user first.")
        pdata = pers.read_json(pers.persona_path(uid))
        if not pdata:
            return await interaction.followup.send("Active persona data missing. Re-create it.")

        # Concurrency guard
        from ..utils.concurrency import SpeakGuard
        import asyncio, time
        async with SpeakGuard(interaction.channel_id if interaction.channel_id else None):
            index_path = INDEX_DIR / f"{uid}.idx"
            retr = Retriever(index_path, embed_fn=llm.embed)
            k = max(1, cfg.rag_k)
            if retr.is_ready():
                try:
                    retrieved = await asyncio.to_thread(
                        lambda: [t[: cfg.rag_snippet_max_chars] for t, _ in retr.query(prompt, k=k)]
                    )
                except Exception:
                    retrieved = []
            else:
                retrieved = []
            style_prompt = pdata.get("style_prompt", "")
            traits = pdata.get("traits", {})
            # Merge in rich traits as an additional style guidance block
            from ..llm.prompting import rich_traits_to_style
            rich_block = rich_traits_to_style(
                pdata.get("username", str(uid)),
                {
                    "text_style": pdata.get("text_style", {}),
                    "personality": pdata.get("personality", {}),
                    "conversation": pdata.get("conversation", {}),
                    "topics": pdata.get("topics", {}),
                    "media": pdata.get("media", {}),
                },
            )
            combined_style = (style_prompt or "")
            if rich_block.strip():
                combined_style = (combined_style + "\n\n" + rich_block).strip()
            if len(combined_style) > cfg.style_max_chars:
                combined_style = combined_style[: cfg.style_max_chars]
            if not style_prompt or "You are to analyze the writing style" in style_prompt or style_prompt.strip().startswith("[stubbed LLM]"):
                # Repair older personas created while LLM was stubbed
                style_prompt = style_from_traits(pdata.get("username", str(uid)), traits if isinstance(traits, dict) else {})
            sprompt = build_speak_prompt(pdata.get("username", str(uid)), combined_style, prompt, retrieved)
            # Streaming or regular generation
            draft = ""
            if cfg.speak_stream:
                # Send initial message via webhook (or fallback) and edit as tokens arrive
                uname = str(pdata.get('username', uid))
                avatar_url = None
                try:
                    if interaction.guild:
                        mem = interaction.guild.get_member(uid) or await interaction.guild.fetch_member(uid)  # type: ignore[arg-type]
                        if mem and getattr(mem, "display_avatar", None):
                            avatar_url = mem.display_avatar.url  # type: ignore[attr-defined]
                except Exception:
                    pass
                msg = await send_via_webhook(
                    interaction.channel,  # type: ignore[arg-type]
                    "…",
                    username=None,
                    avatar_url=avatar_url,
                )
                # If webhook send failed, fallback to a normal followup message we can edit
                if msg is None:
                    msg = await interaction.followup.send("…")
                last_edit = 0.0
                buf = []
                start = time.time()
                def on_delta(s: str):
                    nonlocal last_edit, buf
                    buf.append(s)
                    now = time.time()
                    # Coalesce small chunks and edit at most every interval
                    if (now - last_edit) * 1000.0 >= cfg.stream_edit_interval_ms and sum(len(x) for x in buf) >= cfg.stream_min_chunk_chars:
                        content = "".join(buf)
                        # Schedule edit in event loop
                        try:
                            asyncio.create_task(msg.edit(content=content))
                            last_edit = now
                            buf.clear()
                        except Exception:
                            pass
                # Run stream in thread to avoid blocking
                async with interaction.channel.typing():  # type: ignore
                    draft = await asyncio.to_thread(
                        llm.complete_stream,
                        sprompt,
                        on_delta,
                        time_budget_sec=float(cfg.speak_time_budget_seconds),
                        max_tokens=cfg.speak_max_tokens,
                        temperature=cfg.speak_temperature,
                        num_ctx=cfg.speak_num_ctx,
                    )
                # Flush remaining buffer
                try:
                    if buf:
                        await msg.edit(content="".join(buf))
                except Exception:
                    pass
            else:
                # Typing indicator and offload
                async with interaction.channel.typing():  # type: ignore
                    draft = await asyncio.to_thread(
                        llm.complete,
                        sprompt,
                        max_tokens=cfg.speak_max_tokens,
                        temperature=cfg.speak_temperature,
                        num_ctx=cfg.speak_num_ctx,
                    )

            # Anti-regurgitation
            sim = retr.similarity_to_nearest(draft) if retr.is_ready() else 0.0
            if sim > 0.92:
                sprompt2 = sprompt + "\n\nRephrase completely in your own words and avoid phrases from snippets."
                draft = await asyncio.to_thread(
                    llm.complete,
                    sprompt2,
                    max_tokens=cfg.speak_max_tokens,
                    temperature=cfg.speak_temperature,
                    num_ctx=cfg.speak_num_ctx,
                )

            # If we streamed, the message is already sent/edited. Just send ephemeral ack.
            if cfg.speak_stream:
                try:
                    await interaction.followup.send("Sent.", ephemeral=True)
                except Exception:
                    pass
            else:
                # Non-streaming: send now via webhook
                uname = str(pdata.get('username', uid))
                avatar_url = None
                try:
                    if interaction.guild:
                        mem = interaction.guild.get_member(uid) or await interaction.guild.fetch_member(uid)  # type: ignore[arg-type]
                        if mem and getattr(mem, "display_avatar", None):
                            avatar_url = mem.display_avatar.url  # type: ignore[attr-defined]
                except Exception:
                    pass
                msg2 = await send_via_webhook(
                    interaction.channel,  # type: ignore[arg-type]
                    draft,
                    username=None,
                    avatar_url=avatar_url,
                )
                if msg2 is None:
                    tag = f"Persona Bot (@{uname})"
                    await interaction.followup.send(f"{tag} {draft}")

    @persona.command(name="list", description="List cached personas")
    async def persona_list(self, interaction: discord.Interaction):
        ids = pers.list_personas()
        if not ids:
            return await interaction.response.send_message("No personas cached yet.")
        names = []
        for uid in ids:
            doc = pers.read_json(pers.persona_path(uid)) or {}
            names.append(f"@{doc.get('username', uid)} ({uid})")
        await interaction.response.send_message("Personas: " + ", ".join(names))

    @persona.command(name="erase", description="Delete a persona and its index")
    async def persona_erase(self, interaction: discord.Interaction, user: discord.Member):
        ok = pers.delete_persona(user.id)
        # delete index files
        idx = INDEX_DIR / f"{user.id}.idx"
        npz = idx
        faiss_file = INDEX_DIR / (f"{user.id}.idx.faiss")
        texts_file = INDEX_DIR / (f"{user.id}.idx.texts.json")
        meta_file = INDEX_DIR / (f"{user.id}.idx.meta.json")
        for p in [npz, faiss_file, texts_file, meta_file]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        if ok:
            await interaction.response.send_message(f"Persona for @{user.display_name} erased.")
        else:
            await interaction.response.send_message("Persona not found or failed to delete.")

    @persona.command(name="update", description="Update a persona incrementally (MVP stub)")
    async def persona_update(self, interaction: discord.Interaction, user: discord.Member, since: Optional[int] = None):
        await interaction.response.defer(thinking=True)
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return await interaction.followup.send("Please run in a text channel or thread.")
        if not interaction.client.intents.message_content:
            return await interaction.followup.send("Message Content intent is disabled. Cannot read history.")

        # Determine window (days)
        from datetime import datetime, timedelta, timezone
        days = int(since) if since is not None else 7
        after = datetime.now(timezone.utc) - timedelta(days=max(1, days))

        # Fetch new texts and images
        texts, image_urls = await fetch_texts_and_image_urls_from_channel(channel, user, limit=600, after=after)
        texts = clean_texts(texts)
        if not texts and not image_urls:
            return await interaction.followup.send("No new content found for that window.")

        captions: List[str] = []
        for u in image_urls[:20]:
            try:
                c = llm.vision_describe(u)
                if c:
                    captions.append(c)
            except Exception:
                pass

        # Append to index
        index_path = INDEX_DIR / f"{user.id}.idx"
        retr = Retriever(index_path, embed_fn=llm.embed)
        to_add = texts[-300:] + [f"[img] {c}" for c in captions[:20]]
        if to_add:
            retr.add_texts(to_add)

        # Refresh traits and style (compact)
        traits = extract_basic_traits(texts)
        rich_traits = extract_rich_traits(texts, media_captions=captions)
        pdata = pers.read_json(pers.persona_path(user.id)) or {}

        # Rebuild style via LLM (short), fallback to rich traits summary
        from ..llm.prompting import rich_traits_to_style
        try:
            sprompt = build_persona_creation_prompt(user.display_name, texts[-50:])
            style_prompt = llm.complete(sprompt, max_tokens=cfg.create_max_tokens)
        except Exception:
            style_prompt = ""
        if not style_prompt or style_prompt.strip().startswith("[stubbed LLM]"):
            style_prompt = rich_traits_to_style(user.display_name, rich_traits)
        if len(style_prompt) > cfg.style_max_chars:
            style_prompt = style_prompt[: cfg.style_max_chars]

        # Persist updated persona (merge)
        doc = {
            **pdata,
            "user_id": user.id,
            "username": user.display_name,
            "style_prompt": style_prompt,
            "traits": traits,
            "text_style": rich_traits.get("text_style", {}),
            "personality": rich_traits.get("personality", {}),
            "conversation": rich_traits.get("conversation", {}),
            "topics": rich_traits.get("topics", {}),
            "beliefs": rich_traits.get("beliefs", {}),
            "culture": rich_traits.get("culture", {}),
            "media": rich_traits.get("media", {}),
        }
        pers.write_json(pers.persona_path(user.id), doc)
        await interaction.followup.send(
            f"Updated persona for @{user.display_name} with {len(texts)} new texts and {len(captions)} image captions."
        )

    @persona.command(name="load", description="Load persona into cache (MVP no-op)")
    async def persona_load(self, interaction: discord.Interaction, user: discord.Member):
        if pers.read_json(pers.persona_path(user.id)):
            await interaction.response.send_message("Persona loaded.")
        else:
            await interaction.response.send_message("Persona not found.")

    @persona.command(name="summarize", description="Summarize a user's recent messages (MVP stub)")
    async def persona_summarize(self, interaction: discord.Interaction, user: discord.Member, last: int = 50):
        await interaction.response.defer(thinking=True)
        channel = interaction.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return await interaction.followup.send("Please run in a text channel or thread.")
        if not interaction.client.intents.message_content:
            return await interaction.followup.send("Message Content intent is disabled. Cannot read history.")
        # Progress message in channel
        prog_msg = None
        try:
            prog_msg = await interaction.channel.send(  # type: ignore
                f"Summarizing @{user.display_name}: {progress_bar(0)}"
            )
        except Exception:
            prog_msg = None

        # Scan a larger window to account for non-text messages (attachments, stickers) being skipped
        # Include both the target user's and the invoker's recent textual messages
        author_ids = {user.id, interaction.user.id}
        messages = await fetch_recent_messages_from_channel(
            channel,
            user=None,
            limit=max(50, int(last) * 10),
            user_ids=author_ids,
            include_non_text=False,
        )
        if not messages:
            return await interaction.followup.send("No recent messages found from that user in this channel.")
        texts = clean_texts(messages)
        used = texts[-last:] if last > 0 else texts
        # Truncate each message and enforce a total character budget for speed
        per_msg = max(40, cfg.summarize_msg_max_chars)
        total_cap = max(1000, cfg.summarize_total_max_chars)
        # Compress: merge consecutive very short lines to reduce overhead
        merged: List[str] = []
        buf = ""
        for t in used:
            if len(t) < 40:
                if buf:
                    buf = (buf + " " + t).strip()
                else:
                    buf = t
                if len(buf) >= 100:
                    merged.append(buf)
                    buf = ""
            else:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.append(t)
        if buf:
            merged.append(buf)
        used = [s[:per_msg] for s in merged]
        if prog_msg:
            try:
                await prog_msg.edit(content=f"Summarizing @{user.display_name}: {progress_bar(10)}")
            except Exception:
                pass
        # Optional image captions (parallel with small cache and concurrency)
        img_caps: List[str] = []
        if cfg.summarize_include_images:
            author_ids = {user.id, interaction.user.id}
            items = await fetch_image_items_multi(channel, author_ids, limit=max(100, int(last) * 20))
            cap_n = max(0, cfg.summarize_image_captions)
            to_caption = items[:cap_n]
            # Purge old cache entries occasionally
            try:
                capcache.purge_expired()
            except Exception:
                pass
            # Use cached captions when available
            pending: List[str] = []
            for it in to_caption:
                url = it.get("url", "")
                c = capcache.get(url) if url else None
                if c:
                    img_caps.append(c[: cfg.summarize_image_caption_max_chars])
                else:
                    pending.append(it)

            if pending:
                import asyncio

                sem = asyncio.Semaphore(max(1, cfg.summarize_caption_concurrency))

                async def _cap(it: dict) -> None:
                    async with sem:
                        try:
                            c = await asyncio.to_thread(
                                llm.vision_describe, it.get("url", ""),
                                hint=(it.get("text") or None), filename=(it.get("filename") or None)
                            )
                            if c and cfg.caption_refine:
                                rp = (
                                    "Refine this image caption to be strictly factual and grounded in any context text."
                                    " Remove any guesses. If unsure, say 'uncertain'.\n"
                                    f"Caption: {c}\nContext: {it.get('text','')}"
                                )
                                c = await asyncio.to_thread(
                                    llm.complete,
                                    rp,
                                    max_tokens=cfg.caption_refine_max_tokens,
                                    temperature=0.2,
                                    top_p=0.5,
                                )
                            if c:
                                c = c[: cfg.summarize_image_caption_max_chars]
                                img_caps.append(c)
                                try:
                                    if url:
                                        capcache.set(url, c)
                                except Exception:
                                    pass
                        except Exception:
                            return

                await asyncio.gather(*(_cap(it) for it in pending))
            if prog_msg:
                try:
                    await prog_msg.edit(content=f"Summarizing @{user.display_name}: {progress_bar(40)}")
                except Exception:
                    pass
        # Drop oldest until within total budget
        total_len = sum(len(t) for t in used)
        while used and total_len > total_cap:
            total_len -= len(used[0])
            used.pop(0)
        max_toks = min(cfg.summarize_max_tokens, 128 if cfg.summarize_fast else cfg.summarize_max_tokens)
        # Hierarchical summarization: summarize chunks then merge
        if cfg.summarize_hierarchical and len(used) > 10:
            chunks = max(1, cfg.summarize_chunk_count)
            # Split used into consecutive chunks
            size = max(1, (len(used) + chunks - 1) // chunks)
            chunk_lists = [used[i : i + size] for i in range(0, len(used), size)]

            chunk_summaries: List[str] = []
            for idx, cl in enumerate(chunk_lists, start=1):
                cp = build_summarize_prompt(user.display_name, cl, image_captions=None)
                cs = llm.complete(
                    cp,
                    max_tokens=min(cfg.summarize_chunk_max_tokens, max_toks),
                    model=cfg.summarize_model_name or cfg.text_model_name,
                    temperature=cfg.summarize_temperature,
                    num_ctx=cfg.summarize_num_ctx,
                    top_p=cfg.summarize_top_p,
                )
                chunk_summaries.append(cs)
                if prog_msg:
                    try:
                        pct = 40 + int(idx * 50 / max(1, len(chunk_lists)))
                        await prog_msg.edit(content=f"Summarizing @{user.display_name}: {progress_bar(pct)}")
                    except Exception:
                        pass

            from ..llm.prompting import build_merge_summaries_prompt

            mp = build_merge_summaries_prompt(user.display_name, chunk_summaries, image_captions=img_caps)
            summary = llm.complete(
                mp,
                max_tokens=max_toks,
                model=cfg.summarize_model_name or cfg.text_model_name,
                temperature=cfg.summarize_temperature,
                num_ctx=cfg.summarize_num_ctx,
                top_p=cfg.summarize_top_p,
            )
            if prog_msg:
                try:
                    await prog_msg.edit(content=f"Summarizing @{user.display_name}: {progress_bar(100)}")
                except Exception:
                    pass
        else:
            prompt = build_summarize_prompt(user.display_name, used, image_captions=img_caps)
            summary = llm.complete(
                prompt,
                max_tokens=max_toks,
                model=cfg.summarize_model_name or cfg.text_model_name,
                temperature=cfg.summarize_temperature,
                num_ctx=cfg.summarize_num_ctx,
                top_p=cfg.summarize_top_p,
            )
            if prog_msg:
                try:
                    await prog_msg.edit(content=f"Summarizing @{user.display_name}: {progress_bar(100)}")
                except Exception:
                    pass
        invoker_note = " (+invoker)" if interaction.user.id != user.id else ""
        await interaction.followup.send(
            f"Summary of @{user.display_name}{invoker_note} (requested {last}, included {len(used)} msgs):\n{summary}"
        )


async def setup(bot: commands.Bot):
    import discord as _discord
    from ..config import cfg as _cfg

    cog = PersonaCog(bot)
    await bot.add_cog(cog)
    # Explicitly add the slash command group to the tree, guild-scoped if configured
    try:
        if _cfg.guild_id:
            guild = _discord.Object(id=int(_cfg.guild_id))
            bot.tree.add_command(cog.persona, guild=guild)
        else:
            bot.tree.add_command(cog.persona)
    except Exception:
        # If already added, ignore
        pass
