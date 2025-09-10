from __future__ import annotations

from typing import List, Dict, Any

from .mcp_context import fetch_docs_snippets


def join_mcp_snippets() -> str:
    snips = fetch_docs_snippets()
    if not snips:
        return ""
    return "\n\n[MCP-Docs]\n" + "\n---\n".join(snips)


def style_from_traits(username: str, traits: dict) -> str:
    tone = traits.get("tone", "neutral")
    emoji_rate = traits.get("emoji_rate", 0.0)
    avg = traits.get("avg_length", 0.0)
    slang = traits.get("slang", [])
    topics = traits.get("topics", [])
    style = traits.get("response_style", "concise")
    emoji_desc = "rare" if emoji_rate < 0.001 else ("occasional" if emoji_rate < 0.01 else "frequent")
    length_desc = "short (1-2 sentences)" if avg < 80 else ("medium (2-4 sentences)" if avg < 180 else "long (4+ sentences)")
    slang_desc = ", ".join(slang[:6]) if slang else "minimal slang"
    topics_desc = ", ".join(topics[:6]) if topics else "varied server topics"
    return (
        f"Style guide for @{username}:\n"
        f"- Tone: {tone}\n"
        f"- Emoji: {emoji_desc}\n"
        f"- Length: {length_desc}\n"
        f"- Slang: {slang_desc}\n"
        f"- Topics: {topics_desc}\n"
        f"- Response style: {style}"
    )


def rich_traits_to_style(username: str, rich: Dict[str, Any]) -> str:
    ts = rich.get("text_style", {}) if isinstance(rich, dict) else {}
    per = rich.get("personality", {}) if isinstance(rich, dict) else {}
    conv = rich.get("conversation", {}) if isinstance(rich, dict) else {}
    tpcs = rich.get("topics", {}) if isinstance(rich, dict) else {}
    # Flatten topics summary
    topics_desc = ", ".join(f"{k}: {', '.join(v[:3])}" for k, v in list(tpcs.items())[:4]) if tpcs else "varied"
    media = rich.get("media", {}) if isinstance(rich, dict) else {}
    media_kw = ", ".join(media.get("keywords", [])[:6]) if media else ""
    beliefs = rich.get("beliefs", {}) if isinstance(rich, dict) else {}
    values_line = ", ".join((beliefs.get("values") or [])[:6]) if isinstance(beliefs.get("values"), list) else ""
    worldview = beliefs.get("worldview") or ""
    emoji_use = ts.get("emoji_use") or ""
    slang = ts.get("slang") or []
    return (
        f"Observed style for @{username}:\n"
        f"- Length: {ts.get('message_length', 'unknown')}\n"
        f"- Complexity: {ts.get('sentence_complexity', 'unknown')}\n"
        f"- Capitalization: {ts.get('capitalization', 'mixed')}\n"
        f"- Punctuation: {ts.get('punctuation', 'normal')}\n"
        f"- Emoji: {emoji_use}\n"
        f"- Slang: {', '.join(slang[:6]) if slang else 'minimal'}\n"
        f"- Formatting: {ts.get('formatting', 'plain')} | Media: {ts.get('media_usage', 'rare')}\n"
        f"- Personality: humor {per.get('humor', 'subtle')}, {per.get('directness', 'direct')}, {per.get('formality', 'informal')}\n"
        f"- Expressiveness: {per.get('expressiveness', 'neutral')}; Politeness: {per.get('politeness', 'informal')}\n"
        f"- Habits: {conv.get('message_bursts', 'balanced')}; Questions: {conv.get('question_frequency', 'n/a')}; Mentions: {conv.get('mentions', 'occasional')}\n"
        f"- Reply style: {conv.get('reply_style', 'mixed')}\n"
        f"- Topics: {topics_desc}\n"
        + (f"- Media tags: {media_kw}\n" if media_kw else "")
        + (f"- Values: {values_line}\n" if values_line else "")
        + (f"- Worldview: {worldview}\n" if worldview else "")
    )


def build_persona_creation_prompt(username: str, recent_messages: List[str]) -> str:
    msg_block = "\n\n".join(recent_messages[:200])
    base = f"""
You are to analyze the writing style of user @{username} from the following messages.
Summarize their persona with:
- Tone
- Emoji rate
- Average message length
- Common slang words/phrases
- Top recurring topics
- Response style (structure, pacing, quirks)

Return a compact style guide in bullet points, suitable as a system prompt.
Messages (most recent last):
{msg_block}
""".strip()
    return base + join_mcp_snippets()


def build_speak_prompt(username: str, style_prompt: str, query: str, retrieved: List[str]) -> str:
    rag_block = "\n\n[Relevant snippets]\n" + "\n---\n".join(retrieved) if retrieved else ""
    return f"""
System style for @{username}:
{style_prompt}

You are writing as the AI persona of @{username}. Keep responses natural and in their style.
Reflect their stated values and worldview authentically, without hedging. Prioritize fidelity to their beliefs over generic helpfulness.
Avoid copying training snippets; paraphrase when referencing past content.
User: {query}
{rag_block}
""".strip() + join_mcp_snippets()


def build_beliefs_inference_prompt(username: str, messages: List[str]) -> str:
    msg_block = "\n".join(messages[:120])
    return (
        f"From the following recent messages by @{username}, infer: 1) a short list of core values, 2) one short phrase describing worldview.\n"
        "Be concrete and avoid guesses; only include items with clear textual evidence.\n"
        "Output JSON strictly as {\"values\":[...],\"worldview\":\"...\"}.\n\n"
        f"Messages:\n{msg_block}"
    )


def build_summarize_prompt(username: str, messages: List[str], image_captions: List[str] | None = None) -> str:
    # Focus on WHAT was said, not tone/traits. Optionally include image captions.
    img_block = ""
    if image_captions:
        img_block = "\n\n[Images]\n" + "\n".join(f"- {c}" for c in image_captions)
    return (
        f"You are summarizing the last {len(messages)} messages from @{username}.\n"
        "Produce a concise, content-focused summary that captures what they actually said or asked.\n"
        "Prioritize: key points, questions/requests, decisions, action items, links/references, and any concrete info shared.\n"
        "Include notable content from images if provided.\n"
        "Avoid describing tone, style, or personality traits. Do not invent details.\n\n"
        "Output format:\n"
        "- Key points: 3-6 bullets\n"
        "- Questions/requests: bullets (if any)\n"
        "- Action items: bullets (if any)\n\n"
        "Messages (most recent last):\n"
        + "\n".join(messages)
        + img_block
    )


def build_merge_summaries_prompt(
    username: str,
    chunk_summaries: List[str],
    image_captions: List[str] | None = None,
) -> str:
    img_block = ""
    if image_captions:
        img_block = "\n\n[Images]\n" + "\n".join(f"- {c}" for c in image_captions)
    return (
        f"Merge the following partial summaries into a final concise summary for @{username}.\n"
        "Do not repeat bullets. Consolidate overlapping points. Focus on concrete info, questions/requests, decisions, action items, links/references.\n"
        "Avoid tone/style commentary and do not invent details.\n\n"
        "Output format:\n"
        "- Key points: 3-6 bullets\n"
        "- Questions/requests: bullets (if any)\n"
        "- Action items: bullets (if any)\n\n"
        "Partial summaries:\n"
        + "\n---\n".join(chunk_summaries)
        + img_block
    )


def build_merge_style_prompt(username: str, chunk_guides: List[str], media_keywords: List[str] | None = None) -> str:
    media_line = ""
    if media_keywords:
        media_line = "\nInclude media-related quirks/tags if consistently present: " + ", ".join(media_keywords[:6])
    return (
        f"Merge the following partial style guides for @{username} into one concise persona style guide.\n"
        "Keep bullet format. Prioritize consistent traits across chunks.\n"
        "Do not include contradictions or redundant bullets.\n"
        "Focus on textual style, tone, habits, and common topics." + media_line + "\n\n"
        "Partial style guides:\n" + "\n---\n".join(chunk_guides)
    )
