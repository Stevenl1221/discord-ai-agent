from __future__ import annotations

import re
from typing import List, Dict, Any, Optional


PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b\d{10}\b"),  # 10-digit numbers
]


def scrub_pii(text: str) -> str:
    out = text
    for pat in PII_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def clean_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for t in texts:
        t = t.strip()
        t = scrub_pii(t)
        if t:
            cleaned.append(t)
    return cleaned


def extract_basic_traits(texts: List[str]) -> dict:
    if not texts:
        return {
            "tone": "neutral",
            "emoji_rate": 0.0,
            "avg_length": 0.0,
            "slang": [],
            "topics": [],
            "response_style": "concise",
        }
    total_len = sum(len(t) for t in texts)
    avg_len = total_len / max(1, len(texts))
    emoji_count = sum(sum(1 for ch in t if ch in "😀😃😄😁😆😅😂🙂😉😊😍😘🤔🙃😭🤣✨🔥💯👍🙏❤️") for t in texts)
    emoji_rate = emoji_count / max(1, total_len)
    slang_candidates = ["lol", "brb", "idk", "btw", "omg", "ikr"]
    slang = sorted({w for t in texts for w in slang_candidates if w in t.lower()})
    topics = []  # MVP: empty; could do keyword extraction later
    style = "concise" if avg_len < 80 else "detailed"
    return {
        "tone": "casual",
        "emoji_rate": emoji_rate,
        "avg_length": avg_len,
        "slang": slang,
        "topics": topics,
        "response_style": style,
    }


EMOJI_SET = set("😀😃😄😁😆😅😂🙂😉😊😍😘🤔🙃😭🤣✨🔥💯👍🙏❤️💀😬😎😜😇😏🥲😤😢🤷‍♂️🤷‍♀️😮‍💨")
SLANG_WORDS = [
    "bruh", "ngl", "lowkey", "highkey", "fr", "ong", "tbh", "idk", "ikr", "btw",
    "lol", "lmao", "rofl", "smh", "af", "jk", "imo", "imho", "yeet", "sus",
    "cap", "no cap", "bet", "rip", "brb", "gg", "ggs"
]
POLITENESS = ["please", "pls", "plz", "thank you", "thanks", "ty", "tysm"]
HEDGES = ["maybe", "perhaps", "i think", "kinda", "sort of", "might"]
POS_WORDS = ["great", "good", "nice", "love", "awesome", "cool", "yay", "win", "like"]
NEG_WORDS = ["bad", "hate", "annoying", "ugh", "worst", "lame", "sad", "angry", "fail"]
REGIONAL = {
    "US-South": ["y'all", "fixin'", "ain't"],
    "UK": ["mate", "cheers", "bloody"],
    "AU/NZ": ["mate", "heaps", "keen"],
}
TOPIC_LEXICON: Dict[str, List[str]] = {
    "media": ["anime", "manga", "movie", "show", "season", "episode", "game", "gaming", "lofi", "music", "song", "meme", "memes"],
    "lifestyle": ["gym", "workout", "travel", "trip", "food", "snack", "coffee", "tea", "run", "hike", "bike"],
    "technology": ["ai", "gpt", "llm", "python", "javascript", "crypto", "gpu", "server", "dev", "code"],
    "community": ["school", "work", "job", "team", "fandom", "discord", "guild", "clan"],
    "events": ["news", "politics", "election", "war", "update", "launch", "release"],
}


def extract_rich_traits(texts: List[str], media_captions: Optional[List[str]] = None) -> Dict[str, Any]:
    if not texts:
        return {
            "text_style": {},
            "personality": {},
            "conversation": {},
            "topics": {},
            "beliefs": {},
            "culture": {},
        }

    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    avg_len = total_chars / max(1, len(texts))
    avg_words = total_words / max(1, len(texts))

    # Sentence complexity (very rough): avg words per sentence
    sentences = sum(max(1, len([s for s in re.split(r"[.!?]+", t) if s.strip()])) for t in texts)
    complexity = total_words / max(1, sentences)

    # Capitalization
    lowercase_msgs = sum(1 for t in texts if t and t == t.lower())
    uppercase_words = sum(1 for w in (w for t in texts for w in t.split()) if w.isupper() and len(w) > 1)
    capitalization = (
        "always lowercase" if lowercase_msgs / max(1, len(texts)) > 0.7 else (
            "frequent ALL CAPS" if uppercase_words / max(1, total_words) > 0.05 else "mixed")
    )

    # Punctuation habits
    exclam = sum(t.count("!") for t in texts)
    quest = sum(t.count("?") for t in texts)
    triple_punct = sum(1 for t in texts if "!!" in t or "??" in t or "..." in t)
    punctuation = "frequent !!! and ???" if (exclam + quest) / max(1, len(texts)) > 1.0 else (
        "uses ellipses and repeats" if triple_punct / max(1, len(texts)) > 0.3 else "normal")

    # Emoji
    emoji_counts = [sum(1 for ch in t if ch in EMOJI_SET) for t in texts]
    emoji_total = sum(emoji_counts)
    emoji_freq = emoji_total / max(1, len(texts))
    emoji_end = sum(1 for t in texts if any(ch in EMOJI_SET for ch in t[-3:]))
    emoji_place = "at end of sentences" if emoji_end / max(1, len(texts)) > 0.4 else "inline"
    top_emojis = []
    # simple top emoji extraction
    from collections import Counter

    c = Counter(ch for t in texts for ch in t if ch in EMOJI_SET)
    top_emojis = [e for e, _ in c.most_common(5)]

    # Slang/acronyms
    slang_found = sorted({w for t in texts for w in SLANG_WORDS if w in t.lower()})

    # Typos/misspellings: naive indicators
    elongated = sum(1 for t in texts if re.search(r"([a-zA-Z])\1{2,}", t))
    typos = "some elongated words/typos" if elongated / max(1, len(texts)) > 0.1 else "rare"

    # Formatting quirks
    code_blocks = any("```" in t or "`" in t for t in texts)
    quotes = any(t.strip().startswith(">") for t in texts)
    formatting = ", ".join([s for s in ["code blocks" if code_blocks else "", "quote replies" if quotes else ""] if s]) or "plain"

    # Media usage (based on placeholders)
    media_lines = [t for t in texts if t.startswith("[attachments:")]
    gifs = sum(1 for t in media_lines if ".gif" in t.lower())
    media_usage = (
        "GIFs often" if gifs >= 3 else (
            "images sometimes" if media_lines else "rare")
    )
    media_caps = media_captions or []
    media_keywords: List[str] = []
    if media_caps:
        # pick top tokens as rough tags
        from collections import Counter
        tokens = []
        for c in media_caps:
            tokens += [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", c)]
        stop = {"the", "and", "with", "this", "that", "have", "from", "over", "under", "your", "into", "about"}
        tokens = [w for w in tokens if w not in stop]
        cnt = Counter(tokens)
        media_keywords = [w for w, _ in cnt.most_common(8)]

    # Personality & tone heuristics
    pos = sum(t.lower().count(w) for t in texts for w in POS_WORDS)
    neg = sum(t.lower().count(w) for t in texts for w in NEG_WORDS)
    optimism = "optimistic" if pos > neg * 1.2 else ("pessimistic" if neg > pos * 1.2 else "neutral")
    hedge_rate = sum(1 for t in texts for w in HEDGES if w in t.lower()) / max(1, len(texts))
    directness = "direct" if hedge_rate < 0.1 else "hedged"
    politeness_hits = sum(1 for t in texts for w in POLITENESS if w in t.lower())
    politeness = "polite" if politeness_hits > 0 else "informal"
    expressiveness = "expressive" if emoji_freq > 0.5 or exclam / max(1, len(texts)) > 0.5 else "reserved"
    humor = "meme-heavy" if any(x in (" "+t.lower()+" ") for t in texts for x in [" lol ", " lmao ", " meme "]) else "subtle"
    exaggeration = "frequent" if triple_punct / max(1, len(texts)) > 0.3 or uppercase_words / max(1, total_words) > 0.05 else "rare"

    # Conversational habits
    short_msgs = sum(1 for t in texts if len(t) < 40)
    bursts = "often sends short bursts" if short_msgs / max(1, len(texts)) > 0.6 else "balanced"
    greetings = sum(1 for t in texts if re.match(r"^(yo|hey|hi|sup|hello)\b", t.strip(), re.I))
    initiation = "often starts casually" if greetings / max(1, len(texts)) > 0.2 else "varied"
    question_rate = sum(1 for t in texts if "?" in t) / max(1, len(texts))
    mentions = sum(1 for t in texts if "@" in t or "<@" in t)
    reply_style = "quotes" if quotes else ("short quips" if short_msgs / max(1, len(texts)) > 0.6 else "mixed")
    advice_vs_vent = "advice-giving" if sum(1 for t in texts if re.search(r"\b(you should|try|consider)\b", t, re.I)) > 2 else (
        "venting" if sum(1 for t in texts if re.search(r"\b(i'?m|i am) (tired|annoyed|done)\b", t, re.I)) > 2 else "mixed")

    # Topics
    topics: Dict[str, List[str]] = {}
    for cat, vocab in TOPIC_LEXICON.items():
        found = sorted({w for t in texts for w in vocab if re.search(rf"\b{re.escape(w)}\b", t, re.I)})
        if found:
            topics[cat] = found[:10]

    # Cultural context
    generation = "Gen Z" if any(s in (" "+t.lower()+" ") for t in texts for s in [" bruh ", " ngl ", " lowkey ", " fr "]) else "Millennial"
    regional_refs = []
    for region, words in REGIONAL.items():
        if any(w in t.lower() for t in texts for w in words):
            regional_refs.append(region)
    subcultures = []
    if any(w in t.lower() for t in texts for w in ["anime", "clan", "guild", "discord"]):
        subcultures.append("Discord/anime/gaming culture")

    return {
        "text_style": {
            "message_length": f"avg {avg_words:.1f} words",
            "sentence_complexity": f"avg {complexity:.1f} words/sentence",
            "capitalization": capitalization,
            "punctuation": punctuation,
            "emoji_use": (", ".join(top_emojis) + f"; {emoji_place}") if top_emojis else emoji_place,
            "slang": slang_found,
            "typos": typos,
            "formatting": formatting,
            "media_usage": media_usage,
        },
        "personality": {
            "humor": humor,
            "directness": directness,
            "formality": "informal" if capitalization == "always lowercase" or slang_found else "mixed",
            "optimism": optimism,
            "expressiveness": expressiveness,
            "politeness": politeness,
            "exaggeration": exaggeration,
        },
        "conversation": {
            "message_bursts": bursts,
            "initiation": initiation,
            "question_frequency": f"{question_rate:.0%} of messages",
            "mentions": "frequent" if mentions / max(1, len(texts)) > 0.3 else "occasional",
            "reply_style": reply_style,
            "advice_vs_venting": advice_vs_vent,
        },
        "topics": topics,
        "beliefs": {
            # Hard to infer reliably; leaving placeholders unless explicit cues present
            "values": [],
            "worldview": None,
        },
        "culture": {
            "generation": generation,
            "subcultures": subcultures,
            "regional_references": regional_refs,
        },
        "media": {
            "captions": media_caps[:20],
            "keywords": media_keywords,
        },
    }
