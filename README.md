# Discord AI Persona Bot (MVP)

This project implements a Discord bot that can build and speak as AI personas of server users. It follows the PRD in `PRD.md`.

Key notes:
- Python 3.10+
- discord.py 2.x slash commands
- Local-only storage (JSON + simple vector index, FAISS optional)
- LLM/vision assumed local; HTTP endpoints configurable. If unavailable, code falls back to safe stubs.
- No direct MCP server integrations; optional MCP docs may be included in prompts for context if configured.

## Ollama Setup

This project is compatible with Ollama out of the box.

- Set `LLM_BASE_URL` to your Ollama host (default `http://localhost:11434`).
- Set `TEXT_MODEL_NAME` to your chat model (e.g., `llama3.1`, `qwen2.5`, `mistral`).
- Set `EMBED_MODEL_NAME` to an embedding model (e.g., `mxbai-embed-large` or `nomic-embed-text`).

Quick sanity checks (optional):

```bash
# Generate (non-streaming)
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1","prompt":"Say hi","stream":false}' | jq .response

# Embedding (single string per call)
curl -s http://localhost:11434/api/embeddings \
  -d '{"model":"mxbai-embed-large","input":"hello world"}' | jq .embedding | head -n 3
```

Note: The bot batches embeddings by making one call per text to match Ollama’s embeddings API shape.

### Troubleshooting Ollama
- 404 on `/api/embeddings`: Update Ollama to a recent version and pull an embedding model, e.g. `ollama pull mxbai-embed-large`. Then test:
  - `curl -s http://localhost:11434/api/embeddings -d '{"model":"mxbai-embed-large","input":"hello"}'`
  If it still returns 404, your Ollama build likely predates the embeddings endpoint.
- Slow generations / Discord timeouts: Increase HTTP timeouts via env:
  - `LLM_TIMEOUT=60` (default 60)
  - `EMBED_TIMEOUT=30` (default 30)
  First token on cold start can be slow; subsequent calls are faster.

### Speed/Accuracy Tuning
- Limit generation length: set `SPEAK_MAX_TOKENS` (e.g., 192–256) and `CREATE_MAX_TOKENS`.
- Trim retrieval: set `RAG_K` (default 3) and `RAG_SNIPPET_MAX_CHARS` (default 240) to keep context small but relevant.
- Cap style prompt: set `STYLE_MAX_CHARS` (default 1000) to avoid overly-long system guidance.
- Prewarm the model: set `PREWARM_LLM=true` to issue a tiny request on startup, reducing first-token latency without blocking the bot.

## Setup

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install requirements

```bash
pip install -r requirements.txt
```

3) Configure environment

Copy `.env.example` to `.env` and fill in `DISCORD_TOKEN`. Optionally set `GUILD_ID` for faster slash command sync.

Enable the Message Content Intent for your bot in the Discord Developer Portal. The bot handles missing permissions gracefully but will collect less data without this intent.

4) Quick check

```bash
python hello_bot.py
```

Use `/ping` in your server to see “pong”.

## Full bot with personas

Run the full bot that includes the persona commands:

```bash
python -m src.bot
```

### Slash commands
- `/persona create @username`
- `/persona update @username [since:days]`
- `/persona load @username`
- `/persona list`
- `/persona erase @username`
- `/persona switch @username`
- `/persona speak "your prompt here"`
- `/persona summarize @username last=50`

### Typical flow
1) `/persona create @user` – builds a persona from recent text history (current channel for MVP) and creates a local vector index.
2) `/persona switch @user` – sets that persona active for the channel.
3) `/persona speak "..."` – the bot replies in that persona’s style using RAG retrieval.

### Storage
- Personas: `src/data/personas/{user_id}.json`
- Indexes: `src/data/indexes/{user_id}.idx`
- Active persona per-channel state: `src/data/active_persona.json`

### Privacy & safety
- The bot prepends “Persona Bot (@username)” to outputs.
- Anti-regurgitation checks compare draft output to nearest training snippets; if too similar, it re-generates with stricter guidance.
- Only uses local storage. If LLM/vision endpoints are unreachable, the bot falls back to stubbed behavior with clear messaging.

## Troubleshooting
- If slash commands don’t appear, set `GUILD_ID` in `.env` to your test guild for faster sync and restart the bot.
- If message content isn’t captured, ensure the Message Content intent is enabled for your bot and in code.
- Missing permissions: the bot will skip actions requiring permissions it lacks and respond with a friendly message.
