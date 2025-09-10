import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSONA_DIR = DATA_DIR / "personas"
INDEX_DIR = DATA_DIR / "indexes"
STATE_FILE = DATA_DIR / "active_persona.json"

for d in [DATA_DIR, PERSONA_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    discord_token: str = os.getenv("DISCORD_TOKEN", "")
    # Accept either GUILD_ID or DISCORD_GUILD_ID for convenience
    guild_id: str | None = os.getenv("GUILD_ID") or os.getenv("DISCORD_GUILD_ID")

    # Models and endpoints (local)
    text_model_name: str = os.getenv("TEXT_MODEL_NAME", "local-llm")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "local-embed")
    vision_model_name: str = os.getenv("VISION_MODEL_NAME", "local-vision")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    vision_base_url: str = os.getenv("VISION_BASE_URL", "http://localhost:5000")
    # Network timeouts (seconds)
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    embed_timeout: float = float(os.getenv("EMBED_TIMEOUT", "30"))

    # Toggles
    use_faiss: bool = os.getenv("USE_FAISS", "true").lower() == "true"
    use_mcp_context: bool = os.getenv("USE_MCP_CONTEXT", "false").lower() == "true"

    # Optional MCP doc sources (used only as RAG context snippets)
    mcp_context7_url: str | None = os.getenv("MCP_CONTEXT7_URL")
    mcp_discord_docs_url: str | None = os.getenv("MCP_DISCORD_DOCS_URL")

    # Generation length controls
    speak_max_tokens: int = int(os.getenv("SPEAK_MAX_TOKENS", "256"))
    create_max_tokens: int = int(os.getenv("CREATE_MAX_TOKENS", "384"))
    summarize_max_tokens: int = int(os.getenv("SUMMARIZE_MAX_TOKENS", "192"))
    summarize_msg_max_chars: int = int(os.getenv("SUMMARIZE_MSG_MAX_CHARS", "160"))
    summarize_total_max_chars: int = int(os.getenv("SUMMARIZE_TOTAL_MAX_CHARS", "3000"))
    summarize_fast: bool = os.getenv("SUMMARIZE_FAST", "false").lower() == "true"
    summarize_include_images: bool = os.getenv("SUMMARIZE_INCLUDE_IMAGES", "true").lower() == "true"
    summarize_image_captions: int = int(os.getenv("SUMMARIZE_IMAGE_CAPTIONS", "4"))
    summarize_image_caption_max_chars: int = int(os.getenv("SUMMARIZE_IMAGE_CAPTION_MAX_CHARS", "100"))
    summarize_top_p: float = float(os.getenv("SUMMARIZE_TOP_P", "0.9"))
    summarize_caption_concurrency: int = int(os.getenv("SUMMARIZE_CAPTION_CONCURRENCY", "4"))
    caption_ttl_seconds: int = int(os.getenv("CAPTION_TTL_SECONDS", "86400"))
    # Caption quality controls
    caption_refine: bool = os.getenv("CAPTION_REFINE", "true").lower() == "true"
    caption_refine_max_tokens: int = int(os.getenv("CAPTION_REFINE_MAX_TOKENS", "60"))
    vision_strict_captions: bool = os.getenv("VISION_STRICT_CAPTIONS", "true").lower() == "true"
    # Hierarchical summarization
    summarize_hierarchical: bool = os.getenv("SUMMARIZE_HIERARCHICAL", "true").lower() == "true"
    summarize_chunk_count: int = int(os.getenv("SUMMARIZE_CHUNK_COUNT", "3"))
    summarize_chunk_max_tokens: int = int(os.getenv("SUMMARIZE_CHUNK_MAX_TOKENS", "96"))
    # Per-command model/option overrides
    summarize_model_name: str | None = os.getenv("SUMMARIZE_MODEL_NAME")
    speak_temperature: float = float(os.getenv("SPEAK_TEMPERATURE", "0.7"))
    summarize_temperature: float = float(os.getenv("SUMMARIZE_TEMPERATURE", "0.2"))
    speak_num_ctx: int = int(os.getenv("SPEAK_NUM_CTX", "1536"))
    summarize_num_ctx: int = int(os.getenv("SUMMARIZE_NUM_CTX", "1024"))
    # Optional base system prompt prepended to all LLM prompts
    base_system_prompt: str = os.getenv("BASE_SYSTEM_PROMPT", "").strip()
    # Streaming + early stop for speak
    speak_stream: bool = os.getenv("SPEAK_STREAM", "true").lower() == "true"
    speak_time_budget_seconds: int = int(os.getenv("SPEAK_TIME_BUDGET_SECONDS", "45"))
    stream_edit_interval_ms: int = int(os.getenv("STREAM_EDIT_INTERVAL_MS", "400"))
    stream_min_chunk_chars: int = int(os.getenv("STREAM_MIN_CHUNK_CHARS", "24"))

    # Mention-based conversational speak
    enable_mention_speak: bool = os.getenv("ENABLE_MENTION_SPEAK", "true").lower() == "true"
    session_max_turns: int = int(os.getenv("SESSION_MAX_TURNS", "6"))
    burst_send_delay_ms: int = int(os.getenv("BURST_SEND_DELAY_MS", "350"))
    # Concurrency controls for speak
    speak_global_concurrency: int = int(os.getenv("SPEAK_GLOBAL_CONCURRENCY", "2"))
    speak_channel_exclusive: bool = os.getenv("SPEAK_CHANNEL_EXCLUSIVE", "true").lower() == "true"

    # RAG & prompt size controls
    rag_k: int = int(os.getenv("RAG_K", "3"))
    rag_snippet_max_chars: int = int(os.getenv("RAG_SNIPPET_MAX_CHARS", "240"))
    style_max_chars: int = int(os.getenv("STYLE_MAX_CHARS", "1000"))

    # Optional prewarm to reduce first-token latency
    prewarm_llm: bool = os.getenv("PREWARM_LLM", "false").lower() == "true"

    # Create command tuning
    create_msg_fetch_limit: int = int(os.getenv("CREATE_MSG_FETCH_LIMIT", "400"))
    create_style_msgs: int = int(os.getenv("CREATE_STYLE_MSGS", "50"))
    create_index_snippets: int = int(os.getenv("CREATE_INDEX_SNIPPETS", "180"))
    create_image_captions: int = int(os.getenv("CREATE_IMAGE_CAPTIONS", "12"))
    create_caption_concurrency: int = int(os.getenv("CREATE_CAPTION_CONCURRENCY", "4"))
    index_in_background: bool = os.getenv("INDEX_IN_BACKGROUND", "true").lower() == "true"

    # Embedding concurrency
    embed_concurrency: int = int(os.getenv("EMBED_CONCURRENCY", "4"))

    # Create interaction time budget
    create_time_budget_seconds: int = int(os.getenv("CREATE_TIME_BUDGET_SECONDS", "120"))
    create_fast_mode: bool = os.getenv("CREATE_FAST_MODE", "true").lower() == "true"

    # Hierarchical style for create
    create_hierarchical: bool = os.getenv("CREATE_HIERARCHICAL", "true").lower() == "true"
    create_chunk_count: int = int(os.getenv("CREATE_CHUNK_COUNT", "3"))
    create_chunk_max_tokens: int = int(os.getenv("CREATE_CHUNK_MAX_TOKENS", "96"))
    create_temperature: float = float(os.getenv("CREATE_TEMPERATURE", "0.5"))
    create_num_ctx: int = int(os.getenv("CREATE_NUM_CTX", "1536"))
    create_include_images: bool = os.getenv("CREATE_INCLUDE_IMAGES", "false").lower() == "true"
    create_embed_time_est_ms: int = int(os.getenv("CREATE_EMBED_TIME_EST_MS", "60"))
    create_model_name: str | None = os.getenv("CREATE_MODEL_NAME")
    create_top_p: float = float(os.getenv("CREATE_TOP_P", "0.9"))


cfg = Config()
