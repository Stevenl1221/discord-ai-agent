# Product Requirement Document: Multi-Persona Discord Bot

## Overview
This Discord bot enables **AI personas** that emulate users in a server. Each persona is trained on a user’s historical text messages and images. When prompted, the bot responds **in the style of that user**—including tone, length, emoji use, and content preferences. Personas are **persistently stored**, updated via retrieval-augmented generation (RAG), and can be switched on demand.

The bot will be implemented using **discord.py** for bot logic.  
The **LLM** will integrate with **Context7 MCP Server** and **Discord MCP Server** as **context/documentation bases** during prompt construction and persona updates.  
- **Discord bot does not directly integrate with MCP servers**.  
- **MCP servers serve as knowledge/documentation sources for the LLM**.  

---

## Goals
- Allow one bot to manage **multiple personas** simultaneously.  
- Enable personas to be **trained from user messages and images**.  
- Provide **commands to query personas** for responses in their style.  
- Support **persistent persona descriptions** that evolve as new data is collected.  
- Ensure **efficient storage** by caching persona descriptions rather than refetching raw history.  
- Integrate **MCP documentation/context** into LLM prompts for richer persona building.  
- Follow **consent, transparency, and safety** requirements to avoid misuse.  

---

## Features & Functionality

### 1. Persona Creation
**Description:**  
- Admin or authorized user runs `/persona create @username`.  
- Bot fetches historical messages + image attachments from that user.  
- Text is analyzed for:
  - Average message length  
  - Tone (casual, formal, sarcastic, supportive, etc.)  
  - Emoji frequency and usage patterns  
  - Slang/jargon/phrases  
  - Topic distribution (via embeddings or keyword extraction)  
  - Response latency and length consistency  
- Images are analyzed for:
  - Tags/objects/scenes  
  - Mood/emotion cues  
  - Content type (memes, selfies, screenshots, etc.)  
- A **detailed persona prompt** is generated summarizing qualities and how the persona should respond.  
- When constructing the persona prompt, **LLM can leverage Context7 MCP Server and Discord MCP Server documentation** as contextual references.  

**Priority:** High

---

### 2. Persona Storage & Retrieval
**Description:**  
- Generated persona prompt is **cached to file/database** (JSON).  
- File includes:
  - Persona ID (mapped to user ID)  
  - Traits & descriptors  
  - Representative message examples (few-shot style)  
  - Embeddings for similarity search (for RAG)  
- Supports `/persona load @username` to re-load from cache without reprocessing entire history.  

**Priority:** High

---

### 3. Retrieval-Augmented Generation (RAG)
**Description:**  
- When asked to reply in a persona, the bot:  
  1. Retrieves relevant past messages (via embeddings similarity search).  
  2. Supplies persona prompt + retrieved samples to the LLM.  
  3. LLM incorporates **Context7 MCP Server and Discord MCP Server documentation** as part of its context window.  
  4. Generates response in the persona’s style.  
- New messages/images are incrementally indexed to update persona without reprocessing the entire history.  

**Priority:** High

---

### 4. Persona Interaction
**Description:**  
- Slash commands for interacting with personas:  
  - `/persona speak "your prompt here"` → bot replies using the **currently active persona**.  
  - `/persona summarize @username last=50` → specified persona “summarizes” the recent conversation (last 50 messages, or configurable) in their own style.  
  - `/persona switch @username` → sets the **active persona** for future prompts and `/persona speak` calls.  
- Responses must be clearly labeled as **AI emulation** (to avoid impersonation).  
- Default behavior: if no persona has been set yet, `/persona speak` should instruct the user to select one via `/persona switch`.  

**Priority:** High

---

### 5. Persona Updating
**Description:**  
- Bot can refresh persona with new data:  
  - `/persona update @username` → fetch last N days of messages/images and update traits.  
- Persona prompt is rewritten and persisted.  
- Old prompts archived for versioning.  
- Updates may re-query **MCP servers for contextual knowledge** if relevant.  

**Priority:** Medium

---

### 6. System & Storage
**Description:**  
- **Storage:** free options only (JSON files).  
- **Embedding store:** FAISS (local, free) or SQLite-based vector DB.  
- Configurable cache TTL (e.g., 7 days) for refreshing persona prompts.  

**Priority:** Medium

---

### 7. Safety & Transparency
**Description:**  
- Commands require **admin or user consent** to generate a persona.  
- Clear labeling: every AI response includes `(AI persona of @user)` prefix.  
- Support `/persona erase @username` to delete stored data/persona.  
- Prevent verbatim regurgitation of user content (similarity check before sending).  

**Priority:** High

---

## Non-Goals
- Real-time automatic persona responses without explicit command (avoids spam/impersonation).  
- Support for private DMs (server-only bot).  
- Deep image synthesis (only descriptive analysis, not generative impersonation).  
- Bot does not directly connect to MCP servers; LLM integrates MCP docs only as **contextual references**.  

---

## Technical Requirements

- **Bot Framework:** `discord.py` (Python).  
- **LLM & Vision Models:** Context7 MCP Server (used by LLM as context/documentation).  
- **Discord MCP Server:** integrated as **LLM context**, not as a bot integration.  
- **Storage:** JSON files.  
- **Hosting:** Not a concern for now (local run acceptable).  
- **Gateway Intents:**  
  - `GUILDS`, `GUILD_MESSAGES`, `MESSAGE_CONTENT` (privileged).  
  - `READ_MESSAGE_HISTORY` permission.  

---

## Example User Flow

1. Admin runs `/persona create @Alice`.  
2. Bot fetches Alice’s last 2000 messages + images.  
3. Bot analyzes and writes a persona file:
   ```json
   {
     "user_id": "12345",
     "traits": {
       "tone": "sarcastic but friendly",
       "emoji_rate": "high",
       "message_length": "short quips",
       "topics": ["gaming", "anime", "tech"]
     },
     "examples": [
       {"input": "how’s everyone?", "output": "lol dead chat 😂"}
     ]
   }
