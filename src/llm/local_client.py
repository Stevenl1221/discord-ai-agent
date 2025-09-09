from __future__ import annotations

import json
from typing import List, Dict, Any, Callable, Optional
import base64

import requests

from ..config import cfg
from ..utils.logging import get_logger
import numpy as np


log = get_logger(__name__)


class LocalLLMClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or cfg.llm_base_url

    def complete(self, prompt: str, **kw) -> str:
        # Ollama-compatible non-streaming generate call
        url = f"{self.base_url}/api/generate"
        model = kw.pop("model", None) or cfg.text_model_name
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        # Map common kwargs to Ollama options
        options: Dict[str, Any] = {}
        if "max_tokens" in kw:
            try:
                options["num_predict"] = int(kw.pop("max_tokens"))
            except Exception:
                kw.pop("max_tokens")
        if "temperature" in kw:
            try:
                options["temperature"] = float(kw.pop("temperature"))
            except Exception:
                kw.pop("temperature")
        if "top_p" in kw:
            try:
                options["top_p"] = float(kw.pop("top_p"))
            except Exception:
                kw.pop("top_p")
        if "num_ctx" in kw:
            try:
                options["num_ctx"] = int(kw.pop("num_ctx"))
            except Exception:
                kw.pop("num_ctx")
        if "stop" in kw:
            # Ollama accepts top-level stop (string or list)
            payload["stop"] = kw.pop("stop")
        # Allow passing through an explicit options dict and merge
        if "options" in kw and isinstance(kw["options"], dict):
            options.update(kw.pop("options"))
        if options:
            payload["options"] = options
        # Any remaining kwargs are ignored to avoid API incompatibility
        try:
            resp = requests.post(url, json=payload, timeout=cfg.llm_timeout)
            resp.raise_for_status()
            data = resp.json()
            # expected structure depends on your local server; try common fields
            return data.get("response") or data.get("text") or data.get("output") or ""
        except Exception as e:
            log.warning("LLM unavailable, using fallback stub: %s", e)
            # Fallback: generate a concise, purpose-specific stub response
            pl = prompt.lower()
            # Speak prompt detection
            if "system style for @" in pl and "user:" in pl:
                try:
                    q = prompt.split("User:", 1)[1].splitlines()[0].strip()
                except Exception:
                    q = "your request"
                return f"[stubbed LLM] (in-persona) {q[:160]}"  # brief echo in-character
            # Persona creation detection
            if "you are to analyze the writing style" in pl:
                return (
                    "[stubbed LLM] Style Guide:\n"
                    "- Tone: casual, friendly\n"
                    "- Emoji: occasional\n"
                    "- Length: concise (1-3 sentences)\n"
                    "- Slang: light internet shorthand\n"
                    "- Topics: typical server subjects\n"
                    "- Response style: direct, helpful, playful"
                )
            # Summarize detection
            if "summarize the last" in pl and "messages from @" in pl:
                return "[stubbed LLM] Brief summary of recent messages: upbeat tone, common topics, and quick back-and-forth."
            # Generic
            return "[stubbed LLM] This is a stubbed response for local testing."

    def complete_stream(
        self,
        prompt: str,
        on_delta: Callable[[str], None],
        *,
        time_budget_sec: Optional[float] = None,
        **kw,
    ) -> str:
        """Stream tokens from Ollama /api/generate with stream=true.
        Calls on_delta(text_chunk) as text arrives. Returns the final text.
        Stops early if time_budget_sec elapses.
        """
        url = f"{self.base_url}/api/generate"
        model = kw.pop("model", None) or cfg.text_model_name
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        # options mapping (same as complete)
        options: Dict[str, Any] = {}
        if "max_tokens" in kw:
            try:
                options["num_predict"] = int(kw.pop("max_tokens"))
            except Exception:
                kw.pop("max_tokens")
        if "temperature" in kw:
            try:
                options["temperature"] = float(kw.pop("temperature"))
            except Exception:
                kw.pop("temperature")
        if "top_p" in kw:
            try:
                options["top_p"] = float(kw.pop("top_p"))
            except Exception:
                kw.pop("top_p")
        if "num_ctx" in kw:
            try:
                options["num_ctx"] = int(kw.pop("num_ctx"))
            except Exception:
                kw.pop("num_ctx")
        if options:
            payload["options"] = options
        acc = []
        import time, json as _json
        deadline = time.time() + float(time_budget_sec or 1e9)
        try:
            with requests.post(url, json=payload, stream=True, timeout=cfg.llm_timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        if data.get("done"):
                            break
                        chunk = data.get("response") or data.get("text") or ""
                        if chunk:
                            acc.append(chunk)
                            try:
                                on_delta(chunk)
                            except Exception:
                                pass
                    if time.time() > deadline:
                        break
        except Exception as e:
            log.warning("LLM stream unavailable, falling back to complete: %s", e)
            text = self.complete(prompt, **kw)
            try:
                on_delta(text)
            except Exception:
                pass
            return text
        return "".join(acc)

    def embed(self, texts: List[str]) -> np.ndarray:
        # Ollama embeddings API typically returns one embedding per call.
        # Parallelize requests with bounded concurrency to reduce wall time.
        url = f"{self.base_url}/api/embeddings"
        try:
            import concurrent.futures as _fut

            def _one(idx_text: tuple[int, str]) -> tuple[int, np.ndarray]:
                i, t = idx_text
                payload = {"model": cfg.embed_model_name, "input": t}
                r = requests.post(url, json=payload, timeout=cfg.embed_timeout)
                r.raise_for_status()
                d = r.json()
                if "embedding" in d:
                    vec = np.array(d["embedding"], dtype=np.float32)
                elif "data" in d and isinstance(d["data"], list) and d["data"]:
                    vec = np.array(d["data"][0].get("embedding", []), dtype=np.float32)
                else:
                    raise ValueError("No embedding in response")
                if vec.size == 0:
                    raise ValueError("Empty embedding vector")
                return i, vec

            out_ordered: List[tuple[int, np.ndarray]] = []
            with _fut.ThreadPoolExecutor(max_workers=max(1, cfg.embed_concurrency)) as ex:
                for res in ex.map(_one, list(enumerate(texts))):
                    out_ordered.append(res)
            out_ordered.sort(key=lambda x: x[0])
            embs = [vec for _, vec in out_ordered]
            return np.vstack(embs)
        except Exception as e:
            log.warning("Embeddings unavailable, using random fallback: %s", e)
            # Deterministic-ish random for MVP: hash text -> seed
            arrs = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(seed)
                arrs.append(rng.normal(size=384).astype(np.float32))
            return np.vstack(arrs)

    def vision_describe(self, image_url: str, *, hint: str | None = None, filename: str | None = None) -> str:
        # Preferred path: Ollama multimodal generate (e.g., moondream) on the same server
        try:
            # Download image and base64 encode
            img_resp = requests.get(image_url, timeout=cfg.embed_timeout)
            img_resp.raise_for_status()
            b64 = base64.b64encode(img_resp.content).decode("ascii")
            url = f"{cfg.vision_base_url.rstrip('/')}/api/generate"
            base_rules = (
                "You are an accurate image captioner."
                " Describe the image in one short sentence with concrete, visible facts."
                " Extract on-image text verbatim if clearly legible (OCR)."
                " Do NOT guess identities, locations, or brands that are not explicitly visible."
                " If unsure, say 'uncertain'."
            )
            hint_txt = f"\nContext: {hint}" if hint else ""
            fname_txt = f" (file: {filename})" if filename else ""
            prompt = f"{base_rules}{hint_txt}\nOutput one sentence caption{fname_txt}:"
            payload: Dict[str, Any] = {
                "model": cfg.vision_model_name,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }
            resp = requests.post(url, json=payload, timeout=cfg.llm_timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or data.get("text") or data.get("output") or ""
            if text:
                return text.strip()
        except Exception:
            # fallback to legacy /describe adapter if available
            try:
                url = f"{cfg.vision_base_url.rstrip('/')}/describe"
                payload = {"image_url": image_url, "model": cfg.vision_model_name, "hint": hint, "filename": filename}
                resp = requests.post(url, json=payload, timeout=cfg.embed_timeout)
                if resp.ok:
                    data = resp.json()
                    return (
                        data.get("caption")
                        or data.get("description")
                        or data.get("text")
                        or ""
                    )
            except Exception:
                pass
        return "[image: generic description]"


client = LocalLLMClient()
