from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExamplePair:
    prompt: str
    response: str


@dataclass
class TraitSummary:
    tone: str = ""
    emoji_rate: float = 0.0
    avg_length: float = 0.0
    slang: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    response_style: str = ""


@dataclass
class Persona:
    user_id: int
    username: str
    version: int
    style_prompt: str
    traits: TraitSummary
    examples: List[ExamplePair] = field(default_factory=list)


@dataclass
class IndexMeta:
    user_id: int
    dim: int
    size: int
    backend: str  # "faiss" or "numpy"

