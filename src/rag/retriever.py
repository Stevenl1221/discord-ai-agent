from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..utils.logging import get_logger
from .embedder import SimpleIndex, normalize


log = get_logger(__name__)


class Retriever:
    def __init__(self, index_path: Path, embed_fn):
        self.index_path = index_path
        self.embed_fn = embed_fn
        self.index = SimpleIndex.load(index_path)
        if not self.index:
            log.info("No index at %s yet", index_path)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.index.texts) > 0

    def add_texts(self, texts: List[str], dim_hint: int = 768):
        vecs = normalize(self.embed_fn(texts))
        if self.index is None:
            backend = "faiss" if vecs.shape[1] and vecs.shape[1] > 0 else "numpy"
            self.index = SimpleIndex(vecs.shape[1] or dim_hint, backend=backend)
        self.index.add(vecs, texts)
        self.index.save(self.index_path)

    def query(self, q: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.index or not self.index.texts:
            return []
        qv = normalize(self.embed_fn([q]))
        results = self.index.search(qv, k=k)[0]
        return [(self.index.texts[i], score) for i, score in results]

    def similarity_to_nearest(self, text: str) -> float:
        res = self.query(text, k=1)
        return res[0][1] if res else 0.0

