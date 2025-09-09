from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import cfg
from ..utils.logging import get_logger


log = get_logger(__name__)


class SimpleIndex:
    def __init__(self, dim: int, backend: str = "numpy"):
        self.dim = dim
        self.backend = backend
        self.texts: List[str] = []
        self.vecs: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        if self.backend == "faiss":
            try:
                import faiss  # type: ignore

                self._faiss = faiss.IndexFlatIP(dim)
            except Exception as e:
                log.warning("FAISS unavailable, falling back to numpy: %s", e)
                self.backend = "numpy"

    def add(self, embeddings: np.ndarray, texts: List[str]):
        if self.backend == "faiss" and self._faiss is not None:
            self._faiss.add(embeddings.astype(np.float32))
        else:
            self.vecs = np.vstack([self.vecs, embeddings.astype(np.float32)])
        self.texts.extend(texts)

    def search(self, queries: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        if self.backend == "faiss" and self._faiss is not None:
            import faiss  # type: ignore

            sims, idxs = self._faiss.search(queries.astype(np.float32), k)
            results = []
            for row_sims, row_idxs in zip(sims, idxs):
                results.append([(int(i), float(s)) for i, s in zip(row_idxs, row_sims) if i != -1])
            return results
        else:
            # cosine sim via dot if vectors are normalized
            if self.vecs.shape[0] == 0:
                return [[] for _ in range(queries.shape[0])]
            sims = queries @ self.vecs.T
            results = []
            for srow in sims:
                idxs = np.argsort(-srow)[:k]
                results.append([(int(i), float(srow[i])) for i in idxs])
            return results

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss" and self._faiss is not None:
            import faiss  # type: ignore

            faiss.write_index(self._faiss, str(path) + ".faiss")
            meta = {"backend": "faiss", "dim": self.dim, "size": len(self.texts)}
            (path.parent / (path.name + ".meta.json")).write_text(json.dumps(meta))
            (path.parent / (path.name + ".texts.json")).write_text(json.dumps(self.texts))
        else:
            np.savez_compressed(str(path), vecs=self.vecs, texts=np.array(self.texts, dtype=object))

    @classmethod
    def load(cls, path: Path) -> "SimpleIndex | None":
        if (path.parent / (path.name + ".meta.json")).exists():
            # FAISS backend
            try:
                import faiss  # type: ignore
                meta = json.loads((path.parent / (path.name + ".meta.json")).read_text())
                idx = cls(int(meta["dim"]), backend="faiss")
                idx._faiss = faiss.read_index(str(path) + ".faiss")
                idx.texts = json.loads((path.parent / (path.name + ".texts.json")).read_text())
                return idx
            except Exception as e:
                log.error("Failed to load FAISS index: %s", e)
                return None
        else:
            # numpy backend
            if not path.exists():
                return None
            try:
                data = np.load(str(path), allow_pickle=True)
                vecs = data["vecs"].astype(np.float32)
                texts = list(map(str, data["texts"].tolist()))
                idx = cls(vecs.shape[1], backend="numpy")
                idx.vecs = vecs
                idx.texts = texts
                return idx
            except Exception as e:
                log.error("Failed to load numpy index: %s", e)
                return None


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms

