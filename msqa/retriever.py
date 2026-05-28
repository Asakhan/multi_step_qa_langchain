"""검색기 — 데이터셋 구축 때 만든 ChromaDB 인덱스를 그대로 재사용.

src.rag_index.RagIndex 를 열어 config.yaml 의 rag.top_k(=5),
rag.model(text-embedding-3-small), paths.index(data/index) 설정을 따른다.
이렇게 하면 실험의 검색이 데이터셋 calibration과 완전히 동일해진다.
"""
from __future__ import annotations

from msqa.common import load_config, project_path
from msqa.rag_index import OpenAIEmbedder, RagIndex


class Retriever:
    def __init__(self) -> None:
        cfg = load_config()
        self.top_k: int = int(cfg["rag"]["top_k"])
        index_dir = project_path(cfg["paths"]["index"])
        if not index_dir.exists():
            raise FileNotFoundError(
                f"RAG 인덱스가 없습니다: {index_dir}\n"
                "데이터셋 저장소에서 먼저 다음을 실행해 인덱스를 생성하세요:\n"
                "  python -m msqa.build_index --reset"
            )
        embedder = OpenAIEmbedder(
            model=cfg["rag"]["model"],
            batch_size=int(cfg["rag"].get("embedding_batch_size", 100)),
        )
        self._index = RagIndex(
            persist_dir=index_dir,
            collection_name=cfg["rag"]["collection_name"],
            embedder=embedder,
        )
        n = self._index.count()
        if n == 0:
            raise RuntimeError("인덱스가 비어 있습니다. python -m msqa.build_index 를 먼저 실행하세요.")
        self.n_chunks = n

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """[{chunk_id, text, metadata, distance}, ...] 반환."""
        return self._index.query(query, top_k=top_k or self.top_k)


# ---------------------------------------------------------------------------
# FinQA: 문항별 in-memory 검색기 (전역 인덱스와 동일한 임베딩·코사인 top-k)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from msqa.common import load_config as _load_config  # noqa: E402
from msqa.rag_index import OpenAIEmbedder as _OpenAIEmbedder  # noqa: E402


class FinqaRetriever:
    """FinQA 문항의 self-contained 컨텍스트(text_N/table_N)를 임베딩해
    코사인 유사도 top-k 로 검색한다. K-DART 전역 인덱스와 동일한 임베딩 모델
    (text-embedding-3-small)·동일한 top_k 를 사용하므로 검색기 시그니처가 일치한다.
    """

    _embedder: _OpenAIEmbedder | None = None  # 문항 간 임베더 재사용(클라이언트 1개)

    def __init__(self, context_chunks: list[dict]) -> None:
        cfg = _load_config()
        self.top_k = int(cfg["rag"]["top_k"])
        if FinqaRetriever._embedder is None:
            FinqaRetriever._embedder = _OpenAIEmbedder(
                model=cfg["rag"]["model"],
                batch_size=int(cfg["rag"].get("embedding_batch_size", 100)),
            )
        self._chunks = context_chunks
        texts = [c["text"] for c in context_chunks]
        embs = FinqaRetriever._embedder.embed(texts) if texts else []
        self._mat = np.array(embs, dtype=float) if embs else np.zeros((0, 1))
        # 코사인 유사도용 정규화
        if self._mat.size:
            norms = np.linalg.norm(self._mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._unit = self._mat / norms

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.top_k
        if not self._chunks:
            return []
        q = np.array(FinqaRetriever._embedder.embed([query])[0], dtype=float)
        qn = q / (np.linalg.norm(q) or 1.0)
        sims = self._unit @ qn
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            c = self._chunks[int(i)]
            out.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": {},
                "distance": float(1.0 - sims[int(i)]),  # cosine distance
            })
        return out


def get_retriever(item, kdart_retriever: "Retriever | None"):
    """QAItem.source 에 따라 적절한 검색기를 반환.
    - kdart -> 전역 ChromaDB Retriever (재사용)
    - finqa -> 문항별 FinqaRetriever (새로 구성)
    """
    if item.source == "finqa":
        return FinqaRetriever(item.extra.get("context_chunks", []))
    if kdart_retriever is None:
        kdart_retriever = Retriever()
    return kdart_retriever
