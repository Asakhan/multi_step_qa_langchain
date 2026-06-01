"""검색기 — 데이터셋 구축 때 만든 ChromaDB 인덱스를 그대로 재사용.

src.rag_index.RagIndex 를 열어 config.yaml 의 rag.top_k(=5),
rag.model(text-embedding-3-small), paths.index(data/index) 설정을 따른다.
이렇게 하면 실험의 검색이 데이터셋 calibration과 완전히 동일해진다.
"""
from __future__ import annotations

from msqa.common import load_config, project_path
from msqa.rag_index import OpenAIEmbedder, RagIndex


class Retriever:
    def __init__(
        self,
        top_k_override: int | None = None,
        executor_max: int | None = None,
    ) -> None:
        cfg = load_config()
        # 인덱스에서 실제로 가져오는 개수(검색 강도).
        self.search_top_k: int = int(top_k_override) if top_k_override else int(cfg["rag"]["top_k"])
        # Executor 에게 노출하는 개수(컨텍스트 캡). 미지정 시 search_top_k 와 동일.
        self.executor_max: int = int(executor_max) if executor_max else self.search_top_k
        # 청크 본문 글자수 캡. run.py 가 설정하고 make_tools 가 읽는다(출력 텍스트에만 적용).
        self.chunk_char_cap: int | None = None
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
        """인덱스에서 search_top_k 만큼 가져오되, 호출자에게는 executor_max 만큼만 노출.

        [{chunk_id, text, metadata, distance}, ...] 반환.
        """
        k = top_k or self.search_top_k
        hits = self._index.query(query, top_k=k)
        return hits[: self.executor_max]


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

    def __init__(
        self,
        context_chunks: list[dict],
        top_k_override: int | None = None,
        executor_max: int | None = None,
    ) -> None:
        cfg = _load_config()
        self.search_top_k = int(top_k_override) if top_k_override else int(cfg["rag"]["top_k"])
        self.executor_max = int(executor_max) if executor_max else self.search_top_k
        self.chunk_char_cap: int | None = None
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
        k = top_k or self.search_top_k
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
        # 인덱스 검색은 search_top_k(=k) 기준, Executor 노출은 executor_max 기준.
        return out[: self.executor_max]


def get_retriever(
    item,
    kdart_retriever: "Retriever | None",
    top_k_override: int | None = None,
    executor_max: int | None = None,
):
    """QAItem.source 에 따라 적절한 검색기를 반환.
    - kdart -> 전역 ChromaDB Retriever (재사용)
    - finqa -> 문항별 FinqaRetriever (새로 구성)

    top_k_override/executor_max 가 주어지면 FinQA 검색기에 전달한다(K-DART 전역
    검색기는 호출부에서 동일 override 로 미리 생성해 넘긴다).
    """
    if item.source == "finqa":
        return FinqaRetriever(
            item.extra.get("context_chunks", []),
            top_k_override=top_k_override,
            executor_max=executor_max,
        )
    if kdart_retriever is None:
        kdart_retriever = Retriever(top_k_override=top_k_override, executor_max=executor_max)
    return kdart_retriever
