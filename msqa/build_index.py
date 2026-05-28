"""data/chunks/*.jsonl → ChromaDB 인덱스(data/index) 빌드.

데이터셋 저장소에서 만든 청크 코퍼스(data/chunks)를 이 저장소로 복사한 뒤 실행한다.
임베딩은 text-embedding-3-small(약 2,137청크 ≈ $0.03). 데이터셋 저장소의
03_build_index.py 와 동일한 RagIndex 를 사용하므로 인덱스가 동일하게 재현된다.

사용:
  python -m msqa.build_index            # 증분(upsert)
  python -m msqa.build_index --reset    # 기존 인덱스 삭제 후 재생성
"""
from __future__ import annotations

import argparse
import json
import shutil

from msqa.common import ensure_dir, load_config, project_path
from msqa.rag_index import OpenAIEmbedder, RagIndex, estimate_embedding_cost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="기존 인덱스 삭제 후 재생성")
    ap.add_argument("--yes", action="store_true", help="비용 확인 프롬프트 생략")
    args = ap.parse_args()

    cfg = load_config()
    chunks_dir = project_path(cfg["paths"]["chunks"])
    index_dir = project_path(cfg["paths"]["index"])

    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        print(f"청크 파일이 없습니다: {chunks_dir}/*.jsonl\n"
              f"데이터셋 저장소의 data/chunks/ 를 이 경로로 복사하세요.")
        return 1

    all_chunks: list[dict] = []
    for fp in files:
        for line in fp.open(encoding="utf-8"):
            if line.strip():
                all_chunks.append(json.loads(line))
    print(f"청크 {len(all_chunks):,}개 ({len(files)}개 파일)")

    est = estimate_embedding_cost(
        all_chunks, price_per_1m_tokens_usd=cfg["rag"]["cost_per_1m_tokens_usd"])
    print("임베딩 예상:", est.render())
    if not args.yes:
        if input("진행할까요? [y/N] ").strip().lower() != "y":
            print("중단."); return 0

    embedder = OpenAIEmbedder(
        model=cfg["rag"]["model"],
        batch_size=int(cfg["rag"]["embedding_batch_size"]),
    )
    if args.reset and index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"인덱스 초기화: {index_dir}")
    ensure_dir(index_dir)
    index = RagIndex(persist_dir=index_dir,
                     collection_name=cfg["rag"]["collection_name"],
                     embedder=embedder)
    batch = int(cfg["rag"]["embedding_batch_size"])
    for i in range(0, len(all_chunks), batch):
        index.add_chunks(all_chunks[i:i + batch])
        print(f"  적재 {min(i + batch, len(all_chunks))}/{len(all_chunks)}")
    print(f"완료. 컬렉션 크기 = {index.count():,} → {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
