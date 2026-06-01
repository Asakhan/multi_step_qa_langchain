"""K-DART 검색 reach 진단 — top_k 5/10/20/50/100 에서 gold 청크가 잡히는 비율.

각 문항을 한 번만 임베딩해 top-100 을 가져온 뒤, 그 안에서 잘라 모든 top_k 의
reach 와 gold 첫 등장 위치를 분석한다. E4(검색 실패)가 top_k 상향으로 얼마나
줄어드는지 사전 판정용. 임베딩 호출은 40회뿐(쿼리는 로컬 ChromaDB, 무료).

실행: python -m msqa.diagnose_topk
출력: 콘솔 표 + results/topk_diagnosis.json
"""
import json
import statistics
from collections import Counter

from msqa.common import load_config, project_path
from msqa.rag_index import OpenAIEmbedder, RagIndex

TOP_KS = [5, 10, 20, 50, 100]


def main():
    cfg = load_config()
    embedder = OpenAIEmbedder(model=cfg["rag"]["model"], batch_size=100)
    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=embedder,
    )
    items = [json.loads(l) for l in open(project_path("data/kdart_qa.jsonl"), encoding="utf-8")]
    print(f"K-DART 문항: {len(items)}, 인덱스 청크: {index.count():,}")

    # 각 문항을 한 번씩만 임베딩해서 top-100 가져오기 (가장 큰 k 로 한 번 → 그 안에서 잘라 모든 k 분석)
    results = []
    for it in items:
        hits = index.query(it["question"], top_k=max(TOP_KS))  # top-100
        gold = set(it.get("evidence_chunks") or [])
        per_k = {}
        first_pos = None
        for k in TOP_KS:
            in_topk = bool(gold & set(h["chunk_id"] for h in hits[:k]))
            per_k[k] = in_topk
        # gold 가 처음 등장한 위치
        for i, h in enumerate(hits, start=1):
            if h["chunk_id"] in gold:
                first_pos = i
                break
        results.append({
            "qid": it["id"],
            "task_type": it["task_type"],
            "gold_count": len(gold),
            "first_pos": first_pos,  # gold 가 top-100 안에 처음 등장한 위치 (없으면 None)
            **{f"reach_top{k}": per_k[k] for k in TOP_KS},
        })

    # 보고
    print(f"\n=== reach by top_k ===")
    for k in TOP_KS:
        n = sum(r[f"reach_top{k}"] for r in results)
        print(f"  top_k={k:3d}: {n}/{len(results)} = {n/len(results):.3f}")

    print(f"\n=== task_type × reach_top20 ===")
    for t in sorted(set(r["task_type"] for r in results)):
        sub = [r for r in results if r["task_type"] == t]
        n = sum(r["reach_top20"] for r in sub)
        print(f"  {t}: {n}/{len(sub)}")

    print(f"\n=== gold 첫 등장 위치 분포 ===")
    positions = [r["first_pos"] for r in results if r["first_pos"] is not None]
    no_reach = sum(1 for r in results if r["first_pos"] is None)
    print(f"  top-100 안에 reach: {len(positions)}/{len(results)}, top-100 밖: {no_reach}")
    if positions:
        bins = Counter()
        for p in positions:
            if p <= 5: bins["1-5"] += 1
            elif p <= 10: bins["6-10"] += 1
            elif p <= 20: bins["11-20"] += 1
            elif p <= 50: bins["21-50"] += 1
            else: bins["51-100"] += 1
        print(f"  bins: {dict(bins)}")
        print(f"  median position: {statistics.median(positions)}")

    # JSON 저장
    with open(project_path("results/topk_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: results/topk_diagnosis.json")


if __name__ == "__main__":
    main()
