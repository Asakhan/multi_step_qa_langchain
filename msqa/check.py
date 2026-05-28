"""사전 점검 — 토큰을 쓰지 않고 실험 환경이 준비됐는지 확인한다.

확인 항목:
  1) OPENAI_API_KEY 존재
  2) K-DART-QA(40) / FinQA(20) 로드 가능 + 개수
  3) ChromaDB 인덱스 존재 + 청크 수 (임베딩 쿼리는 호출하지 않음)
  4) LangChain 도구 2종 구성 가능
"""
from __future__ import annotations

import os
import sys

from .datasets import load_items


def main() -> int:
    ok = True

    # 1) API key
    if os.environ.get("OPENAI_API_KEY"):
        print("[1] OPENAI_API_KEY: 설정됨")
    else:
        print("[1] OPENAI_API_KEY: 없음 — .env 에 추가하세요")
        ok = False

    # 2) datasets
    try:
        kd = load_items("kdart")
        print(f"[2] K-DART-QA 로드: {len(kd)}문항 (기대 40)")
        import collections
        print("    유형분포:", dict(collections.Counter(i.task_type for i in kd)))
    except Exception as e:  # noqa: BLE001
        print(f"[2] K-DART-QA 로드 실패: {e}")
        ok = False
    try:
        fq = load_items("finqa")
        print(f"[2] FinQA 로드: {len(fq)}문항 (기대 20)")
    except Exception as e:  # noqa: BLE001
        print(f"[2] FinQA 로드 실패(아직 prepare_finqa 미실행?): {e}")

    # 3) index
    try:
        from .retriever import Retriever
        r = Retriever()
        print(f"[3] ChromaDB 인덱스: 청크 {r.n_chunks:,}개")
    except Exception as e:  # noqa: BLE001
        print(f"[3] 인덱스 점검 실패: {e}")
        ok = False

    # 4) tools
    try:
        from .tools import make_tools
        tools = make_tools(retriever=None, evidence_log=[])
        print(f"[4] 도구 구성: {[t.name for t in tools]}")
    except Exception as e:  # noqa: BLE001
        print(f"[4] 도구 구성 실패: {e}")
        ok = False

    print("\n결과:", "준비 완료 ✅" if ok else "미흡 ❌ (위 항목 확인)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
