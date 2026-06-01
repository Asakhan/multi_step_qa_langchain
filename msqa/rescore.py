"""재채점 — 기존 V0/V1/V2 JSONL 을 수정된 grade() 로 다시 채점한다(토큰 0).

LLM 호출 없이 기존 레코드의 predicted/v0_predicted/retry_predicted 와 gold_answer 만
새 grade() 규칙(비율-퍼센트 정규화 포함)으로 재비교한다. 원본은 건드리지 않고
별도 *_rescored.jsonl 로 저장하며, 재채점 전 값은 original_correct 에 보존한다.

수정 필드: correct, v0_correct, retry_correct, rescored(=True), original_correct.
그 외 필드(predicted, gold_answer, reasoning_trace 등)는 일절 변경하지 않는다.

사용:
  python -m msqa.rescore --input results/langchain_v0.jsonl --output results/langchain_v0_rescored.jsonl
"""
from __future__ import annotations

import argparse
import json

from msqa.common import project_path
from msqa.grading import grade


def _rescore_record(r: dict) -> dict:
    gold = r.get("gold_answer")
    r["original_correct"] = r.get("correct")
    r["rescored"] = True

    # V1/V2 레코드: v0_correct / retry_correct 보유 → 단계별 재채점 후 최종 채택.
    if "retried" in r:
        r["v0_correct"] = grade(r.get("v0_predicted") or "", gold)
        if r.get("retried"):
            r["retry_correct"] = grade(r.get("retry_predicted") or "", gold)
            r["correct"] = r["retry_correct"]
        else:
            # 재시도 없으면 retry_correct 는 그대로 None, 최종은 첫 시도.
            r["retry_correct"] = None
            r["correct"] = r["v0_correct"]
    else:
        # V0 레코드: 단일 시도.
        r["correct"] = grade(r.get("predicted") or "", gold)

    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = project_path(args.input)
    out_path = project_path(args.output)

    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    orig_ok = sum(1 for r in rows if r.get("correct"))

    rescored = [_rescore_record(r) for r in rows]
    new_ok = sum(1 for r in rescored if r["correct"])

    with out_path.open("w", encoding="utf-8") as f:
        for r in rescored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    label = in_path.stem
    delta = new_ok - orig_ok
    print(f"{label}: {orig_ok}→{new_ok} 정답 (+{delta})  n={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
