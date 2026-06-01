"""오답 오류유형 태깅 — 토큰 0 (LLM 호출 없이 기존 JSONL 레코드만 분석).

오답 케이스를 NON_NUM(비수치 gold)/E4(검색실패)/E2(계산오류)/E1(근거선택)/OTHER 로
자동 분류하고, source × task_type 매트릭스와 검증자(V2/V3)의 이론적 상한을 추정한다.

실행: python -m msqa.analyze_errors [--input ...] [--output ...]
기본 입력: results/langchain_v1_rescored.jsonl (읽기 전용)
기본 출력: results/error_tagging_rescored.json
"""
from __future__ import annotations

import argparse
import json
import re

from msqa.common import project_path

# 모델 포기 표현 — 수치 정규식에 걸리더라도 비숫자로 강제(E2 오분류 방지).
_GIVE_UP = ("정보 없음", "정보없음", "데이터 없음", "데이터없음", "확인 불가", "확인불가",
            "값 없음", "값없음", "정보 불충분", "정보불충분")


def _looks_numeric(s) -> bool:
    if s is None:
        return False
    text = str(s)
    if any(k in text for k in _GIVE_UP):
        return False
    return bool(re.search(r"-?\d[\d,\.]*", text))


def _gold_is_numeric(gold) -> bool:
    """gold 가 수치 답인가. 항목명/순서명(제N기·…부문 등)은 비수치로 본다."""
    s = str(gold).strip()
    if not re.search(r"-?\d", s):
        return False
    if re.match(r"^제?\d+[기차회월일]", s):
        return False
    if s.endswith(("부문", "업종", "사업부", "회사", "법인")):
        return False
    return True


def classify(record) -> str:
    gold = record.get("gold_answer")

    # NON_NUM: gold 자체가 비수치(항목명/순서명) → 계산검증 대상 아님.
    if not _gold_is_numeric(gold):
        return "NON_NUM"

    gold_chunks = set(record.get("gold_evidence_chunks") or [])
    v0_retrieved = set(e["chunk_id"] for e in (record.get("v0_retrieved_evidence") or []))
    retry_retrieved = set(e["chunk_id"] for e in (record.get("retry_retrieved_evidence") or []))
    all_retrieved = v0_retrieved | retry_retrieved

    # E4: gold 청크가 v0/retry 어느 쪽에도 안 들어옴.
    if gold_chunks and not (gold_chunks & all_retrieved):
        return "E4"

    # gold 가 검색됐지만 오답 — E2 vs E1 구분.
    used_calc = _used_calc(record)
    pred_is_numeric = _looks_numeric(record.get("predicted"))

    if used_calc and pred_is_numeric:
        return "E2"  # 계산 시도 + 답 숫자 → 계산 단계 오류
    return "E1"      # 계산 안 했거나 답이 비숫자 → 근거 선택/포기


def _gold_reached(record) -> bool:
    gold_chunks = set(record.get("gold_evidence_chunks") or [])
    v0_retrieved = set(e["chunk_id"] for e in (record.get("v0_retrieved_evidence") or []))
    retry_retrieved = set(e["chunk_id"] for e in (record.get("retry_retrieved_evidence") or []))
    return bool(gold_chunks & (v0_retrieved | retry_retrieved))


def _used_calc(record) -> bool:
    v0 = any(s.get("tool") == "calculator" for s in (record.get("v0_reasoning_trace") or []))
    rt = any(s.get("tool") == "calculator" for s in (record.get("retry_reasoning_trace") or []))
    return v0 or rt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/langchain_v1_rescored.jsonl")
    ap.add_argument("--output", default="results/error_tagging_rescored.json")
    args = ap.parse_args()

    in_path = project_path(args.input)
    out_path = project_path(args.output)
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    total_runs = len(rows)

    wrong = [r for r in rows if not r["correct"]]
    regressed = [r for r in wrong if r.get("v0_correct")]

    tagged = []
    for r in wrong:
        tagged.append({
            "qid": r["qid"], "source": r["source"], "task_type": r["task_type"],
            "error_type": classify(r),
            "gold": r.get("gold_answer"), "predicted": r.get("predicted"),
            "gold_chunks_reached": _gold_reached(r),
            "used_calculator": _used_calc(r),
            "regressed": bool(r.get("v0_correct")),
        })

    ETYPES = ["E4", "E2", "E1", "NON_NUM", "OTHER"]
    LABELS = {"E4": "E4 (검색실패)", "E2": "E2 (계산오류)", "E1": "E1 (근거선택)",
              "NON_NUM": "NON_NUM (비수치)", "OTHER": "OTHER"}
    TASKS = ["T1", "T2", "T3", "T4", "T5"]
    src_total = {s: sum(1 for r in rows if r["source"] == s) for s in ("kdart", "finqa")}

    # --- 표 1: 전체 분포 ---
    print(f"\n오답 분류 (재채점 후, n={len(wrong)})")
    print(f"{'':<18}{'kdart':<9}{'finqa':<9}{'전체':<9}")
    for et in ETYPES:
        kd = sum(1 for t in tagged if t["error_type"] == et and t["source"] == "kdart")
        fq = sum(1 for t in tagged if t["error_type"] == et and t["source"] == "finqa")
        print(f"  {LABELS[et]:<16}{kd:<9}{fq:<9}{kd + fq:<9}")
    print(f"  {'(regressed)':<16}{len(regressed)}")

    # --- 표 2: task_type × error_type ---
    print(f"\ntask_type × error_type 매트릭스")
    print(f"{'':<8}{'E4':<6}{'E2':<6}{'E1':<6}{'NON_NUM':<9}{'OTHER':<7}{'n':<5}")
    for tt in TASKS:
        row = {et: sum(1 for t in tagged if t["task_type"] == tt and t["error_type"] == et) for et in ETYPES}
        n = sum(row.values())
        print(f"  {tt:<6}{row['E4']:<6}{row['E2']:<6}{row['E1']:<6}{row['NON_NUM']:<9}{row['OTHER']:<7}{n:<5}")
    totrow = {et: sum(1 for t in tagged if t["error_type"] == et) for et in ETYPES}
    print(f"  {'전체':<6}{totrow['E4']:<6}{totrow['E2']:<6}{totrow['E1']:<6}"
          f"{totrow['NON_NUM']:<9}{totrow['OTHER']:<7}{len(wrong):<5}")

    # --- 표 3: 검증자 이론적 상한 ---
    tsr = sum(r["correct"] for r in rows) / total_runs
    v2_ceiling = totrow["E2"] / total_runs
    v3_ceiling = (totrow["E1"] + totrow["E2"]) / total_runs
    print(f"\nV2 상한 = E2/180 = {totrow['E2']}/{total_runs} = {v2_ceiling:.3f}  → TSR {tsr:.3f} 대비 상한 {tsr + v2_ceiling:.3f}")
    print(f"V3 상한 = (E1+E2)/180 = {totrow['E1'] + totrow['E2']}/{total_runs} = {v3_ceiling:.3f}  → TSR {tsr:.3f} 대비 상한 {tsr + v3_ceiling:.3f}")

    # --- 표 4: 샘플 케이스 (각 유형 3건) ---
    print(f"\n샘플 케이스 (각 유형 3건 — 분류 검증용)")
    for et in ETYPES:
        ex = [t for t in tagged if t["error_type"] == et][:3]
        print(f"  [{et}]")
        if not ex:
            print("    (없음)")
        for t in ex:
            print(f"    {t['qid']:<26} ({t['source']}/{t['task_type']}) "
                  f"gold={t['gold']!r} pred={t['predicted']!r} "
                  f"reached={t['gold_chunks_reached']} calc={t['used_calculator']}")

    out_path.write_text(json.dumps(tagged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}  ({len(tagged)} 오답 레코드)")


if __name__ == "__main__":
    main()
