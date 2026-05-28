"""FinQA test.json -> data/finqa.jsonl 변환 (발표 논문[3] 20문항 구성).

발표 논문[3]은 '단일 근거 수치 질의 10개(=program 1-step) + 다단계 계산 10개(=program 2-step 이상)'
를 사용한다. 본 스크립트는 같은 기준으로 결정론적으로 20개를 선택한다.
정확히 동일한 20개 id 목록이 있으면 --ids 로 지정해 고정할 수 있다.

각 문항은 self-contained 하다: pre_text/post_text/table 을 context_chunks 로 함께 저장하여
FinqaRetriever 가 문항별 RAG 검색을 수행한다.

사용:
  python -m experiment.prepare_finqa --finqa-test /path/to/FinQA/dataset/test.json
  python -m experiment.prepare_finqa --finqa-test ... --ids id1,id2,...   # 명시 선택
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from msqa.common import project_path


def _remove_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def table_row_to_text(header: list[str], row: list[str]) -> str:
    """FinQA general_utils.table_row_to_text 와 동일한 행 선형화."""
    res = ""
    if header and header[0]:
        res += header[0] + " "
    for head, cell in zip(header[1:], row[1:]):
        res += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return _remove_space(res).strip()


def build_context_chunks(item: dict) -> list[dict]:
    """pre_text/post_text 문장 + table 행을 FinQA 인덱싱 규약(text_N/table_N)으로 청크화."""
    chunks: list[dict] = []
    for i, t in enumerate(item.get("pre_text", [])):
        if t and t.strip():
            chunks.append({"chunk_id": f"text_{i}", "text": _remove_space(t)})
    n_pre = len(item.get("pre_text", []))
    table = item.get("table", [])
    if table:
        header = table[0]
        for r, row in enumerate(table[1:], start=1):
            txt = table_row_to_text(header, row)
            if txt:
                chunks.append({"chunk_id": f"table_{r}", "text": txt})
    for j, t in enumerate(item.get("post_text", [])):
        if t and t.strip():
            chunks.append({"chunk_id": f"text_{n_pre + j}", "text": _remove_space(t)})
    return chunks


def n_steps(item: dict) -> int:
    qa = item["qa"]
    steps = qa.get("steps")
    if isinstance(steps, list):
        return len(steps)
    prog = qa.get("program", "") or ""
    return max(1, prog.count("(") )


def to_record(item: dict) -> dict:
    qa = item["qa"]
    gold = qa.get("exe_ans", qa.get("answer"))
    return {
        "id": item["id"],
        "question": qa["question"],
        "gold_answer": str(gold),
        "gold_answer_unit": None,
        "evidence_chunks": list((qa.get("gold_inds") or {}).keys()),
        "task_type": "T3" if n_steps(item) <= 1 else "T4",
        "context_chunks": build_context_chunks(item),
        "n_steps": n_steps(item),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finqa-test", required=True, help="FinQA dataset/test.json 경로")
    ap.add_argument("--ids", default="", help="쉼표구분 id 목록(지정 시 그대로 사용)")
    ap.add_argument("--n-single", type=int, default=10)
    ap.add_argument("--n-multi", type=int, default=10)
    args = ap.parse_args()

    data = json.load(open(args.finqa_test, encoding="utf-8"))
    by_id = {d["id"]: d for d in data}

    if args.ids.strip():
        chosen = [by_id[i.strip()] for i in args.ids.split(",") if i.strip() in by_id]
    else:
        singles, multis = [], []
        for d in sorted(data, key=lambda x: x["id"]):
            if "qa" not in d or "question" not in d["qa"]:
                continue
            if n_steps(d) <= 1 and len(singles) < args.n_single:
                singles.append(d)
            elif n_steps(d) >= 2 and len(multis) < args.n_multi:
                multis.append(d)
            if len(singles) >= args.n_single and len(multis) >= args.n_multi:
                break
        chosen = singles + multis

    out_path = project_path("data/finqa.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in chosen:
            f.write(json.dumps(to_record(d), ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} FinQA items -> {out_path}")


if __name__ == "__main__":
    main()
