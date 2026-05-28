"""실험 러너 — LangChain Agent × V0.

문항 × 반복(run_index) 을 순회하며 1건당 1 JSONL 레코드를 results/ 에 기록한다.
중단 후 재실행하면 이미 완료된 (qid, run_index) 는 건너뛴다(이어하기).

사용 예:
  # 1) 토큰 안 쓰는 사전 점검
  python -m experiment.check

  # 2) 스모크: K-DART 1문항 1회만
  python -m experiment.run --dataset kdart --limit 1 --repeats 1

  # 3) 본 실행: 60문항 × 3반복 (LangChain × V0)
  python -m experiment.run --dataset all --repeats 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from msqa.common import ensure_dir, project_path

from .datasets import load_items
from .executor_langchain import run_one
from .retriever import Retriever, get_retriever


def _done_keys(out_path: Path) -> set[tuple[str, int]]:
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["qid"], r["run_index"]))
            except Exception:  # noqa: BLE001
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["kdart", "finqa", "all"])
    ap.add_argument("--repeats", type=int, default=3, help="문항당 반복 횟수(실행 일관성용)")
    ap.add_argument("--limit", type=int, default=0, help="문항 수 제한(0=전체)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--framework", default="langchain")
    ap.add_argument("--verifier", default="V0")
    ap.add_argument("--out", default="results/langchain_v0.jsonl")
    args = ap.parse_args()

    items = load_items(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]

    out_path = project_path(args.out)
    ensure_dir(out_path.parent)
    done = _done_keys(out_path)

    # K-DART 전역 검색기는 1회만 연다(FinQA 는 문항별로 새로 만든다).
    kdart_retriever = None
    if any(it.source == "kdart" for it in items):
        kdart_retriever = Retriever()
        print(f"[index] ChromaDB chunks = {kdart_retriever.n_chunks:,}")

    total = len(items) * args.repeats
    print(f"[plan] items={len(items)} × repeats={args.repeats} = {total} runs "
          f"(framework={args.framework}, verifier={args.verifier}, model={args.model})")

    n_done = n_ok = 0
    fout = out_path.open("a", encoding="utf-8")
    t_start = time.perf_counter()
    for it in items:
        retr = get_retriever(it, kdart_retriever)
        for run_index in range(1, args.repeats + 1):
            if (it.qid, run_index) in done:
                continue
            rec = run_one(
                it, retr,
                run_index=run_index, model=args.model,
                framework=args.framework, verifier=args.verifier,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            n_ok += int(bool(rec["correct"]))
            status = "OK " if rec["correct"] else ("ERR" if rec["error"] else "x  ")
            print(f"  [{n_done}/{total - len(done)}] {status} {it.qid} run{run_index} "
                  f"tok={rec['total_tokens']} t={rec['time_sec']}s pred={rec['predicted']!r}")
    fout.close()

    elapsed = time.perf_counter() - t_start
    if n_done:
        print(f"[done] {n_done} new runs, TSR(new)={n_ok}/{n_done}={n_ok/n_done:.3f}, "
              f"elapsed={elapsed:.1f}s -> {out_path}")
    else:
        print(f"[done] 새로 실행할 항목 없음(모두 완료됨) -> {out_path}")


if __name__ == "__main__":
    main()
