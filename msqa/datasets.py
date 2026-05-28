"""60문항 로더 — K-DART-QA(한국어 40) + FinQA(영어 20)를 공통 스키마로 정규화.

K-DART-QA: data/kdart_qa.jsonl  (gpt-4o-mini calibration 통과본, 40문항)
FinQA    : data/finqa.jsonl          (발표 논문[3]에서 채택한 20문항 — 사용자가 배치)

FinQA jsonl 의 기대 스키마(최소):
  {"id": "...", "question": "...", "gold_answer": "...", "task_type": "T3|T4", ...}
필드명이 다르면 _load_finqa 의 매핑만 고치면 된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from msqa.common import project_path


@dataclass
class QAItem:
    qid: str
    question: str
    gold_answer: str
    gold_unit: str | None
    gold_evidence_chunks: list[str]  # E1/E4 오류 분석용 gold chunk id
    task_type: str                   # T1~T5
    lang: str                        # "ko" | "en"
    source: str                      # "kdart" | "finqa"
    extra: dict = field(default_factory=dict)


def _load_kdart(path: Path) -> list[QAItem]:
    items: list[QAItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(
                QAItem(
                    qid=r["id"],
                    question=r["question"],
                    gold_answer=str(r["gold_answer"]),
                    gold_unit=r.get("gold_answer_unit"),
                    gold_evidence_chunks=list(r.get("evidence_chunks") or []),
                    task_type=r.get("task_type", "T?"),
                    lang="ko",
                    source="kdart",
                    extra={
                        "reasoning_hops": r.get("reasoning_hops"),
                        "source_company": r.get("source_company"),
                        "source_report": r.get("source_report"),
                    },
                )
            )
    return items


def _load_finqa(path: Path) -> list[QAItem]:
    items: list[QAItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            items.append(
                QAItem(
                    qid=r["id"],
                    question=r["question"],
                    gold_answer=str(r["gold_answer"]),
                    gold_unit=r.get("gold_answer_unit"),
                    gold_evidence_chunks=list(r.get("evidence_chunks") or []),
                    task_type=r.get("task_type", "T?"),
                    lang="en",
                    source="finqa",
                    extra={
                        "n_steps": r.get("n_steps"),
                        # FinQA 문항별 RAG 검색용 self-contained 컨텍스트
                        "context_chunks": r.get("context_chunks") or [],
                    },
                )
            )
    return items


def load_items(dataset: str = "all") -> list[QAItem]:
    """dataset in {"kdart", "finqa", "all"} -> QAItem 리스트."""
    kdart_path = project_path("data/kdart_qa.jsonl")
    finqa_path = project_path("data/finqa.jsonl")

    items: list[QAItem] = []
    if dataset in ("kdart", "all"):
        if not kdart_path.exists():
            raise FileNotFoundError(f"K-DART-QA 파일이 없습니다: {kdart_path}")
        items += _load_kdart(kdart_path)
    if dataset in ("finqa", "all"):
        if finqa_path.exists():
            items += _load_finqa(finqa_path)
        elif dataset == "finqa":
            raise FileNotFoundError(
                f"FinQA 파일이 없습니다: {finqa_path}\n"
                "발표 논문[3]의 FinQA 20문항을 위 경로에 JSONL 로 배치하세요."
            )
        else:
            print(f"[warn] FinQA 파일 없음({finqa_path}) — K-DART-QA만 로드합니다.")
    return items
