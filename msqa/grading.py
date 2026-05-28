"""채점 로직 — src/calibrator.py 와 동일(verbatim).

실험의 TSR(Task Success Rate)을 데이터셋 calibration과 정확히 같은 기준으로
매기기 위해, calibrator의 parse_number / normalize_text / grade 를 그대로 복제한다.
(import 대신 복제하는 이유: src.calibrator 는 google-generativeai 를 모듈 로드 시점에
 import 하므로, 실험에서 불필요한 의존성을 끌어오지 않기 위함.)
"""
from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def parse_number(s: str) -> float | None:
    """콤마·단위 접미사를 제거하고 한국어 수치 표현을 float 으로 파싱."""
    if s is None:
        return None
    m = _NUM_RE.search(str(s).replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def grade(predicted: str, gold: Any, *, rel_tol: float = 0.01) -> bool:
    """수치면 ±rel_tol 상대오차 비교, 아니면 정규화 문자열 정확일치(EM)."""
    if predicted is None or predicted == "":
        return False
    gnum = parse_number(gold) if not isinstance(gold, (int, float)) else float(gold)
    pnum = parse_number(predicted)
    if gnum is not None and pnum is not None:
        if gnum == 0:
            return abs(pnum) < rel_tol
        return abs(pnum - gnum) / abs(gnum) <= rel_tol
    return normalize_text(predicted) == normalize_text(gold)


# Executor 출력에서 "최종답: <값>" 을 뽑아내는 정규식 (calibrator와 동일 규약).
# 빈 최종답("최종답:" 뒤 공백/빈줄) 도 "매치"로 잡기 위해 캡처를 (.*?) 로 둔다.
# (calibrator 의 (.+?) 와 달리 빈 값을 매치로 인식하되, 빈 값은 extract_final 에서 None 처리.)
FINAL_ANSWER_RE = re.compile(r"최종답\s*[:：]\s*(.*?)\s*$", re.MULTILINE)


def extract_final(text: str) -> str | None:
    """모델 출력 텍스트에서 최종답을 추출.

    - "최종답:" 매치가 있고 값이 (공백 제거 후) 비어있지 않으면 그 값을 반환.
    - "최종답:" 매치가 있으나 값이 비어 있으면 None 반환(윗줄을 주워오지 않음).
    - "최종답:" 매치가 아예 없을 때만 폴백(마지막 비어있지 않은 줄)을 사용.
    """
    if not text:
        return None
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        val = matches[-1].strip()
        return val if val else None
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return None
