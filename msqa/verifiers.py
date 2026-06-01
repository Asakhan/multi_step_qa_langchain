"""검증자(Verifier) — 논문 3.4 의 V1(LLM 검증자).

V1 은 Executor(V0) 가 생성한 답을 입력으로만 받아(=도구 호출 없음) 세 가지 기준
(Evidence Grounding / Logical Consistency / Constraint Satisfaction)을 LLM 으로
검토하고 PASS/FAIL 과 (FAIL 시) critique 를 반환한다. 도구·검색기·새로운 사실은
일절 쓰지 않는다(이 점이 V2 와의 구분이다).

토큰 집계: 검증은 단일 ChatOpenAI 호출(prompt → 응답)이며, UsageMetadataCallbackHandler
로 Executor 와 분리해 직접 합산한다(run_one 의 콜백 방식과 동일).
"""
from __future__ import annotations

import re
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .executor_langchain import EXECUTOR_MODEL, NUMERIC_REL_TOL, _sum_usage
from .grading import grade
from .tools import calculate

VERIFIER_MODEL = EXECUTOR_MODEL  # 논문 통제변수: Executor 와 동일 gpt-4o-mini

# --- V1 프롬프트 상수 (부록 A 확정본) ---
VERIFIER_SYSTEM_PROMPT = (
    "당신은 한국어 재무 공시 문서 QA 의 엄격한 검증자입니다. 답변이 검색된 청크(evidence)에서 "
    "직접 도출됐는지를 chunk_id 단위로 확인하세요.\n\n"
    "검증 원칙:\n"
    "- evidence 밖의 지식이나 추론으로 답을 정당화하지 마세요. evidence 청크에 명시되지 않은 "
    "사실은 PASS 사유가 될 수 없습니다.\n"
    "- 답변에 인용된 모든 수치·항목은 evidence 청크 어느 하나의 본문에서 직접 찾을 수 있어야 합니다.\n"
    "- 그 청크가 질문이 묻는 회사·연도·항목을 명시적으로 다루고 있어야 합니다(예: 청크 id 와 본문이 "
    "그 회사·연도를 표기).\n"
    "- 모델이 \"정보 없음\"이라고 답안 중간에 적었거나 evidence 가 질문의 연도를 다루지 않는데 최종 "
    "수치를 제시했다면 즉시 FAIL 입니다.\n"
    "- critique 는 항상 구체적이어야 합니다. 추상적 지시(\"다시 계산하세요\", \"자료를 확인하세요\")는 금지."
)

VERIFIER_USER_TEMPLATE = """
Question: {query}

Candidate Answer: {candidate_answer}

Reasoning Trace:
{reasoning_trace_text}

Retrieved Evidence (chunk_id: text):
{evidence_text}

세 가지 기준을 각각 채점하세요. 각 기준에서 PASS 라면 그 근거가 된 chunk_id 를 본문에서 인용하세요.

1) Evidence Grounding
   - 답변에 나오는 모든 수치/항목이 evidence 청크의 본문에서 직접 찾을 수 있는가?
   - 그 수치를 담은 청크의 본문이 질문이 묻는 "회사·연도·항목"과 일치하는가? (예: 질문이 "POSCO홀딩스 2023 비유동부채" 이면, 인용 청크 본문이 POSCO홀딩스의 2023년 재무상태표를 다뤄야 함)
   - 위 둘 중 하나라도 어긋나면 FAIL. 답변 수치가 evidence 청크들 중 어디에도 없거나, 있더라도 다른 회사/연도/항목의 청크에서 가져왔으면 FAIL.

2) Logical Consistency
   - reasoning 단계가 서로 모순 없는가?
   - 답변 본문에서 모델이 스스로 "정보 없음/데이터 부족"이라고 인정한 경우, 최종답에 구체 수치가 있으면 FAIL(모순).

3) Constraint Satisfaction
   - 질문의 명시적 제약(연도, 회사, 별도/연결, 단위, "전년 대비/증감률" 같은 계산 방향)을 답변이 정확히 만족하는가?
   - evidence 청크들이 질문의 연도·회사를 실제로 다루지 않는데(예: 2024년을 물었으나 2022~2023 청크만 검색됨) 답이 그 연도의 구체 수치를 주장하면 FAIL.

응답 형식(반드시 이대로, 각 줄 끝에 사유 또는 인용 chunk_id 포함):
Grounding: PASS|FAIL — <사유. PASS 면 인용 chunk_id 와 해당 본문 일부; FAIL 이면 어떤 수치가 어느 청크와 어긋나는지>
Consistency: PASS|FAIL — <사유>
Constraint: PASS|FAIL — <사유. PASS 면 어떤 청크가 질문의 연도/회사를 cover 하는지; FAIL 이면 누락된 제약>
Verdict: PASS|FAIL
Critique: <FAIL 일 때만. 형식: "[chunk_id]에 따르면 X인데 답변은 Y를 주장한다" 또는 "evidence 청크 어느 것도 질문의 [연도/회사/항목]을 다루지 않는다 — 답변은 추정이다. Executor 가 [구체적 다음 검색어 또는 다음 단계]를 시도해야 한다." 추상적 표현 금지.>
"""
# --- V1 프롬프트 상수 끝 ---


def _format_trace(trace: list | None) -> str:
    """reasoning_trace(list of step dict) 를 사람이 읽을 수 있는 텍스트로 직렬화."""
    if not trace:
        return "(추론 단계 없음)"
    lines = []
    for s in trace:
        tool = s.get("tool")
        tin = s.get("tool_input")
        obs = s.get("observation")
        lines.append(f"{s.get('step')}. {tool}({tin}) -> {obs}")
    return "\n".join(lines)


def _format_evidence(evidence: list | None) -> str:
    """retrieved_evidence(list of {chunk_id,text,...}) 를 'chunk_id: text' 로 직렬화."""
    if not evidence:
        return "(검색된 근거 없음)"
    return "\n\n".join(
        f"{e.get('chunk_id')}: {e.get('text')}" for e in evidence
    )


def _parse_verdict(text: str) -> tuple[str, str]:
    """LLM 응답에서 Verdict 와 Critique 를 추출.

    Verdict 가 정확히 'PASS' 일 때만 PASS, 그 외(부분 PASS·표기 변형 포함)는 FAIL.
    Critique 는 'Critique:' 마커 이후 전체(여러 문장)를 취하며, PASS 면 비운다.
    """
    verdict = "FAIL"
    # 마지막으로 등장한 Verdict 라인을 채택.
    for m in re.finditer(r"^\s*Verdict\s*[:：]\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE):
        val = m.group(1).strip().upper()
        verdict = "PASS" if val == "PASS" else "FAIL"

    critique = ""
    cm = re.search(r"Critique\s*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if cm:
        critique = cm.group(1).strip()

    if verdict == "PASS":
        critique = ""
    return verdict, critique


def verify(query: str, v0_record: dict, *, model: str = VERIFIER_MODEL) -> dict:
    """V0 실행 결과(v0_record)를 보고 V1 판정을 수행.

    반환: {verdict, critique, raw, tokens_in, tokens_out, tokens_total, time_sec}
    토큰은 이 검증 호출만 집계하여 Executor 와 섞이지 않는다.
    """
    candidate = v0_record.get("predicted")
    trace_text = _format_trace(v0_record.get("reasoning_trace"))
    evidence_text = _format_evidence(v0_record.get("retrieved_evidence"))
    user_msg = VERIFIER_USER_TEMPLATE.format(
        query=query,
        candidate_answer=candidate,
        reasoning_trace_text=trace_text,
        evidence_text=evidence_text,
    )

    # 도구 없이 단일 ChatOpenAI 호출(prompt → 응답). run_one 과 동일한 콜백 집계 방식.
    llm = ChatOpenAI(model=model)
    usage_cb = UsageMetadataCallbackHandler()
    t0 = time.perf_counter()
    resp = llm.invoke(
        [SystemMessage(content=VERIFIER_SYSTEM_PROMPT), HumanMessage(content=user_msg)],
        config={"callbacks": [usage_cb]},
    )
    elapsed = time.perf_counter() - t0

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    verdict, critique = _parse_verdict(content)
    in_tok, out_tok, tot_tok = _sum_usage(usage_cb.usage_metadata)

    return {
        "verdict": verdict,
        "critique": critique,
        "raw": content,
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "tokens_total": tot_tok,
        "time_sec": round(elapsed, 4),
    }


# =====================================================================
# V2 — 외부 계산기 검증자 (논문 3.4.4)
# =====================================================================
# V2 의 본질: 검증자가 reasoning 에서 산술식을 추출(LLM 1회)하고, 그 식을 Python
# 안전 평가기(msqa.tools.calculate)로 직접 재계산한 뒤, 모델 답과 불일치하면
# 재계산 값을 critique 에 그대로 박아 retry 가 그 값을 쓰도록 한다. 수치 계산
# 오답(E2)에만 개입하며, 비수치/추출불가/계산일치는 PASS(개입 안 함)로 둔다.

V2_EXTRACT_SYSTEM_PROMPT = (
    "당신은 한국어 재무 공시 QA 의 계산 검증자입니다. 모델의 reasoning 과 최종답을 보고, "
    "모델이 어떤 산술식으로 답을 만들었는지 추출하세요. 추측·창작 금지: reasoning 에 실제로 "
    "등장한 수치와 연산만 사용하세요."
)

V2_EXTRACT_USER_TEMPLATE = """
Question: {query}

Reasoning Trace:
{reasoning_trace_text}

Answer (raw):
{answer_raw}

Retrieved Evidence (chunk_id: text):
{evidence_text}

다음 형식으로 정확히 응답하세요(다른 텍스트 금지):

Question_Type: CALC_REQUIRED 또는 NON_NUMERIC
Extracted_Formula: <파이썬 산술식. 예: (765124-498879)/498879*100. 비수치 질문이면 비움.>
Extracted_Operands: <피연산자마다 "값: chunk_id - 설명" 한 줄. 비수치면 비움.>
Final_Numeric_Answer: <answer_raw 의 '최종답:' 뒤 수치. 콤마·% 제거 후 순수 숫자. 없으면 NONE.>

판단 규칙:
- 답이 수치(숫자/퍼센트/비율)이고 reasoning 에 계산이 있으면 CALC_REQUIRED.
- 답이 항목/부문/이름 같은 비수치 또는 "정보 없음/데이터 부족" 류면 NON_NUMERIC.
- reasoning 에 명시된 수식이 없으면(모델이 단순히 청크 값을 그대로 갖다 붙임) NON_NUMERIC.
"""


def _extract_field(text: str, key: str) -> str:
    """'Key: value' 한 줄에서 value 를 추출(없으면 빈 문자열)."""
    m = re.search(rf"^\s*{key}\s*[:：]\s*(.*?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def verify_v2(query: str, v0_record: dict, *, model: str = VERIFIER_MODEL) -> dict:
    """V2 계산 검증. V1 의 verify() 와 동일 시그니처/반환 키 + v2 전용 2키.

    반환: {verdict, critique, raw, tokens_in/out/total, time_sec,
           recalculated, skip_reason}
    skip_reason ∈ {"non_numeric_gold","non_numeric","parse_fail","calc_match", None(=FAIL)}
    """
    # 비수치 gold 가드 — 추출 LLM 호출 전(토큰 0). gold 가 항목명/순서명이면
    # 계산 검증 대상이 아니므로 즉시 PASS(개입 안 함). answer-기반 non_numeric
    # 가드보다 한 단계 앞에서 gold-기반으로 차단(둘 다 PASS 라 충돌 없음).
    gold = v0_record.get("gold_answer")
    if gold is not None:
        gold_str = str(gold).strip()
        if (not re.search(r"\d", gold_str)) or \
           re.match(r"^제?\d+[기차회월일]", gold_str) or \
           gold_str.endswith(("부문", "업종", "사업부", "회사", "법인")):
            return {
                "verdict": "PASS",
                "critique": "",
                "raw": "",
                "tokens_in": 0, "tokens_out": 0, "tokens_total": 0,
                "time_sec": 0.0,
                "skip_reason": "non_numeric_gold",
                "recalculated": None,
            }

    trace_text = _format_trace(v0_record.get("reasoning_trace"))
    evidence_text = _format_evidence(v0_record.get("retrieved_evidence"))
    answer_raw = v0_record.get("answer_raw") or ""
    user_msg = V2_EXTRACT_USER_TEMPLATE.format(
        query=query,
        reasoning_trace_text=trace_text,
        answer_raw=answer_raw,
        evidence_text=evidence_text,
    )

    # 1단계 — 산술식 추출 (LLM 1회, 토큰 발생).
    llm = ChatOpenAI(model=model)
    usage_cb = UsageMetadataCallbackHandler()
    t0 = time.perf_counter()
    resp = llm.invoke(
        [SystemMessage(content=V2_EXTRACT_SYSTEM_PROMPT), HumanMessage(content=user_msg)],
        config={"callbacks": [usage_cb]},
    )
    elapsed = time.perf_counter() - t0
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    in_tok, out_tok, tot_tok = _sum_usage(usage_cb.usage_metadata)

    def _ret(verdict, critique, recalculated, skip_reason):
        return {
            "verdict": verdict,
            "critique": critique,
            "raw": content,
            "tokens_in": in_tok,
            "tokens_out": out_tok,
            "tokens_total": tot_tok,
            "time_sec": round(elapsed, 4),
            "recalculated": recalculated,
            "skip_reason": skip_reason,
        }

    qtype = _extract_field(content, "Question_Type").upper()
    formula = _extract_field(content, "Extracted_Formula")
    final_numeric = _extract_field(content, "Final_Numeric_Answer")

    # 비수치 질문 / 계산 미사용 → V2 개입 대상 아님 → PASS.
    if qtype != "CALC_REQUIRED":
        return _ret("PASS", "", None, "non_numeric")

    # 2단계 — Python 재계산. 식이 비었거나 최종답 추출 불가면 개입 불가 → PASS.
    if not formula or final_numeric.upper() == "NONE" or not final_numeric:
        return _ret("PASS", "", None, "parse_fail")

    recalc = calculate(formula)
    if recalc.startswith("계산 오류"):  # 안전 평가기 실패 → 개입 불가 → PASS.
        return _ret("PASS", "", None, "parse_fail")

    # 모델 답 vs 재계산 값 비교 (±1% 수치 비교 재사용).
    if grade(final_numeric, recalc, rel_tol=NUMERIC_REL_TOL):
        return _ret("PASS", "", None, "calc_match")

    # 불일치 → FAIL. 재계산 값을 critique 에 직접 박아 retry 가 그대로 쓰게 한다.
    critique = (
        f"당신의 reasoning 에 따른 계산을 외부 계산기로 재현했습니다.\n"
        f"식: {formula}\n"
        f"재계산 결과: {recalc}\n"
        f"당신이 제시한 답: {final_numeric}\n"
        f"두 값이 일치하지 않습니다(상대오차 > 1%). "
        f"재계산 결과인 {recalc} 를 최종답으로 사용하세요. "
        f"마지막 줄은 반드시 '최종답: {recalc}' 형식이어야 합니다."
    )
    return _ret("FAIL", critique, recalc, None)


# =====================================================================
# V3 — 결합 검증자 (논문 3.4.5): V2(계산) 먼저, PASS 면 V1(근거) 보충
# =====================================================================
# V2 가 산술 오류를 잡으면(재계산값 박은 critique) 그대로 사용. V2 가 PASS/비개입이면
# V1 으로 근거·일관성·제약을 보충 검증한다. V1/V2 함수·프롬프트는 재사용만 한다.

def verify_v3(query: str, v0_record: dict, *, model: str = VERIFIER_MODEL) -> dict:
    """V1 + V2 결합. V2 먼저, PASS 면 V1 추가.

    반환: verify() 의 키 + {verifier_used, v1_verdict, v2_verdict,
    v2_skip_reason, v2_recalculated}. verifier_used ∈ {"V2","V1","PASS"}.
    """
    v2_result = verify_v2(query, v0_record, model=model)

    # V2 가 FAIL 이면 그 결과(재계산값 박은 critique)를 그대로 사용 — V1 호출 안 함.
    if v2_result["verdict"] == "FAIL":
        return {
            "verdict": "FAIL",
            "critique": v2_result["critique"],
            "raw": v2_result.get("raw", ""),
            "tokens_in": v2_result["tokens_in"],
            "tokens_out": v2_result["tokens_out"],
            "tokens_total": v2_result["tokens_total"],
            "time_sec": v2_result["time_sec"],
            "verifier_used": "V2",
            "v1_verdict": None,
            "v2_verdict": "FAIL",
            "v2_skip_reason": v2_result.get("skip_reason"),
            "v2_recalculated": v2_result.get("recalculated"),
        }

    # V2 PASS/비개입 → V1 보충 검증.
    v1_result = verify(query, v0_record, model=model)
    return {
        "verdict": v1_result["verdict"],          # 최종 판정은 V1 기준
        "critique": v1_result["critique"],
        "raw": v1_result.get("raw", ""),
        "tokens_in": v1_result["tokens_in"] + v2_result["tokens_in"],
        "tokens_out": v1_result["tokens_out"] + v2_result["tokens_out"],
        "tokens_total": v1_result["tokens_total"] + v2_result["tokens_total"],
        "time_sec": round(v1_result["time_sec"] + v2_result["time_sec"], 4),
        "verifier_used": "V1" if v1_result["verdict"] == "FAIL" else "PASS",
        "v1_verdict": v1_result["verdict"],
        "v2_verdict": v2_result["verdict"],
        "v2_skip_reason": v2_result.get("skip_reason"),
        "v2_recalculated": v2_result.get("recalculated"),
    }
