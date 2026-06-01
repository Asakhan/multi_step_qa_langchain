"""LangChain Agent (A1) Executor — V0(검증 없음) 기준선.

단일 LLM 추론 루프가 doc_search/calculator 도구를 순차 호출하여 답을 만든다.
실행 로그: query, answer, reasoning_trace, retrieved_evidence, tokens(in/out/total),
time, framework, verifier, run_index, 그리고 채점 결과(correct).

V1/V2/V3 검증은 이 Executor 의 출력(answer + reasoning_trace + retrieved_evidence)을
입력받는 별도 래퍼로 추가한다(다음 단계). 본 파일은 V0 경로를 완성한다.
"""
from __future__ import annotations

import time

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .datasets import QAItem
from .grading import extract_final, grade
from .tools import make_tools

EXECUTOR_MODEL = "gpt-4o-mini"   # 논문 v5 통제변수
MAX_ITERATIONS = 12              # 도구 호출 루프 상한
NUMERIC_REL_TOL = 0.01           # TSR 채점 허용 상대오차(1%)

SYSTEM_PROMPT = (
    "당신은 재무 공시 문서 QA 전문가입니다. 반드시 doc_search 도구로 근거를 검색한 뒤 답하세요. "
    "수치 계산이 필요하면 calculator 도구를 사용하세요. 추측하지 말고 검색된 근거에 기반하세요.\n"
    "답변에는 reasoning step(어떤 근거를 어떻게 계산했는지)을 간결히 포함하고, "
    "마지막 줄은 반드시 다음 형식으로 끝내세요:\n"
    "최종답: <단일 값 또는 항목>"
)


def _build_agent(tools, model: str = EXECUTOR_MODEL) -> AgentExecutor:
    # temperature 등 생성 파라미터는 모델 기본값을 사용(논문 통제변수: 모든 실행 동일 기본값).
    llm = ChatOpenAI(model=model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        verbose=False,
    )


def _sum_usage(usage_by_model: dict) -> tuple[int, int, int]:
    """get_usage_metadata_callback 의 모델별 usage 를 합산 -> (in, out, total)."""
    pin = pout = ptot = 0
    for u in (usage_by_model or {}).values():
        pin += int(u.get("input_tokens", 0) or 0)
        pout += int(u.get("output_tokens", 0) or 0)
        ptot += int(u.get("total_tokens", 0) or 0)
    if ptot == 0:
        ptot = pin + pout
    return pin, pout, ptot


def _serialize_steps(intermediate_steps) -> list[dict]:
    out = []
    for i, (action, observation) in enumerate(intermediate_steps, start=1):
        out.append({
            "step": i,
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "observation": str(observation)[:2000],  # 로그 비대화 방지
        })
    return out


def run_one(
    item: QAItem,
    retriever,
    *,
    run_index: int,
    model: str = EXECUTOR_MODEL,
    framework: str = "langchain",
    verifier: str = "V0",
    input_text: str | None = None,
) -> dict:
    """문항 1건을 실행하고 전체 실행 로그 dict 를 반환.

    Executor 에게 전달하는 입력은 기본적으로 item.question 이지만,
    input_text 가 주어지면(예: V1 재시도용 critique 포함 프롬프트) 그 문자열을
    대신 보낸다. 시스템 프롬프트·도구·검색 로직은 두 경로에서 완전히 동일하며,
    채점은 항상 item.gold_answer 기준이다.
    """
    evidence_log: list[dict] = []
    tools = make_tools(retriever, evidence_log)
    # make_tools 는 list 호환 _ToolList 를 반환하고 검색 통계를 .search_stats 로 노출한다.
    # (튜플 언패킹을 쓰지 않는 이유: check.py 가 반환값을 그대로 순회하므로 list 호환을 유지.)
    search_stats = getattr(tools, "search_stats", {"search_calls": 0, "cache_hits": 0})
    agent = _build_agent(tools, model=model)

    t0 = time.perf_counter()
    error = None
    n_steps = 0
    stopped_max_iter = False
    try:
        usage_cb = UsageMetadataCallbackHandler()
        result = agent.invoke(
            {"input": input_text if input_text is not None else item.question},
            config={"callbacks": [usage_cb]},
        )
        raw_output = result.get("output", "") or ""
        raw_steps = result.get("intermediate_steps", [])
        n_steps = len(raw_steps)
        steps = _serialize_steps(raw_steps)
        in_tok, out_tok, tot_tok = _sum_usage(usage_cb.usage_metadata)
        # max_iterations 도달 판정: (a) 종료 문구 또는 (b) 스텝 수가 상한 이상.
        stopped_max_iter = (
            "stopped due to max iterations" in raw_output
            or n_steps >= MAX_ITERATIONS
        )
    except Exception as e:  # noqa: BLE001
        raw_output, steps = "", []
        in_tok = out_tok = tot_tok = 0
        error = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    predicted = extract_final(raw_output)
    correct = grade(predicted or "", item.gold_answer, rel_tol=NUMERIC_REL_TOL)

    return {
        "qid": item.qid,
        "source": item.source,
        "lang": item.lang,
        "task_type": item.task_type,
        "framework": framework,
        "verifier": verifier,
        "run_index": run_index,
        "model": model,
        # 입력/출력
        "query": item.question,
        "gold_answer": item.gold_answer,
        "answer_raw": raw_output,
        "predicted": predicted,
        "correct": correct,
        # 추론·근거
        "reasoning_trace": steps,
        "retrieved_evidence": evidence_log,
        "gold_evidence_chunks": item.gold_evidence_chunks,
        # 계측(보강): max_iterations 도달 여부·스텝 수·검색/캐시 통계
        "stopped_max_iter": stopped_max_iter,
        "n_steps": n_steps,
        "search_calls": search_stats.get("search_calls", 0),
        "cache_hits": search_stats.get("cache_hits", 0),
        # 비용·지연
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": tot_tok,
        "time_sec": round(elapsed, 4),
        "error": error,
    }


# V1 재시도용 입력 프롬프트 — 원래 질문 + 검증자 critique + 재작성 지시.
RETRY_INPUT_TEMPLATE = (
    "{query}\n\n"
    "[이전 답변에 대한 검증자 피드백]\n"
    "{critique}\n\n"
    "위 피드백을 반영해 처음부터 다시 답변하세요. "
    "마지막 줄은 반드시 '최종답: <값>' 형식으로 끝내야 합니다."
)


def run_one_retry(
    item: QAItem,
    retriever,
    critique: str,
    *,
    run_index: int,
    model: str = EXECUTOR_MODEL,
    framework: str = "langchain",
    verifier: str = "V1",
) -> dict:
    """V1 FAIL 후 재시도 1회 실행.

    입력 문자열만 RETRY_INPUT_TEMPLATE 로 다르고, 그 외 동작(새 AgentExecutor·
    새 도구·새 evidence_log·새 캐시)은 run_one 과 완전히 동일하다.
    """
    retry_input = RETRY_INPUT_TEMPLATE.format(query=item.question, critique=critique or "")
    return run_one(
        item, retriever,
        run_index=run_index, model=model,
        framework=framework, verifier=verifier,
        input_text=retry_input,
    )
