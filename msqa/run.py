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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from msqa.common import ensure_dir, project_path

from .datasets import load_items
from .executor_langchain import run_one, run_one_retry
from .retriever import Retriever, get_retriever
from .verifiers import verify, verify_v2, verify_v3

try:
    from openai import RateLimitError
except Exception:  # noqa: BLE001 — openai 미존재/구버전 시 문자열 매칭으로 폴백
    RateLimitError = None


def _is_rate_limit(obj) -> bool:
    """예외/에러 메시지에 레이트리밋 신호가 있으면 True."""
    s = str(obj)
    return "Rate limit" in s or "429" in s or "RateLimitError" in s


def _is_context_overflow(obj) -> bool:
    """예외/에러 메시지가 '검증자 입력 과대'(영구적으로 처리 불가)면 True.

    두 가지를 같은 부류로 본다(둘 다 exec=15 누적 evidence 가 너무 커서 생김):
    - 400 BadRequestError context_length_exceeded: 'context_length_exceeded' /
      'maximum context length' (128k 컨텍스트 초과).
    - 429 'Request too large': 단일 요청이 org 의 요청당 토큰 한도를 초과(예: 300k>200k).
      'Request too large' / 'must be reduced'. 재시도해도 같은 크기라 절대 성공 못 하므로
      transient 레이트리밋과 달리 재시도 대상이 아니다.

    그 외 400/429/예외는 오버플로가 아니므로 상위로 전파해(감추지 않음) 데이터 오염을 막는다.
    transient TPM 429('Rate limit reached ... Please try again in Xs')는 여기서 False 가
    되어 기존 60초 백오프 재시도 경로를 탄다.
    """
    if obj is None:
        return False
    s = str(obj)
    return (
        "context_length_exceeded" in s
        or "maximum context length" in s
        or "Request too large" in s
        or "must be reduced" in s
    )


def _failure_record(it, run_index, model, framework, verifier, err) -> dict:
    """레이트리밋 재시도 소진 시 기록할 정상 실패 레코드(run_one 스키마 호환)."""
    return {
        "qid": it.qid, "source": it.source, "lang": it.lang,
        "task_type": it.task_type, "framework": framework, "verifier": verifier,
        "run_index": run_index, "model": model,
        "query": it.question, "gold_answer": it.gold_answer,
        "answer_raw": "", "predicted": None, "correct": False,
        "reasoning_trace": [], "retrieved_evidence": [],
        "gold_evidence_chunks": it.gold_evidence_chunks,
        "stopped_max_iter": False, "n_steps": 0,
        "search_calls": 0, "cache_hits": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "time_sec": 0.0, "error": f"{type(err).__name__}: {err}",
    }


def _run_one_with_retry(it, retr, *, run_index, model, framework, verifier) -> dict:
    """run_one 호출. 429(RateLimit) 면 60초 대기 후 최대 2회 재시도.

    run_one 은 내부에서 예외를 삼키고 error 필드로 반환할 수 있으므로
    (a) 발생 예외와 (b) 반환 레코드의 error 문자열을 모두 검사한다.
    재시도를 모두 소진하면 정상 실패 레코드를 반환하고 호출부는 다음 문항으로 진행.
    """
    attempt = 0
    while True:
        try:
            rec = run_one(
                it, retr,
                run_index=run_index, model=model,
                framework=framework, verifier=verifier,
            )
        except Exception as e:  # noqa: BLE001
            is_rl = (RateLimitError is not None and isinstance(e, RateLimitError)) \
                or _is_rate_limit(e)
            if is_rl and attempt < 2:
                attempt += 1
                print(f"  RateLimit hit -> sleeping 60s (retry {attempt}/2)")
                time.sleep(60)
                continue
            if is_rl:
                return _failure_record(it, run_index, model, framework, verifier, e)
            raise
        if rec.get("error") and _is_rate_limit(rec["error"]) and attempt < 2:
            attempt += 1
            print(f"  RateLimit hit -> sleeping 60s (retry {attempt}/2)")
            time.sleep(60)
            continue
        return rec


def _retry_one_with_retry(it, retr, critique, *, run_index, model, framework, verifier) -> dict:
    """run_one_retry 호출에 _run_one_with_retry 와 동일한 429 백오프를 적용."""
    attempt = 0
    while True:
        try:
            rec = run_one_retry(
                it, retr, critique,
                run_index=run_index, model=model,
                framework=framework, verifier=verifier,
            )
        except Exception as e:  # noqa: BLE001
            is_rl = (RateLimitError is not None and isinstance(e, RateLimitError)) \
                or _is_rate_limit(e)
            if is_rl and attempt < 2:
                attempt += 1
                print(f"  RateLimit hit (retry) -> sleeping 60s (retry {attempt}/2)")
                time.sleep(60)
                continue
            if is_rl:
                return _failure_record(it, run_index, model, framework, verifier, e)
            raise
        if rec.get("error") and _is_rate_limit(rec["error"]) and attempt < 2:
            attempt += 1
            print(f"  RateLimit hit (retry) -> sleeping 60s (retry {attempt}/2)")
            time.sleep(60)
            continue
        return rec


def _verify_with_retry(query, v0_record, *, model, verify_fn=verify) -> dict:
    """검증(verify/verify_v2) 호출에 429 백오프 + 컨텍스트 오버플로(400) 처리를 적용.

    검증 단계에서 컨텍스트 오버플로(128k 초과)가 나면 재시도 없이 검증 실패 vres 를
    반환한다(verdict="ERROR_CONTEXT_OVERFLOW", 토큰 0). 호출부(run_one_verified)는
    이를 보고 v0 답을 최종 채택하고 error="context_overflow_in_verify" 로 기록한다.
    verifier 토큰은 분리 기록한다.
    """
    attempt = 0
    t0 = time.perf_counter()
    while True:
        try:
            return verify_fn(query, v0_record, model=model)
        except Exception as e:  # noqa: BLE001
            # 검증자 입력 과대(영구)는 재시도가 무의미 → 즉시 실패 vres 로 기록.
            if _is_context_overflow(e):
                return {
                    "verdict": "ERROR_CONTEXT_OVERFLOW",
                    "critique": "",
                    "raw": "",
                    "tokens_in": 0, "tokens_out": 0, "tokens_total": 0,
                    "time_sec": round(time.perf_counter() - t0, 4),
                    "_context_overflow": True,
                }
            is_rl = (RateLimitError is not None and isinstance(e, RateLimitError)) \
                or _is_rate_limit(e)
            if is_rl and attempt < 2:
                attempt += 1
                print(f"  RateLimit hit (verify) -> sleeping 60s (retry {attempt}/2)")
                time.sleep(60)
                continue
            raise


def run_one_verified(it, retr, *, run_index, model, framework, verifier) -> dict:
    """V1/V2 공통 파이프라인: V0 실행 → 검증 → (FAIL 시) 재시도 1회 → 최종 채택.

    verifier="V1" 이면 LLM 검증자(verify), "V2" 면 외부 계산기 검증자(verify_v2)를
    사용한다. 호출 횟수: 검증 1회 + (FAIL 시) Executor 재시도 1회. 다회 재시도 금지.
    레코드는 v0_* / verifier_* / retry_* 로 단계별 계측을 모두 보존하고,
    최종 predicted/correct/total_tokens/time_sec 는 채택 답 기준으로 채운다.
    V2 는 v2_recalculated / v2_skip_reason 두 필드를 추가로 기록한다.
    """
    verify_fn = {"V1": verify, "V2": verify_v2, "V3": verify_v3}[verifier]

    # 1) 첫 시도(V0 경로 그대로).
    v0 = _run_one_with_retry(
        it, retr,
        run_index=run_index, model=model,
        framework=framework, verifier=verifier,
    )

    # 2) 검증 1회.
    vres = _verify_with_retry(it.question, v0, model=model, verify_fn=verify_fn)
    verdict = vres["verdict"]
    critique = vres["critique"]
    retried = (verdict == "FAIL")

    # 3) FAIL 이면 재시도 1회.
    retry = None
    if retried:
        retry = _retry_one_with_retry(
            it, retr, critique,
            run_index=run_index, model=model,
            framework=framework, verifier=verifier,
        )

    # 컨텍스트 오버플로(400) 판정: 검증 단계(vres) / 재시도 Executor(retry.error).
    verify_overflow = bool(vres.get("_context_overflow"))
    retry_overflow = bool(
        retried and retry is not None and _is_context_overflow(retry.get("error"))
    )
    # 재시도가 오버플로로 실패하면 그 레코드를 채택하지 않고 v0 로 폴백(retry_* 는 None/0).
    retry_eff = retry if (retry is not None and not retry_overflow) else None

    # 최종 채택: 정상 재시도면 재시도 결과, 그 외(미재시도·검증오버플로·재시도오버플로)는 v0.
    if retried and retry_eff is not None:
        final_predicted = retry_eff["predicted"]
        final_correct = retry_eff["correct"]
    else:
        final_predicted = v0["predicted"]
        final_correct = v0["correct"]

    retry_total = retry_eff["total_tokens"] if retry_eff else 0
    total_tokens = v0["total_tokens"] + vres["tokens_total"] + retry_total
    time_sec = round(
        v0["time_sec"] + vres["time_sec"] + (retry_eff["time_sec"] if retry_eff else 0.0), 4
    )

    context_error = None
    if verify_overflow:
        context_error = "context_overflow_in_verify"
    elif retry_overflow:
        context_error = "context_overflow_in_retry"

    rec = {
        # --- 식별/기본 (기존 필드 보존) ---
        "qid": it.qid, "source": it.source, "lang": it.lang,
        "task_type": it.task_type, "framework": framework,
        "verifier": verifier, "run_index": run_index, "model": model,
        "query": it.question, "gold_answer": it.gold_answer,
        "gold_evidence_chunks": it.gold_evidence_chunks,
        # --- 최종 채택 ---
        "predicted": final_predicted,
        "correct": final_correct,
        "total_tokens": total_tokens,
        "time_sec": time_sec,
        # --- 첫 시도(V0) ---
        "v0_predicted": v0["predicted"],
        "v0_correct": v0["correct"],
        "v0_answer_raw": v0["answer_raw"],
        "v0_tokens_in": v0["prompt_tokens"],
        "v0_tokens_out": v0["completion_tokens"],
        "v0_tokens_total": v0["total_tokens"],
        "v0_time_sec": v0["time_sec"],
        "v0_reasoning_trace": v0["reasoning_trace"],
        "v0_retrieved_evidence": v0["retrieved_evidence"],
        "v0_stopped_max_iter": v0["stopped_max_iter"],
        "v0_n_steps": v0["n_steps"],
        "v0_search_calls": v0["search_calls"],
        "v0_cache_hits": v0["cache_hits"],
        "v0_error": v0["error"],
        # --- 검증자 ---
        "verifier_verdict": verdict,
        "verifier_critique": critique if retried else "",
        "verifier_tokens_in": vres["tokens_in"],
        "verifier_tokens_out": vres["tokens_out"],
        "verifier_tokens_total": vres["tokens_total"],
        "verifier_time_sec": vres["time_sec"],
        # --- 재시도 ---
        "retried": retried,
        "retry_predicted": retry_eff["predicted"] if retry_eff else None,
        "retry_correct": retry_eff["correct"] if retry_eff else None,
        "retry_tokens_in": retry_eff["prompt_tokens"] if retry_eff else 0,
        "retry_tokens_out": retry_eff["completion_tokens"] if retry_eff else 0,
        "retry_tokens_total": retry_eff["total_tokens"] if retry_eff else 0,
        "retry_time_sec": retry_eff["time_sec"] if retry_eff else 0.0,
        "retry_reasoning_trace": retry_eff["reasoning_trace"] if retry_eff else None,
        "retry_retrieved_evidence": retry_eff["retrieved_evidence"] if retry_eff else None,
        "retry_stopped_max_iter": retry_eff["stopped_max_iter"] if retry_eff else None,
        "retry_n_steps": retry_eff["n_steps"] if retry_eff else None,
        "retry_search_calls": retry_eff["search_calls"] if retry_eff else 0,
        "retry_cache_hits": retry_eff["cache_hits"] if retry_eff else 0,
        "retry_error": retry_eff["error"] if retry_eff else None,
        "error": context_error,
    }

    # V2 전용 필드: 재계산 값(FAIL 시만) + PASS 사유.
    if verifier == "V2":
        rec["v2_recalculated"] = vres.get("recalculated")
        rec["v2_skip_reason"] = vres.get("skip_reason")

    # V3 전용 필드: 어느 검증자가 FAIL 을 냈는지 + 각 단계 verdict + V2 부산물.
    if verifier == "V3":
        rec["verifier_used"] = vres.get("verifier_used")
        rec["v1_verdict"] = vres.get("v1_verdict")
        rec["v2_verdict"] = vres.get("v2_verdict")
        rec["v2_skip_reason"] = vres.get("v2_skip_reason")
        rec["v2_recalculated"] = vres.get("v2_recalculated")

    return rec


def run_one_v1(it, retr, *, run_index, model, framework) -> dict:
    """V1 진입점(하위호환 별칭) — run_one_verified(verifier='V1')."""
    return run_one_verified(
        it, retr, run_index=run_index, model=model,
        framework=framework, verifier="V1",
    )


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


def _run_one_task(it, retr, *, run_index, model, framework, verifier) -> dict:
    """(qid, run_index) 1건을 처리해 결과 레코드를 반환.

    400 컨텍스트 오버플로 처리는 run_one_verified(V1/V2/V3) 및 run_one 내부
    try/except(V0) 가 담당하므로 여기서는 단순 디스패치만 한다. 워커 스레드에서
    호출되며, run_one 이 매 호출마다 자기만의 tools/evidence_log/AgentExecutor 를
    만들기 때문에 스레드 간 상태 공유가 없다(자연 격리).
    """
    if verifier in ("V1", "V2", "V3"):
        return run_one_verified(
            it, retr, run_index=run_index, model=model,
            framework=framework, verifier=verifier,
        )
    return _run_one_with_retry(
        it, retr, run_index=run_index, model=model,
        framework=framework, verifier=verifier,
    )


def _format_progress(rec: dict, idx: int, n_total: int, verifier: str) -> str:
    """진행 1줄 출력 문자열(순차/병렬 공통)."""
    if verifier in ("V1", "V2", "V3"):
        final_maxit = (
            rec["retry_stopped_max_iter"] if rec["retried"]
            else rec["v0_stopped_max_iter"]
        )
        ov_err = rec.get("error")
        if ov_err == "context_overflow_in_verify":
            status = "OVERFLOW"
        elif ov_err == "context_overflow_in_retry":
            status = "RETRY_OVERFLOW"
        else:
            status = "OK" if rec["correct"] else ("MAXIT" if final_maxit else "x")
        extra = ""
        if verifier == "V2":
            extra = f" skip={rec.get('v2_skip_reason')} recalc={rec.get('v2_recalculated')}"
        elif verifier == "V3":
            extra = f" used={rec.get('verifier_used')} v2={rec.get('v2_verdict')} v1={rec.get('v1_verdict')}"
        return (
            f"  [{idx}/{n_total}] {status} {rec['qid']} run{rec['run_index']} "
            f"tok={rec['total_tokens']} t={rec['time_sec']}s "
            f"verdict={rec['verifier_verdict']} "
            f"retried={'Y' if rec['retried'] else 'N'}{extra} "
            f"pred={rec['predicted']!r}"
        )
    status = "OK " if rec["correct"] else ("ERR" if rec["error"] else "x  ")
    maxit = " MAXIT" if rec.get("stopped_max_iter") else ""
    return (
        f"  [{idx}/{n_total}] {status} {rec['qid']} run{rec['run_index']} "
        f"tok={rec['total_tokens']} t={rec['time_sec']}s "
        f"steps={rec.get('n_steps')}{maxit} pred={rec['predicted']!r}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["kdart", "finqa", "all"])
    ap.add_argument("--repeats", type=int, default=3, help="문항당 반복 횟수(실행 일관성용)")
    ap.add_argument("--limit", type=int, default=0, help="문항 수 제한(0=전체)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--framework", default="langchain")
    ap.add_argument("--verifier", default="V0", choices=["V0", "V1", "V2", "V3"])
    ap.add_argument("--top-k", type=int, default=None,
                    help="검색기 top_k 오버라이드(기본=config 의 5). 인덱스에서 가져오는 개수.")
    ap.add_argument("--executor-max", type=int, default=None,
                    help="Executor 에게 노출할 청크 수(기본=top_k 동일). 컨텍스트 캡.")
    ap.add_argument("--chunk-char-cap", type=int, default=None,
                    help="각 청크 본문 글자수 캡(기본=자르지 않음). Executor 출력 텍스트에만 적용.")
    ap.add_argument("--parallel", type=int, default=1,
                    help="문항 단위 동시 실행 워커 수(기본 1=순차). Tier1 안전선상 2 권장, 3 이상 금지.")
    ap.add_argument("--out", default=None,
                    help="기본값: V0/V1/V2/V3 -> results/langchain_v{0,1,2,3}.jsonl")
    args = ap.parse_args()

    if args.out is None:
        base = {
            "V0": "results/langchain_v0.jsonl",
            "V1": "results/langchain_v1.jsonl",
            "V2": "results/langchain_v2.jsonl",
            "V3": "results/langchain_v3.jsonl",
        }[args.verifier]
        # 이어하기 충돌 방지를 위해 비기본 설정은 파일명 접미사로 구분한다.
        suffix = ""
        if args.top_k is not None and args.top_k != 5:
            suffix += f"_topk{args.top_k}"
        if args.executor_max is not None and args.executor_max != (args.top_k or 5):
            suffix += f"_exec{args.executor_max}"
        if args.chunk_char_cap is not None:
            suffix += f"_cap{args.chunk_char_cap}"
        args.out = base.replace(".jsonl", f"{suffix}.jsonl")

    items = load_items(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]

    out_path = project_path(args.out)
    ensure_dir(out_path.parent)
    done = _done_keys(out_path)

    # K-DART 전역 검색기는 1회만 연다(FinQA 는 문항별로 새로 만든다).
    kdart_retriever = None
    if any(it.source == "kdart" for it in items):
        kdart_retriever = Retriever(top_k_override=args.top_k, executor_max=args.executor_max)
        kdart_retriever.chunk_char_cap = args.chunk_char_cap
        print(f"[index] ChromaDB chunks = {kdart_retriever.n_chunks:,}  "
              f"search_top_k={kdart_retriever.search_top_k}  executor_max={kdart_retriever.executor_max}")

    total = len(items) * args.repeats
    print(f"[plan] items={len(items)} × repeats={args.repeats} = {total} runs "
          f"(framework={args.framework}, verifier={args.verifier}, model={args.model}, "
          f"top_k={args.top_k or 'config(5)'}, executor_max={args.executor_max or 'top_k'}, "
          f"chunk_char_cap={args.chunk_char_cap or 'none'})")

    # FinQA 검색기의 공유 임베더를 메인 스레드에서 미리 초기화(워커 간 lazy-init 경쟁 제거).
    if any(it.source == "finqa" for it in items):
        from .retriever import FinqaRetriever
        if FinqaRetriever._embedder is None:
            FinqaRetriever([], top_k_override=args.top_k, executor_max=args.executor_max)

    workers = max(1, args.parallel)

    def process_one(it, run_index) -> dict:
        """워커에서 실행: 검색기 준비 → 1건 처리 → (워커 페이싱) sleep(2.0)."""
        retr = get_retriever(it, kdart_retriever, top_k_override=args.top_k,
                             executor_max=args.executor_max)
        # chunk_char_cap 을 검색기에 실어 make_tools(executor 내부 호출)가 읽게 한다.
        retr.chunk_char_cap = args.chunk_char_cap
        rec = _run_one_task(
            it, retr, run_index=run_index, model=args.model,
            framework=args.framework, verifier=args.verifier,
        )
        # 워커 내 다음 문항 시작 전 레이트리밋 완화(직렬 때의 문항 사이 sleep 과 동일 의도).
        time.sleep(2.0)
        return rec

    # 이어하기: 아직 안 된 (qid, run_index) 만 작업 큐로.
    tasks = [
        (it, ri)
        for it in items
        for ri in range(1, args.repeats + 1)
        if (it.qid, ri) not in done
    ]
    n_total = len(tasks)

    n_done = n_ok = 0
    n_overflow = 0           # 컨텍스트 오버플로(검증/재시도) 누적 — 5건이면 중단
    file_lock = Lock()       # JSONL 동시 write 안전성
    fout = out_path.open("a", encoding="utf-8")
    t_start = time.perf_counter()
    print(f"[exec] parallel workers={workers}, 신규 작업 {n_total}건 (skip {len(done)}건)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, (it, ri) in enumerate(tasks):
            # 첫 workers 개는 1초 간격으로 stagger 하여 동시 spike(429) 방지.
            if workers > 1 and 0 < i < workers:
                time.sleep(1.0)
            fut = pool.submit(process_one, it, ri)
            futures[fut] = (it.qid, ri)

        stopped = False
        for fut in as_completed(futures):
            qid, ri = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 — 예상 못한 예외는 감추지 않고 표시
                print(f"  [{qid} run{ri}] CRASH: {type(e).__name__}: {e}")
                continue
            with file_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            n_done += 1
            n_ok += int(bool(rec["correct"]))
            print(_format_progress(rec, n_done, n_total, args.verifier))
            if str(rec.get("error") or "").startswith("context_overflow"):
                n_overflow += 1
                if n_overflow >= 5 and not stopped:
                    print(f"[STOP] context_overflow {n_overflow}건 발생 — 중단. "
                          f"그때까지 기록된 {n_done}개 레코드는 {out_path} 에 보존됨.")
                    stopped = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    fout.close()

    elapsed = time.perf_counter() - t_start
    if n_done:
        print(f"[done] {n_done} new runs, TSR(new)={n_ok}/{n_done}={n_ok/n_done:.3f}, "
              f"elapsed={elapsed:.1f}s -> {out_path}")
    else:
        print(f"[done] 새로 실행할 항목 없음(모두 완료됨) -> {out_path}")


if __name__ == "__main__":
    main()
