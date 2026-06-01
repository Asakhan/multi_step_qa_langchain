"""LangChain 도구 — 세 프레임워크에 동일 시그니처로 제공할 (1) 문서 검색, (2) 수치 계산.

make_tools(retriever, evidence_log) 는 도구 호출 시 검색 결과를 evidence_log 에 누적하여
실행 로그의 retrieved_evidence 를 구성할 수 있게 한다.
"""
from __future__ import annotations

import ast
import operator as op

from langchain_core.tools import StructuredTool

# ---- 안전한 산술 평가기 (eval 금지, AST 화이트리스트) ----
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("허용되지 않은 수식입니다.")


def calculate(expression: str) -> str:
    """사칙연산·거듭제곱·나머지만 허용하는 계산기. 예: '(5829-5735)/5735*100'."""
    try:
        expr = expression.replace(",", "").replace("×", "*").replace("÷", "/")
        val = _safe_eval(ast.parse(expr, mode="eval"))
        return f"{val}"
    except Exception as e:  # noqa: BLE001
        return f"계산 오류: {e}"


_CACHE_NOTICE = (
    "(이미 동일/유사 질의로 검색함 — 새 근거가 필요하면 다른 키워드로 검색하거나 "
    "calculator 로 계산하세요)"
)


class _ToolList(list):
    """도구 리스트(list) 그대로 동작하되 검색 통계를 .search_stats 로 함께 노출.

    기존 호출부(run.py/check.py)는 list 처럼 순회만 하므로 영향이 없고,
    run_one 은 .search_stats({"search_calls","cache_hits"}) 를 결과에 기록한다.
    """

    search_stats: dict


def make_tools(retriever, evidence_log: list[dict], chunk_char_cap: int | None = None):
    """retriever.search 를 감싼 doc_search + calculator 를 LangChain 도구로 반환.

    반환: _ToolList (list 호환). tool.search_stats 에 {"search_calls","cache_hits"}
    가변 dict 가 실려, doc_search 호출 횟수와 동일 질의 캐시 히트 횟수를 누적한다
    (문항 1건 범위). 동일(정규화) 질의를 다시 검색하면 retriever.search 를
    재호출하지 않고 직전 청크를 안내 문구와 함께 반환하며, evidence_log 에는
    중복 기록하지 않는다.

    chunk_char_cap 이 주어지면 Executor 가 보는 출력 텍스트의 각 청크 본문을 그
    글자수까지 자른다(잘림 표시 부착). 명시하지 않으면 retriever.chunk_char_cap
    에서 가져오므로 호출부(executor) 시그니처 변경 없이도 캡을 흘려보낼 수 있다.
    evidence_log 에 적재되는 text 는 항상 잘리지 않은 원본을 유지한다(사후 분석용).
    """
    if chunk_char_cap is None:
        chunk_char_cap = getattr(retriever, "chunk_char_cap", None)
    # 캐시·통계는 문항별 호출(run_one 1회)마다 새로 만들어지므로 자연히 초기화된다.
    _cache: dict[str, list] = {}
    _result_keys: set[tuple] = set()   # 이미 본 결과집합(top-k chunk_id 튜플)
    _stats = {"search_calls": 0, "cache_hits": 0}

    def _norm(query: str) -> str:
        # strip + 내부 공백 1칸 + 소문자
        return " ".join(str(query).split()).lower()

    def _fmt(h: dict) -> str:
        text = h.get("text", "") or ""
        if chunk_char_cap and len(text) > chunk_char_cap:
            text = text[:chunk_char_cap] + "... [잘림]"
        return f"[{h.get('chunk_id')}] {text}"

    def _format(hits: list) -> str:
        if not hits:
            return "검색 결과 없음."
        return "\n\n".join(_fmt(h) for h in hits)

    def doc_search(query: str) -> str:
        """재무 공시 문서에서 질의와 관련된 근거 청크를 검색한다.
        숫자·표·항목을 찾을 때 이 도구로 먼저 근거를 확보하라."""
        key = _norm(query)
        # 1) 정규화 query 히트: retriever.search 를 건너뛴다 → search_calls 증가 없음.
        if key in _cache:
            _stats["cache_hits"] += 1
            return f"{_CACHE_NOTICE}\n\n{_format(_cache[key])}"

        # 2) 실제 검색 수행(검색이 일어났으므로 search_calls +1).
        _stats["search_calls"] += 1
        hits = retriever.search(query)
        result_key = tuple(h.get("chunk_id") for h in hits)  # top-k 순서 보존

        # 2a) 결과집합을 이미 본 적 있으면 중복: evidence_log 중복 기록 없음.
        if result_key in _result_keys:
            _stats["cache_hits"] += 1
            return f"{_CACHE_NOTICE}\n\n{_format(hits)}"

        # 2b) 정상 첫 등장.
        _cache[key] = hits
        _result_keys.add(result_key)
        for h in hits:
            evidence_log.append({
                "query": query,
                "chunk_id": h.get("chunk_id"),
                "distance": h.get("distance"),
                "text": h.get("text"),
            })
        return _format(hits)

    doc_search_tool = StructuredTool.from_function(
        func=doc_search,
        name="doc_search",
        description=(
            "재무 공시 문서에서 질의와 관련된 근거 텍스트/표 청크를 검색한다. "
            "입력: 검색 질의(query) 문자열. 출력: 상위 청크들의 [chunk_id] 텍스트."
        ),
    )
    calc_tool = StructuredTool.from_function(
        func=calculate,
        name="calculator",
        description=(
            "산술식을 계산한다. 입력: 파이썬 산술식 문자열(expression), "
            "예 '(5829-5735)/5735*100'. 사칙연산·거듭제곱(**)·나머지(%)만 지원."
        ),
    )
    tools = _ToolList([doc_search_tool, calc_tool])
    tools.search_stats = _stats
    return tools
