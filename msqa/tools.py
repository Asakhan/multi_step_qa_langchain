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


def make_tools(retriever, evidence_log: list[dict]):
    """retriever.search 를 감싼 doc_search + calculator 를 LangChain 도구로 반환."""

    def doc_search(query: str) -> str:
        """재무 공시 문서에서 질의와 관련된 근거 청크를 검색한다.
        숫자·표·항목을 찾을 때 이 도구로 먼저 근거를 확보하라."""
        hits = retriever.search(query)
        for h in hits:
            evidence_log.append({
                "query": query,
                "chunk_id": h.get("chunk_id"),
                "distance": h.get("distance"),
                "text": h.get("text"),
            })
        if not hits:
            return "검색 결과 없음."
        return "\n\n".join(
            f"[{h.get('chunk_id')}] {h.get('text')}" for h in hits
        )

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
    return [doc_search_tool, calc_tool]
