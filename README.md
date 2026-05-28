# multi_step_qa_langchain

검증 메커니즘이 LLM 에이전트의 정확도·비용에 미치는 영향을 측정하는 실험 저장소.
**1단계: LangChain Agent(A1) × V0(검증 없음) 기준선**을 자체 완결로 구현한다.

- 데이터: K-DART-QA 40문항 + FinQA 20문항 = **60문항**(이미 `data/` 에 포함).
- 검색: ChromaDB + OpenAI `text-embedding-3-small`, top-5 (데이터셋 calibration과 동일).
- Executor: OpenAI `gpt-4o-mini`, 도구 `doc_search`(RAG)·`calculator`.
- 로그: 1실행당 1 JSONL 레코드(query·answer·reasoning_trace·retrieved_evidence·tokens·time·correct).

## 폴더 구조

```
multi_step_qa_langchain/
├─ config.yaml              # rag/executor 설정
├─ requirements.txt
├─ .env.example             # 복사 → .env 에 OPENAI_API_KEY
├─ data/
│  ├─ kdart_qa.jsonl        # 40문항 (포함)
│  ├─ finqa.jsonl           # 20문항 (포함)
│  ├─ index/                # ChromaDB ← 데이터셋 저장소에서 복사 (gitignore)
│  └─ chunks/               # (대안) 청크 코퍼스 ← 복사 후 build_index (gitignore)
└─ msqa/
   ├─ common.py  rag_index.py  grading.py  datasets.py
   ├─ retriever.py  tools.py  executor_langchain.py
   └─ build_index.py  prepare_finqa.py  check.py  run.py
```

---

## Windows + 가상환경(venv) 실행 가이드

아래는 PowerShell 기준. (명령 프롬프트 cmd 는 활성화 줄만 다름)

### 1) 저장소 준비

이미 `https://github.com/Asakhan/multi_step_qa_langchain` 를 만들었다면, 이 폴더 내용을
그 저장소 루트에 넣고 커밋한다.

```powershell
cd C:\work
git clone https://github.com/Asakhan/multi_step_qa_langchain.git
# (이 zip 의 파일들을 multi_step_qa_langchain\ 안에 복사)
cd multi_step_qa_langchain
```

### 2) 가상환경 생성·활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> PowerShell 에서 `Activate.ps1` 실행이 막히면(보안정책) 한 번만:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> cmd 를 쓰면: `.\.venv\Scripts\activate.bat`

활성화되면 프롬프트 앞에 `(.venv)` 가 붙는다.

### 3) 의존성 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) API 키 설정

`.env.example` 를 `.env` 로 복사하고 키를 채운다.

```powershell
copy .env.example .env
notepad .env      # OPENAI_API_KEY=sk-... 저장
```

### 5) RAG 인덱스 가져오기 (둘 중 하나)

K-DART 문서 청크는 용량이 커서 이 저장소에 포함하지 않는다. 데이터셋 저장소(`kdart_qa`)에서
로컬로 만든 결과물을 복사한다.

**(권장·무료) 이미 만든 ChromaDB 인덱스를 그대로 복사** — 재임베딩 없음:

```powershell
xcopy /E /I C:\path\to\kdart_qa\data\index .\data\index
```

**(대안) 청크만 복사 후 재빌드** — 약 $0.03 재임베딩:

```powershell
xcopy /E /I C:\path\to\kdart_qa\data\chunks .\data\chunks
python -m msqa.build_index --reset
```

> FinQA 는 문항이 자체 컨텍스트를 들고 있어 인덱스가 필요 없다(`data\finqa.jsonl` 포함).
> 정확히 동일한 FinQA 20개 id 로 다시 만들려면:
> `python -m msqa.prepare_finqa --finqa-test C:\path\to\FinQA\dataset\test.json --ids id1,id2,...`

### 6) 사전 점검 (토큰 사용 안 함)

```powershell
python -m msqa.check
```

`[1]~[4]` 가 모두 정상이면 "준비 완료 ✅".

### 7) 스모크 테스트 — K-DART 1문항 1회 (몇 센트 미만)

```powershell
python -m msqa.run --dataset kdart --limit 1 --repeats 1
```

`results\langchain_v0.jsonl` 에 레코드 1줄이 생기고, `pred=...` 와 토큰·시간이 출력된다.

### 8) 본 실행 — 60문항 × 3반복 (LangChain × V0)

```powershell
python -m msqa.run --dataset all --repeats 3
```

중단되어도 다시 실행하면 이미 끝난 `(qid, run_index)` 는 건너뛴다(이어하기).
`--dataset kdart` / `--dataset finqa` 로 분리 실행도 가능.

---

## 출력 스키마 (results/*.jsonl, 1줄 = 1실행)

`qid, source, lang, task_type, framework, verifier, run_index, model,
query, gold_answer, answer_raw, predicted, correct,
reasoning_trace[], retrieved_evidence[], gold_evidence_chunks[],
prompt_tokens, completion_tokens, total_tokens, time_sec, error`

논문 3.5절의 전수 로그 요건을 그대로 충족한다.

## 다음 단계

1. **V1/V2/V3 검증 래퍼** (`msqa/verifiers.py`): `run_one()` 출력을 받아 PASS/FAIL·critique
   → 1회 재시도. 공통 인터페이스(논문 3.4.1).
2. **CrewAI(A2)/LangGraph(A3) Executor**: 동일한 `tools.make_tools` 를 묶어 추가,
   `run.py --framework` 로 분기.
3. **분석**: 카이제곱(H1) · 이원 ANOVA on log(TPS)(H2) · 부트스트랩 CI.
