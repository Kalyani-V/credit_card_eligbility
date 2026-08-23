# Credit Card Eligibility Assessment

An end-to-end **multi-agent** credit-card underwriting system built on
[CrewAI](https://github.com/crewAIInc/crewAI), **Google Gemini** (via the
`google-genai` client and CrewAI's LiteLLM-backed `LLM`), and a **hybrid
keyword + FAISS vector search + LLM rerank** retrieval pipeline over a
versioned credit-policy corpus.

The system takes a raw applicant submission and produces a policy-grounded
`APPROVE` / `DECLINE` / `MANUAL_REVIEW` decision plus an applicant-facing
adverse-action / approval letter — while keeping cost down by skipping the
LLM entirely for clear-cut cases.

---

## System Architecture

```

High-level control flow for a single applicant, then the batch/eval paths.

                         ┌──────────────────────────┐
                         │      Applicant Input     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────--┐
                         │   Deterministic Router     │
                         │    quick_prescreen()       │
                         └───────┬──────--┬───────────┘
                                 │        │
                    AUTO_DECLINE │        │ BORDERLINE / FAST TRACK
                                 │        │
                                 ▼        ▼
                         ┌──────────┐  ┌─────────────────────────┐
                         │ Decline  │  │      CrewAI Crew        │
                         │ No LLM   │  │                         │
                         └──────────┘  │  1. Intake Specialist   │
                                       │          ↓              │
                                       │  2. Policy Research     │
                                       │          ↓              │
                                       │  3. Senior Underwriter  │
                                       │          ↓              │
                                       │  4. Risk Reviewer       │
                                       │          ↓              │
                                       │  5. Compliance Officer  │
                                       └────────────┬────────────┘
                                                    │
                                  ┌─────────────────┼─────────────────┐
                                  │                 │                 │
                                  ▼                 ▼                 ▼
                         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                         │ Bureau Tool  │  │ Income Tool  │  │ DTI Tool     │
                         └──────────────┘  └──────────────┘  └──────────────┘
                                  │                 │                 │
                                  └─────────────────┼─────────────────┘
                                                    ▼
                                      ┌────────────────────────┐
                                      │       Policy RAG       │
                                      │                        │
                                      │ FAISS Vector Search    │
                                      │ + Keyword Search       │
                                      │ + LLM Reranking        │
                                      │ + Revision Filtering   │
                                      └────────────┬───────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────┐
                                      │ Grounded Decision +    │
                                      │ Policy Citations       │
                                      └────────────┬───────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────┐
                                      │ Guardrail Validation   │
                                      │ + Human Review         │
                                      └────────────┬───────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────┐
                                      │ Final Decision JSON    │
                                      └────────────┬───────────┘
                                                   │
                                                   ▼
                                      ┌────────────────────────┐
                                      │ Reflection / Revision  │
                                      │ Applicant Letter       │
                                      └────────────────────────┘
```

Supporting layers used throughout the pipeline:

| Layer | Implementation |
|---|---|
| **Policy RAG** | Keyword search ∪ FAISS vector search → LLM rerank → revision filter → grounded answer with citation or abstention |
| **Applicant tools** | Credit bureau lookup, income verification, deterministic DTI calculator |
| **Memory** | CrewAI short-term, long-term (SQLite), and entity memory (Chroma) |

---

## Agentic Design Patterns Implemented

| Pattern | Where it shows up |
|---|---|
| Prompt chaining & routing | Sequential task `context=[...]` handoffs; `quick_prescreen()` short-circuits `AUTO_DECLINE` cases before any LLM call |
| Reflection / self-critique | `t4_critique` task in the crew; `reflect_and_revise()` loop for applicant letters |
| Tool use (4 tools) | `policy_retriever`, `credit_bureau_lookup`, `income_verification`, `dti_calculator` |
| Multi-agent planning | 5 specialized agents; the Senior Underwriter uses explicit ReAct (Thought → Action → Observation) prompting |
| Memory (3 types) | CrewAI short-term, long-term (SQLite), and entity (Chroma) memory, backed by a Gemini embedder |
| Hybrid RAG + citations | Keyword ∪ vector search → LLM rerank → `ground()` with mandatory citations or abstention |
| Guardrails & evaluation | Schema + fair-lending guardrail on the final decision; labeled router-accuracy test harness |

---

## Agents

| Agent | Responsibility | Tools |
|---|---|---|
| **Application Intake Specialist** | Parses raw submissions into structured fields | None (flash LLM) |
| **Policy Research Analyst** | Grounds every rule in the corpus; always cites `doc_id` | `policy_retriever` |
| **Senior Underwriter (ReAct)** | Thought → Action → Observation for each requirement | All 4 tools |
| **Independent Risk Reviewer** | Critiques math, citations, fair-lending risk, flags borderline cases | `policy_retriever` |
| **Compliance Officer** | Emits the final JSON decision; escalates borderline cases via `human_input` | Guardrail only |

Agents are rebuilt fresh (`build_agents()`) for every crew run rather than
reused as module-level singletons, so concurrent crews never share an
executor instance.

---

## Policy Corpus

A small versioned JSON corpus (`POLICY_CORPUS`) with fields `doc_id`,
`section`, `revision`, `effective`, and `text`. Notably, **`POL-003`
(income requirements) has two revisions** — a 2020 version ($25k / $60k
minimums) and a 2025 version ($35k / $80k) — used to test that the system
applies the currently-effective rule and never a superseded one.

Example entries:

```json
{"doc_id": "POL-002", "section": "Minimum Credit Score", "revision": "2020-01",
 "text": "A minimum FICO-equivalent credit score of 650 is required for the
 Standard card. The Platinum card requires a score of 750 or above."}

{"doc_id": "POL-004", "section": "Debt-to-Income Ratio", "revision": "2020-01",
 "text": "Applicants with a DTI above 45% are automatically declined
 regardless of credit score. DTI between 36% and 45% requires manual
 underwriting review."}
```

---

## Hybrid RAG Pipeline

1. **Keyword search** — exact token overlap; rewards precise identifiers like `POL-003`.
2. **Vector search** — FAISS `IndexFlatIP` over L2-normalized `gemini-embedding-001` embeddings.
3. **LLM rerank** — structured 0–1 relevance scores via `generate_json`.
4. **`drop_superseded()`** — per `doc_id`, keeps only the newest revision effective on or before today.
5. **`ground()`** — answers only from retrieved passages, citing a span, or returns `not_covered` below a 0.5 relevance threshold.

---

## Applicant Evidence Tools

| Tool | Signature | Notes |
|---|---|---|
| `credit_bureau_lookup` | `applicant_id → score, delinquencies, bankruptcy, watchlist` | Simulated bureau DB, 5 labeled applicants |
| `income_verification` | `applicant_id → annual_income, monthly_debt, employment` | Feeds the DTI calculator |
| `dti_calculator` | `annual_income, monthly_debt → DTI %` | Deterministic arithmetic — avoids LLM math errors |
| `policy_retriever` | `question → grounded answer + citation (or NOT_COVERED)` | Wraps the hybrid RAG pipeline as a `@tool` |

---

## Deterministic Routing (`quick_prescreen`)

A cheap, non-LLM heuristic runs before any crew is invoked:

| Route | Trigger | Action |
|---|---|---|
| `AUTO_DECLINE` | Watchlist, bankruptcy, delinquency in last 24mo, or DTI > 45% | Returns `DECLINE` immediately — zero LLM calls |
| `BORDERLINE` | DTI 36–45%, or score within 20 pts of the 650/750 cutoff | Full crew + `human_input=True` (per POL-008 manual review) |
| `FAST_TRACK_APPROVE` | Score ≥ 650 and no hard-decline flags | Full crew, no human gate |

---

## Guardrails & Fair-Lending Controls (`eligibility_output_guardrail`)

Runs against the compliance officer's final task output before CrewAI accepts it:

- Valid JSON (rejects prose-only output)
- Required keys present: `applicant_id`, `decision`, `card_tier`, `reasons`, `citations`, `confidence`
- `decision` restricted to `APPROVE | DECLINE | MANUAL_REVIEW`
- At least one policy `doc_id` citation required
- Rejects protected-attribute language: race, gender, religion, ethnicity, national origin, marital status

---

## Reflection Loop & Memory

`reflect_and_revise()` drafts an applicant letter, then has the **same**
agent self-critique it against Regulation B adverse-action requirements
(missing reasons, protected-attribute language, tone) and revise — capped at
`max_iterations=2` to prevent uncontrolled loops.

CrewAI memory is configured with a Gemini embedder and provides three types:

- **Short-term** — recent task context (Chroma)
- **Long-term** — cross-run history (SQLite)
- **Entity** — named entities across tasks (Chroma)

---

## Evaluation Harness

`run_evaluation_harness()` runs 5 labeled `TEST_CASES` against the router and reports:

- **Router accuracy:** 100% (12 labeled cases in the deck's evaluation; 5 in the notebook's `TEST_CASES`)
- **Auto-decline rate:** 60% of cases resolved with zero LLM calls

**Ablation — with vs. without hybrid RAG memory:**

| Metric | With Hybrid RAG + Memory (default) | Without |
|---|---|---|
| Task success | 91.7% | 58.3% |
| Faithfulness | 0.94 | 0.68 |
| Failure mode | — | Hallucinated terms & DTI rules |

---

## End-to-End Workflow

`generate_letters_for_all_applicants()` runs the full pipeline —
**router → crew → guardrail → reflection letter** — concurrently for every
applicant via `asyncio.gather`, and returns each applicant's route, decision,
and generated letter.

---

## Setup

### Requirements

```bash
pip install crewai crewai-tools google-genai faiss-cpu numpy litellm pandas python-dotenv chromadb
```

### API Key

You need a **Gemini API key**.

- **Google Colab:** add it via Secrets (key icon) as `GEMINI_API_KEY`.
- **Local:** set the `GEMINI_API_KEY` environment variable (a `.env` file is supported via `python-dotenv`).

### Running

Open `final_credit_card_eligibility.ipynb` and run all cells top to bottom.
Sections build on each other in this order:

1. Foundation — models, dependencies, runtime config
2. RAG — policy corpus, embeddings, FAISS index
3. Tool use — applicant evidence tools
4. Guardrails — structured decision + fair-lending validation
5. Multi-agent architecture — the 5 agents
6. Prompt chaining — sequential task workflow
7. Memory & crew orchestration
8. Deterministic routing
9. Reflection / self-critique
10. Evaluation harness
11. End-to-end workflow (concurrent, all applicants)

---

## Key Design Notes

- **Fresh agents per crew run** (`build_agents()`) — prevents concurrent
  crews from sharing an executor instance, which otherwise raises
  `"Executor is already running"`.
- **Facts vs. reasoning are separated** — numeric values (score, income, DTI)
  always come from tools, never from the LLM, to eliminate hallucinated numbers.
- **Cost-aware by design** — the deterministic router resolves the majority
  of clear-cut cases (60% in testing) without any LLM invocation.

---

## Takeaways

1. **Architecture over prompts** — routing, specialized agents, tools, and guardrails do more for reliability than a single clever prompt.
2. **Grounded policy decisions** — hybrid RAG with revision filtering and abstention prevents both hallucination and use of superseded rules.
3. **Cost-aware control flow** — 60% of cases resolved with zero LLM calls via deterministic hard-decline/fast-track screens.
4. **Auditability by design** — every final decision carries citations; fair-lending language is blocked at the guardrail boundary.
5. **Production-minded patterns** — reflection loops, multi-type memory, an evaluation harness, and human-in-the-loop for borderline cases.
