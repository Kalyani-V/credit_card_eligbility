# Credit Card Eligibility Assessment — System Architecture

**Stack:** CrewAI (multi-agent orchestration) · Google Gemini (`google-genai` + LiteLLM) · FAISS (vector search) · Hybrid RAG (keyword + vector + LLM rerank)

This document describes the architecture implemented in `credit_card_eligibility_assessment.ipynb`: a multi-agent underwriting pipeline that decides whether an applicant is eligible for a credit card, grounded in a versioned policy corpus and backed by deterministic guardrails.

---

## 1. System Diagram

```
                                 ┌───────────────────────────-┐
                                 │       Applicant Input      │
                                 └──────────────┬─────────────┘
                                                │
                                                ▼
                                 ┌─────────────────────────-─-─┐
                                 │   Deterministic Router      │
                                 │    quick_prescreen()        │   (no LLM call — rules only)
                                 └───────┬──────────-┬─────────┘
                          AUTO_DECLINE   │           │  BORDERLINE / FAST_TRACK_APPROVE
                                         ▼           ▼
                                 ┌──────────┐  ┌──────────────────────────────────---──┐
                                 │ Decline  │  │           CrewAI Crew                 │
                                 │ (no LLM) │  │  (Process.sequential, memory=True)    │
                                 └──────────┘  │                                       │
                                               │  1. Intake Specialist                 │
                                               │        │ context handoff              │
                                               │        ▼                              │
                                               │  2. Policy Research Analyst           │
                                               │        │  (uses policy_retriever /    │
                                               │        │   hybrid RAG tool)           │
                                               │        ▼                              │
                                               │  3. Senior Underwriter (ReAct)        │
                                               │        │  (uses credit_bureau_lookup, │
                                               │        │   income_verification,       │
                                               │        │   dti_calculator)            │
                                               │        ▼                              │
                                               │  4. Independent Risk Reviewer         │
                                               │        │  (self-critique of t3)       │
                                               │        ▼                              │
                                               │  5. Compliance Officer                │
                                               │        │  → final decision JSON       │
                                               └────────┼─────────────────────────────-┘
                                                        ▼
                                               ┌───────────────────────────────--─┐
                                               │   Guardrail Validator            │
                                               │   eligibility_output_guardrail() │  (deterministic,
                                               │   - valid JSON / required keys   │   non-LLM control
                                               │   - allowed decision enum        │   boundary)
                                               │   - policy citation present      │
                                               │   - fair-lending term screen     │
                                               └───────────────┬──────────────────┘
                                                               ▼
                                               ┌────────────────────────────────--┐
                                               │  Reflection Loop (letter agent)  │
                                               │  Draft → Self-Critique → Revise  │
                                               │  (bounded by max_iterations)     │
                                               └───────────────┬──────────────────┘
                                                               ▼
                                               ┌──────────────────────────────--──┐
                                               │   Applicant-Facing Letter        │
                                               │   (Reg B compliant, decision +   │
                                               │    principal reasons)            │
                                               └─────────────────────────────--───┘

  Cross-cutting:  Batch/parallel execution (process_batch / generate_letters_for_all_applicants)
                  Evaluation harness (run_evaluation_harness) checks router accuracy offline
```

**RAG sub-pipeline** (used by the Policy Research Analyst and Senior Underwriter):

```
Query ──► Keyword Search ──┐
                            ├──► Candidate Set ──► drop_superseded() ──► LLM Rerank ──► Relevance Gate ──► GroundedAnswer + Citation
Query ──► FAISS Vector Search ──┘                                                            │
                                                                                    (below bar → "not_covered", abstain)
```

---

## 2. Component Descriptions

| Component | File location | Responsibility |
|---|---|---|
| **Runtime foundation** | §Foundation cell | Loads `GEMINI_API_KEY`, constructs the `genai.Client`, two CrewAI `LLM` instances (`llm`, `flash_llm`), defines `_call()` (429-retry wrapper), `generate_json()` (Pydantic-constrained structured output), and `embed()` (L2-normalized embeddings). All other components depend on this layer. |
| **Policy corpus & FAISS index** | §1 RAG | A small in-memory list of policy passages (`POLICY_CORPUS`), each with `doc_id`, `section`, `revision`, `effective` date, and `text`. Each passage is embedded with `gemini-embedding-001` and indexed in FAISS for cosine-similarity search. Multiple revisions of `POL-003` are included deliberately to exercise version handling. |
| **Semantic search (`search`)** | §RAG Retrieval | Embeds a query and performs FAISS nearest-neighbor lookup against the policy index. |
| **Keyword search (`keyword_search`)** | §Hybrid RAG | Cheap lexical token-overlap scorer that reliably surfaces exact identifiers (e.g., "POL-003") that pure vector search can miss. |
| **Hybrid retrieval + rerank (`hybrid`, `rerank`)** | §Hybrid RAG | Unions keyword and vector candidates, optionally drops superseded policy revisions, then asks the LLM to score each candidate 0–1 for relevance to the question (`generate_json` + `RerankScore`). |
| **Grounding layer (`ground`)** | §RAG Reliability | Applies a relevance-bar gate (`RELEVANCE_BAR = 0.5`); if nothing clears the bar it returns `status="not_covered"` instead of guessing. If a passage governs the question, it returns a structured `GroundedAnswer` with a `Citation` (doc_id, section, revision, exact span). |
| **Applicant tools** | §2 Tool Use | Three `@tool`-decorated functions — `credit_bureau_lookup`, `income_verification`, `dti_calculator` — that return applicant facts from simulated databases or perform a deterministic DTI calculation. These keep verifiable numeric/factual data out of the LLM's hands. |
| **Guardrail validator** | §3 Guardrails | `eligibility_output_guardrail()` is a CrewAI task-output guardrail: parses JSON, checks required keys (`applicant_id, decision, card_tier, reasons, citations, confidence`), validates the decision enum, screens `reasons` text against `PROHIBITED_TERMS` (protected-attribute language), and requires at least one policy citation. Runs deterministically, outside the LLM. |
| **Agents** | §4 Multi-Agent | Five role-specialized `Agent`s built fresh per run by `build_agents()`: Application Intake Specialist, Policy Research Analyst, Senior Underwriter (ReAct planner), Independent Risk Reviewer, Compliance Officer. Each has only the tools/model it needs. |
| **Task chain** | §5 Prompt Chaining | `build_tasks()` wires the agents into a sequential pipeline (Extraction → Policy Research → Underwriting → Critique → Final Decision) using `context=[...]` to hand off prior outputs. |
| **Crew & memory** | §6 Memory + Orchestration | `build_crew()` assembles a `Crew` per applicant with `Process.sequential` and `memory=True`, configured with a Gemini embedder. This gives the crew short-term (Chroma), long-term (SQLite), and entity (Chroma) memory, in addition to the explicit policy RAG. |
| **Router** | §7 Routing | `quick_prescreen()` is a zero-LLM heuristic classifier (`AUTO_DECLINE` / `BORDERLINE` / `FAST_TRACK_APPROVE`) based on watchlist/bankruptcy/delinquency flags, DTI, and credit score. `route_applicant()` short-circuits `AUTO_DECLINE` cases (no crew invocation) and otherwise kicks off the crew asynchronously. |
| **Reflection loop** | §8 Reflection | `reflect_and_revise()` has a dedicated Correspondence Writer agent draft an applicant letter, then critique and revise its own draft (bounded by `max_iterations`) for missing decline reasons, protected-attribute language, and tone. |
| **Evaluation harness** | §9 Evaluation | `run_evaluation_harness()` runs labeled `TEST_CASES` through the router (optionally the full LLM path) and reports routing accuracy and the zero-LLM auto-decline rate — a lightweight regression test. |
| **End-to-end orchestration** | §10 | `generate_letter_for_applicant()` / `generate_letters_for_all_applicants()` chain routing → decision → reflection → letter for one or many applicants, with per-applicant error isolation and support for concurrent execution. |

---

## 4. Patterns-Used Map

| Pattern | Implementation |
|---|---|
| **Prompt chaining** | `build_tasks()` — sequential tasks linked via `context=[...]` handoffs (Extraction → Policy → Underwriting → Critique → Decision). |
| **Routing** | `quick_prescreen()` — non-LLM conditional dispatch to `AUTO_DECLINE` / `BORDERLINE` / `FAST_TRACK_APPROVE`. |
| **Parallelization** | `process_batch()`, `generate_letters_for_all_applicants()` — concurrent per-applicant execution. |
| **Reflection / self-critique** | `t4_critique` task (crew critiques its own preliminary decision); `reflect_and_revise()` (draft → self-critique → revise loop, bounded by `max_iterations`). |
| **Tool use (≥2 tools)** | `policy_retriever` (RAG tool) plus `credit_bureau_lookup`, `income_verification`, `dti_calculator` — 4 `@tool`-decorated functions. |
| **Planning (ReAct)** | Senior Underwriter task (`t3_reason`) explicitly prompted to run Thought → Action → Observation over each policy requirement. |
| **Multi-agent collaboration** | 5 specialized agents (`build_agents()`) chained via `context=[...]` in `build_tasks()`. |
| **Memory management (≥2 types)** | `build_crew()` — `memory=True` + Gemini embedder enables CrewAI short-term (Chroma), long-term (SQLite), and entity (Chroma) memory. |
| **RAG** | Policy corpus → FAISS embeddings → hybrid keyword+vector retrieval → LLM rerank → relevance-gated, cited answers (`ground()`). |
| **Guardrails** | `eligibility_output_guardrail()` — schema validation, decision-enum check, citation requirement, fair-lending term screen. |
| **Evaluation harness** | `run_evaluation_harness()` — labeled test cases scored against router output; reports accuracy and auto-decline rate. |

---

## 5. Model Choice

| Model | Used for | Rationale |
|---|---|---|
| `gemini/gemini-3.5-flash` (CrewAI `LLM`, `temperature=0.3`) | Policy Research Analyst, Senior Underwriter, Independent Risk Reviewer — the agents doing multi-step reasoning, tool orchestration, and critique. | Higher-capability model where reasoning quality and multi-step tool planning matter most; low temperature favors consistent, defensible underwriting output over creativity. |
| `gemini/gemini-3.1-flash-lite` (CrewAI `LLM`, `temperature=0.3`) | Application Intake Specialist, Correspondence Writer, and other lightweight/high-volume tasks. | Cheaper, lower-latency model for tasks that are largely extraction/formatting rather than judgment, keeping per-applicant cost down since these run on every request. |
| `gemini-3.1-flash-lite` (raw `google-genai` client, via `MODEL`) | Direct generation and rerank calls (`generate_json`, hybrid rerank) outside CrewAI. | Same cost/latency rationale, used directly rather than through LiteLLM where CrewAI's agent abstraction isn't needed (e.g., scoring rerank candidates). |
| `gemini-embedding-001` (`EMBED_MODEL`) | Policy corpus embeddings, query embeddings, and CrewAI's memory embedder. | Purpose-built embedding model; used consistently across both the explicit RAG index and CrewAI's internal memory store so retrieval behavior is uniform. |

Two-tier model selection (a stronger model for judgment-bearing steps, a lighter model for extraction/formatting/high-volume steps) is the core cost/quality lever in this design — every applicant pays for the lite model at least twice (intake, letter) but the full-strength model only for the steps that actually decide eligibility.

---

## 6. Secret Handling

- The **Gemini API key** (`GEMINI_API_KEY`) is never hardcoded. It is loaded from:
  - **Google Colab Secrets** via `userdata.get("GEMINI_API_KEY")` (the primary documented path — the notebook instructs users to add it via the Secrets key icon), or
  - A local **environment variable** / **`.env` file** loaded with `load_dotenv()`, for non-Colab execution.
- If no key is found, the notebook raises `RuntimeError` immediately at setup rather than proceeding with a missing/empty key.
- The key is read into a single `api_key` variable and passed directly to `genai.Client(api_key=api_key)` and the two CrewAI `LLM` objects; it is not written to disk, logged, or included in any prompt, tool output, or generated letter.
- Logging is deliberately quieted (`crewai.flow.runtime`, `LiteLLM`, `litellm`, tracing listeners set to `CRITICAL`/suppressed) so verbose framework output doesn't incidentally surface request/response internals — this also reduces the chance of accidental credential leakage into notebook output.

---

## 7. Limitations

- **Default data stores.** `SIMULATED_BUREAU_DB` and `SIMULATED_INCOME_DB` contain a single applicant/few applicants ; there is no real credit-bureau or income-verification integration, which cannot be fixed as the data is highly confidential.
- **Static policy corpus.** `POLICY_CORPUS` is six hardcoded passages held in memory; there is no ingestion pipeline, real document store, or automatic re-embedding on policy updates. Version handling (`drop_superseded`) is demonstrated but only exercised on one duplicated doc_id (`POL-003`). - This can be handled using preloaded table with the rules implementing scd2 logic.
- **Minimal evaluation coverage.** `TEST_CASES` contains a single labeled example; the "evaluation harness" pattern is demonstrated structurally but does not provide statistically meaningful accuracy or regression coverage.- There are many other factors such as ssn match, personal details of the applicant match which contributes to the accuracy of the test cases.
- **Guardrail scope is narrow.** The fair-lending screen is a fixed keyword blocklist (`race, gender, religion, ethnicity, national origin, marital status`) — it catches explicit mentions but not proxy discrimination (e.g., zip-code-based redlining) or subtler disparate-impact language.- For smaller chunk of input, this is the appropriate selection.
