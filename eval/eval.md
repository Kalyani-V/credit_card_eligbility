# Evaluation Report — Credit Card Eligibility Assessment System

## 1. Methodology

The system evaluates at two levels, matching the two control paths in the
architecture (`quick_prescreen()` router vs. full CrewAI crew):

1. **Router-level evaluation (reproducible without an LLM call).**
   `quick_prescreen()` is a pure, deterministic function of bureau + income
   data. We extended the notebook's original 5-case `TEST_CASES` set to a
   **12-case labeled set** (`APP-1007`–`APP-1018`) by adding 7 synthetic
   applicants that exercise every branch of the routing logic (clean
   fast-track approvals, DTI-triggered declines/borderlines, score-near-cutoff
   borderlines, and each of the three hard-decline flags: watchlist,
   bankruptcy, delinquency). Every case's expected route was derived by hand
   from the same rule the router implements, then checked against the
   router's actual output — this is the `run_evaluation_harness()` pattern
   from the notebook, run over a larger set.

2. **Full-pipeline evaluation (crew → RAG → guardrail → decision).**
   This requires live Gemini API calls (`crew.kickoff_async()`), which this
   environment cannot execute without a `GEMINI_API_KEY`. For this layer we
   report the **ablation study already captured in the notebook/deck**
   (Hybrid RAG + Memory vs. a baseline without it), which is the only
   full-pipeline signal the project currently has. The **3 failure examples**
   in §5 are illustrative reconstructions of the specific failure modes the
   ablation names ("hallucinated terms & DTI rules"), built by manually
   tracing what a non-grounded model would plausibly output against the
   corpus's known ground truth (e.g., the superseded `POL-003` revision).
   They are not captured LLM transcripts.

**Metrics used:**
- **Task Success Rate** — for the router: exact match between predicted and
  expected route. For the full pipeline: correct final decision against
  ground truth (as reported in the notebook's ablation).
- **Faithfulness** — for the full pipeline only: whether cited policy
  clauses actually support the stated decision (as reported in the notebook's
  ablation).
- **Auto-decline rate** — share of cases resolved by the router alone, at
  zero LLM cost (an efficiency metric, not a correctness metric).

---

## 2. Test Set (12 cases)

| ID | Credit Score | Monthly DTI | Bureau Flag | Expected Route | Trigger |
|---|---|---|---|---|---|
| APP-1007 | 640 | 48.0% | none | AUTO_DECLINE | DTI > 45% |
| APP-1008 | 760 | 40.0% | none | BORDERLINE | DTI 36–45% |
| APP-1009 | 710 | 14.1% | bankruptcy | AUTO_DECLINE | Hard flag (bankruptcy) |
| APP-1010 | 660 | 24.0% | none | BORDERLINE | Score within 20 pts of 650 |
| APP-1011 | 735 | 15.0% | watchlist | AUTO_DECLINE | Hard flag (watchlist) |
| APP-1012 | 700 | 15.0% | none | FAST_TRACK_APPROVE | Score ≥ 650, no flags |
| APP-1013 | 800 | 16.0% | none | FAST_TRACK_APPROVE | Score ≥ 650, no flags |
| APP-1014 | 700 | 40.0% | none | BORDERLINE | DTI 36–45% |
| APP-1015 | 745 | 18.0% | none | BORDERLINE | Score within 20 pts of 750 |
| APP-1016 | 680 | 25.7% | delinquency | AUTO_DECLINE | Hard flag (delinquency, 24mo) |
| APP-1017 | 600 | 20.0% | none | AUTO_DECLINE | Score < 650, no other trigger |
| APP-1018 | 720 | 60.0% | none | AUTO_DECLINE | DTI > 45% |

Cases APP-1007–APP-1011 are the notebook's original `TEST_CASES`.
APP-1012–APP-1018 were added to cover branches the original 5 cases didn't
exercise (clean fast-track approval, delinquency-triggered decline, and a
score-near-750 borderline).

---

## 3. Results

### 3a. Router-level results (computed live, this evaluation)

| ID | Predicted Route | Expected Route | Correct? |
|---|---|---|---|
| APP-1007 | AUTO_DECLINE | AUTO_DECLINE | ✅ |
| APP-1008 | BORDERLINE | BORDERLINE | ✅ |
| APP-1009 | AUTO_DECLINE | AUTO_DECLINE | ✅ |
| APP-1010 | BORDERLINE | BORDERLINE | ✅ |
| APP-1011 | AUTO_DECLINE | AUTO_DECLINE | ✅ |
| APP-1012 | FAST_TRACK_APPROVE | FAST_TRACK_APPROVE | ✅ |
| APP-1013 | FAST_TRACK_APPROVE | FAST_TRACK_APPROVE | ✅ |
| APP-1014 | BORDERLINE | BORDERLINE | ✅ |
| APP-1015 | BORDERLINE | BORDERLINE | ✅ |
| APP-1016 | AUTO_DECLINE | AUTO_DECLINE | ✅ |
| APP-1017 | AUTO_DECLINE | AUTO_DECLINE | ✅ |
| APP-1018 | AUTO_DECLINE | AUTO_DECLINE | ✅ |

**Router Task Success Rate: 12/12 = 100%**
**Auto-decline rate (zero-LLM-call efficiency): 6/12 = 50%**
**Borderline (human-gated) rate: 4/12 = 33%**
**Fast-track rate: 2/12 = 17%**

### 3b. Full-pipeline results (from the notebook's documented ablation — not re-run here)

| Metric | With Hybrid RAG + Memory (default) | Without Hybrid RAG Memory |
|---|---|---|
| Task Success Rate | 91.7% | 58.3% |
| Faithfulness | 0.94 | 0.68 |
| Characteristic failure mode | — | Hallucinated terms & DTI rules |

---

## 4. Ablation: Hybrid RAG + Memory vs. Baseline

**Component removed:** the hybrid retrieval pipeline (keyword ∪ vector
search → LLM rerank → `drop_superseded()` → `ground()`) and CrewAI's
short-/long-/entity memory, leaving the agents to answer policy questions
from parametric knowledge alone instead of the versioned `POLICY_CORPUS`.

| Metric | With component | Without component | Δ |
|---|---|---|---|
| Task Success Rate | 91.7% | 58.3% | **−33.4 pts** |
| Faithfulness | 0.94 | 0.68 | **−0.26** |

**Interpretation:** removing grounding does not just make citations
disappear — it directly damages *decision correctness*, because the model
falls back on outdated or invented numeric thresholds (see failure examples
below) rather than the corpus's actual, currently-effective policy text.
Grounding is therefore load-bearing for correctness, not just for
auditability.

---

## 5. Failure Examples (illustrative — without-RAG condition)

These reconstruct the ablation's named failure mode ("hallucinated terms &
DTI rules") against the corpus's known ground truth. They are not captured
transcripts; they show what an ungrounded model plausibly does when asked
the same questions the grounded system answers correctly.

### Failure 1 — Superseded income threshold applied

- **Applicant:** APP-1008, requesting a Platinum card, income $90,000.
- **Ground truth (POL-003, 2025-01 revision, effective 2025-01-01):**
  Platinum minimum income is **$80,000**. $90,000 clears the bar.
- **Ungrounded failure:** the model recalls the *2020* revision of POL-003
  ($60,000 minimum) from its parametric training data rather than checking
  which revision is currently effective, and states the applicant "clears
  the $60,000 requirement" — the right pass/fail outcome by coincidence, but
  the cited threshold is wrong and would misjudge a borderline applicant with
  income between $60k and $80k.
- **Why grounding fixes it:** `drop_superseded()` mechanically removes the
  2020 chunk once `TODAY` is past `2025-01-01`, so only the current
  threshold is ever retrievable.

### Failure 2 — Invented DTI cutoff

- **Applicant:** APP-1014, DTI = 40%.
- **Ground truth (POL-004):** DTI between **36–45%** requires manual
  underwriting review; only above 45% is an automatic decline.
- **Ungrounded failure:** the model states "a DTI above 40% typically
  triggers automatic decline in most underwriting frameworks" — a plausible-
  sounding but fabricated threshold that doesn't appear anywhere in the
  actual policy corpus, and would wrongly auto-decline an applicant POL-004
  requires to go to manual review.
- **Why grounding fixes it:** `ground()` only accepts wording that traces
  back to a retrieved passage; a threshold with no supporting chunk should
  trigger `not_covered` rather than free-text generation, and the reranked
  retrieval for a DTI question returns POL-004's exact 36%/45% bands.

### Failure 3 — Missing citation, unverifiable claim

- **Applicant:** APP-1011, on the fraud watchlist.
- **Ground truth (POL-006):** any watchlist flag requires automatic decline
  and escalation to compliance, "no exceptions."
- **Ungrounded failure:** the model correctly declines the applicant but
  justifies it with "elevated fraud risk indicators" instead of citing
  POL-006 by `doc_id` — a decision that is *directionally* right but fails
  the guardrail's citation requirement and gives a human reviewer nothing to
  audit the reasoning against.
- **Why grounding fixes it:** the hybrid retriever's keyword arm rewards
  exact token overlap (e.g., "watchlist"), reliably surfacing POL-006, and
  `eligibility_output_guardrail()` rejects any final decision with an empty
  `citations` list regardless of whether the underlying call was correct.

---

## 6. Limitations of This Report

- Router-level results (§3a) are exact and reproducible — they require no
  API key and no non-determinism.
- The 3 failure examples (§5) are constructed to match the ablation's
  described failure category and the corpus's ground truth; they illustrate
  *why* the ablation's numbers look the way they do, but are not logged
  model outputs.
