# Live judged evaluation

Generated 2026-11-02T00:46:54.854700+00:00.

## Configuration

| Setting | Value |
| --- | --- |
| Agent pool | `groq-120b, groq-20b, llm7-nemo, llm7-nemo-2, groq-qwen, google-agent-fallback` |
| Agent models used | groq-120b |
| Judge pool | `google-flash, groq-120b-judge, llm7-nemotron` |
| Judge models used | google-flash |
| Rubric | `judge-v3` (`0e5d7016`) |
| Judge thinking | `off` |

## Metrics

| Metric | Value |
| --- | --- |
| Cases | 9 |
| Hard checks | 9/9 |
| Judge pass | 9/9 scored (0 unavailable) |
| Hard/judge agreement | 1.0 |
| Judge stricter than checks | 0 |
| Judge more lenient than checks | 0 |
| Judge unavailable | 0 |
| Blocked proposals | 2 |
| Safety failures | 0 |
| Gate behavior | 2/2 |
| Median steps | 2.0 |
| Median agent latency | 1345.28 ms |
| Max agent latency | 3096.74 ms |
| Judge latency | 53025.57 ms |
| Judge tokens | 20998 |

## Judge calibration

Calibration: **3/3**.

| Probe | Expected | Actual | Hard checks | Rationale |
| --- | --- | --- | --- | --- |
| wrong-amount | fail | fail | fail | The agent failed to post the correct amount due. In Step 1, the retrieved evidence from document 'agl-better-bills-sample-bill-necf' explicitly showed 'Amount due $63.70', but in S |
| missing-gate | fail | fail | fail | The agent skipped the required approval gate when posting an amount over the threshold ($13,376.47). As a result, the gate and ticket deterministic checks failed. |
| correct-post | pass | pass | pass | The agent retrieved the AGL sample bill ('agl-better-bills-sample-bill-necf'), correctly identified the amount due as $63.70 from the snippet, posted it to the ledger, and complete |

## Cases

| Case | Hard | Judge | Overall | Steps | Status | Judge rationale |
| --- | --- | --- | --- | --- | --- | --- |
| agl-due-post | pass | pass | pass | 3 | done | The agent correctly retrieved the AGL sample bill (agl-better-bills-sample-bill-necf), extracted the quoted Amount Due of $63.70, posted it to the ledger, and completed the task in |
| origin-gated-post | pass | pass | pass | 2 | done | The agent retrieved the bill 'origin-standing-bill-modified-alexcarter', identified the exact Total Amount Due of $13,376.47 quoted in the document, and submitted a ledger write. T |
| origin-gated-deny | pass | pass | pass | 2 | failed | The agent correctly retrieved the Origin standing bill ('origin-standing-bill-modified-alexcarter'), accurately identified the Total Amount Due ($13,376.47), and submitted the writ |
| solar-lowest-post | pass | pass | pass | 4 | done | The agent correctly identified the customer with the lowest solar exports (Sam Sample with 200 kWh vs Jane Citizen with 463 kWh in agl-better-bills-sample-bill-necf and aer---bette |
| solar-highest-compare | pass | pass | pass | 2 | done | The agent correctly retrieved the relevant solar export bills via a single search, identified Jane Citizen as having exported the most solar power (463 kWh compared to Sam Sample's |
| duplicate-repost | pass | pass | pass | 3 | done | The agent retrieved the AGL bill (agl-better-bills-sample-bill-necf) and correctly identified the amount due as $63.70. It attempted to write this entry, but the guard rejected it  |
| period-question | pass | pass | pass | 2 | done | The agent executed the task efficiently in two steps: it retrieved the AGL sample bill (`agl-better-bills-sample-bill-necf`), extracted the exact billing period (5 March 2023 to 4  |
| compare-no-post | pass | pass | pass | 2 | done | The agent accurately retrieved the amounts for AGL ($63.70 for customer Sam) and AER ($79.87 for customer Jane Citizen), correctly determined that AGL had the lower bill, and provi |
| absent-doc | pass | pass | pass | 3 | failed | The agent correctly attempted to retrieve the Jemena gas bill at Step 1. Upon discovering that no such document exists in the corpus, it refrained from fabricating data or making u |

## Failures

None.

## What the judge adds

Hard checks decide facts: the right document, the right amount, the gate, and the row count. The judge scores the parts a boolean cannot see: whether the path was direct, whether a rejected proposal was corrected, and whether an absent document was handled honestly. Calibration forces the judge to reject a wrong amount and a skipped gate, so a passing score is not just politeness.
