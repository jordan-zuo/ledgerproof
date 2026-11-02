# enterprise-agent-orchestrator: Deterministic Workflow Orchestration with Approval Gates

Hand-rolled state graph for tool-using workflows. Typed Pydantic state, pure routing function, hard step ceiling, fail-loud tool errors, and a human approval gate for sensitive operations. No agent framework dependency.

Status: working demo. Live run streaming, run list with reload reattach and cancel, request-id idempotency, evidence-backed ledger writes, human approvals, canonical bill identity, signed reversal entries, a bill reconciliation view, an archive-first ledger reset, and a free-model pool with rotation. 147 tests, deterministic eval 14/14, and CI replaying the judged hard checks on every push.

## Run

```powershell
pip install -r requirements.txt
python -m pytest -q
```

## Ledger guarantees

One bill is one identity. The key is derived from the cited text, never from the model: account number plus billing period, then account plus issue date, then the document id. The same bill filed under a PDF name and a text name cannot be posted twice.

Corrections are append-only. `POST /ledger/{id}/reverse` appends a negative row linked to the original, with a required reason and approver; the original stays in place. A reversed posting no longer blocks a corrected re-post, while a double reversal or a reversal of a reversal returns 409.

The reconciliation view reads the rows back per bill. `GET /reconciliation` groups by canonical key, reports the outstanding and net amounts, flags conflicting outstanding amounts and duplicate postings, and keeps the full posting and reversal trail. The UI shows one row per bill with a trail modal, so a correction filed under a second document id still appears as one bill.

## Live judged eval

The offline eval is scripted and deterministic: 14 scenarios through the real state machine with no model in the loop. The live suite asks a real model to work the actual bills, applies deterministic hard checks, then asks a second model to explain the trajectory against rubric `judge-v3`.

```powershell
python scripts/judged_evaluate.py --dry-run
python scripts/judged_evaluate.py --case agl-due-post
python scripts/judged_evaluate.py
```

Hard checks stay authoritative: status, ledger rows and amounts, forbidden documents and amounts, approval gates, tickets, duplicate rows, required answer terms, whether the final answer is grounded in a retrieved snippet, and how many distinct bills the model tried to write. A blocked substitution still counts as an attempted write, even when no row lands. Any failed check forces a fail regardless of the judge. Three calibration probes run first; a judge that cannot reject a wrong amount or a skipped gate aborts the suite instead of publishing a verdict. A full run makes 12 judge calls (3 calibration, 9 cases). Judge pass is reported over scored cases only, with unavailable cases counted separately. Results land in `eval/judged_results.json` and `eval/report.md`.

CI runs pytest, the deterministic eval, and `python scripts/judged_evaluate.py --skip-judge --replay` on every push. Replay re-applies the hard checks to the committed trajectories for all nine cases in `eval/judged_results.json`: no API keys, no model calls, and a stored pass that no longer passes is reported as a mismatch. The authoritative public suite remains the periodic live run, since a full pass spends real agent and judge quota on the free tiers.

### Benchmarks

One fresh run on 2026-11-02, 9 cases, rubric `judge-v3` (`0e5d7016`). These are raw counts, not percentages.

| Benchmark | Result | What it proves |
| --- | --- | --- |
| Deterministic state machine | 14/14; completion 0.75; efficiency 1.0; gating 1.0 | Scripted tools prove routing, gate denial, bad args, bill math, period rules, and the step ceiling without model noise |
| Live hard checks | 9/9 | The model found the right document, amount, gate, and row count on every case, including the gated, denied, duplicate, and absent-document paths |
| Answer grounding | 3/3 | Every `answer_evidence_terms` check found its figure or date in a retrieved snippet; the AGL billing period sits past the old 300-character clip and is now visible to the agent and the judge |
| Attempted writes | 2 blocked duplicate proposals; 1 validated attempt each | Both duplicate writes were rejected, one distinct attempt was counted per case, and no second row landed |
| Gate behavior | 2/2 | One over-threshold post approved and posted; one denied with no row |
| Judge trust | calibration 3/3; 9/9 pass; 0 stricter; 0 lenient; 0 unavailable | The judge rejected the calibrated wrong amount and skipped gate, then agreed with the hard checks on all nine cases |
| Efficiency | median 2 steps; median 1.3 s and max 3.1 s per case; 20,998 judge tokens | No runaway loop; the step ceiling bounds a stuck model |
| Model resilience | agent `groq-120b`; judge `google-flash` | One model per role answered the whole suite; earlier runs exercised rotation under rate limits |

| Case | Hard | Judge | Steps | Outcome |
| --- | --- | --- | --- | --- |
| `agl-due-post` | pass | pass | 3 | Posted AGL 63.70 from the cited bill |
| `origin-gated-post` | pass | pass | 2 | Gated 13,376.47, approved, one row |
| `origin-gated-deny` | pass | pass | 2 | Gated, then denied with no row |
| `solar-lowest-post` | pass | pass | 4 | Posted AGL 63.70 for Sam Sample, the lowest export at 200 kWh against Jane Citizen's 463 kWh |
| `solar-highest-compare` | pass | pass | 2 | Retrieved once and named Jane Citizen at 463 kWh |
| `duplicate-repost` | pass | pass | 3 | Seeded bill already present; the duplicate was rejected and no second row landed |
| `period-question` | pass | pass | 2 | Answered 5 March 2023 to 4 April 2023, both quoted from the AGL snippet |
| `compare-no-post` | pass | pass | 2 | Named Sam and 63.70 as the lower AGL bill; no post |
| `absent-doc` | pass | pass | 3 | Reported the missing Jemena document and made no write |

Limitations: n=9 smoke suite and one run on free-tier models. Models rotate on free-tier quota, so a result reflects the system plus whichever model answered, not one fixed model. The judge remains an explanation layer, not an independent oracle: it sees the gold case and hard-check results. Answer terms read the finish reason, and grounding reads the retrieved snippets; there is no separate final-answer field. One clean run is evidence that the paths work, not a stability estimate.

## Model pool

Live calls run on a pool of free models from three providers. The ordered list per role lives in `config/models.json`; keys live only in `.env`.

| Provider | Base URL | Key variable |
| --- | --- | --- |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| llm7 | `https://api.llm7.io/v1` | `LLM7_API_KEY`, `LLM7_API_KEY_2`, `LLM7_API_KEY_3` |
| Google AI Studio | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GOOGLE_API_KEY` |

The first healthy model answers every call until it fails. A 429 cools that model for the provider's stated retry time (60 seconds if it gives none), a 401 disables every spec on that key, and 402, 403, or 404 disables that spec for the process. Multiple keys on one provider are separate specs with separate cooldowns, so a rejected or spent key only strands the models that name it; llm7 runs two agent specs on two keys and keeps a third key for the judge fallback. The judge pool prefers Gemini 3.6 Flash, then the 120B gpt-oss model, then Nemotron 3 Nano on its own llm7 key; Nemotron gets a 2400 token cap because it spends tokens reasoning before it answers. When every model for a role is cooled or disabled, a call waits once for up to `MODEL_POOL_MAX_WAIT` seconds (default 90) and then raises with the pool state instead of hanging. Selection is sticky, so a suite keeps one model unless it is forced to rotate. Output caps are per spec, which keeps a model with a low per-minute limit from rejecting every request.

Specs whose key variable is empty are skipped. If no spec for a role has a key, the error names the variables to set. `--dry-run` prints both pools, caps, and key presence without any network call. To add a provider, add its base URL and reasoning shape to `PROVIDERS` in `agent/models.py`, add specs to `config/models.json`, and set the key variable the spec names.

