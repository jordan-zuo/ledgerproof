# Ledgerproof: Deterministic Bill Posting with an Approval Gate

Hand-rolled state graph for tool-using workflows. Typed Pydantic state, pure routing function, hard step ceiling, fail-loud tool errors, and a human approval gate for sensitive operations. No agent framework dependency.

Status: working demo. Live run streaming, run list with reload reattach and cancel, a database-backed run queue with leases and restart recovery, request-id idempotency, evidence-backed ledger writes, human approvals, canonical bill identity, signed reversal entries, a bill reconciliation view, accounting-period close with append-only close/reopen actions, generic and Xero bill CSV exports, an archive-first ledger reset, token-gated writes with a read-only demo mode, inbox ingestion with identity and duplicate detection, a free-model pool with rotation, and a one-command launcher that runs on a native Python host without Docker. 176 tests, deterministic eval 14/14, and CI replaying the judged hard checks on every push.

## Run

```powershell
pip install -r requirements.txt
python run.py
```

`run.py` loads `.env`, creates the database, prints the access mode, and opens `http://127.0.0.1:8000/ui`. `python run.py --check` prints an install report and exits without a server: Python version, database path, corpus size, UI presence, access mode, and how many model specs have keys. `--read-only` refuses writes, `--port` and `--host` set the bind, and `--reload` restarts on edits.

```powershell
python -m pytest -q
python scripts/evaluate.py
```

Live public demo: https://ledgerproof-orchestrator.onrender.com/ui

[DEMO.md](DEMO.md) is the two-minute walkthrough. [CASE_STUDY.md](CASE_STUDY.md) covers the design, the evidence, and the limits.

## Deploy without Docker

The app is one Python process over SQLite, so it runs on a native Python runtime. There is no Dockerfile, no container build, and no image registry. `render.yaml` is a Render blueprint for its Python runtime, and `Procfile` is the same start command for Railway, Heroku, or any platform that can run `python run.py`.

[Deploy the blueprint on Render](https://render.com/deploy?repo=https://github.com/jordan-zuo/ledgerproof)

The blueprint runs one Python process with the default ephemeral SQLite database and the inbox watcher on, so a fresh instance ingests the three sample PDFs at startup. The hosted instance allows public writes: anyone can run a task, approve the gate, and close a period. Data resets when the instance restarts or redeploys.

| Setting | What it does |
| --- | --- |
| `PORT` | Injected by the platform. `run.py` reads it when `--port` is absent. |
| `APP_DB` | Database path. Defaults to `data/app.db`, which is ephemeral on a host without a disk. |
| `APP_INBOX_WATCH` | Inbox poller. On by default; startup ingest fills the document catalog. |
| `GROQ_API_KEY`, `LLM7_API_KEY`, `LLM7_API_KEY_2`, `LLM7_API_KEY_3`, `GOOGLE_API_KEY` | Model pool keys. Required for runs. |
| `APP_TOKEN` | Optional. When set, every write needs an `X-API-Token` header. |
| `READ_ONLY_DEMO=1` | Optional. Refuses every write with 403 while reads stay open. |

The free Render plan has an ephemeral filesystem and sleeps when idle. A restart starts with an empty ledger and no closed periods, and the startup ingest rebuilds the document catalog from the committed PDFs. The first visit after a pause can take 30 to 60 seconds. For durable writes, add a disk mounted at `/var/data` and set `APP_DB=/var/data/app.db`; Railway needs the same volume. The corpus, UI, and model config ship in the repo, so the database is the only state that needs a disk. The hosted demo is connected through the Render GitHub App, so a push to `main` deploys it automatically.

Before a public deploy, run the check that matters:

```powershell
python run.py --check
```

Then confirm the hosted instance reports the mode you expect at `/config`: `open` for the public demo, `read_only` when that flag is set. HTTPS, the public hostname, and restart supervision belong to the platform; the app binds `0.0.0.0` only because the platform forwards to it.

## Run queue

`POST /run` writes a pending row and wakes a worker. Workers claim one run at a time with a conditional SQLite update, so several processes pointed at the same database cannot run the same job twice. A claim carries a 30-second lease and a 5-second heartbeat; if a process dies, the lease lapses and another worker resumes the run from step zero. The ledger's bill identity guard keeps a resumed run from posting the same bill again. A run is retried at most three times, then failed, so a crash loop cannot spend quota forever. Cancellation writes a durable flag as well as waking the local worker, so a cancel reaches whichever process holds the run.

```powershell
python -m uvicorn scripts.api:app --port 8000 --workers 2
```

The queue is SQLite, so this covers a handful of workers on one machine or a shared volume, not a distributed fleet. A worker stalled longer than its lease can be resumed while it is still alive; the ledger rejects the duplicate posting, but that run may record the resumed attempt.

## Access control

Writes are open by default so local development stays frictionless. Two environment variables change that:

| Setting | Effect |
| --- | --- |
| neither set | Open. Anyone who can reach the port can run tasks and change the ledger. |
| `APP_TOKEN` set | Every write needs an `X-API-Token` header. Reads stay open. |
| `READ_ONLY_DEMO=1` | Every write returns 403. Reads stay open. |
| both set | Visitors are read-only, and the owner can still write with the token. |

The UI reads `GET /config` and hides Run, Approve, Reverse, Cancel, Clear, Ingest, Close, and Reopen while writes are locked. Exports and views stay readable.

The hosted demo is open on purpose:

```powershell
python -m uvicorn scripts.api:app --port 8000
```

Set `APP_TOKEN` (and optionally `READ_ONLY_DEMO=1`) when the instance must be locked. A token-only instance expects the header on every write, because the UI has no token prompt.

Anyone can browse the ledger, reconciliation, and run history; only a request carrying the token can start or change anything. Keep the token out of URLs and out of git. Binding the server to `127.0.0.1`, the default, keeps it off the network entirely. Use `--host 0.0.0.0` or a tunnel only when you mean to serve it.

## Ingestion

The inbox is `data/inbox` (`APP_INBOX` moves it) and the corpus is `data/corpus.json` (`APP_CORPUS` moves it). Drop in `.txt`, `.md`, or `.pdf` files and either `POST /ingest`, run `python scripts/ingest.py`, or let the watcher do it. The watcher starts with the server, polls every 5 seconds (`APP_INGEST_INTERVAL`), and ingests only when a file name, size, or modification time changed. `APP_INBOX_WATCH=0` disables it.

One ingest does four things:

- `doc_id` is the file stem, lowercased, with whitespace collapsed to hyphens.
- Two files with the same `doc_id` and the same text are one source; the first file name owns it and the other is reported as a duplicate source.
- Two files with the same `doc_id` and different text are a collision. The corpus keeps one text, the other file is named in the report, and nothing is overwritten silently. Rename the ignored file to give it its own id.
- Canonical bill identity is computed and stored at ingest time: account plus period, then account plus issue date, then the document fallback. Two doc ids for the same bill are reported and marked in `GET /documents`, so a duplicate is visible before an agent runs.

`GET /documents` is the catalog the UI uses: identity basis, account, period, source file, text size, a preview, and duplicate flags. `GET /documents/{doc_id}/file` serves the original inbox file, so the catalog and evidence citations link the exact PDF the extraction came from. Unsupported, empty, and no-text PDF files are reported as skipped. Corpus writes are atomic, and a lock file serializes ingest across processes, so several uvicorn workers can watch one inbox. `POST /ingest` is gated by the same write rules as every other mutation. Existing corpus documents without an inbox file stay in place, and clearing the ledger keeps document metadata.

## Ledger guarantees

One bill is one identity. The key is derived from the cited text, never from the model: account number plus billing period, then account plus issue date, then the document id. The same bill filed under a PDF name and a text name cannot be posted twice.

Corrections are append-only. `POST /ledger/{id}/reverse` appends a negative row linked to the original, with a required reason and approver; the original stays in place. A reversed posting no longer blocks a corrected re-post, while a double reversal or a reversal of a reversal returns 409.

The reconciliation view reads the rows back per bill. `GET /reconciliation` groups by canonical key, reports the outstanding and net amounts, flags conflicting outstanding amounts and duplicate postings, and keeps the full posting and reversal trail. The UI shows one row per bill with a trail modal, so a correction filed under a second document id still appears as one bill.

## Export and period close

A bill is booked to the month its billing period ends, then to its issue month, and only then to the month its ledger row was written. Server time cannot be backdated, so a posting-month close would say nothing about the bills it was meant to finalise. `GET /periods` lists those months with open or closed state, bills, outstanding and net totals per currency, and any bill still flagged conflict or duplicate.

`POST /periods/{period}/close` appends a close action with the totals it froze. While a month is closed, a new posting or a reversal for a bill in that month is refused with 409; other months continue. `POST /periods/{period}/reopen` appends a reopen action so a correction can be recorded, and the close action plus its snapshot stay in the history. The UI shows the same view and applies the write rules to Close and Reopen.

Two exports read the same rows:

| Endpoint | Shape |
| --- | --- |
| `GET /export/ledger.csv?period=YYYY-MM&scope=all\|outstanding` | Every audit field: canonical bill key, accounting period and its basis, account and bill dates, amount, currency, bill status, run, entry, reversal link, reason, approver, recorded time, and whether the period is closed. |
| `GET /export/xero.csv?period=YYYY-MM&contact=&account_code=&tax_type=&due_days=` | Xero bill import columns for outstanding bills only. A corrected bill exports its live replacement; the reversed row is absent. |

Xero needs a supplier contact, account code, and tax type. The ledger stores evidence and amounts, not your chart of accounts, so the endpoint returns 400 and names the missing fields instead of inventing them. Set `XERO_CONTACT`, `XERO_ACCOUNT_CODE`, `XERO_TAX_TYPE`, and `XERO_DUE_DAYS` to make them defaults. Invoice date is the issue date, then the period end, then the posting date; due date is the invoice date plus `due_days` (default 0). Exports are reads and stay open in read-only mode.

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

