# Ledgerproof

An approval-gated agent runtime for accounts payable. The model reads a bill and proposes one step. The runtime checks the number against the source document, enforces the ledger rules, and stops for a human before money moves.

**[Open the live demo](https://ledgerproof-orchestrator.onrender.com/ui)**. No login needed. Writes are real, and the ledger resets when the instance redeploys. The first load after idle can take 30 to 60 seconds.

[![CI](https://github.com/jordan-zuo/ledgerproof/actions/workflows/ci.yml/badge.svg)](https://github.com/jordan-zuo/ledgerproof/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ledgerproof-orchestrator.onrender.com/ui)

## What it does

1. A task asks for a bill to be posted.
2. A worker claims the run and asks the model for one JSON step: retrieve, write to the ledger, request approval, or finish.
3. Retrieval is the only source of evidence. A write must quote the amount in the cited text, match known bill arithmetic, name the payable amount rather than a subtotal, match the bill period, and survive the duplicate check.
4. Above the threshold, or when the model asks, the runtime opens a real approval ticket and ends the step. Approval re-runs the checks before anything is written.
5. Ledger rows are append-only. A correction is a signed negative row linked to the original. Closing a month freezes a snapshot and refuses later postings in that month.
6. Every step is stored and streamed over SSE. A run says `done` only when the requested end state actually exists.

## Results

| Gate | Command | Latest result |
| --- | --- | --- |
| Tests | `python -m pytest -q` | 193 passed |
| Deterministic scenarios | `python scripts/evaluate.py` | 14/14 |
| Judged hard checks, offline replay | `python scripts/judged_evaluate.py --skip-judge --replay` | 9/9 |
| Live agent and judge suite | `python scripts/judged_evaluate.py` | 9/9 hard, 9/9 judge, 3/3 calibration |

CI runs the first three on every push. The live suite is manual because it spends real free-tier model quota. Replay re-applies the hard checks to committed trajectories with no API keys and no model calls. The generated report is in [eval/report.md](eval/report.md).

### Deterministic suite

Fourteen scenarios run through the real state machine with scripted tools and no model in the loop.

| Metric | Result |
| --- | --- |
| Pass rate | 14/14 |
| Completion | 0.75 |
| Trajectory efficiency | 1.0 |
| Gating accuracy | 1.0 |

Completion is 0.75 on purpose. Three of the four completion cases finish, and the fourth is a stubborn tool that the step ceiling has to stop. The suite covers routing, the step ceiling, malformed arguments, grounding, bill arithmetic, period rules, gate denial, gate approval, and ledger row counts.

### Live judged suite

Nine cases on 2026-11-02, rubric `judge-v3` (`0e5d7016`), agent `groq-120b` (`openai/gpt-oss-120b`), judge `google-flash` (`gemini-3.6-flash`). Raw counts, not percentages.

| Metric | Result |
| --- | --- |
| Hard checks | 9/9 |
| Judge pass | 9/9 scored, 0 unavailable |
| Hard and judge agreement | 1.0, 0 stricter, 0 more lenient |
| Judge calibration | 3/3, rejected a wrong amount and a skipped gate |
| Blocked proposals | 2 duplicate writes, no second row landed |
| Safety failures | 0 |
| Gate behavior | 2/2: one approved and posted, one denied with no row |
| Answer grounding | 3/3 required figures found in a retrieved snippet |
| Efficiency | median 2 steps, median 1.3 s per case, max 3.1 s |
| Judge latency | 53.1 s total, 20,998 tokens |

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

## Architecture

```text
browser (static HTML, SSE)
          |
          v
  FastAPI routes (scripts/api.py)
  auth, limits, body cap
          |
          v
  SQLite in WAL mode  <-->  dispatcher (agent/dispatch.py)
  runs, events,             claim, heartbeat, cancel
  approvals, ledger,              |
  documents, closures             v
                            runner (agent/runner.py)
                            step loop and ceiling
                                  |
                                  v
                            router, guardrails, bill identity,
                            ledger flow, retrieval, model pool
```

| Path | Responsibility |
| --- | --- |
| `agent/states.py` | typed state, evidence, and step decisions |
| `agent/router.py` | the pure routing function |
| `agent/runner.py` | the step loop, ceiling, and cancellation checks |
| `agent/guardrails.py` | grounding, payable amount, period, approval threshold |
| `agent/ledger_flow.py` | approval resume and the ledger write path |
| `agent/store.py` | schema, queue, ledger, events, tickets, closures, reset |
| `agent/dispatch.py` | claims, leases, heartbeat, durable cancellation |
| `agent/tools.py` | retrieval and ledger tools |
| `agent/driver.py` | prompt, JSON step parsing, and one retry |
| `agent/models.py` | provider pool, cooldowns, and key handling |
| `scripts/api.py` | HTTP surface, SSE, exports, access control, abuse limits |
| `static/index.html` | the entire UI, no build step |

## Why I built this

A language model can read a utility bill and type a number into a ledger. That is the easy part. The hard part is everything that makes the number safe to post: it has to come from the document, it has to be the payable amount rather than a subtotal, the bill must not already be in the ledger, a large payment has to wait for a person, a mistake has to be reversible without erasing history, and a closed month has to stay closed.

Ledgerproof is a working answer to that. It is a small runtime built around one rule: the model proposes, the runtime decides. There is no agent framework and no container runtime. One Python process, one SQLite file, one static HTML page.

## The parts that were hard

### The model proposes, the runtime decides

Routing is a pure function over typed Pydantic state. Tool arguments are validated before execution. The step ceiling belongs to the runner, not the prompt. A bad amount, a missing quote, or a period mismatch raises inside the tool and comes back as a failed step, not a silent no-op. The model never gets to decide that a write was allowed.

### Evidence before money

The ledger write requires a quoted snippet, an account, a period, and arithmetic that matches the bill. The grounding check matches a complete numeric token, not a substring: `13.00` is not grounded by a document that only says `113.00`, while a sentence ending in `$114.87.` still counts. The payable-amount check rejects a subtotal. The bill-math check reconciles the known bills. The period check rejects a bill from the wrong month. A plausible guess cannot reach the ledger.

### One bill, one row

Bill identity comes from the cited text, never from the model: account plus billing period, then account plus issue date, then the document id. The same bill filed under a PDF name and a text name collapses to one key. A run that is resumed after a crash cannot post the bill twice. The claim itself is a conditional SQLite update inside `BEGIN IMMEDIATE`, so two workers pointing at the same database cannot own the same run.

### Approvals are a state, not a chat message

A ticket is a row in SQLite with the evidence packet attached. The gate ends the step immediately. Approval resumes the run and re-runs grounding, math, amount due, period, and the closed-month guard before the write. Denial ends the run with no row. A second decision on the same ticket returns 409. Deciding a ticket on a `done`, `failed`, or `cancelled` run is refused, so a terminal status stays true. Cancelling at a gate denies the ticket and reports it, even if the exiting worker got there first.

### A run has to survive its process

`POST /run` writes a `pending` row and wakes a worker. A claim carries a 30 second lease and a 5 second heartbeat. If the process dies, the lease lapses and another worker resumes the run from step zero, where the bill identity guard makes replay safe. A run is retried at most three times, then failed with a recorded reason. Cancellation is a database flag as well as a local event, so the click reaches whichever process holds the run.

### A closed month has to mean something

A bill books to the month its billing period ends, then its issue month, then the posting month. Server time cannot be backdated, so closing the posting month would say nothing about the bills it was meant to finalise. A close appends a snapshot of bills, totals, and exceptions. New postings and reversals in a closed month are refused inside the same write transaction, so a close cannot race a write. A reopen appends an action and keeps the close history.

## Run it

```powershell
git clone https://github.com/jordan-zuo/ledgerproof.git
cd ledgerproof
pip install -r requirements.txt
python run.py
```

Before the first run, copy `.env.example` to `.env` and add at least one provider key. `run.py` loads the env file, creates the database, ingests the three committed PDFs when the inbox watcher is on, prints the access mode, and opens `http://127.0.0.1:8000/ui`.

```powershell
python run.py --check          # install report, no server and no network call
python run.py --read-only      # every write returns 403
python run.py --host 0.0.0.0   # bind for a host
```

`--port` sets the port and `--reload` restarts on edits. The [two-minute walkthrough](DEMO.md) follows the same flow on the hosted instance or locally.

## Deploy without Docker

The app is one Python process over SQLite, so it runs on a native Python runtime. There is no Dockerfile and no image registry. `render.yaml` is a Render blueprint; `Procfile` is the same start command for Railway, Heroku, or any host that runs Python. [Deploy the blueprint on Render](https://render.com/deploy?repo=https://github.com/jordan-zuo/ledgerproof).

| Setting | Effect |
| --- | --- |
| `PORT` | Injected by the platform. `run.py` uses it when `--port` is absent. |
| `APP_DB` | Database path. Defaults to `data/app.db`, which is ephemeral on a host without a disk. |
| `APP_INBOX_WATCH` | Inbox poller. On by default; startup ingest fills the catalog. |
| `GROQ_API_KEY`, `LLM7_API_KEY`, `LLM7_API_KEY_2`, `LLM7_API_KEY_3`, `GOOGLE_API_KEY` | Model pool keys. At least one is required for runs. |
| `APP_TOKEN` | When set, every write needs an `X-API-Token` header. |
| `READ_ONLY_DEMO=1` | Refuses every write with 403 while reads stay open. |

For durable writes, mount a disk at `/var/data` and set `APP_DB=/var/data/app.db`. The corpus, UI, and model config ship in the repo, so the database is the only state that needs a disk. The hosted instance is connected through the Render GitHub App, so a push to `main` deploys it.

## Access and abuse controls

| Configuration | Write behavior |
| --- | --- |
| Neither set | Open. Anyone who can reach the port can change the ledger. |
| `APP_TOKEN` set | Every write needs `X-API-Token`; reads stay open. |
| `READ_ONLY_DEMO=1` | Every write returns 403; reads stay open. |
| Both set | Visitors are read-only, and the owner can still write with the token. |

The UI reads `GET /config` and hides Run, Approve, Reverse, Cancel, Clear, Ingest, Close, and Reopen when writes are locked. Hiding a button is cosmetic; the 401 or 403 comes from the route.

The public instance also has process-local per-client limits, a global run cap, an active-run cap, a stream cap, a 32 KB request body cap, input length caps, CSV formula neutralization, and browser hardening headers. Per-client identity uses the rightmost `X-Forwarded-For` entry, which a caller can spoof, so those limits are a speed bump. The global caps are the hard bound. Limits reset when the process restarts.

## The rest of the ledger

**Corrections.** `POST /ledger/{id}/reverse` appends a negative row linked to the original with a reason and an approver. SQLite triggers refuse updates and deletes on ledger rows and events. A reversed posting no longer blocks a corrected re-post; a double reversal or a reversal of a reversal returns 409. `GET /reconciliation` folds the rows into one row per canonical bill with outstanding and net amounts, conflict and duplicate flags, and the full trail.

**Periods and exports.** `GET /periods` lists each accounting month with its open or closed state, bills, totals per currency, and any bill still flagged. `GET /export/ledger.csv` carries the audit fields, including the period basis, canonical key, run, entry, reversal link, and approver. `GET /export/xero.csv` produces Xero bill import columns for outstanding bills. Xero needs a contact, account code, and tax type; the endpoint returns 400 and names the missing fields rather than inventing them.

**Ingestion.** The inbox is `data/inbox` and the corpus is `data/corpus.json`. Drop in `.txt`, `.md`, or `.pdf` files and either `POST /ingest`, run `python scripts/ingest.py`, or let the watcher do it. `doc_id` comes from the file stem. Two files with the same id and the same text are one source; two files with the same id and different text are a collision that is reported, never silently overwritten. Canonical bill identity is computed at ingest, so two doc ids for one bill are visible in `GET /documents` before a run. Corpus writes are atomic and a lock file serializes ingest across processes.

**Model pool.** Live calls run on a pool of free models from Groq, llm7, and Google AI Studio. The ordered list per role lives in `config/models.json`; keys live only in `.env`. A 429 cools one model for the provider's retry window, a 401 disables the specs on that key, and when every model for a role is unavailable the call waits once for a bounded time and then fails with the pool state. Specs with no key are skipped. `--dry-run` prints both pools without a network call.

## Where it stops

- The queue is one SQLite database. It covers a handful of workers on one machine or a shared volume, not a fleet. A worker stalled past its lease can be resumed while still alive; the ledger rejects the duplicate row, but the run records the resumed attempt.
- There is one shared token at most. No accounts, sessions, or token expiry.
- The live suite is nine cases. It covers the paths end to end; a wider corpus and repeated runs are the next evidence to collect. The judge explains the trajectory and sees the hard-check results, so it is not an independent oracle.
- Amount-due parsing takes the first payable-looking figure. A document with several conflicting due lines could be read wrong.
- The Xero export is one supplier mapping and a bill import shape. It is not a chart of accounts, tax engine, or payment run.
- Period assignment falls back to the posting month when neither the bill period nor the issue date can be read. The export names the basis so the choice is visible.

## Repo map

| Path | What is there |
| --- | --- |
| `agent/` | runtime, state graph, guardrails, store, dispatcher, model pool |
| `scripts/` | API entry point, evaluators, ingest CLI |
| `static/` | the single-page UI, no build step |
| `tests/` | 22 test files, fixture corpus, no live database |
| `eval/` | committed live results and the generated report |
| `scenarios/` | deterministic eval cases |
| `data/` | inbox, corpus JSON, and committed sample PDFs |
| `demo/` | a snapshot ledger for inspection |

The tests use temp databases and a fixture corpus. They never touch `data/app.db` or the live documents. The live judged eval reads the real corpus read-only.

## Docs

- [DEMO.md](DEMO.md): the two-minute walkthrough, hosted or local.
- [CASE_STUDY.md](CASE_STUDY.md): the problem, the decisions, the limits, and what I would build next.
- [eval/report.md](eval/report.md): the generated live judged report.
- [config/models.json](config/models.json): the model pool, including which spec serves which role.
