# Case study: an approval-gated accounting agent

## The problem

Utility bills arrive as PDFs. A language model can read one and propose a posting, but the proposal alone is not accounting. The amount has to come from the document, a duplicate has to be impossible, a large payment has to wait for a person, a mistake has to be reversed rather than erased, and a closed month has to stay closed. The interesting work is not the model call. It is the state machine, the evidence, and the ledger rules around it.

This project is a working answer to that: a small agent runtime that reads three sample bills, proves the figure it wants to post, gates anything over a threshold, and keeps an append-only ledger that can be reconciled, closed, and exported.

## What it is

One Python process. FastAPI for the HTTP surface and SSE, SQLite in WAL mode for runs, events, tickets, documents, ledger rows, and period closures, Pydantic for typed state and tool arguments, and a hand-rolled state graph. There is no agent framework and no container runtime. The UI is one static HTML file.

The loop is deliberately boring:

```text
retrieve -> evidence -> decide -> execute tool | await approval | terminal
```

The model proposes one step at a time. Code owns the ceiling, the routing, the argument validation, and the write rules. A run that cannot finish within the step ceiling, repeats the same tool with the same arguments, or claims completion without settling the task is stopped by the runner, not by the prompt.

## Decisions that shaped it

**The model proposes, the runtime disposes.** Routing is a pure function over typed state. Tool arguments are validated before execution. Every failure is loud: a bad amount, a missing quote, or a period mismatch raises inside the tool and comes back as a failed step, not a silent no-op.

**A write needs evidence from the document.** The ledger tool requires a quoted snippet, an account, a period, and math that matches the bill. The amount is checked against the cited text before a ticket or a row exists. A model that guesses a plausible number cannot post it.

**Bill identity comes from the text, not the model.** The canonical key is account plus billing period, then account plus issue date, then the document id. The same bill filed under a PDF name and a text name collapses to one key. That is what makes a duplicate repost fail even after a restart or a resumed run.

**Corrections are appended.** There is no update or delete for ledger rows; SQLite triggers refuse both. A reversal is a negative row linked to the original, with a reason and an approver. Reconciliation groups the trail per bill, so a corrected bill shows its live replacement and its reversal without losing history.

**A month closes on the bill period, not the posting time.** Server time cannot be backdated, so closing the month a row was written would say nothing about the bills it was meant to finalise. A bill books to the month its billing period ends, then its issue month, then the posting month. A close appends a snapshot of bills, outstanding, net, and exceptions. The store refuses new postings and reversals in a closed month inside `BEGIN IMMEDIATE`, so a close cannot race a write. A reopen appends an action and keeps the close snapshot in the history.

**The gate is a real stop.** Amounts over the threshold create a ticket and end the step. Approval resumes the run; denial ends it with no row. The ticket is not consumed when the bill's month is closed, so the operator can reopen and approve later.

**The model pool is a reliability layer.** Free-tier models rotate per role with per-spec cooldowns and a bounded wait instead of an open-ended retry. A 429 cools one model, a rejected key disables only the specs that name it. A suite keeps one model unless it is forced to rotate, which makes a comparison between runs meaningful.

**No Docker.** The whole state is one SQLite file and the code is one Python process. A native Python runtime is fewer moving parts than an image build: `render.yaml` for Render, `Procfile` for Railway or Heroku, and `python run.py` everywhere else. The tradeoff is a dependency on the host's Python runtime, which `.python-version` pins.

## Evidence

The offline suite is deterministic: 14 scenarios run through the real state machine with scripted tools. The live suite asks a real model to work the actual bills, applies deterministic hard checks, then asks a second model to explain the trajectory against a fixed rubric. Hard checks win every disagreement.

| Check | Result |
| --- | --- |
| Pytest | 173 tests |
| Deterministic eval | 14/14, completion 0.75, efficiency 1.0, gating 1.0 |
| Live hard checks | 9/9 on 2026-11-02 |
| Judge calibration | 3/3 probes rejected the wrong amount and the skipped gate |
| Answer grounding | 3/3 required figures found in a retrieved snippet |
| Duplicate attempts | 2 blocked, no second row landed |
| Gate behavior | 1 approved and posted, 1 denied with no row |
| Efficiency | median 2 steps, median 1.3 s per case, max 3.1 s |
| CI offline replay | 9/9 hard checks re-applied to committed trajectories, no keys |

The benchmark is a nine-case smoke suite on free-tier models, not a stability study. One clean run proves the paths work. It does not estimate variance, and the judge is an explanation layer that sees the gold case, not an independent oracle. The raw numbers and per-case table live in the [README](README.md) and [eval/report.md](eval/report.md).

Access control is tested the same way: a read-only demo returns 403 on every write while reads stay open, a configured token rejects a missing or wrong header, and the owner token unlocks a read-only instance. Before a public deploy, `python run.py --check` reports the database path, corpus size, access mode, and which model specs have keys, without a network call.

## Failure modes and limits

- SQLite covers a handful of workers on one machine or a shared volume, not a distributed fleet. A worker stalled past its lease can be resumed while it is still alive; the ledger rejects the duplicate posting, but the run records the resumed attempt.
- Render's free plan has an ephemeral filesystem. A ledger that must survive a deploy needs a disk and `APP_DB` pointing at it.
- Period assignment is best effort when neither the bill period nor the issue date can be read from the text. The fallback is the posting month, and the export names the basis so a reviewer can see it.
- The Xero export is one supplier mapping and a bill import shape. It is not a chart of accounts, tax engine, or payment run.
- Per-user accounts, token expiry, and rate limits are not built. The public posture is one owner token plus a read-only default, not multi-tenant auth.
- The judged suite is nine cases. A wider corpus and repeated runs are the next evidence to collect.

## What I would build next

A retrieval cache keyed by document and query, so repeated runs do not re-extract the same PDF. A Postgres adapter behind the same store interface for a real multi-worker deployment. Per-user sessions and rate limits before any instance accepts writes from more than one operator. And a wider judged suite with repeated runs, so the pass rate becomes a distribution instead of a single clean result.

## Links

- Repository: https://github.com/jordan-zuo/ledgerproof
- Live read-only instance: https://ledgerproof-d1ve.onrender.com/ui
- Two-minute walkthrough: [DEMO.md](DEMO.md)
- Live judged report: [eval/report.md](eval/report.md)
