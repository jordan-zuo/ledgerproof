# Two-minute demo

Live instance: https://enterprise-agent-orchestrator.onrender.com/ui

Two tracks. Track A is for a visitor on the hosted URL, where writes are disabled. Track B is the full flow on a local instance, where you own the token.

## Track A: hosted tour, 120 seconds

The public instance runs `READ_ONLY_DEMO=1`. Every write returns 403, so Run, Approve, Reverse, Cancel, Clear, Ingest, Close, and Reopen stay hidden. Everything below is a read of stored state.

| Time | Where | What to show |
| --- | --- | --- |
| 0:00 | `/ui` | The navbar badge says Read-only demo. The hero states the claim: deterministic steps, evidence, and a human gate. |
| 0:10 | Financial Ledger | One row per bill, not per document. Open a trail to show the posting, its evidence snippet, and any reversal. |
| 0:30 | Bill Reconciliation | Status per bill: posted, corrected, reversed, conflict, duplicate. Point at outstanding versus net. |
| 0:50 | Period Close & Export | Months sorted newest first, with bills, outstanding, and review exceptions. Open the ledger CSV in a new tab; show the period basis column. |
| 1:10 | Run Timeline | Pick a run and replay the event stream. The gate event is visible: the run paused for approval instead of posting. |
| 1:30 | Document Catalog | Identity basis, account, period, source file, and duplicate flags. This is the check that runs before the agent sees a bill. |
| 1:45 | `/docs` | The API surface: runs, SSE stream, approvals, ledger, reconciliation, periods, exports. |
| 2:00 | Stop | Close on the guarantee: nothing posts without evidence, a gate, and a bill identity that is not already in the ledger. |

If the instance is empty, the tour still works through Document Catalog and `/docs`, but the ledger, reconciliation, and period views have nothing to show. Start a run once as the owner, or point the host at a disk and seed it from a local run.

## Track B: full flow, local, 120 seconds

```powershell
pip install -r requirements.txt
python run.py
```

The browser opens `http://127.0.0.1:8000/ui`; writes are open because no token is set. Put a model key in `.env` first.

| Time | Action |
| --- | --- |
| 0:00 | Run Task: `Find the AGL sample bill and post the amount due.` The timeline streams retrieval, evidence, and one ledger write for 63.70 AUD. |
| 0:20 | Run Task: `Post the Origin standing bill total amount due.` The amount is over the gate threshold, so the run parks in Pending Approvals with 13,376.47 AUD. Approve it. One row lands. Deny another attempt to show that denial writes nothing. |
| 0:45 | Rerun the AGL task. The duplicate is rejected by the canonical bill key, and no second row appears. |
| 1:00 | Period Close: close 2023-04, the month the AGL bill period ends in. Try to reverse that row and show the 409. Reopen, reverse with a reason, and watch the corrected bill in Reconciliation. |
| 1:30 | Export: download the generic ledger CSV and the Xero CSV. The Xero export refuses to run until contact, account code, and tax type are set, so it cannot invent supplier data. |
| 1:50 | Clear the ledger. The reset archives the database first and keeps document metadata. |
| 2:00 | Stop. |

For a hosted owner session, set `APP_TOKEN` on the host, click Owner login, and paste the token. It stays in that tab's sessionStorage and is sent only on write requests.
