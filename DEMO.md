# Two-minute demo

Live instance: https://ledgerproof-orchestrator.onrender.com/ui

The hosted instance allows public writes, so the full flow runs in the browser. Writes are real and the database is ephemeral: a restart clears the ledger and closes nothing. The first visit after an idle pause can take 30 to 60 seconds.

## Hosted walkthrough, 120 seconds

| Time | Where | What to show |
| --- | --- | --- |
| 0:00 | `/ui` | The task box and write controls are visible. The hero states the claim: deterministic steps, evidence, and a human gate. |
| 0:10 | Run Task | `Find the AGL sample bill and post the amount due.` The timeline streams retrieval, evidence, and one ledger write for 63.70 AUD. |
| 0:40 | Run Task | `Post the Origin standing bill total amount due.` The amount is over the gate threshold, so the run parks in Pending Approvals at 13,376.47 AUD. Approve it and one row lands. |
| 1:00 | Financial Ledger | One row per bill. Open a trail to show the posting, its evidence snippet, and any reversal. |
| 1:15 | Bill Reconciliation | Status per bill: posted, corrected, reversed, conflict, duplicate. Point at outstanding versus net. |
| 1:30 | Period Close & Export | Close 2023-04, the month the AGL bill period ends in. Reversing the AGL row returns 409. Reopen, reverse, and watch the corrected bill. Download the ledger CSV; the Xero CSV returns 400 until contact, account code, and tax type are set. |
| 1:50 | Document Catalog | Identity basis, account, period, source file, and duplicate flags. This is the check that runs before the agent sees a bill. |
| 2:00 | Stop | Close on the guarantee: nothing posts without evidence, a gate, and a bill identity that is not already in the ledger. |

A restart starts with an empty ledger and no closed periods. The startup ingest rebuilds the document catalog from the three committed PDFs, so the catalog is ready even when the ledger is empty.

## Local, same flow

```powershell
pip install -r requirements.txt
python run.py
```

The browser opens `http://127.0.0.1:8000/ui`. Put a model key in `.env` first. Locally the database is `data/app.db` and survives a restart until you use Clear ledger.
