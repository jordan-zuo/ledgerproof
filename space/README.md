---
title: Enterprise Agent Orchestrator
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
python_version: "3.12"
pinned: false
short_description: Approval-gated bill agent with an append-only ledger
---

# Enterprise Agent Orchestrator

A read-only demo of an approval-gated agent that reads utility bills, proves the
amount it wants to post, pauses for a human above a threshold, and writes an
append-only ledger.

This Space is served by the project's FastAPI app on a native Python runtime.
Writes are disabled with `READ_ONLY_DEMO=1`, so visitors can browse the ledger,
reconciliation, period close, exports, runs, and the document catalog.

The ledger is a snapshot of real agent runs on the three sample bills in the
source repository. Closing a month, reversing a posting, or running a new task
requires the owner token and is not available here.

- Source and full setup: https://github.com/jordan-zuo/enterprise-agent-orchestrator
- Two-minute walkthrough: https://github.com/jordan-zuo/enterprise-agent-orchestrator/blob/main/DEMO.md
- Case study: https://github.com/jordan-zuo/enterprise-agent-orchestrator/blob/main/CASE_STUDY.md
