import pytest
from pydantic import ValidationError

from agent.states import AgentState
from agent.tools import LedgerArgs, LedgerWriteTool, RetrievalArgs, RetrievalTool

DOCS = {
    "q3-financials": "enterprise revenue was 48.2 million with margin 18.4 percent",
    "security-policy": "api keys must be rotated every 90 days with hardware mfa",
}


def test_retrieval_returns_top_doc() -> None:
    tool = RetrievalTool(DOCS)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="api key rotation"))
    assert "security-policy" in event


def test_retrieval_folds_plurals() -> None:
    tool = RetrievalTool(DOCS)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="keys rotated"))
    assert "security-policy" in event


def test_retrieval_splits_underscores_and_hyphens() -> None:
    tool = RetrievalTool(DOCS)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="api_keys-rotated"))
    assert "security-policy" in event


def test_retrieval_no_match_says_so() -> None:
    tool = RetrievalTool(DOCS)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="penguin migration"))
    assert event == "retrieve [penguin migration]: no matching document"


def test_retrieval_names_an_absent_task_entity_instead_of_substituting() -> None:
    docs = {"gas-bill": "greenhouse gas emissions and an amount due of 79.87"}
    state = AgentState(task="Post the amount due from the Jemena gas bill.")
    event = RetrievalTool(docs).execute(state, RetrievalArgs(query="gas bill"))
    assert "no document mentions" in event
    assert "jemena" in event
    assert state.evidence == []


def test_retrieval_does_not_treat_a_month_as_an_absent_entity() -> None:
    docs = {"agl-bill": "Bill period: 5 March 2023 to 4 April 2023. Amount due 63.70"}
    state = AgentState(task="Does the AGL bill cover December 2023?")
    event = RetrievalTool(docs).execute(state, RetrievalArgs(query="AGL December 2023"))
    assert "agl-bill" in event
    assert len(state.evidence) == 1


def test_retrieval_blank_query_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalArgs(query="   ")


def test_ledger_appends_and_receipts() -> None:
    ledger = LedgerWriteTool()
    args = LedgerArgs(entry_id="j-001", amount=48.2, source_doc="q3-financials")
    event = ledger.execute(AgentState(task="t"), args)
    assert event == "ledger_write: q3-financials 48.20 AUD"
    assert len(ledger.entries) == 1


def test_ledger_rejects_bad_args() -> None:
    with pytest.raises(ValidationError):
        LedgerArgs(entry_id="", amount=-5, source_doc="")


def test_ledger_blocks_a_second_identical_post_in_one_run() -> None:
    ledger = LedgerWriteTool()
    args = LedgerArgs(entry_id="j-001", amount=63.70, source_doc="agl")
    ledger.execute(AgentState(task="t"), args)
    assert ledger.already_posted(args) is not None


def test_retrieval_snippet_preserves_query_matches_and_account_number() -> None:
    corpus = {
        "bill-1": (
            "Electricity tax invoice. Account number 123456. Name Jane Citizen. "
            "Supply address 1 Street Road. Amount due 79.87 dollars. "
            "Solar exports feed-in tariff 463 kWh at 0.09 dollars per kWh. "
            "Previous reading 8763 current reading 9226."
        )
    }
    tool = RetrievalTool(corpus)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="463 kWh solar exports"))
    assert "123456" in event
    assert "463" in event
    assert "Jane Citizen" in event
    assert "bill-1" in event


def test_retrieval_snippet_keeps_due_figure_past_the_header() -> None:
    filler = "Payment options and contact details for your account. " * 30
    corpus = {
        "bill-long": (
            "Tax invoice. Account number 123456. Amount due Due date. "
            + filler
            + " Total new charges (including GST) = $63.70 Amount Due = $63.70."
        )
    }
    tool = RetrievalTool(corpus)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="amount due"))
    assert "123456" in event
    assert "63.70" in event


def test_retrieval_snippet_keeps_query_match_and_identity_figure() -> None:
    filler = "Payment options and contact details for your account. " * 20
    corpus = {
        "bill-long": (
            "Tax invoice. Name Sam Sample. Account number 123456. "
            + filler
            + " Solar exports feed-in tariff 200 kWh at 0.05 dollars per kWh. "
            "Amount due 63.70 dollars."
        )
    }
    tool = RetrievalTool(corpus)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="solar export"))
    assert "Sam Sample" in event
    assert "200 kWh" in event
    assert "63.70" in event


def test_evidence_keeps_the_snippet_the_model_saw() -> None:
    filler = "Payment options and contact details for your account. " * 30
    corpus = {
        "bill-long": (
            "Tax invoice. Account number 123456. "
            + filler
            + " Name Jane Citizen. Solar exports feed-in tariff 463 kWh."
        )
    }
    state = AgentState(task="t")
    RetrievalTool(corpus).execute(state, RetrievalArgs(query="solar export"))
    item = state.evidence[0]
    assert "Jane Citizen" in item.snippet
    assert "Jane Citizen" not in item.text[:500]


def test_retrieval_ranks_multiple_candidates_by_relevance() -> None:
    corpus = {
        "bill-sam": "Electricity invoice for Sam Sample. Solar export 200 kWh.",
        "bill-jane": "Electricity invoice for Jane Citizen. Solar exports feed-in tariff 463 kWh. Average daily solar exports.",
        "bill-water": "Water service quarterly invoice.",
    }
    tool = RetrievalTool(corpus)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="solar export", top_k=2))
    assert "bill-jane" in event
    assert "bill-sam" in event
    assert "bill-water" not in event


def test_retrieval_prefers_a_named_document_over_a_body_match() -> None:
    """Asking for AER must not return the AGL bill just because both say due."""
    body = "Tax invoice. Amount due 79.87 dollars."
    tool = RetrievalTool({"agl-sample-bill": body, "aer-example-bill": body})
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="AER amount due", top_k=1))
    assert event.startswith("retrieve [AER amount due]: aer-example-bill")


def test_retrieval_snippet_keeps_the_billing_period_when_asked() -> None:
    filler = "Payment options and contact details for your account. " * 30
    corpus = {
        "bill-long": (
            "Tax invoice. Account number 123456. Amount due 63.70 dollars. "
            + filler
            + " Bill period: 5 March 2023 to 4 April 2023 (31 Days)."
        )
    }
    tool = RetrievalTool(corpus)
    event = tool.execute(AgentState(task="t"), RetrievalArgs(query="billing period"))
    assert "5 March 2023" in event
    assert "4 April 2023" in event
