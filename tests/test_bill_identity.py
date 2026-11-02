from agent.bill_identity import identify_bill


def test_the_same_bill_under_two_doc_ids_shares_an_identity() -> None:
    """The AGL PDF and its text version are one bill, not two."""
    pdf = (
        "AGL electricity tax invoice. Account number 123 4567 891X. "
        "Issue date 5 Apr 2023. Bill period: 5 March 2023 to 4 April 2023."
    )
    text = (
        "AGL electricity tax invoice. Account number 1234567891X. "
        "Bill issue date 5 Apr 2023. Bill period 5 March 2023 to 4 April 2023."
    )
    first = identify_bill("agl-better-bills-sample-bill-necf", pdf)
    second = identify_bill("bill-agl-apr23", text)

    assert first.key == second.key
    assert first.basis == "account+period"
    assert first.key == "acct:1234567891X|period:2023-03-05:2023-04-04"


def test_period_spelling_and_separator_do_not_change_identity() -> None:
    long_form = "Account number: 123456 Bill period 1 March 2022 to 31 March 2022"
    short_form = "Account No 123456 Bill period 1 Mar 2022 – 31 Mar 2022"

    assert identify_bill("a", long_form).key == identify_bill("b", short_form).key


def test_a_different_period_is_a_different_bill() -> None:
    march = "Account number 123456 Bill period 1 March 2022 to 31 March 2022"
    april = "Account number 123456 Bill period 1 April 2022 to 30 April 2022"

    assert identify_bill("a", march).key != identify_bill("b", april).key


def test_unparseable_metadata_falls_back_to_the_document() -> None:
    """Without an account and period, two docs must not collide by accident."""
    first = identify_bill("doc-a", "Amount due 10.00. Payment assistance available.")
    second = identify_bill("doc-b", "Amount due 10.00. Payment assistance available.")

    assert first.basis == "document"
    assert first.key != second.key
