from scripts.prepare_vlm_bundle import conversation


def test_vlm_conversation_is_label_first_and_evidence_grounded() -> None:
    authentic = conversation({"label": 0})
    forged = conversation({"label": 1, "field_name": "total"})

    assert authentic[1]["content"][0]["text"].startswith("VERDICT: AUTHENTIC")
    assert forged[1]["content"][0]["text"].startswith("VERDICT: FORGED")
    assert "total" in forged[1]["content"][0]["text"]
    assert authentic[0]["content"][0] == {"type": "image"}
