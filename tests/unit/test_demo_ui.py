from pathlib import Path


def test_demo_has_plain_language_result_card() -> None:
    html = (Path(__file__).resolve().parents[2] / "demo" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="plain"' in html
    assert "What this means" in html
    assert "not the real chance of forgery" in html
    assert "white mask are not proof of forgery" in html.lower()
