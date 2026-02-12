from __future__ import annotations

from kb_project.utils.messages import content_to_text


def test_content_to_text_uses_last_incremental_item():
    content = [
        "The capital of France is Paris.",
        "The capital of France is Paris. It is in Europe.",
        "The capital of France is Paris. It is in Europe. It is also the largest city.",
    ]
    text = content_to_text(content)
    assert text == content[-1]


def test_content_to_text_handles_typed_blocks():
    content = [
        {"type": "output_text", "text": "Marie Curie was born on November 7, 1867."},
        {"type": "output_text", "text": "She discovered polonium and radium."},
    ]
    text = content_to_text(content)
    assert "Marie Curie was born on November 7, 1867." in text
    assert "She discovered polonium and radium." in text
