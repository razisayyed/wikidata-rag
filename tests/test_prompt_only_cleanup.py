from __future__ import annotations

from kb_project.prompt_only_llm import answer_question_prompt_only


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeAgent:
    def stream(self, _payload):
        yield {
            "model": {
                "messages": [
                    _FakeMessage(
                        "I cannot verify a real-world collaboration between A and B in Wikidata, based on the search results."
                    )
                ]
            }
        }


def test_prompt_only_response_uses_same_final_cleanup():
    answer = answer_question_prompt_only(
        "Tell me about the collaboration between A and B.",
        llm=_FakeAgent(),
        verbose=False,
    )
    assert "wikidata" not in answer.lower()
    assert "search results" not in answer.lower()
    assert ",." not in answer
    assert "I cannot verify a real-world collaboration" in answer
