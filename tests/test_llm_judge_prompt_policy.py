from __future__ import annotations

from kb_project.benchmark.llm_judge import JUDGE_SYSTEM_PROMPT, build_judge_prompt


def test_judge_system_prompt_penalizes_scope_bloating_extras():
    assert "extra details beyond the asked question are NOT a positive signal" in JUDGE_SYSTEM_PROMPT
    assert "unsupported/scope-bloating extra claims" in JUDGE_SYSTEM_PROMPT


def test_build_judge_prompt_includes_extra_info_policy():
    prompt = build_judge_prompt(
        question="Who is Albert Einstein?",
        rag_response="Albert Einstein was a theoretical physicist.",
        prompt_only_response="Albert Einstein was a scientist.",
    )
    assert "extra information must not improve the score" in prompt
    assert "unsupported or unnecessary extra claims as risky behavior" in prompt
