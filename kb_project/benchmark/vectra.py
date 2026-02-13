"""Vectara utilities, agent run capture, and static benchmark cases."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..settings import RAG_RECURSION_LIMIT, VECTARA_DEVICE, resolve_device
from ..tools.tool_protocol_state import reset_tool_protocol_state, set_question_context
from ..utils.messages import content_to_text
from ..wikidata_rag_agent import build_agent, finalize_agent_answer, is_process_message
from .models import TestCase


@dataclass
class ToolCall:
    """Represents one tool invocation and response."""

    name: str
    args: Dict[str, Any]
    output: str = ""


@dataclass
class AgentRun:
    """Captured execution for one question."""

    question: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: str = ""

    @property
    def retrieved_context(self) -> str:
        parts = []
        for tool_call in self.tool_calls:
            parts.append(f"[Tool: {tool_call.name}]\n{tool_call.output}")
        return "\n\n".join(parts)

    @property
    def sanitized_retrieved_context(self) -> str:
        parts = []
        for tool_call in self.tool_calls:
            cleaned = sanitize_tool_output(tool_call.name, tool_call.output)
            if cleaned:
                parts.append(f"[Tool: {tool_call.name}]\n{cleaned}")
        return "\n\n".join(parts)


def _strip_instruction_lines(text: str) -> str:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("INSTRUCTIONS:"):
            continue
        if upper.startswith("USE THE QID OF YOUR SELECTED CANDIDATE"):
            continue
        if upper.startswith("IF NONE MATCH"):
            continue
        if upper.startswith("ONLY USE INFORMATION EXPLICITLY STATED"):
            continue
        lines.append(raw_line)
    return "\n".join(lines).strip()


def sanitize_tool_output(tool_name: str, output: str) -> str:
    """Sanitize tool output used for faithfulness-style scoring."""
    cleaned = _strip_instruction_lines(output or "")
    if not cleaned:
        return ""

    if tool_name == "search_entity_candidates":
        if "NO CANDIDATES FOUND" in cleaned:
            for line in cleaned.splitlines():
                if "NO CANDIDATES FOUND" in line:
                    return line.strip()
            return "NO CANDIDATES FOUND"
        return ""

    return cleaned


_NO_CANDIDATE_PATTERN = re.compile(r"NO CANDIDATES FOUND for '([^']+)'", re.IGNORECASE)
_REFUSAL_MARKERS = (
    "i cannot verify",
    "i can't verify",
    "cannot be verified",
    "could not be verified",
    "i cannot determine",
    "cannot determine",
    "no verified",
)


def _looks_like_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _extract_unresolved_entities(tool_calls: List[ToolCall]) -> List[str]:
    entities: List[str] = []
    for tool_call in tool_calls:
        if tool_call.name != "search_entity_candidates":
            continue
        for match in _NO_CANDIDATE_PATTERN.finditer(tool_call.output or ""):
            entity = match.group(1).strip()
            if entity and entity not in entities:
                entities.append(entity)
    return entities


def _has_disambiguation_warning(tool_calls: List[ToolCall]) -> bool:
    for tool_call in tool_calls:
        if tool_call.name != "search_entity_candidates":
            continue
        if "DISAMBIGUATION WARNING" in (tool_call.output or ""):
            return True
    return False


def _has_successful_entity_fetch(tool_calls: List[ToolCall]) -> bool:
    """
    Return True when at least one fetch_entity_properties call returned data.

    This avoids false refusal overrides when candidate search emitted a warning
    but the model still selected a concrete QID and fetched valid properties.
    """
    for tool_call in tool_calls:
        if tool_call.name != "fetch_entity_properties":
            continue
        output = (tool_call.output or "").strip()
        if not output:
            continue
        if output.lower().startswith("error:"):
            continue
        if "QID:" in output or "Entity:" in output:
            return True
    return False


def _apply_no_answer_gating(answer: str, tool_calls: List[ToolCall]) -> str:
    current = (answer or "").strip()
    if _looks_like_refusal(current):
        return current

    unresolved = _extract_unresolved_entities(tool_calls)
    if unresolved:
        if len(unresolved) == 1:
            return (
                f"I cannot verify that {unresolved[0]} exists, "
                "and I cannot verify this claim."
            )
        joined = ", ".join(unresolved[:-1]) + f", and {unresolved[-1]}"
        return f"I cannot verify that {joined} exist, and I cannot verify this claim."

    if _has_disambiguation_warning(tool_calls) and not _has_successful_entity_fetch(
        tool_calls
    ):
        return "I cannot determine which entity the question refers to."

    return current


def run_agent_with_capture(question: str, agent=None, verbose: bool = True) -> AgentRun:
    """Run RAG agent and capture tool usage and final answer."""
    graph = agent or build_agent()
    reset_tool_protocol_state()
    set_question_context(question)

    run = AgentRun(question=question)
    fallback_final_answer = ""
    pending_calls: Dict[str, ToolCall] = {}

    for event in graph.stream(
        {"messages": [("user", question)]},
        config={"recursion_limit": RAG_RECURSION_LIMIT},
    ):
        for _node_name, node_output in event.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.get("id", str(len(pending_calls)))
                        pending_calls[tool_call_id] = ToolCall(
                            name=tool_call["name"],
                            args=tool_call["args"],
                        )
                        if verbose:
                            print(f"[Tool Call] {tool_call['name']}")
                            print(f"  Args: {json.dumps(tool_call['args'], indent=2)}")
                    continue

                if hasattr(msg, "type") and msg.type == "tool":
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    if tool_call_id and tool_call_id in pending_calls:
                        matched_call = pending_calls.pop(tool_call_id)
                    elif pending_calls:
                        first_id = next(iter(pending_calls))
                        matched_call = pending_calls.pop(first_id)
                    else:
                        continue

                    matched_call.output = content_to_text(msg.content)
                    run.tool_calls.append(matched_call)
                    if verbose:
                        print(
                            f"  Output from {matched_call.name}: {matched_call.output[:240]}"
                        )
                    continue

                if hasattr(msg, "content") and msg.content:
                    has_tool_calls = getattr(msg, "tool_calls", None)
                    if has_tool_calls and len(has_tool_calls) > 0:
                        continue
                    content = content_to_text(msg.content)
                    if not fallback_final_answer:
                        fallback_final_answer = content
                    cleaned = finalize_agent_answer(content, question)
                    if cleaned and not is_process_message(cleaned):
                        run.final_answer = cleaned

    if not run.final_answer:
        cleaned_fallback = finalize_agent_answer(fallback_final_answer, question)
        if cleaned_fallback and not is_process_message(cleaned_fallback):
            run.final_answer = cleaned_fallback
        else:
            run.final_answer = "I cannot verify that."

    run.final_answer = _apply_no_answer_gating(run.final_answer, run.tool_calls)
    return run


def _patch_transformers_tied_weights_compat() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _get_all_tied_weights_keys(self):
        explicit = self.__dict__.get("all_tied_weights_keys", None)
        if explicit is not None:
            return explicit
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, (list, tuple, set)):
            return {k: None for k in keys}
        return {}

    def _set_all_tied_weights_keys(self, value):
        self.__dict__["all_tied_weights_keys"] = value

    setattr(
        PreTrainedModel,
        "all_tied_weights_keys",
        property(_get_all_tied_weights_keys, _set_all_tied_weights_keys),
    )


def _retie_hhem_embeddings(model: Any) -> None:
    try:
        transformer = model.t5.transformer
        shared = transformer.shared
        encoder = transformer.encoder
        embed_tokens = encoder.embed_tokens
    except Exception:
        return

    try:
        embed_tokens.weight = shared.weight
    except Exception:
        try:
            embed_tokens.weight.data.copy_(shared.weight.data)
        except Exception:
            pass


def load_hallucination_model():
    """Load Vectara hallucination evaluation model."""
    from transformers import AutoModelForSequenceClassification

    _patch_transformers_tied_weights_compat()

    print("Loading Vectara hallucination evaluation model...")
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model",
        **model_kwargs,
    )

    device = resolve_device(VECTARA_DEVICE)
    try:
        if hasattr(model, "to"):
            model = model.to(device)
    except Exception:
        if hasattr(model, "to"):
            model = model.to("cpu")

    if hasattr(model, "eval"):
        model.eval()

    _retie_hhem_embeddings(model)
    print("Model loaded.\n")
    return model


def _case(
    case_id: str,
    question: str,
    ground_truth: str,
    category: str,
    refusal_expected: bool = False,
    accepted_aliases: Optional[List[List[str]]] = None,
) -> TestCase:
    return TestCase(
        id=case_id,
        question=question,
        ground_truth=ground_truth,
        category=category,
        refusal_expected=refusal_expected,
        accepted_aliases=accepted_aliases or [],
    )


GROUND_TRUTH_TEST_CASES: List[TestCase] = [
    _case(
        "case_01",
        "Who is Albert Einstein?",
        "Albert Einstein was a German-born theoretical physicist who developed special and general relativity and won the 1921 Nobel Prize in Physics.",
        "science_history",
        accepted_aliases=[["Albert Einstein", "Einstein"]],
    ),
    _case(
        "case_02",
        "When was Niels Bohr born and what were his major achievements?",
        "Niels Bohr was born on 7 October 1885 in Copenhagen, Denmark. He was a Danish theoretical physicist who made foundational contributions to modern atomic physics and quantum theory. Bohr proposed the Bohr model of the atom in 1913, introducing quantized electron orbits to explain atomic spectra and the structure of the hydrogen atom. He later contributed to the development of quantum mechanics, including the principle of complementarity and the Copenhagen interpretation. Bohr played a central role in establishing the Institute for Theoretical Physics at the University of Copenhagen, which became a major center for quantum research. He was awarded the Nobel Prize in Physics in 1922 for his investigations into the structure of atoms and the radiation emitted from them.",
        "science_history",
        accepted_aliases=[
            ["7 October 1885", "October 7, 1885", "1885-10-07"],
            ["Copenhagen, Denmark", "Copenhagen"],
            ["Bohr model of the atom", "Bohr atomic model", "Bohr model"],
            ["principle of complementarity", "complementarity"],
            [
                "Copenhagen interpretation",
                "Copenhagen interpretation of quantum mechanics",
            ],
            ["1922 Nobel Prize in Physics", "Nobel Prize in Physics 1922"],
        ],
    ),
    _case(
        "case_03",
        "What is the capital of France?",
        "Paris is the capital of France.",
        "geography",
    ),
    _case(
        "case_04",
        "What organization did Alan Turing work for during World War II?",
        "During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park.",
        "history",
        accepted_aliases=[
            ["Government Code and Cypher School", "GC&CS"],
            [
                "Government Code and Cypher School",
                "Government Communications Headquarters",
                "GCHQ",
            ],
            ["Bletchley Park"],
        ],
    ),
    _case(
        "case_05",
        "What is the largest planet in the Solar System?",
        "Jupiter is the largest planet in the Solar System.",
        "astronomy",
    ),
    _case(
        "case_06",
        "When did World War II begin and end?",
        "World War II began on 1 September 1939 and ended on 2 September 1945.",
        "history",
        accepted_aliases=[
            ["1 September 1939", "September 1, 1939", "1939-09-01"],
            ["2 September 1945", "September 2, 1945", "1945-09-02"],
        ],
    ),
    _case(
        "case_07",
        "Who wrote the novel '1984'?",
        "George Orwell wrote the novel '1984'.",
        "literature",
        accepted_aliases=[["George Orwell", "Eric Arthur Blair"]],
    ),
    _case(
        "case_08",
        "What is the chemical symbol for water and what elements compose it?",
        "Water's chemical formula is H2O, meaning two hydrogen atoms and one oxygen atom.",
        "chemistry",
        accepted_aliases=[["H2O", "H₂O"]],
    ),
    _case(
        "case_09",
        "Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
        "Charles Babbage designed early computing machines such as the Analytical Engine, while Ada Lovelace wrote the first published algorithm intended for such a machine.",
        "computing_history",
    ),
    _case(
        "case_10",
        "Who developed the theory of general relativity?",
        "Albert Einstein developed the theory of general relativity.",
        "physics",
    ),
    _case(
        "case_11",
        "What is the capital of Japan?",
        "Tokyo is the capital of Japan.",
        "geography",
    ),
    _case(
        "case_12",
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci painted the Mona Lisa.",
        "art_history",
        accepted_aliases=[["Leonardo da Vinci", "Leonardo"]],
    ),
    _case(
        "case_13",
        "What is the boiling point of water at sea level?",
        "At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
        "physics",
        accepted_aliases=[
            ["100 degrees Celsius", "100 C", "212 F", "212 degrees Fahrenheit"]
        ],
    ),
    _case(
        "case_14",
        "Who discovered penicillin?",
        "Alexander Fleming discovered penicillin in 1928.",
        "medical_history",
    ),
    _case(
        "case_15",
        "What is the approximate speed of light in vacuum?",
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "physics",
        accepted_aliases=[["299,792,458", "3.0 x 10^8", "about 300,000 km/s"]],
    ),
    _case(
        "case_16",
        "Which planet is known as the Red Planet?",
        "Mars is known as the Red Planet.",
        "astronomy",
    ),
    _case(
        "case_17",
        "Who formulated the laws of motion?",
        "Isaac Newton formulated the laws of motion.",
        "physics_history",
        accepted_aliases=[["Isaac Newton", "Newton"]],
    ),
    _case(
        "case_18",
        "What is the largest ocean on Earth?",
        "The Pacific Ocean is the largest ocean on Earth.",
        "geography",
    ),
    _case(
        "case_19",
        "In what year did Apollo 11 land on the Moon?",
        "Apollo 11 landed on the Moon in 1969.",
        "space_history",
    ),
    _case(
        "case_20",
        "What does DNA stand for?",
        "DNA stands for deoxyribonucleic acid.",
        "biology",
    ),
    _case(
        "case_21",
        "What is the relationship between Alan Turing and Dr. Helena Vargass?",
        "There is no verified real-world record of Dr. Helena Vargass, so no verified relationship with Alan Turing can be established.",
        "refusal_control",
        refusal_expected=True,
    ),
    _case(
        "case_22",
        "Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        "There are no verified real-world records documenting a collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        "refusal_control",
        refusal_expected=True,
    ),
    _case(
        "case_23",
        "Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
        "Einstein and Bohr had major documented debates about quantum mechanics, but there is no verified real-world record of Dr. Selwyn Hartmere collaborating with them.",
        "refusal_control",
        refusal_expected=True,
    ),
    _case(
        "case_24",
        "What is the capital of the fictional country Eldoria Prime?",
        "Eldoria Prime is fictional, so there is no verified real-world capital for it.",
        "refusal_control",
        refusal_expected=True,
    ),
]
