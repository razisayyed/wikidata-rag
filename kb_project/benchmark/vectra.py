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
        "multi_hop_001",
        "What capital corresponds to the country where Frida Kahlo was born?",
        "Frida Kahlo was born in Coyoacán. Coyoacán is in Mexico. The capital of Mexico is Mexico City.",
        "multi_hop",
        accepted_aliases=[
            [
                "Mexico City",
                "Mexico D.F.",
                "Ciudad de México",
                "City of Mexico",
                "Mexico City, Mexico",
                "CDMX",
                "Mexico",
            ]
        ],
    ),
    _case(
        "multi_hop_002",
        "For The Great Gatsby, what is the capital of its author's citizenship country?",
        "The Great Gatsby was authored by F. Scott Fitzgerald. F. Scott Fitzgerald has country of citizenship United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_003",
        "What is the capital of the nation where Microsoft's founder was born?",
        "Microsoft was founded by Bill Gates. Bill Gates was born in Seattle. Seattle is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_004",
        "Identify the continent of Karlovac Gymnasium's country for Nikola Tesla.",
        "Nikola Tesla was educated at Karlovac Gymnasium. Karlovac Gymnasium is in Croatia. Croatia is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_005",
        "Which capital is tied to the country of birth of Mexico's head of state?",
        "The head of state of Mexico is Claudia Sheinbaum. Claudia Sheinbaum was born in Mexico City. Mexico City is in Mexico. The capital of Mexico is Mexico City.",
        "multi_hop",
        accepted_aliases=[
            [
                "Mexico City",
                "Mexico D.F.",
                "Ciudad de México",
                "City of Mexico",
                "Mexico City, Mexico",
                "CDMX",
                "Mexico",
            ]
        ],
    ),
    _case(
        "multi_hop_006",
        "Identify the capital of the nation associated with Congressional Gold Medal that George Washington received.",
        "George Washington received Congressional Gold Medal. Congressional Gold Medal is associated with United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_007",
        "Which capital belongs to the country of Ada Lovelace's birthplace?",
        "Ada Lovelace was born in London. London is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_008",
        "What capital is associated with the author's country in Don Quixote?",
        "Don Quixote was authored by Miguel de Cervantes. Miguel de Cervantes has country of citizenship Crown of Castile. The capital of Crown of Castile is Madrid.",
        "multi_hop",
        accepted_aliases=[["Madrid"]],
    ),
    _case(
        "multi_hop_009",
        "What is the capital of the country where the founder of Siemens was born?",
        "Siemens was founded by Werner von Siemens. Werner von Siemens was born in Lenthe. Lenthe is in Germany. The capital of Germany is Berlin.",
        "multi_hop",
        accepted_aliases=[["Berlin", "Berlin, Germany", "DE-BE"]],
    ),
    _case(
        "multi_hop_010",
        "Identify the continent of Science Faculty of Paris's country for Marie Curie.",
        "Marie Curie was educated at Science Faculty of Paris. Science Faculty of Paris is in France. France is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_011",
        "What is the capital of the country where the head of state of South Africa was born?",
        "The head of state of South Africa is Cyril Ramaphosa. Cyril Ramaphosa was born in Soweto. Soweto is in South Africa. The capital of South Africa is Pretoria.",
        "multi_hop",
        accepted_aliases=[
            [
                "Pretoria",
                "Pretoria, S. Africa",
                "Pretoria, S Africa",
                "Pretoria, Gauteng, South Africa",
                "Pretoria, Gauteng",
                "Pretoria, South Africa",
            ]
        ],
    ),
    _case(
        "multi_hop_012",
        "Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country?",
        "Alan Turing received Officer of the Order of the British Empire. Officer of the Order of the British Empire is associated with United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_013",
        "The birthplace of George Washington is in which country's capital city?",
        "George Washington was born in Westmoreland County. Westmoreland County is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_014",
        "For Don Quixote, what is the capital of its author's citizenship country?",
        "Don Quixote was authored by Miguel de Cervantes. Miguel de Cervantes has country of citizenship Crown of Castile. The capital of Crown of Castile is Valladolid.",
        "multi_hop",
        accepted_aliases=[["Valladolid", "Pucela"]],
    ),
    _case(
        "multi_hop_015",
        "What is the capital of the country where the founder of Meta was born?",
        "Meta was founded by Dustin Moskovitz. Dustin Moskovitz was born in Gainesville. Gainesville is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_016",
        "Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman.",
        "Richard Feynman was educated at Massachusetts Institute of Technology. Massachusetts Institute of Technology is in United States. United States is in North America.",
        "multi_hop",
        accepted_aliases=[
            [
                "North America",
                "NA",
                "Turtle Island",
                "North and Central America",
                "N. America",
                "North American continent",
                "North america",
                "North-America",
                "North America (continent)",
                "North America (region)",
                "North America (Americas)",
                "North American Continent",
                "N america",
                "003 (UN M.49 code)",
                "N America",
                "Amérique du Nord",
                "N Am",
                "North Am",
                "NoAm",
                "NOAM",
                "North Amer",
            ]
        ],
    ),
    _case(
        "multi_hop_017",
        "Name the capital city of the birth country of Brazil's head of state.",
        "The head of state of Brazil is Luiz Inácio Lula da Silva. Luiz Inácio Lula da Silva was born in Caetés. Caetés is in Brazil. The capital of Brazil is Brasília.",
        "multi_hop",
        accepted_aliases=[["Brasília"]],
    ),
    _case(
        "multi_hop_018",
        "Isaac Newton received Knight Bachelor; what is the capital of the award's country?",
        "Isaac Newton received Knight Bachelor. Knight Bachelor is associated with United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_019",
        "What is the capital of the country where Charles Darwin was born?",
        "Charles Darwin was born in The Mount. The Mount is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_020",
        "Identify the capital of the nation linked to the author of Moby-Dick.",
        "Moby-Dick was authored by Herman Melville. Herman Melville has country of citizenship United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_021",
        "Which capital corresponds to the founder-birth country of Cisco?",
        "Cisco was founded by Sandra Lerner. Sandra Lerner was born in California. California is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_022",
        "The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent?",
        "Albert Einstein was educated at ETH Zurich. ETH Zurich is in Switzerland. Switzerland is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_023",
        "What is the capital of the country where the head of state of Australia was born?",
        "The head of state of Australia is Charles III. Charles III was born in Buckingham Palace. Buckingham Palace is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_024",
        "Charles Darwin received Copley Medal; what is the capital of the award's country?",
        "Charles Darwin received Copley Medal. Copley Medal is associated with United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_025",
        "What capital corresponds to the country where Nelson Mandela was born?",
        "Nelson Mandela was born in Mvezo. Mvezo is in South Africa. The capital of South Africa is Bloemfontein.",
        "multi_hop",
        accepted_aliases=[["Bloemfontein"]],
    ),
    _case(
        "multi_hop_026",
        "Name the capital city of the author's country for The Brothers Karamazov.",
        "The Brothers Karamazov was authored by Fyodor Dostoyevsky. Fyodor Dostoyevsky has country of citizenship Russian Empire. The capital of Russian Empire is Saint Petersburg.",
        "multi_hop",
        accepted_aliases=[
            [
                "Saint Petersburg",
                "St. Petersburg",
                "Petrograd",
                "Leningrad",
                "Petersburg",
                "Sankt-Peterburg",
                "St Petersburg",
                "St.Petersburg",
            ]
        ],
    ),
    _case(
        "multi_hop_027",
        "Which capital corresponds to the founder-birth country of Samsung Electronics?",
        "Samsung Electronics was founded by Lee Byung-chul. Lee Byung-chul was born in Uiryeong County. Uiryeong County is in South Korea. The capital of South Korea is Seoul.",
        "multi_hop",
        accepted_aliases=[
            [
                "Seoul",
                "Seoul Special City",
                "Sŏul T'ŭkpyŏlsi",
                "Wiryeseong",
                "Namgyeong",
                "Hanseong",
                "Hanyang",
                "Keijō",
                "Keijou",
                "Gyeongseong",
            ]
        ],
    ),
    _case(
        "multi_hop_028",
        "Identify the continent of Charles University's country for Nikola Tesla.",
        "Nikola Tesla was educated at Charles University. Charles University is in Czech Republic. Czech Republic is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_029",
        "For United Kingdom, which capital belongs to its head of state's birth country?",
        "The head of state of United Kingdom is Charles III. Charles III was born in Buckingham Palace. Buckingham Palace is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_030",
        "Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru, received by Angela Merkel.",
        "Angela Merkel received Grand Cross of the Order of the Sun of Peru. Grand Cross of the Order of the Sun of Peru is associated with Peru. The capital of Peru is Lima.",
        "multi_hop",
        accepted_aliases=[["Lima", "City of the Kings"]],
    ),
    _case(
        "multi_hop_031",
        "Which capital belongs to the country of Abraham Lincoln's birthplace?",
        "Abraham Lincoln was born in Sinking Spring Farm. Sinking Spring Farm is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_032",
        "What is the capital of the nation where Intel's founder was born?",
        "Intel was founded by Robert Noyce. Robert Noyce was born in Burlington. Burlington is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_033",
        "Which continent includes the country where Stephen Hawking attended Trinity Hall?",
        "Stephen Hawking was educated at Trinity Hall. Trinity Hall is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_034",
        "The head of state of United States was born in a country with what capital?",
        "The head of state of United States is Donald Trump. Donald Trump was born in Jamaica Hospital Medical Center. Jamaica Hospital Medical Center is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_035",
        "What capital is associated with the country tied to Time Person of the Year received by Mahatma Gandhi?",
        "Mahatma Gandhi received Time Person of the Year. Time Person of the Year is associated with United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_036",
        "What is the capital of the country where Barack Obama was born?",
        "Barack Obama was born in Kapiolani Medical Center for Women and Children. Kapiolani Medical Center for Women and Children is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_037",
        "Identify the capital of the country containing the founder's birthplace of Intel.",
        "Intel was founded by Gordon Moore. Gordon Moore was born in San Francisco. San Francisco is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_038",
        "On which continent is the country where Isaac Newton studied at University of Cambridge?",
        "Isaac Newton was educated at University of Cambridge. University of Cambridge is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_039",
        "What is the capital of the country where the head of state of Spain was born?",
        "The head of state of Spain is Felipe VI of Spain. Felipe VI of Spain was born in Madrid. Madrid is in Spain. The capital of Spain is Madrid.",
        "multi_hop",
        accepted_aliases=[["Madrid"]],
    ),
    _case(
        "multi_hop_040",
        "Which capital belongs to the country connected to Willard Gibbs Award received by Marie Curie?",
        "Marie Curie received Willard Gibbs Award. Willard Gibbs Award is associated with United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_041",
        "What is the capital of the country where Grace Hopper was born?",
        "Grace Hopper was born in New York City. New York City is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_042",
        "What is the capital of the country where the founder of Nvidia was born?",
        "Nvidia was founded by Jensen Huang. Jensen Huang was born in Tainan. Tainan is in Taiwan. The capital of Taiwan is Taipei.",
        "multi_hop",
        accepted_aliases=[
            [
                "Taipei",
                "The City of Azaleas",
                "Taipei City",
                "Taibei",
                "City of Taipei",
                "Tanshoui",
                "Tai Pei",
            ]
        ],
    ),
    _case(
        "multi_hop_043",
        "Identify the continent of University of Cambridge's country for Niels Bohr.",
        "Niels Bohr was educated at University of Cambridge. University of Cambridge is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_044",
        "For Japan, which capital belongs to its head of state's birth country?",
        "The head of state of Japan is Naruhito. Naruhito was born in Hospital of the Imperial Household. Hospital of the Imperial Household is in Japan. The capital of Japan is Tokyo.",
        "multi_hop",
        accepted_aliases=[
            [
                "Tokyo",
                "Tōkyō",
                "Tôkyô",
                "Tokyo-to",
                "Tokyo Metropolitan prefecture",
                "Tōkyō-to",
                "Tôkyô-to",
                "Tokyo Metropolis",
                "Tokio",
                "Tokyo Prefecture",
                "Tokyo, Japan",
                "Tokei",
                "Tokyo (Japan)",
                "Edo",
                "Jedo",
                "Yedo",
            ]
        ],
    ),
    _case(
        "multi_hop_045",
        "What is the capital of the country associated with the award Angela Merkel received (Grand Cross of the Order of Prince Henry)?",
        "Angela Merkel received Grand Cross of the Order of Prince Henry. Grand Cross of the Order of Prince Henry is associated with Portugal. The capital of Portugal is Lisbon.",
        "multi_hop",
        accepted_aliases=[["Lisbon", "Lisboa"]],
    ),
    _case(
        "multi_hop_046",
        "Name the capital city of the nation in which Galileo Galilei was born.",
        "Galileo Galilei was born in Pisa. At the time of his birth in 1564, Pisa was part of the Duchy of Florence. The capital of the Duchy of Florence was Florence.",
        "multi_hop",
        accepted_aliases=[
            [
                "Florence",
                "Firenze",
                "Florentia",
                "City of Flowers",
                "The Flowering City",
            ]
        ],
    ),
    _case(
        "multi_hop_047",
        "What is the capital of the nation where Oracle Corporation's founder was born?",
        "Oracle Corporation was founded by Bob Miner. Bob Miner was born in Cicero. Cicero is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_048",
        "Identify the continent of King's College's country for Alan Turing.",
        "Alan Turing was educated at King's College. King's College is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_049",
        "For France, which capital belongs to its head of state's birth country?",
        "The head of state of France is Emmanuel Macron. Emmanuel Macron was born in Amiens. Amiens is in France. The capital of France is Paris.",
        "multi_hop",
        accepted_aliases=[["Paris", "City of Light", "City of Love", "Lutetia"]],
    ),
    _case(
        "multi_hop_050",
        "For Charles Darwin and the award Pour le Mérite for Sciences and Arts order, what is the country's capital?",
        "Charles Darwin received Pour le Mérite for Sciences and Arts order. Pour le Mérite for Sciences and Arts order is associated with Prussia. The capital of Prussia is Berlin.",
        "multi_hop",
        accepted_aliases=[["Berlin", "Berlin, Germany", "DE-BE"]],
    ),
    _case(
        "multi_hop_051",
        "Identify the capital of the country containing Winston Churchill's birth location.",
        "Winston Churchill was born in Blenheim Palace. Blenheim Palace is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_052",
        "What is the capital of the country where the founder of Tesla, Inc. was born?",
        "Tesla, Inc. was founded by Marc Tarpenning. Marc Tarpenning was born in Sacramento. Sacramento is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_053",
        "Name the continent of the country that contains University of Paris, where Marie Curie was educated.",
        "Marie Curie was educated at University of Paris. University of Paris is in France. France is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_054",
        "What capital corresponds to the birth-country of the current head of state of Netherlands?",
        "The head of state of Netherlands is William Alexander of the Netherlands. William Alexander of the Netherlands was born in University Medical Center Utrecht. University Medical Center Utrecht is in Netherlands. The capital of Netherlands is Amsterdam.",
        "multi_hop",
        accepted_aliases=[
            [
                "Amsterdam",
                "Mokum",
                "Amsterdam, North Holland",
                "Amsterdam, NL",
                "Amsterdam, Netherlands",
                "A'dam",
            ]
        ],
    ),
    _case(
        "multi_hop_055",
        "For Albert Einstein and the award Barnard Medal for Meritorious Service to Science, what is the country's capital?",
        "Albert Einstein received Barnard Medal for Meritorious Service to Science. Barnard Medal for Meritorious Service to Science is associated with United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_056",
        "The birthplace of Alan Turing is in which country's capital city?",
        "Alan Turing was born in Maida Vale. Maida Vale is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_057",
        "Name the capital city of the founder's birth country for Meta.",
        "Meta was founded by Mark Zuckerberg. Mark Zuckerberg was born in White Plains. White Plains is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_058",
        "Stephen Hawking studied at St Albans School in a country on which continent?",
        "Stephen Hawking was educated at St Albans School. St Albans School is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_059",
        "For Canada, which capital belongs to its head of state's birth country?",
        "The head of state of Canada is Charles III. Charles III was born in Buckingham Palace. Buckingham Palace is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_060",
        "What is the capital of the country associated with the award Albert Einstein received (Nobel Prize in Physics)?",
        "Albert Einstein received Nobel Prize in Physics. Nobel Prize in Physics is associated with Sweden. The capital of Sweden is Stockholm.",
        "multi_hop",
        accepted_aliases=[["Stockholm", "Sthlm", "STHLM"]],
    ),
    _case(
        "multi_hop_061",
        "What capital corresponds to the country where Mahatma Gandhi was born?",
        "Mahatma Gandhi was born in Porbandar. Porbandar is in India. The capital of India is New Delhi.",
        "multi_hop",
        accepted_aliases=[
            [
                "New Delhi",
                "Nayi Dilli",
                "New Delhi Municipal Council Area",
                "NDMC area",
                "Nai Dilli",
            ]
        ],
    ),
    _case(
        "multi_hop_062",
        "What is the capital of the nation where Sony Group's founder was born?",
        "Sony Group was founded by Akio Morita. Akio Morita was born in Nagoya. Nagoya is in Japan. The capital of Japan is Tokyo.",
        "multi_hop",
        accepted_aliases=[
            [
                "Tokyo",
                "Tōkyō",
                "Tôkyô",
                "Tokyo-to",
                "Tokyo Metropolitan prefecture",
                "Tōkyō-to",
                "Tôkyô-to",
                "Tokyo Metropolis",
                "Tokio",
                "Tokyo Prefecture",
                "Tokyo, Japan",
                "Tokei",
                "Tokyo (Japan)",
                "Edo",
                "Jedo",
                "Yedo",
            ]
        ],
    ),
    _case(
        "multi_hop_063",
        "Which continent includes the country where Marie Curie attended Flying University?",
        "Marie Curie was educated at Flying University. Flying University is in Congress Poland. Congress Poland is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_064",
        "For India, which capital belongs to its head of state's birth country?",
        "The head of state of India is Droupadi Murmu. Droupadi Murmu was born in Mayurbhanj district. Mayurbhanj district is in India. The capital of India is New Delhi.",
        "multi_hop",
        accepted_aliases=[
            [
                "New Delhi",
                "Nayi Dilli",
                "New Delhi Municipal Council Area",
                "NDMC area",
                "Nai Dilli",
            ]
        ],
    ),
    _case(
        "multi_hop_065",
        "What capital is associated with the country tied to Fellow of the Royal Society received by Winston Churchill?",
        "Winston Churchill received Fellow of the Royal Society. Fellow of the Royal Society is associated with United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_066",
        "Marie Curie was born in a country whose capital is what?",
        "Marie Curie was born in Warsaw. Warsaw is in Poland. The capital of Poland is Warsaw.",
        "multi_hop",
        accepted_aliases=[["Warsaw", "Warszawa"]],
    ),
    _case(
        "multi_hop_067",
        "What is the capital of the nation where Cisco's founder was born?",
        "Cisco was founded by Leonard Bosack. Leonard Bosack was born in Philadelphia. Philadelphia is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_068",
        "Name the continent of the country that contains Newnham College, where Rosalind Franklin was educated.",
        "Rosalind Franklin was educated at Newnham College. Newnham College is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_069",
        "For Italy, which capital belongs to its head of state's birth country?",
        "The head of state of Italy is ሴርጆ ማታሬላ. ሴርጆ ማታሬላ was born in Palermo. Palermo is in Italy. The capital of Italy is Rome.",
        "multi_hop",
        accepted_aliases=[
            ["Rome", "The Eternal City", "Roma", "Rome, Italy", "City of Seven Hills"]
        ],
    ),
    _case(
        "multi_hop_070",
        "For Martin Luther King Jr. and the award Jawaharlal Nehru Award for International Understanding, what is the country's capital?",
        "Martin Luther King Jr. received Jawaharlal Nehru Award for International Understanding. Jawaharlal Nehru Award for International Understanding is associated with India. The capital of India is New Delhi.",
        "multi_hop",
        accepted_aliases=[
            [
                "New Delhi",
                "Nayi Dilli",
                "New Delhi Municipal Council Area",
                "NDMC area",
                "Nai Dilli",
            ]
        ],
    ),
    _case(
        "multi_hop_071",
        "Identify the capital of the country containing Martin Luther King Jr.'s birth location.",
        "Martin Luther King Jr. was born in Atlanta. Atlanta is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_072",
        "What is the capital of the country where the founder of Intel was born?",
        "Intel was founded by Andrew Grove. Andrew Grove was born in Budapest. Budapest is in Hungary. The capital of Hungary is Budapest.",
        "multi_hop",
        accepted_aliases=[
            [
                "Budapest",
                "Buda Pest",
                "Buda-Pest",
                "Budapešť",
                "Budapesta",
                "Budapeszt",
                "Buda",
                "Ofen",
                "Budín",
                "Budim",
                "Budon",
                "Pest",
                "Pešť",
                "Pešta",
                "Alt-Ofen",
                "Budapest, Hungary",
                "Buda-Pesth",
            ]
        ],
    ),
    _case(
        "multi_hop_073",
        "Richard Feynman studied at Princeton University in a country on which continent?",
        "Richard Feynman was educated at Princeton University. Princeton University is in United States. United States is in North America.",
        "multi_hop",
        accepted_aliases=[
            [
                "North America",
                "NA",
                "Turtle Island",
                "North and Central America",
                "N. America",
                "North American continent",
                "North america",
                "North-America",
                "North America (continent)",
                "North America (region)",
                "North America (Americas)",
                "North American Continent",
                "N america",
                "003 (UN M.49 code)",
                "N America",
                "Amérique du Nord",
                "N Am",
                "North Am",
                "NoAm",
                "NOAM",
                "North Amer",
            ]
        ],
    ),
    _case(
        "multi_hop_074",
        "What is the capital of the country where the head of state of South Africa was born?",
        "The head of state of South Africa is Cyril Ramaphosa. Cyril Ramaphosa was born in Soweto. Soweto is in South Africa. The capital of South Africa is Bloemfontein.",
        "multi_hop",
        accepted_aliases=[["Bloemfontein"]],
    ),
    _case(
        "multi_hop_075",
        "What capital is associated with the country tied to Gold Medal of the Royal Astronomical Society received by Albert Einstein?",
        "Albert Einstein received Gold Medal of the Royal Astronomical Society. Gold Medal of the Royal Astronomical Society is associated with United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_076",
        "What capital corresponds to the country where Isaac Newton was born?",
        "Isaac Newton was born in Woolsthorpe Manor. Woolsthorpe Manor is in United Kingdom. The capital of United Kingdom is London.",
        "multi_hop",
        accepted_aliases=[
            [
                "London",
                "London, UK",
                "London, United Kingdom",
                "London, England",
                "London UK",
                "London U.K.",
                "Londinium",
                "Loñ",
                "Lundenwic",
                "Londinio",
                "Londini",
                "Londiniensium",
                "Augusta",
                "Trinovantum",
                "Kaerlud",
                "Karelundein",
                "Lunden",
                "Big Smoke",
                "the Big Smoke",
                "Lundenburh",
                "Lundenburgh",
                "Llyn Dain",
                "Llan Dian",
                "Londinion",
                "Loniniensi",
                "Lon.",
                "Loñ.",
                "Lond.",
                "LDN",
            ]
        ],
    ),
    _case(
        "multi_hop_077",
        "The founder of Google was born in a country whose capital is what?",
        "Google was founded by Larry Page. Larry Page was born in East Lansing. East Lansing is in United States. The capital of United States is Washington, D.C..",
        "multi_hop",
        accepted_aliases=[
            [
                "Washington, D.C.",
                "Washington, District of Columbia",
                "D.C. Washington",
                "The District",
                "District of Columbia",
                "City of Washington, D.C.",
                "Washington City, D.C.",
                "Nation's Capital (D.C.)",
                "Federal City (D.C.)",
                "Columbia District",
            ]
        ],
    ),
    _case(
        "multi_hop_078",
        "The country of Stephen Hawking's alma mater, University College, Oxford, lies on what continent?",
        "Stephen Hawking was educated at University College, Oxford. University College, Oxford is in United Kingdom. United Kingdom is in Europe.",
        "multi_hop",
        accepted_aliases=[
            [
                "Europe",
                "European continent",
                "The Old Continent",
                "European Continent",
                "European Peninsula",
                "Old Continent",
            ]
        ],
    ),
    _case(
        "multi_hop_079",
        "Which capital is tied to the country of birth of Germany's head of state?",
        "The head of state of Germany is Frank-Walter Steinmeier. Frank-Walter Steinmeier was born in Detmold. Detmold is in Germany. The capital of Germany is Berlin.",
        "multi_hop",
        accepted_aliases=[["Berlin", "Berlin, Germany", "DE-BE"]],
    ),
    _case(
        "multi_hop_080",
        "Identify the capital of the nation associated with Nobel Prize in Chemistry that Marie Curie received.",
        "Marie Curie received Nobel Prize in Chemistry. Nobel Prize in Chemistry is associated with Sweden. The capital of Sweden is Stockholm.",
        "multi_hop",
        accepted_aliases=[["Stockholm", "Sthlm", "STHLM"]],
    ),
    # _case(
    #     "multi_hop_01",
    #     "What is the capital of the country where Tesla was born?",
    #     "Nikola Tesla was born in Croatia. The capital of Croatia is Zagreb.",
    #     "multi_hop",
    #     accepted_aliases=[["Zagreb"]],  # Accept just the capital name as well
    # ),
    # _case(
    #     "multi_hop_02",
    #     "What is the capital of the country where Frida Kahlo was born?",
    #     "Frida Kahlo was born in Mexico. The capital of Mexico is Mexico City.",
    #     "multi_hop",
    #     accepted_aliases=[["Mexico City", "Mexico D.F.", "Ciudad de México"]],
    # ),
    # _case(
    #     "multi_hop_03",
    #     "What is the capital of the country where Albert Einstein was born?",
    #     "Albert Einstein was born in Germany. The capital of Germany is Berlin.",
    #     "multi_hop",
    #     accepted_aliases=[["Berlin"]],
    # ),
    # _case(
    #     "multi_hop_04",
    #     "Which currency is used in the country where Leonardo da Vinci was born?",
    #     "The currency used in the country where Leonardo da Vinci was born is Euro.",
    #     "multi_hop",
    #     accepted_aliases=[["Euro", "EUR"]],
    # ),
    # _case(
    #     "multi_hop_05",
    #     "Who is the founder of the company that created the iPhone?",
    #     "The iPhone was created by Apple. Apple was co-founded by Steve Jobs (along with Steve Wozniak and Ronald Wayne).",
    #     "multi_hop",
    #     accepted_aliases=[["Steve Jobs", "Steven Jobs"]],
    # ),
    # _case(
    #     "multi_hop_06",
    #     "What is the capital of the country whose official language is Japanese?",
    #     "Japanese is the official language of Japan. The capital of Japan is Tokyo.",
    #     "multi_hop",
    #     accepted_aliases=[["Tokyo"]],
    # ),
    # _case(
    #     "multi_hop_07",
    #     "Which continent is the country located on where Nelson Mandela was born?",
    #     "Nelson Mandela was born in South Africa. South Africa is located on the continent of Africa.",
    #     "multi_hop",
    #     accepted_aliases=[["Africa"]],
    # ),
    # _case(
    #     "multi_hop_08",
    #     "Who is the author of the book that features the character Harry Potter?",
    #     "Harry Potter is a character in books written by J. K. Rowling. The author is J. K. Rowling.",
    #     "multi_hop",
    #     accepted_aliases=[["J. K. Rowling", "Joanne Rowling"]],
    # ),
    # _case(
    #     "multi_hop_09",
    #     "What is the capital of the country where the Amazon River is primarily located?",
    #     "The Amazon River is primarily located in Brazil. The capital of Brazil is Brasília.",
    #     "multi_hop",
    #     accepted_aliases=[["Brasília", "Brasilia"]],
    # ),
    # _case(
    #     "multi_hop_10",
    #     "Which planet is named after the Roman god who is the god of war?",
    #     "Mars is named after the Roman god Mars, who is the god of war. The planet is Mars.",
    #     "multi_hop",
    #     accepted_aliases=[["Mars"]],
    # ),
    # _case(
    #     "multi_hop_11",
    #     "What is the capital of the country where the company Samsung is headquartered?",
    #     "Samsung is headquartered in South Korea. The capital of South Korea is Seoul.",
    #     "multi_hop",
    #     accepted_aliases=[["Seoul"]],
    # ),
    # _case(
    #     "multi_hop_12",
    #     "Which ocean borders the country where Mahatma Gandhi was born?",
    #     "Mahatma Gandhi was born in India. India borders the Indian Ocean.",
    #     "multi_hop",
    #     accepted_aliases=[["Indian Ocean"]],
    # ),
    # _case(
    #     "multi_hop_13",
    #     "What is the capital of the country where Cristiano Ronaldo was born?",
    #     "Cristiano Ronaldo was born in Portugal. The capital of Portugal is Lisbon.",
    #     "multi_hop",
    #     accepted_aliases=[["Lisbon"]],
    # ),
    # _case(
    #     "multi_hop_14",
    #     "Who discovered the element named after the planet Uranus?",
    #     "Uranium is named after the planet Uranus. Uranium was discovered by Martin Heinrich Klaproth.",
    #     "multi_hop",
    #     accepted_aliases=[["Martin Heinrich Klaproth", "Klaproth"]],
    # ),
    # _case(
    #     "multi_hop_15",
    #     "What is the Capital of the Country where the person who discovered penicillin was born?",
    #     "Alexander Fleming discovered penicillin in 1928. He was born in Scotland. The capital of Scotland is Edinburgh.",
    #     "multi_hop (3 hops)",
    # ),
    # _case(
    #     "multi_answer_01",
    #     "What are Nelson Mandela's occupations?",
    #     "Nelson Mandela's occupations are autobiographer, lawyer, political activist, political prisoner, politician, and screenwriter.",
    #     "multi_answer",
    # ),
    # _case(
    #     "multi_answer_02",
    #     "What are Ada Lovelace's occupations?",
    #     "Ada Lovelace's occupations are computer scientist, engineer, inventor, mathematician, poet, programmer, translator, and writer.",
    #     "multi_answer",
    # ),
    # # _case(
    # #     "case_01",
    # #     "Who is Albert Einstein?",
    # #     "Albert Einstein was a German-born theoretical physicist who developed special and general relativity and won the 1921 Nobel Prize in Physics.",
    # #     "science_history",
    # #     accepted_aliases=[["Albert Einstein", "Einstein"]],
    # # ),
    # # _case(
    # #     "case_02",
    # #     "When was Niels Bohr born and what were his major achievements?",
    # #     "Niels Bohr was born on 7 October 1885 in Copenhagen, Denmark. He was a Danish theoretical physicist who made foundational contributions to modern atomic physics and quantum theory. Bohr proposed the Bohr model of the atom in 1913, introducing quantized electron orbits to explain atomic spectra and the structure of the hydrogen atom. He later contributed to the development of quantum mechanics, including the principle of complementarity and the Copenhagen interpretation. Bohr played a central role in establishing the Institute for Theoretical Physics at the University of Copenhagen, which became a major center for quantum research. He was awarded the Nobel Prize in Physics in 1922 for his investigations into the structure of atoms and the radiation emitted from them.",
    # #     "science_history",
    # #     accepted_aliases=[
    # #         [
    # #             "7 October 1885",
    # #             "October 7, 1885",
    # #             "1885-10-07",
    # #         ],
    # #         ["Copenhagen, Denmark", "Copenhagen"],
    # #         ["Bohr model of the atom", "Bohr atomic model", "Bohr model"],
    # #         ["principle of complementarity", "complementarity"],
    # #         ["correspondence principle in atomic physics", "correspondence principle"],
    # #         [
    # #             "Copenhagen interpretation",
    # #             "Copenhagen interpretation of quantum mechanics",
    # #         ],
    # #         ["1922 Nobel Prize in Physics", "Nobel Prize in Physics 1922"],
    # #     ],
    # # ),
    # # _case(
    # #     "case_03",
    # #     "What is the capital of France?",
    # #     "Paris is the capital of France.",
    # #     "geography",
    # # ),
    # # _case(
    # #     "case_04",
    # #     "What organization did Alan Turing work for during World War II?",
    # #     "During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park.",
    # #     "history",
    # #     accepted_aliases=[
    # #         ["Government Code and Cypher School", "GC&CS"],
    # #         [
    # #             "Government Code and Cypher School",
    # #             "Government Communications Headquarters",
    # #             "GCHQ",
    # #         ],
    # #         ["Bletchley Park"],
    # #     ],
    # # ),
    # # _case(
    # #     "case_05",
    # #     "When did World War II begin and end?",
    # #     "World War II began on 1 September 1939 and ended on 2 September 1945.",
    # #     "history",
    # #     accepted_aliases=[
    # #         ["1 September 1939", "September 1, 1939", "1939-09-01"],
    # #         ["2 September 1945", "September 2, 1945", "1945-09-02"],
    # #     ],
    # # ),
    # # _case(
    # #     "case_07",
    # #     "What is the chemical symbol for water and what elements compose it?",
    # #     "Water's chemical formula is H2O, meaning two hydrogen atoms and one oxygen atom.",
    # #     "chemistry",
    # #     accepted_aliases=[["H2O", "H₂O"]],
    # # ),
    # # _case(
    # #     "case_09",
    # #     "Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
    # #     "Charles Babbage designed early computing machines such as the Analytical Engine, while Ada Lovelace wrote the first published algorithm intended for such a machine.",
    # #     "computing_history",
    # # ),
    # # _case(
    # #     "case_12",
    # #     "Who painted the Mona Lisa?",
    # #     "Leonardo da Vinci painted the Mona Lisa.",
    # #     "art_history",
    # #     accepted_aliases=[["Leonardo da Vinci", "Leonardo"]],
    # # ),
    # # _case(
    # #     "case_13",
    # #     "What is the boiling point of water at sea level?",
    # #     "At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
    # #     "physics",
    # #     accepted_aliases=[
    # #         ["100 degrees Celsius", "100 C", "212 F", "212 degrees Fahrenheit"]
    # #     ],
    # # ),
    # # _case(
    # #     "case_16",
    # #     "Which planet is known as the Red Planet?",
    # #     "Mars is known as the Red Planet.",
    # #     "astronomy",
    # # ),
    # # _case(
    # #     "case_19",
    # #     "In what year did Apollo 11 land on the Moon?",
    # #     "Apollo 11 landed on the Moon in 1969.",
    # #     "space_history",
    # # ),
    # # _case(
    # #     "case_21",
    # #     "What is the relationship between Alan Turing and Dr. Helena Vargass?",
    # #     "There is no verified real-world record of Dr. Helena Vargass, so no verified relationship with Alan Turing can be established.",
    # #     "refusal_control",
    # #     refusal_expected=True,
    # # ),
    # # _case(
    # #     "case_22",
    # #     "Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
    # #     "There are no verified real-world records documenting a collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
    # #     "refusal_control",
    # #     refusal_expected=True,
    # # ),
]
