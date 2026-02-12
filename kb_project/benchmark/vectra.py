"""
Test Wikidata Agent for Hallucinations
======================================
Uses the Vectara hallucination evaluation model to check if the agent's
responses are grounded in the Wikidata facts it retrieves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Agent imports
# ─────────────────────────────────────────────────────────────────────────────
from ..settings import RAG_RECURSION_LIMIT, VECTARA_DEVICE, resolve_device
from ..utils.messages import content_to_text
from ..wikidata_rag_agent import build_agent, finalize_agent_answer, is_process_message
from ..tools.tool_protocol_state import reset_tool_protocol_state

# ─────────────────────────────────────────────────────────────────────────────
# Data structures for capturing agent execution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """Represents a single tool invocation."""

    name: str
    args: Dict[str, Any]
    output: str = ""


@dataclass
class AgentRun:
    """Captures a full agent execution for evaluation."""

    question: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: str = ""

    @property
    def retrieved_context(self) -> str:
        """Combine all tool outputs as the 'context' for hallucination check."""
        parts = []
        for tc in self.tool_calls:
            parts.append(f"[Tool: {tc.name}]\n{tc.output}")
        return "\n\n".join(parts)

    @property
    def sanitized_retrieved_context(self) -> str:
        """
        Context sanitized for faithfulness scoring.

        Removes candidate-list chatter and instruction/meta fragments while
        keeping concrete retrieved facts and hard no-candidate signals.
        """
        parts = []
        for tc in self.tool_calls:
            cleaned = sanitize_tool_output(tc.name, tc.output)
            if cleaned:
                parts.append(f"[Tool: {tc.name}]\n{cleaned}")
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
    """Sanitize individual tool output for retrieval-faithfulness evaluation."""
    clean_output = _strip_instruction_lines(output or "")
    if not clean_output:
        return ""

    if tool_name == "search_entity_candidates":
        # Candidate rankings are disambiguation hints, not factual evidence.
        if "NO CANDIDATES FOUND" in clean_output:
            for line in clean_output.splitlines():
                if "NO CANDIDATES FOUND" in line:
                    return line.strip()
            return "NO CANDIDATES FOUND"
        return ""

    return clean_output


# ─────────────────────────────────────────────────────────────────────────────
# Run agent and capture context + response
# ─────────────────────────────────────────────────────────────────────────────


def run_agent_with_capture(question: str, agent=None, verbose: bool = True) -> AgentRun:
    """
    Execute the Wikidata agent and capture:
      - All tool calls (name, args, outputs)
      - The final response

    Returns an AgentRun object suitable for hallucination evaluation.
    """
    # Set question context for entity disambiguation (used by selector LLM)
    # set_question_context(question)

    graph = agent or build_agent()
    reset_tool_protocol_state()
    run = AgentRun(question=question)
    fallback_final_answer = ""

    # Track pending tool calls by their ID (supports multiple concurrent calls)
    pending_tool_calls: Dict[str, ToolCall] = {}

    if verbose:
        print("\n" + "=" * 60)
        print("Running agent...")
        print("=" * 60 + "\n")

    for event in graph.stream(
        {"messages": [("user", question)]},
        config={"recursion_limit": RAG_RECURSION_LIMIT},
    ):
        for node_name, node_output in event.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                # Agent emits tool call(s)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_id = tc.get("id", str(len(pending_tool_calls)))
                        pending_tool_calls[tool_call_id] = ToolCall(
                            name=tc["name"],
                            args=tc["args"],
                        )
                        if verbose:
                            print(f"[Tool Call] {tc['name']}")
                            print(f"  Args: {json.dumps(tc['args'], indent=4)}")

                # Tool response - match by tool_call_id
                elif hasattr(msg, "type") and msg.type == "tool":
                    tool_call_id = getattr(msg, "tool_call_id", None)

                    if tool_call_id and tool_call_id in pending_tool_calls:
                        # Match response to its tool call by ID
                        matched_call = pending_tool_calls.pop(tool_call_id)
                        matched_call.output = content_to_text(msg.content)
                        run.tool_calls.append(matched_call)
                    elif pending_tool_calls:
                        # Fallback: pop the first pending call (for older LangGraph versions)
                        first_id = next(iter(pending_tool_calls))
                        matched_call = pending_tool_calls.pop(first_id)
                        matched_call.output = content_to_text(msg.content)
                        run.tool_calls.append(matched_call)

                    if verbose:
                        tool_text = content_to_text(msg.content)
                        snippet = tool_text[:300] + (
                            "..." if len(tool_text) > 300 else ""
                        )
                        print(f"  Output: {snippet}\n")

                # Final answer (AI message without tool calls)
                elif hasattr(msg, "content") and msg.content:
                    has_tool_calls = getattr(msg, "tool_calls", None)
                    if not has_tool_calls or len(has_tool_calls) == 0:
                        content = content_to_text(msg.content)
                        if not fallback_final_answer:
                            fallback_final_answer = content
                        cleaned = finalize_agent_answer(content, question)
                        if cleaned and not is_process_message(cleaned):
                            run.final_answer = cleaned

    if verbose:
        print("=" * 60)
        print(f"Agent finished. Captured {len(run.tool_calls)} tool call(s).")
        print("=" * 60 + "\n")

    if not run.final_answer:
        cleaned_fallback = finalize_agent_answer(fallback_final_answer, question)
        run.final_answer = cleaned_fallback or "I cannot verify that."
    return run


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination evaluation using Vectara model
# ─────────────────────────────────────────────────────────────────────────────


def _patch_transformers_tied_weights_compat() -> None:
    """
    Compatibility patch for custom HF models that only define `_tied_weights_keys`.

    Newer transformers code paths may access `all_tied_weights_keys`. Some remote-code
    model classes still rely on the older private field only.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _get_all_tied_weights_keys(self):  # type: ignore[no-redef]
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

    def _set_all_tied_weights_keys(self, value):  # type: ignore[no-redef]
        self.__dict__["all_tied_weights_keys"] = value

    setattr(
        PreTrainedModel,
        "all_tied_weights_keys",
        property(_get_all_tied_weights_keys, _set_all_tied_weights_keys),
    )


def _retie_hhem_embeddings(model: Any) -> None:
    """
    Repair missing embedding tie for HHEM custom model on some transformers versions.

    We observed checkpoints loading with:
      t5.transformer.encoder.embed_tokens.weight -> MISSING
    while t5.transformer.shared.weight is loaded.
    """
    try:
        transformer = model.t5.transformer
        shared = transformer.shared
        encoder = transformer.encoder
        embed_tokens = encoder.embed_tokens
    except Exception:
        return

    # Make encoder embedding share the same parameter object as shared embeddings.
    try:
        embed_tokens.weight = shared.weight
    except Exception:
        # Fallback to value copy if direct tying fails.
        try:
            embed_tokens.weight.data.copy_(shared.weight.data)
        except Exception:
            pass


def _sanity_check_hhem_model(model: Any) -> None:
    """
    Quick health check using known examples from model card.

    If the model returns almost identical scores, print a warning.
    """
    try:
        pairs = [
            ("The capital of France is Berlin.", "The capital of France is Paris."),
            ("I am in California", "I am in United States."),
        ]
        scores = model.predict(pairs)
        s0 = float(scores[0].item() if hasattr(scores[0], "item") else scores[0])
        s1 = float(scores[1].item() if hasattr(scores[1], "item") else scores[1])
        if abs(s0 - s1) < 0.02:
            print(
                "Warning: Vectara model sanity-check scores are very close "
                f"({s0:.4f} vs {s1:.4f}). Results may be unreliable."
            )
    except Exception:
        # Non-fatal: keep benchmark running.
        pass


def load_hallucination_model():
    """
    Load the Vectara hallucination evaluation model.
    Returns the model ready for prediction.
    """
    from transformers import AutoModelForSequenceClassification

    _patch_transformers_tied_weights_compat()

    print("Loading Vectara hallucination evaluation model...")
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
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
    except Exception as exc:
        print(
            f"Warning: could not move Vectara model to device '{device}' ({exc}). "
            "Using CPU."
        )
        if hasattr(model, "to"):
            model = model.to("cpu")
        device = "cpu"

    if hasattr(model, "eval"):
        model.eval()

    _retie_hhem_embeddings(model)
    _sanity_check_hhem_model(model)
    print(f"Vectara model device: {device}")
    print("Model loaded.\n")
    return model


def evaluate_hallucination(
    context: str,
    response: str,
    model,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate if *response* is grounded in *context*.

    Args:
        context: The retrieved Wikidata facts (tool outputs).
        response: The agent's final answer.
        model: Loaded Vectara model.
        threshold: Score below this is flagged as potential hallucination.

    Returns:
        dict with score, is_hallucination flag, and interpretation.
    """
    # Model expects list of [context, response] pairs
    score = model.predict([[context, response]])[0]

    is_hallucination = score < threshold

    return {
        "score": float(score),
        "threshold": threshold,
        "is_hallucination": is_hallucination,
        "interpretation": (
            "POTENTIAL HALLUCINATION: Response may contain unsupported claims."
            if is_hallucination
            else "GROUNDED: Response appears consistent with retrieved context."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined test runner
# ─────────────────────────────────────────────────────────────────────────────


def test_agent(
    question: str,
    hallucination_model=None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full test pipeline:
      1. Run agent on question
      2. Capture tool outputs (context) and final answer (response)
      3. Score with hallucination model

    Args:
        question: User question to test.
        hallucination_model: Preloaded model (will load if None).
        threshold: Hallucination threshold.
        verbose: Print step-by-step info.

    Returns:
        dict with question, context, response, and evaluation results.
    """
    # Run agent
    run = run_agent_with_capture(question, verbose=verbose)

    context = run.retrieved_context
    response = run.final_answer

    if verbose:
        print("\n" + "-" * 60)
        print("RETRIEVED CONTEXT (Wikidata tool outputs):")
        print("-" * 60)
        print(context[:2000] + ("..." if len(context) > 2000 else ""))
        print()

        print("-" * 60)
        print("AGENT FINAL ANSWER:")
        print("-" * 60)
        print(response)
        print()

    # Evaluate
    if hallucination_model is None:
        hallucination_model = load_hallucination_model()

    eval_result = evaluate_hallucination(
        context=context,
        response=response,
        model=hallucination_model,
        threshold=threshold,
    )

    if verbose:
        print("-" * 60)
        print("HALLUCINATION EVALUATION:")
        print("-" * 60)
        print(f"  Score: {eval_result['score']:.4f}")
        print(f"  Threshold: {eval_result['threshold']}")
        print(f"  Result: {eval_result['interpretation']}")
        print()

    return {
        "question": question,
        "context": context,
        "response": response,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "output": tc.output}
            for tc in run.tool_calls
        ],
        "evaluation": eval_result,
    }


def run_test_suite(
    questions: List[str],
    threshold: float = 0.5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple questions through the test pipeline.
    Loads the hallucination model once and reuses it.
    """
    model = load_hallucination_model()
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'#' * 60}")
        print(f"# TEST {i}/{len(questions)}")
        print(f"{'#' * 60}")
        print(f"Question: {question}\n")

        result = test_agent(
            question=question,
            hallucination_model=model,
            threshold=threshold,
            verbose=verbose,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    hallucinated = sum(1 for r in results if r["evaluation"]["is_hallucination"])
    grounded = len(results) - hallucinated

    for i, r in enumerate(results, 1):
        status = (
            "❌ HALLUCINATION" if r["evaluation"]["is_hallucination"] else "✅ GROUNDED"
        )
        score = r["evaluation"]["score"]
        q_short = (
            r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
        )
        print(f"  {i}. {status} (score={score:.3f}) — {q_short}")

    print(
        f"\nTotal: {grounded}/{len(results)} grounded, {hallucinated}/{len(results)} potential hallucinations"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Test Cases
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """A benchmark case with concise reference answer and optional structure."""

    __test__ = False

    question: str
    ground_truth: str
    description: str = ""
    key_facts: List[str] = field(default_factory=list)
    accepted_aliases: List[List[str]] = field(default_factory=list)
    refusal_expected: bool = False


# The ground truth should contain ONLY factual, verifiable information
# Ground truth scope should MATCH the question scope (simple question = simple answer)
GROUND_TRUTH_TEST_CASES: List[TestCase] = [
    TestCase(
        question="Who is Albert Einstein?",
        ground_truth="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist known for developing the theories of special and general relativity, making foundational contributions to modern physics, and winning the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect.",
        description="Comprehensive biographical and scientific identity question about Albert Einstein",
        key_facts=[
            "Full name: Albert Einstein.",
            "Profession: Theoretical physicist.",
            "Birth date: 14 March 1879.",
            "Birthplace: Ulm, Kingdom of Württemberg, German Empire.",
            "Death date: 18 April 1955.",
            "Death place: Princeton, New Jersey, United States.",
            "Cause of death: Rupture of an abdominal aortic aneurysm.",
            "Age at death: 76.",
            "Born into a Jewish family.",
            "Widely regarded as one of the most influential physicists in history.",
            "Major contributor to modern physics.",
            "Developed the theory of special relativity (1905).",
            "Developed the theory of general relativity (1915).",
            "Special relativity introduced a new understanding of space and time.",
            "General relativity describes gravity as curvature of spacetime.",
            "Formulated mass–energy equivalence.",
            "Mass–energy equivalence equation: E = mc^2.",
            "Explained the photoelectric effect using light quanta (photons).",
            "Provided theoretical explanation of Brownian motion.",
            "Work on Brownian motion supported the existence of atoms and molecules.",
            "Contributed to quantum theory.",
            "Contributed to statistical mechanics.",
            "Contributed to cosmology.",
            "Worked on unified field theory attempts.",
            "Published four major scientific papers in 1905.",
            "1905 papers covered the photoelectric effect, Brownian motion, special relativity, and mass–energy equivalence.",
            "The year 1905 is known as his annus mirabilis (miracle year).",
            "Awarded the Nobel Prize in Physics in 1921.",
            "Nobel Prize awarded for explanation of the photoelectric effect and contributions to theoretical physics.",
            "Nobel Prize was not awarded for relativity.",
            "Studied at the Swiss Federal Polytechnic (ETH Zurich).",
            "Graduated from ETH Zurich in 1900.",
            "Worked at the Swiss Patent Office in Bern from 1902 to 1909.",
            "Held academic positions at the University of Zurich.",
            "Held academic position at Charles University in Prague.",
            "Held academic position at ETH Zurich.",
            "Held academic position at the University of Berlin.",
            "Emigrated from Germany in 1933 due to the rise of Adolf Hitler and Nazi anti-Jewish policies.",
            "Moved to the United States in 1933.",
            "Worked at the Institute for Advanced Study in Princeton from 1933 until his death.",
            "Born a citizen of the German Empire.",
            "Renounced German citizenship in 1896 and became stateless.",
            "Became a Swiss citizen in 1901.",
            "Held German citizenship again from 1914 to 1933.",
            "Became a United States citizen in 1940.",
            "Retained Swiss citizenship after becoming a U.S. citizen.",
            "Signed a 1939 letter to U.S. President Franklin D. Roosevelt warning about potential nuclear weapons development in Nazi Germany.",
            "The 1939 letter is commonly known as the Einstein–Szilárd letter.",
            "Supported civil rights in the United States.",
            "Spoke publicly against racism in the United States.",
            "His theories remain foundational to modern physics and cosmology.",
            "His name is commonly used as a synonym for genius in popular culture.",
        ],
        accepted_aliases=[
            ["Albert Einstein", "Einstein"],
            ["14 March 1879", "March 14, 1879", "1879-03-14"],
            ["18 April 1955", "April 18, 1955", "1955-04-18"],
            ["Ulm, Kingdom of Württemberg, German Empire", "Ulm, Germany", "Ulm"],
            [
                "Princeton, New Jersey, United States",
                "Princeton, New Jersey",
                "Princeton",
            ],
            ["E = mc^2", "E=mc^2", "mass-energy equivalence"],
            ["1921 Nobel Prize in Physics", "Nobel Prize in Physics 1921"],
            ["special relativity", "theory of special relativity"],
            ["general relativity", "theory of general relativity"],
            ["Institute for Advanced Study", "IAS Princeton"],
            ["annus mirabilis", "miracle year 1905"],
        ],
    ),
    TestCase(
        question="When was Niels Bohr born and what were his major achievements?",
        ground_truth="Niels Bohr (7 October 1885 - 18 November 1962) was a Danish physicist who made foundational contributions to atomic structure and quantum theory, developed the Bohr model of the atom, and received the 1922 Nobel Prize in Physics for his work on the structure of atoms and radiation.",
        description="Biographical and scientific achievements question about Niels Bohr",
        key_facts=[
            "Full name: Niels Henrik David Bohr.",
            "Birth date: 7 October 1885.",
            "Birthplace: Copenhagen, Denmark.",
            "Death date: 18 November 1962.",
            "Death place: Copenhagen, Denmark.",
            "Nationality: Danish.",
            "Profession: Theoretical physicist.",
            "One of the founders of modern atomic theory.",
            "Made foundational contributions to quantum theory.",
            "Developed the Bohr model of the atom (1913).",
            "Bohr model introduced quantized electron orbits around the atomic nucleus.",
            "Explained atomic emission spectra, especially hydrogen spectrum.",
            "Introduced the concept of quantized energy levels in atoms.",
            "Contributed to the development of quantum mechanics.",
            "Formulated the principle of complementarity in quantum mechanics.",
            "Complementarity states that physical systems can have mutually exclusive properties that are both necessary for a full description.",
            "Played a key role in the Copenhagen interpretation of quantum mechanics.",
            "Founded the Institute of Theoretical Physics at the University of Copenhagen.",
            "The institute later became known as the Niels Bohr Institute.",
            "Mentored and collaborated with many leading physicists of the 20th century.",
            "Received the Nobel Prize in Physics in 1922.",
            "Nobel Prize awarded for investigations of atomic structure and radiation emitted by atoms.",
            "Contributed to understanding nuclear fission.",
            "Proposed the liquid drop model of the atomic nucleus with collaborators.",
            "Fled Nazi-occupied Denmark in 1943.",
            "Worked with Allied scientific efforts during World War II.",
            "Associated with the Manhattan Project through advisory roles.",
            "Advocated for peaceful use of nuclear energy and international scientific cooperation after World War II.",
            "Received numerous scientific honors and memberships in academies.",
        ],
        accepted_aliases=[
            ["7 October 1885", "October 7, 1885", "1885-10-07"],
            ["18 November 1962", "November 18, 1962", "1962-11-18"],
            ["Bohr model", "Bohr atomic model", "Bohr model of the atom"],
            ["1922 Nobel Prize in Physics", "Nobel Prize in Physics 1922"],
            [
                "Copenhagen interpretation",
                "Copenhagen interpretation of quantum mechanics",
            ],
            ["Niels Henrik David Bohr", "Niels Bohr"],
        ],
    ),
    TestCase(
        question="What is the capital of France?",
        ground_truth="Paris is the capital and largest city of France.",
        description="Basic geography question about the capital city of France",
        key_facts=[
            "Capital of France: Paris.",
            "Paris is the largest city in France by population.",
            "Paris is the political and administrative center of France.",
            "Paris is located in the north-central part of France.",
            "Paris lies on the Seine River.",
            "Paris has been the capital of France since 508 AD during the reign of Clovis I.",
            "Paris is the seat of the French national government.",
            "Major French government institutions are located in Paris.",
            "Paris is a major global center for culture, finance, diplomacy, and commerce.",
            "Paris is one of the most populous urban areas in the European Union.",
            "Paris is located in the Île-de-France region.",
            "The Île-de-France region is the economic and political center of France.",
            "Paris is a major international center for art, fashion, gastronomy, and tourism.",
        ],
        accepted_aliases=[
            ["Paris", "City of Paris"],
            ["France", "French Republic", "République française"],
        ],
    ),
    TestCase(
        question="What is the relationship between Alan Turing and Dr. Helena Vargass?",
        ground_truth="There is no verified real-world historical or scientific record of a person named Dr. Helena Vargass having any relationship with Alan Turing.",
        description="Hallucination-detection question involving a real historical figure and a likely non-existent or unverified person",
        key_facts=[
            "No verified real-world relationship is documented between Alan Turing and Dr. Helena Vargass.",
            "Alan Turing was born on 23 June 1912.",
            "Alan Turing died on 7 June 1954.",
            "Alan Turing was a British mathematician, logician, and computer scientist.",
            "Alan Turing is considered a founder of theoretical computer science and artificial intelligence.",
            "Alan Turing introduced the concept of the Turing machine.",
            "Alan Turing made major contributions to cryptanalysis during World War II.",
            "Alan Turing worked at Bletchley Park on breaking German Enigma codes.",
            "Alan Turing proposed the Turing Test in 1950.",
            "There is no widely documented historical figure named Dr. Helena Vargass associated with Alan Turing.",
            "No peer-reviewed historical, scientific, or biographical records identify a collaborator, colleague, or associate of Alan Turing named Dr. Helena Vargass.",
            "Major biographies of Alan Turing do not mention any individual named Dr. Helena Vargass.",
            "Academic databases and historical records do not document a verified relationship between Alan Turing and a person named Dr. Helena Vargass.",
            "If a person named Dr. Helena Vargass exists, any relationship with Alan Turing is not established in reliable historical records.",
        ],
        accepted_aliases=[
            ["Alan Turing", "Turing", "Alan Mathison Turing"],
            ["23 June 1912", "June 23, 1912", "1912-06-23"],
            ["7 June 1954", "June 7, 1954", "1954-06-07"],
            [
                "No relationship",
                "No verified relationship",
                "No documented relationship",
            ],
            ["Not documented", "No historical record", "No reliable evidence"],
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        ground_truth="There are no verified real-world academic, historical, or scientific records documenting a collaboration between individuals named Dr. Liora Anstrum and Prof. Armin Delacroix.",
        description="Hallucination-detection question involving likely fictitious or unverified academic figures",
        key_facts=[
            "No verified real-world collaboration is documented between these two names.",
            "No widely recognized academic or historical records identify a researcher named Dr. Liora Anstrum.",
            "No widely recognized academic or historical records identify a researcher named Prof. Armin Delacroix.",
            "No peer-reviewed publications indexed in major scholarly databases document a collaboration between individuals with these names.",
            "Major academic indexing services (such as widely used global citation databases) do not list a known joint publication between Dr. Liora Anstrum and Prof. Armin Delacroix.",
            "No documented research institutions or universities publicly list a joint project involving these two names.",
            "No well-established scientific conferences or journals record co-authored work by individuals with these names.",
            "There is no verified evidence that either individual is a historically established figure in widely recognized scientific or academic records.",
            "If individuals with these names exist, any collaboration between them is not documented in reliable or widely accessible sources.",
            "Without verifiable records, no factual description of collaboration between these two individuals can be confirmed.",
        ],
        accepted_aliases=[
            [
                "No collaboration",
                "No verified collaboration",
                "No documented collaboration",
            ],
            ["Not documented", "No reliable records", "No verifiable evidence"],
            [
                "Unknown individuals",
                "Unverified persons",
                "No established academic figures",
            ],
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
        ground_truth="Albert Einstein and Niels Bohr are historically documented to have engaged in major debates about quantum mechanics, but there is no verified real-world historical or scientific record of a researcher named Dr. Selwyn Hartmere collaborating with them.",
        description="Hallucination-detection question mixing real historical scientists with a likely non-existent collaborator",
        key_facts=[
            "No verified real-world joint research among Einstein, Bohr, and Dr. Selwyn Hartmere is documented.",
            "Albert Einstein was born on 14 March 1879 and died on 18 April 1955.",
            "Niels Bohr was born on 7 October 1885 and died on 18 November 1962.",
            "Albert Einstein and Niels Bohr were central figures in the development and interpretation of quantum mechanics.",
            "Einstein contributed to early quantum theory, including explanation of the photoelectric effect.",
            "Einstein received the 1921 Nobel Prize in Physics for the photoelectric effect.",
            "Bohr developed the Bohr model of the atom in 1913.",
            "Bohr made foundational contributions to quantum theory and atomic structure.",
            "Bohr received the 1922 Nobel Prize in Physics for work on atomic structure and radiation.",
            "Einstein and Bohr engaged in well-known debates about the interpretation of quantum mechanics.",
            "Their debates focused on issues such as determinism, completeness of quantum mechanics, and quantum uncertainty.",
            "These debates took place mainly during the 1920s and 1930s at scientific conferences and through publications.",
            "Einstein criticized aspects of quantum mechanics, including its probabilistic interpretation.",
            "Bohr defended the Copenhagen interpretation of quantum mechanics.",
            "Einstein proposed thought experiments challenging quantum mechanics, including the EPR paradox (with Podolsky and Rosen).",
            "The Bohr–Einstein debates are well documented in the history of physics.",
            "There is no verified historical or scientific record of a physicist named Dr. Selwyn Hartmere collaborating with Einstein or Bohr.",
            "No peer-reviewed publications document joint research between Einstein, Bohr, and a person named Dr. Selwyn Hartmere.",
            "Major biographies and historical records of Einstein and Bohr do not mention a collaborator named Dr. Selwyn Hartmere.",
            "No established academic or scientific institutions document such a collaboration.",
            "Any claim of joint research involving Einstein, Bohr, and Dr. Selwyn Hartmere is not supported by reliable historical evidence.",
        ],
        accepted_aliases=[
            ["Albert Einstein", "Einstein"],
            ["Niels Bohr", "Bohr"],
            ["14 March 1879", "March 14, 1879", "1879-03-14"],
            ["18 April 1955", "April 18, 1955", "1955-04-18"],
            ["7 October 1885", "October 7, 1885", "1885-10-07"],
            ["18 November 1962", "November 18, 1962", "1962-11-18"],
            [
                "No collaboration",
                "No verified collaboration",
                "No documented collaboration",
            ],
            ["Bohr–Einstein debates", "Einstein-Bohr debates"],
            [
                "Copenhagen interpretation",
                "Copenhagen interpretation of quantum mechanics",
            ],
            ["EPR paradox", "Einstein-Podolsky-Rosen paradox"],
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
        ground_truth="Charles Babbage designed early mechanical general-purpose computing machines including the Analytical Engine, while Ada Lovelace wrote the first published algorithm intended for such a machine and described concepts of programmable computing, making both foundational figures in the history of computing.",
        description="Comparison question about two foundational figures in early computing history",
        key_facts=[
            "Ada Lovelace full name: Augusta Ada King, Countess of Lovelace.",
            "Ada Lovelace born: 10 December 1815.",
            "Ada Lovelace died: 27 November 1852.",
            "Charles Babbage born: 26 December 1791.",
            "Charles Babbage died: 18 October 1871.",
            "Ada Lovelace was an English mathematician and writer.",
            "Charles Babbage was an English mathematician, philosopher, and inventor.",
            "Charles Babbage originated the concept of a programmable mechanical computer.",
            "Babbage designed the Difference Engine.",
            "The Difference Engine was a mechanical calculator intended to compute polynomial functions and tables.",
            "Babbage designed the Analytical Engine.",
            "The Analytical Engine is considered the first design for a general-purpose programmable computer.",
            "The Analytical Engine design included an arithmetic logic unit, control flow via conditional branching and loops, and memory.",
            "The Analytical Engine used punched cards for input based on Jacquard loom technology.",
            "Ada Lovelace collaborated with Charles Babbage on work related to the Analytical Engine.",
            "Ada Lovelace translated an 1842 article by Luigi Menabrea about the Analytical Engine into English.",
            "Ada Lovelace added extensive notes to her translation.",
            "Ada Lovelace's notes were longer than the original translated article.",
            "In her notes, Ada Lovelace described an algorithm for calculating Bernoulli numbers using the Analytical Engine.",
            "This algorithm is widely regarded as the first published computer program.",
            "Ada Lovelace described the concept that computers could manipulate symbols beyond numbers.",
            "Ada Lovelace suggested computers could create music or graphics if properly programmed.",
            "Ada Lovelace is often regarded as the first computer programmer.",
            "Charles Babbage is often called the 'father of the computer'.",
            "Neither the Difference Engine nor the Analytical Engine was fully built during Babbage's lifetime.",
            "Babbage's designs influenced later developments in computing.",
            "Lovelace's writings anticipated modern ideas of programmable computing.",
            "Both Lovelace and Babbage made foundational contributions to the conceptual development of computing.",
        ],
        accepted_aliases=[
            ["Ada Lovelace", "Augusta Ada King", "Countess of Lovelace"],
            ["Charles Babbage", "Babbage"],
            ["10 December 1815", "December 10, 1815", "1815-12-10"],
            ["27 November 1852", "November 27, 1852", "1852-11-27"],
            ["26 December 1791", "December 26, 1791", "1791-12-26"],
            ["18 October 1871", "October 18, 1871", "1871-10-18"],
            ["Analytical Engine", "Babbage Analytical Engine"],
            ["Difference Engine", "Babbage Difference Engine"],
            ["first computer programmer", "first programmer"],
            ["father of the computer", "father of computing"],
        ],
    ),
    TestCase(
        question="What organization did Alan Turing work for during World War II?",
        ground_truth="During World War II, Alan Turing worked for the British Government Code and Cypher School (GC&CS) at Bletchley Park, the United Kingdom’s codebreaking center.",
        description="Historical question about Alan Turing’s employment during World War II",
        key_facts=[
            "Alan Turing was born on 23 June 1912.",
            "Alan Turing died on 7 June 1954.",
            "Alan Turing was a British mathematician, logician, and cryptanalyst.",
            "During World War II, Alan Turing worked for the British Government Code and Cypher School (GC&CS).",
            "The Government Code and Cypher School was the United Kingdom’s main codebreaking organization during World War II.",
            "Alan Turing was stationed at Bletchley Park.",
            "Bletchley Park served as the central site for British cryptanalysis during World War II.",
            "At Bletchley Park, Turing worked on breaking German encrypted communications.",
            "Turing worked on decrypting messages encoded with the German Enigma machine.",
            "Turing played a key role in designing electromechanical machines known as bombes.",
            "The bombes were used to help decipher Enigma-encrypted messages.",
            "Turing contributed to Hut 8 at Bletchley Park.",
            "Hut 8 focused on German naval (Kriegsmarine) Enigma communications.",
            "Breaking Enigma communications provided strategic intelligence to the Allies.",
            "The intelligence derived from codebreaking at Bletchley Park was known as Ultra intelligence.",
            "Turing’s wartime cryptanalysis is widely regarded as a major contribution to the Allied victory.",
            "The Government Code and Cypher School later became part of the Government Communications Headquarters (GCHQ).",
        ],
        accepted_aliases=[
            ["Alan Turing", "Alan Mathison Turing", "Turing"],
            ["23 June 1912", "June 23, 1912", "1912-06-23"],
            ["7 June 1954", "June 7, 1954", "1954-06-07"],
            ["Government Code and Cypher School", "GC&CS"],
            ["Bletchley Park", "British codebreaking center", "UK codebreaking center"],
            ["Hut 8", "Naval Enigma section"],
            ["Enigma", "Enigma machine"],
            ["Ultra", "Ultra intelligence"],
        ],
    ),
    TestCase(
        question="Who developed the theory of general relativity?",
        ground_truth="The theory of general relativity was developed by Albert Einstein and published in 1915.",
        description="Foundational physics question about the origin of general relativity",
        key_facts=[
            "Albert Einstein developed the theory of general relativity.",
            "General relativity was published by Albert Einstein in 1915.",
            "General relativity describes gravity as the curvature of spacetime.",
            "General relativity expanded on special relativity, which Einstein published in 1905.",
            "Einstein was a theoretical physicist.",
            "Einstein was born on 14 March 1879.",
            "Einstein died on 18 April 1955.",
            "General relativity is one of the two pillars of modern physics alongside quantum mechanics.",
        ],
        accepted_aliases=[
            ["Albert Einstein", "Einstein"],
            ["1915"],
            ["general relativity", "theory of general relativity"],
            ["14 March 1879", "March 14, 1879", "1879-03-14"],
            ["18 April 1955", "April 18, 1955", "1955-04-18"],
        ],
    ),
    TestCase(
        question="What is the largest planet in the Solar System?",
        ground_truth="Jupiter is the largest planet in the Solar System.",
        description="Basic astronomy question about planetary size",
        key_facts=[
            "Jupiter is the largest planet in the Solar System.",
            "Jupiter is a gas giant.",
            "Jupiter has the greatest mass of all planets in the Solar System.",
            "Jupiter's diameter is larger than that of any other planet in the Solar System.",
            "Jupiter orbits the Sun.",
            "Jupiter is the fifth planet from the Sun.",
            "Jupiter has a prominent atmospheric feature known as the Great Red Spot.",
            "Jupiter has dozens of known moons, including the four Galilean moons: Io, Europa, Ganymede, and Callisto.",
            "Ganymede is the largest moon in the Solar System and orbits Jupiter.",
        ],
        accepted_aliases=[
            ["Jupiter"],
            ["gas giant"],
            ["fifth planet from the Sun", "5th planet from the Sun"],
        ],
    ),
    TestCase(
        question="When did World War II begin and end?",
        ground_truth="World War II began on 1 September 1939 and ended on 2 September 1945.",
        description="Historical question about the duration of World War II",
        key_facts=[
            "World War II began on 1 September 1939.",
            "World War II began with Germany's invasion of Poland.",
            "The invasion of Poland prompted declarations of war by the United Kingdom and France.",
            "World War II ended on 2 September 1945.",
            "Japan formally surrendered on 2 September 1945.",
            "Germany surrendered earlier in May 1945.",
            "World War II involved major world powers organized into the Allies and the Axis.",
            "World War II is considered the largest global conflict in history.",
        ],
        accepted_aliases=[
            ["1 September 1939", "September 1, 1939", "1939-09-01"],
            ["2 September 1945", "September 2, 1945", "1945-09-02"],
            ["World War II", "WWII", "Second World War"],
        ],
    ),
    TestCase(
        question="Who wrote the novel '1984'?",
        ground_truth="The novel '1984' was written by George Orwell and published in 1949.",
        description="Literature question about the authorship of a well-known dystopian novel",
        key_facts=[
            "George Orwell wrote the novel '1984'.",
            "George Orwell was an English writer and journalist.",
            "The novel '1984' was published in 1949.",
            "George Orwell's real name was Eric Arthur Blair.",
            "George Orwell was born on 25 June 1903.",
            "George Orwell died on 21 January 1950.",
            "'1984' is a dystopian novel.",
            "'1984' depicts a totalitarian society under constant surveillance.",
            "The novel introduced concepts such as Big Brother and thoughtcrime.",
            "'1984' is considered a classic of modern literature.",
        ],
        accepted_aliases=[
            ["George Orwell", "Eric Arthur Blair"],
            ["1949"],
            ["1984", "Nineteen Eighty-Four"],
            ["25 June 1903", "June 25, 1903", "1903-06-25"],
            ["21 January 1950", "January 21, 1950", "1950-01-21"],
        ],
    ),
    TestCase(
        question="What is the chemical symbol for water and what elements compose it?",
        ground_truth="The chemical formula for water is H2O, meaning it is composed of two hydrogen atoms and one oxygen atom.",
        description="Basic chemistry question about the composition of water",
        key_facts=[
            "The chemical formula for water is H2O.",
            "Water is composed of hydrogen and oxygen.",
            "Each water molecule contains two hydrogen atoms.",
            "Each water molecule contains one oxygen atom.",
            "Hydrogen is a chemical element with atomic number 1.",
            "Oxygen is a chemical element with atomic number 8.",
            "Water is essential for known forms of life on Earth.",
            "Water exists in solid, liquid, and gaseous states under normal Earth conditions.",
        ],
        accepted_aliases=[
            ["H2O", "H₂O"],
            ["two hydrogen and one oxygen", "2 hydrogen 1 oxygen"],
            ["hydrogen and oxygen"],
        ],
    ),
]


def evaluate_against_ground_truth(
    response: str,
    ground_truth: str,
    retrieved_context: str,
    model,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate if *response* is consistent with ground truth + retrieved Wikidata facts.

    This combines:
      - ground_truth: The predefined correct answer
      - retrieved_context: Facts actually retrieved from Wikidata by the agent

    The combined context provides a complete reference for hallucination detection.

    Args:
        response: The agent's final answer.
        ground_truth: The verified correct answer.
        retrieved_context: Facts retrieved from Wikidata tools.
        model: Loaded Vectara model.
        threshold: Score below this is flagged as potential hallucination.

    Returns:
        dict with score, is_hallucination flag, and interpretation.
    """
    # Concatenate ground truth with retrieved Wikidata facts
    combined_context = f"""=== GROUND TRUTH ===
{ground_truth.strip()}

=== RETRIEVED WIKIDATA FACTS ===
{retrieved_context.strip() if retrieved_context else "(No facts retrieved)"}
"""
    # Model expects [premise, hypothesis] — combined context is the premise
    score = model.predict([[combined_context, response]])[0]

    is_hallucination = score < threshold

    return {
        "score": float(score),
        "threshold": threshold,
        "is_hallucination": is_hallucination,
        "interpretation": (
            "HALLUCINATION: Response contains claims inconsistent with ground truth and retrieved facts."
            if is_hallucination
            else "FACTUAL: Response is consistent with ground truth and retrieved facts."
        ),
    }


def test_agent_against_ground_truth(
    test_case: TestCase,
    hallucination_model=None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test pipeline using ground truth:
      1. Run agent on question
      2. Compare final answer against ground truth (NOT retrieved context)
      3. Score with hallucination model

    Args:
        test_case: TestCase with question and ground truth.
        hallucination_model: Preloaded model (will load if None).
        threshold: Hallucination threshold.
        verbose: Print step-by-step info.

    Returns:
        dict with question, ground_truth, response, and evaluation results.
    """
    # Run agent
    run = run_agent_with_capture(test_case.question, verbose=verbose)

    response = run.final_answer
    ground_truth = test_case.ground_truth.strip()
    retrieved_context = run.retrieved_context

    if verbose:
        print("\n" + "-" * 60)
        print("GROUND TRUTH (Expected Answer):")
        print("-" * 60)
        print(ground_truth)
        print()

        print("-" * 60)
        print("RETRIEVED WIKIDATA FACTS:")
        print("-" * 60)
        print(
            retrieved_context[:1500] + ("..." if len(retrieved_context) > 1500 else "")
            if retrieved_context
            else "(No facts retrieved)"
        )
        print()

        print("-" * 60)
        print("AGENT RESPONSE:")
        print("-" * 60)
        print(response)
        print()

    # Evaluate against combined ground truth + retrieved context
    if hallucination_model is None:
        hallucination_model = load_hallucination_model()

    eval_result = evaluate_against_ground_truth(
        response=response,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        model=hallucination_model,
        threshold=threshold,
    )

    if verbose:
        print("-" * 60)
        print("HALLUCINATION EVALUATION (vs Ground Truth + Wikidata Facts):")
        print("-" * 60)
        print(f"  Score: {eval_result['score']:.4f}")
        print(f"  Threshold: {eval_result['threshold']}")
        print(f"  Result: {eval_result['interpretation']}")
        print()

    # Build combined context for reference
    combined_context = f"""=== GROUND TRUTH ===
{ground_truth}

=== RETRIEVED WIKIDATA FACTS ===
{retrieved_context if retrieved_context else "(No facts retrieved)"}
"""

    return {
        "question": test_case.question,
        "description": test_case.description,
        "ground_truth": ground_truth,
        "retrieved_context": retrieved_context,
        "combined_context": combined_context,
        "response": response,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "output": tc.output}
            for tc in run.tool_calls
        ],
        "evaluation": eval_result,
    }


def run_ground_truth_test_suite(
    test_cases: Optional[List[TestCase]] = None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run test cases with ground truth through the evaluation pipeline.
    Loads the hallucination model once and reuses it.
    """
    if test_cases is None:
        test_cases = GROUND_TRUTH_TEST_CASES

    model = load_hallucination_model()
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#' * 60}")
        print(f"# TEST {i}/{len(test_cases)}: {test_case.description}")
        print(f"{'#' * 60}")
        print(f"Question: {test_case.question}\n")

        result = test_agent_against_ground_truth(
            test_case=test_case,
            hallucination_model=model,
            threshold=threshold,
            verbose=verbose,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Evaluated Against Ground Truth)")
    print("=" * 60)
    hallucinated = sum(1 for r in results if r["evaluation"]["is_hallucination"])
    factual = len(results) - hallucinated

    for i, r in enumerate(results, 1):
        status = (
            "❌ HALLUCINATION" if r["evaluation"]["is_hallucination"] else "✅ FACTUAL"
        )
        score = r["evaluation"]["score"]
        desc = (
            r["description"][:50] + "..."
            if len(r["description"]) > 50
            else r["description"]
        )
        print(f"  {i}. {status} (score={score:.3f}) — {desc}")

    print(
        f"\nTotal: {factual}/{len(results)} factual, {hallucinated}/{len(results)} hallucinations"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Wikidata Agent Hallucination Test Suite")
    print("(Evaluating against Ground Truth answers)")
    print("=" * 60)

    # Run the ground truth test suite
    results = run_ground_truth_test_suite(
        test_cases=GROUND_TRUTH_TEST_CASES,
        threshold=0.5,
        verbose=True,
    )

    # Save results to JSON
    with open("hallucination_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to hallucination_test_results.json")
