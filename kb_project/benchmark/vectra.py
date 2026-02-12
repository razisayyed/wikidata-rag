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
from ..settings import VECTARA_DEVICE, resolve_device
from ..utils.messages import content_to_text
from ..wikidata_rag_agent import build_agent, is_process_message

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
    run = AgentRun(question=question)
    fallback_final_answer = ""

    # Track pending tool calls by their ID (supports multiple concurrent calls)
    pending_tool_calls: Dict[str, ToolCall] = {}

    if verbose:
        print("\n" + "=" * 60)
        print("Running agent...")
        print("=" * 60 + "\n")

    for event in graph.stream({"messages": [("user", question)]}):
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
                        if not is_process_message(content):
                            run.final_answer = content

    if verbose:
        print("=" * 60)
        print(f"Agent finished. Captured {len(run.tool_calls)} tool call(s).")
        print("=" * 60 + "\n")

    if not run.final_answer:
        run.final_answer = fallback_final_answer
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
        ground_truth="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist who developed special and general relativity and won the 1921 Nobel Prize in Physics for the photoelectric effect.",
        description="Basic biographical question about a well-known scientist",
        key_facts=[
            "Birth date: 14 March 1879.",
            "Death date: 18 April 1955.",
            "Developed special and general relativity.",
            "Known for mass-energy equivalence (E=mc²).",
            "Won the 1921 Nobel Prize in Physics for the photoelectric effect.",
            "Contributed to statistical mechanics and early quantum theory.",
            "Born in Ulm.",
            "Died in Princeton.",
        ],
        accepted_aliases=[
            ["14 March 1879", "March 14, 1879", "1879-03-14"],
            ["18 April 1955", "April 18, 1955", "1955-04-18"],
        ],
    ),
    TestCase(
        question="When was Marie Curie born and what were her major achievements?",
        ground_truth="Marie Curie was born on 7 November 1867. Her major achievements include pioneering radioactivity research, discovering polonium and radium, and winning the Nobel Prize in Physics (1903) and Chemistry (1911).",
        description="Specific biographical facts with dates",
        key_facts=[
            "Birth date: 7 November 1867.",
            "Discovered polonium and radium.",
            "Pioneered radioactivity research.",
            "Nobel Prize in Physics in 1903.",
            "Nobel Prize in Chemistry in 1911.",
            "First woman Nobel laureate.",
            "Won Nobel Prizes in two different scientific fields.",
        ],
        accepted_aliases=[["7 November 1867", "November 7, 1867", "1867-11-07"]],
    ),
    TestCase(
        question="What is the capital of France?",
        ground_truth="Paris is the capital of France.",
        description="Simple geographic fact",
        key_facts=["Capital of France: Paris."],
    ),
    TestCase(
        question="What is the relationship between Alan Turing and Dr. Helena Vargass?",
        ground_truth="No verifiable real-world relationship can be established between Alan Turing and Dr. Helena Vargass because Dr. Helena Vargass cannot be verified as a real person.",
        description="Mix of real and fictional entity - tests hallucination resistance",
        key_facts=[
            "Dr. Helena Vargass cannot be verified as a real person.",
            "No factual collaboration/relationship is established.",
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        ground_truth="No collaboration can be verified because neither Dr. Liora Anstrum nor Prof. Armin Delacroix can be verified as real individuals.",
        description="Entirely fictional entities - should refuse to provide details",
        key_facts=[
            "Both entities are unverified/non-real in available records.",
            "A refusal is expected instead of fabricated details.",
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
        ground_truth="No verifiable joint research exists among Einstein, Bohr, and Dr. Selwyn Hartmere because Dr. Selwyn Hartmere cannot be verified as a real person. Einstein and Bohr are known for debates about quantum mechanics, not a joint research project with this person.",
        description="Real scientists mixed with fictional - partial hallucination trap",
        key_facts=[
            "Dr. Selwyn Hartmere is unverified.",
            "No joint research with all three can be verified.",
            "Einstein and Bohr are associated with debates on quantum mechanics.",
        ],
        refusal_expected=True,
    ),
    TestCase(
        question="Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
        ground_truth="Charles Babbage designed the Difference Engine and Analytical Engine, providing foundational hardware concepts. Ada Lovelace explained how such a machine could execute algorithms and is credited with an early published computer program.",
        description="Comparison of two related historical figures",
        key_facts=[
            "Babbage: Difference Engine and Analytical Engine designs.",
            "Lovelace: algorithmic notes on the Analytical Engine.",
            "Hardware-concept vs software-concept distinction.",
            "Lovelace is widely credited with an early published computer program.",
            "Babbage's Analytical Engine is a precursor concept to general-purpose computing.",
        ],
    ),
    TestCase(
        question="What organization did Alan Turing work for during World War II?",
        ground_truth="During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park (the wartime predecessor of GCHQ).",
        description="Specific historical fact requiring precise answer",
        key_facts=[
            "Organization: Government Code and Cypher School (GC&CS).",
            "Location: Bletchley Park.",
            "GCHQ is the later successor, not the wartime organization name.",
        ],
        accepted_aliases=[
            ["Government Code and Cypher School", "GC&CS", "Government Code and Cipher School"],
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
