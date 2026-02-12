"""
Test Wikidata Agent for Hallucinations
======================================
Uses the Vectara hallucination evaluation model to check if the agent's
responses are grounded in the Wikidata facts it retrieves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Agent imports
# ─────────────────────────────────────────────────────────────────────────────
from ..wikidata_rag_agent import build_agent

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
                        matched_call.output = msg.content
                        run.tool_calls.append(matched_call)
                    elif pending_tool_calls:
                        # Fallback: pop the first pending call (for older LangGraph versions)
                        first_id = next(iter(pending_tool_calls))
                        matched_call = pending_tool_calls.pop(first_id)
                        matched_call.output = msg.content
                        run.tool_calls.append(matched_call)

                    if verbose:
                        snippet = msg.content[:300] + (
                            "..." if len(msg.content) > 300 else ""
                        )
                        print(f"  Output: {snippet}\n")

                # Final answer (AI message without tool calls)
                elif hasattr(msg, "content") and msg.content:
                    has_tool_calls = getattr(msg, "tool_calls", None)
                    if not has_tool_calls or len(has_tool_calls) == 0:
                        run.final_answer = msg.content

    if verbose:
        print("=" * 60)
        print(f"Agent finished. Captured {len(run.tool_calls)} tool call(s).")
        print("=" * 60 + "\n")

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination evaluation using Vectara model
# ─────────────────────────────────────────────────────────────────────────────


def load_hallucination_model():
    """
    Load the Vectara hallucination evaluation model.
    Returns the model ready for prediction.
    """
    from transformers import AutoModelForSequenceClassification

    print("Loading Vectara hallucination evaluation model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model",
        trust_remote_code=True,
    )
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
    """A test case with question and ground truth answer."""

    question: str
    ground_truth: str
    description: str = ""


# The ground truth should contain ONLY factual, verifiable information
# Ground truth scope should MATCH the question scope (simple question = simple answer)
GROUND_TRUTH_TEST_CASES: List[TestCase] = [
    TestCase(
        question="Who is Albert Einstein?",
        ground_truth="Albert Einstein was a German-born theoretical physicist who developed the theories of special relativity and general relativity, provided the quantum explanation of the photoelectric effect, and contributed foundational work to statistical mechanics, Brownian motion, and early quantum theory. He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. He was born on 14 March 1879 in Ulm, Kingdom of Württemberg (German Empire), and died on 18 April 1955 in Princeton, New Jersey, United States. He was a human, male. His parents were Hermann Einstein and Pauline Koch. He was married to Mileva Marić (1903–1919) and later to Elsa Einstein (1919–1936). His children were Hans Albert Einstein, Eduard Einstein, and his daughter Lieserl Einstein, whose early life is documented through correspondence. He held German citizenship at birth, became Swiss in 1901, and American in 1940. He studied at ETH Zurich, earning a teaching diploma in physics and mathematics, and completed his doctorate at the University of Zurich under Alfred Kleiner. He worked at the Swiss Patent Office in Bern and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics in Berlin, and later the Institute for Advanced Study in Princeton. His notable scientific contributions include special relativity, general relativity, mass–energy equivalence (E=mc²), the Einstein field equations, the explanation of the photoelectric effect, theoretical work on Brownian motion, Einstein coefficients, Bose–Einstein statistics (with S. N. Bose), and the Einstein–Rosen bridge (with Nathan Rosen). His doctoral students included Nathan Rosen and Ernst G. Straus, and he collaborated with scientists such as Marcel Grossmann, S. N. Bose, and Leo Szilard. He received major recognitions including the Copley Medal in 1925 and the Max Planck Medal in 1929. In 1905, his Annus Mirabilis, he published four revolutionary papers that transformed modern physics. He advocated for civil rights in the United States, supported W. E. B. Du Bois, signed the Einstein–Szilard letter in 1939 urging U.S. research into nuclear chain reactions, supported the establishment of the Hebrew University of Jerusalem, and later declined the presidency of Israel in 1952. His brain was removed for scientific study after his death, an action that is historically documented and widely discussed. The chemical element einsteinium (atomic number 99) was named in his honor.",
        description="Basic biographical question about a well-known scientist",
    ),
    TestCase(
        question="When was Marie Curie born and what were her major achievements?",
        ground_truth="Marie Curie was born on 7 November 1867 in Warsaw, then part of the Russian Empire (Congress Poland). She was a pioneering physicist and chemist best known for discovering the elements polonium and radium, developing the theory of radioactivity, establishing experimental techniques for isolating radioactive isotopes, and laying the foundations of nuclear physics, radiochemistry, and medical radiation therapy. She was a human, female. She died on 4 July 1934 in Passy, France, from aplastic anemia caused by prolonged exposure to radiation. Her parents were Władysław Skłodowski and Bronisława Skłodowska. She married Pierre Curie in 1895 and remained married until his death in 1906. She had two daughters, Irène Joliot-Curie and Ève Curie. She studied at the clandestine Flying University in Warsaw and later at the University of Paris (Sorbonne), earning degrees in physics and mathematics. She held academic positions at the University of Paris and later directed research at the Radium Institute in Paris, becoming the first woman professor at the Sorbonne. Her scientific fields included radioactivity, experimental physics, and radiation chemistry. She received the 1903 Nobel Prize in Physics (shared with Pierre Curie and Henri Becquerel) for research on radiation phenomena, and the 1911 Nobel Prize in Chemistry for the discovery of radium and polonium and the isolation of radium. She was the first woman to win a Nobel Prize, the first person to win two Nobel Prizes, and the only person to win Nobel Prizes in two different scientific fields (Physics and Chemistry). She directed the development and deployment of mobile X-ray units during World War I, personally training operators and assisting with battlefield radiology. She also played a central role in establishing two major research centers—the Radium Institute in Paris and a parallel institute in Warsaw—both of which became leading hubs for nuclear science and medical radiology.",
        description="Specific biographical facts with dates",
    ),
    TestCase(
        question="What is the capital of France?",
        ground_truth="Paris is the capital and largest city of France.",
        description="Simple geographic fact",
    ),
    TestCase(
        question="What is the relationship between Alan Turing and Dr. Helena Vargass?",
        ground_truth="There is no relationship between Alan Turing and Dr. Helena Vargass because Alan Turing is a real historical figure while Dr. Helena Vargass is not a real person. No historical, academic, scientific, or biographical records contain this name, so no connection of any kind exists. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist. He was born on 23 June 1912 in Maida Vale, London, England, and died on 7 June 1954 in Wilmslow, Cheshire, England. He was male and held British citizenship. His parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. He studied at Sherborne School, then at King's College, Cambridge, where he completed major mathematical work, and later earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park during World War II, the National Physical Laboratory, and the University of Manchester. His fields of work included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His notable achievements included the formulation of the universal Turing machine, foundational contributions to computation theory, major contributions to breaking the German Enigma cipher during World War II, and the proposal of the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in 1951. In 1952, Turing was prosecuted for homosexual acts and forced to undergo chemical castration; he died by suicide in 1954, and was posthumously pardoned by the British government in 2013. Dr. Helena Vargass is fictional, and therefore no factual properties can be provided for this entity.",
        description="Mix of real and fictional entity - tests hallucination resistance",
    ),
    TestCase(
        question="Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
        ground_truth="There is no collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix because both names refer to fictional individuals. No historical, academic, scientific, or publicly documented persons exist with these names, and therefore no factual properties, biographical details, or real-world collaborations can be provided.",
        description="Entirely fictional entities - should refuse to provide details",
    ),
    TestCase(
        question="Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
        ground_truth="There was no joint research between Albert Einstein, Niels Bohr, and Dr. Selwyn Hartmere because Dr. Selwyn Hartmere is not a real person and therefore no collaboration involving this fictional name ever existed. Albert Einstein and Niels Bohr were real physicists, but they did not conduct joint research together; instead, they engaged in famous scientific and philosophical debates about quantum mechanics, especially at the Solvay Conferences in the 1920s and 1930s. Their interactions were intellectual arguments over foundational issues such as determinism and the Copenhagen interpretation, not co-authored work or shared experiments. Albert Einstein was a human theoretical physicist born on 14 March 1879 in Ulm, Kingdom of Württemberg, German Empire, and he died on 18 April 1955 in Princeton, New Jersey, United States. He held German, Swiss, and American citizenships, was male, and his occupations included theoretical physicist and professor. His parents were Hermann Einstein and Pauline Koch. He married Mileva Marić and later Elsa Einstein and had two sons, Hans Albert Einstein and Eduard Einstein. He studied at ETH Zurich and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics, and the Institute for Advanced Study in Princeton. His scientific fields included relativity, quantum theory, statistical physics, and cosmology, and his major achievements included special relativity, general relativity, the photoelectric effect, and foundational contributions to early quantum physics. He received the 1921 Nobel Prize in Physics, the Copley Medal, and the Max Planck Medal. Niels Bohr was a human theoretical physicist born on 7 October 1885 in Copenhagen, Denmark, where he also died on 18 November 1962. He held Danish citizenship, was male, and his parents were Christian Bohr and Ellen Adler Bohr. He married Margrethe Nørlund and had children including Aage Bohr, who became a Nobel laureate. Niels Bohr studied at the University of Copenhagen and worked with J. J. Thomson and Ernest Rutherford in Cambridge and Manchester. He founded the Institute for Theoretical Physics (now the Niels Bohr Institute) at the University of Copenhagen. His scientific fields included atomic structure, quantum theory, and nuclear physics. His major achievements included the Bohr model of the atom, foundational contributions to quantum mechanics, and the principle of complementarity. He received the 1922 Nobel Prize in Physics and numerous international honors, and he later contributed as a scientific advisor to the Allied nuclear effort during World War II. Because Dr. Selwyn Hartmere does not exist, no factual properties or collaborations can be attributed to this name.",
        description="Real scientists mixed with fictional - partial hallucination trap",
    ),
    TestCase(
        question="Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
        ground_truth="Ada Lovelace and Charles Babbage made foundational but distinct contributions to the early history of computing. Charles Babbage, a human mathematician, inventor, and mechanical engineer, conceived the Difference Engine and designed the Analytical Engine, introducing key architectural principles of modern computers including a memory store, a processing unit (the mill), conditional branching, loops, and programmable operations. (His pioneering work has earned him recognition as the 'father of the computer'.) Ada Lovelace, a human mathematician and writer, studied the Analytical Engine, translated and expanded Luigi Menabrea’s paper on it, and added extensive annotations that included the algorithm for computing Bernoulli numbers—widely regarded as the first published computer program, making Lovelace widely regarded as the world’s first computer programmer. Lovelace also articulated the theoretical insight that the machine could manipulate symbols beyond numerical calculation, envisioning general-purpose computing and early notions of software. Ada Lovelace was born on 10 December 1815 in London, England, died on 27 November 1852 in Marylebone, London, held British citizenship, was female, and her parents were Lord George Gordon Byron and Anne Isabella Milbanke Byron. She married William King-Noel, later Earl of Lovelace, and had three children: Byron, Anne Isabella, and Ralph. She studied mathematics privately under mentors such as Augustus De Morgan and Mary Somerville and worked primarily as an independent thinker. Her fields included mathematics and early computational theory. Charles Babbage was born on 26 December 1791 in London, England, died on 18 October 1871 in London, held British citizenship, was male, and his parents were Benjamin Babbage and Betsy Plumleigh Teape. He married Georgiana Whitmore and had several children. He studied at Trinity College and Peterhouse, Cambridge, later became Lucasian Professor of Mathematics, and worked on mechanical computation, mathematical tables, and precision engineering. He is often regarded as the 'father of the computer' for his pioneering inventions. His designs for the Analytical Engine influenced later generations of computer scientists and historians. Babbage provided the conceptual hardware architecture of computing, while Lovelace contributed the earliest software concepts and theoretical understanding of symbolic computation, and they collaborated intellectually on the Analytical Engine project.",
        description="Comparison of two related historical figures",
    ),
    TestCase(
        question="What organization did Alan Turing work for during World War II?",
        ground_truth="During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park, the British government's cryptanalytic center responsible for signals intelligence and codebreaking. The organization was founded in 1919, operated at Bletchley Park during the war, and later became the foundation of GCHQ. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist born on 23 June 1912 in Maida Vale, London, and he died on 7 June 1954 in Wilmslow, Cheshire. He held British citizenship, was male, and his parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. Turing studied at Sherborne School, King's College Cambridge, and earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park, the National Physical Laboratory, and the University of Manchester. His fields included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His major achievements included formulating the universal Turing machine, foundational contributions to computation theory, breaking key components of the Enigma cipher during World War II, and proposing the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in recognition of his scientific contributions. In 1952, Turing was convicted under British law for homosexual acts and subjected to chemical castration; two years later, in 1954, he died by suicide. In 2013, he received a posthumous royal pardon. ",
        description="Specific historical fact requiring precise answer",
    ),
]


# # The ground truth should contain ONLY factual, verifiable information
# # Ground truth scope should MATCH the question scope (simple question = simple answer)
# GROUND_TRUTH_TEST_CASES: List[TestCase] = [
#     # ─────────────────────────────────────────────────────────────────────────
#     # Simple factual questions (agent should answer correctly)
#     # ─────────────────────────────────────────────────────────────────────────
#     TestCase(
#         question="Who is Albert Einstein?",
#         ground_truth="Albert Einstein was a German-born theoretical physicist who developed the theories of special relativity and general relativity, provided the quantum explanation of the photoelectric effect, and contributed foundational work to statistical mechanics, Brownian motion, and early quantum theory. He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. He was born on 14 March 1879 in Ulm, Kingdom of Württemberg (German Empire), and died on 18 April 1955 in Princeton, New Jersey, United States. He was a human, male. His parents were Hermann Einstein and Pauline Koch. He was married to Mileva Marić (1903–1919) and later to Elsa Einstein (1919–1936). His children were Hans Albert Einstein, Eduard Einstein, and his daughter Lieserl Einstein, whose early life is documented through correspondence. He held German citizenship at birth, became Swiss in 1901, and American in 1940. He studied at ETH Zurich, earning a teaching diploma in physics and mathematics, and completed his doctorate at the University of Zurich under Alfred Kleiner. He worked at the Swiss Patent Office in Bern and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics in Berlin, and later the Institute for Advanced Study in Princeton. His notable scientific contributions include special relativity, general relativity, mass–energy equivalence (E=mc²), the Einstein field equations, the explanation of the photoelectric effect, theoretical work on Brownian motion, Einstein coefficients, Bose–Einstein statistics (with S. N. Bose), and the Einstein–Rosen bridge (with Nathan Rosen). His doctoral students included Nathan Rosen and Ernst G. Straus, and he collaborated with scientists such as Marcel Grossmann, S. N. Bose, and Leo Szilard. He received major recognitions including the Copley Medal in 1925 and the Max Planck Medal in 1929. In 1905, his Annus Mirabilis, he published four revolutionary papers that transformed modern physics. He advocated for civil rights in the United States, supported W. E. B. Du Bois, signed the Einstein–Szilard letter in 1939 urging U.S. research into nuclear chain reactions, supported the establishment of the Hebrew University of Jerusalem, and later declined the presidency of Israel in 1952. His brain was removed for scientific study after his death, an action that is historically documented and widely discussed.",
#         description="Basic biographical question about a well-known scientist",
#     ),
#     TestCase(
#         question="When was Marie Curie born and what were her major achievements?",
#         ground_truth="Marie Curie was born on 7 November 1867 in Warsaw, then part of the Russian Empire (Congress Poland). She was a pioneering physicist and chemist best known for discovering the elements polonium and radium, developing the theory of radioactivity, establishing experimental techniques for isolating radioactive isotopes, and laying the foundations of nuclear physics, radiochemistry, and medical radiation therapy. She was a human, female. She died on 4 July 1934 in Passy, France, from aplastic anemia caused by prolonged exposure to radiation. Her parents were Władysław Skłodowski and Bronisława Skłodowska. She married Pierre Curie in 1895 and remained married until his death in 1906. She had two daughters, Irène Joliot-Curie and Ève Curie. She studied at the clandestine Flying University in Warsaw and later at the University of Paris (Sorbonne), earning degrees in physics and mathematics. She held academic positions at the University of Paris and later directed research at the Radium Institute in Paris, becoming the first woman professor at the Sorbonne. Her scientific fields included radioactivity, experimental physics, and radiation chemistry. She received the 1903 Nobel Prize in Physics (shared with Pierre Curie and Henri Becquerel) for research on radiation phenomena, and the 1911 Nobel Prize in Chemistry for the discovery of radium and polonium and the isolation of radium. She directed the development and deployment of mobile X-ray units during World War I, personally training operators and assisting with battlefield radiology. She also played a central role in establishing two major research centers—the Radium Institute in Paris and a parallel institute in Warsaw—both of which became leading hubs for nuclear science and medical radiology.",
#         description="Specific biographical facts with dates",
#     ),
#     TestCase(
#         question="What is the capital of France?",
#         ground_truth="Paris is the capital of France.",
#         description="Simple geographic fact",
#     ),
#     # ─────────────────────────────────────────────────────────────────────────
#     # Questions with fictional entities (agent should NOT hallucinate)
#     # ─────────────────────────────────────────────────────────────────────────
#     TestCase(
#         question="What is the relationship between Alan Turing and Dr. Helena Vargass?",
#         ground_truth="There is no relationship between Alan Turing and Dr. Helena Vargass because Alan Turing is a real historical figure while Dr. Helena Vargass is not a real person. No historical, academic, scientific, or biographical records contain this name, so no connection of any kind exists. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist. He was born on 23 June 1912 in Maida Vale, London, England, and died on 7 June 1954 in Wilmslow, Cheshire, England. He was male and held British citizenship. His parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. He studied at Sherborne School, then at King's College, Cambridge, where he completed major mathematical work, and later earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park during World War II, the National Physical Laboratory, and the University of Manchester. His fields of work included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His notable achievements included the formulation of the universal Turing machine, foundational contributions to computation theory, major contributions to breaking the German Enigma cipher during World War II, and the proposal of the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in 1951. Dr. Helena Vargass is fictional, and therefore no factual properties can be provided for this entity.",
#         description="Mix of real and fictional entity - tests hallucination resistance",
#     ),
#     TestCase(
#         question="Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.",
#         ground_truth="There is no collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix because both names refer to fictional individuals. No historical, academic, scientific, or publicly documented persons exist with these names, and therefore no factual properties, biographical details, or real-world collaborations can be provided.",
#         description="Entirely fictional entities - should refuse to provide details",
#     ),
#     TestCase(
#         question="Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.",
#         ground_truth="There was no joint research between Albert Einstein, Niels Bohr, and Dr. Selwyn Hartmere because Dr. Selwyn Hartmere is not a real person and therefore no collaboration involving this fictional name ever existed. Albert Einstein and Niels Bohr were real physicists, but they did not conduct joint research together; instead, they engaged in famous scientific and philosophical debates about quantum mechanics, especially at the Solvay Conferences in the 1920s and 1930s. Their interactions were intellectual arguments over foundational issues such as determinism and the Copenhagen interpretation, not co-authored work or shared experiments. Albert Einstein was a human theoretical physicist born on 14 March 1879 in Ulm, Kingdom of Württemberg, German Empire, and he died on 18 April 1955 in Princeton, New Jersey, United States. He held German, Swiss, and American citizenships, was male, and his occupations included theoretical physicist and professor. His parents were Hermann Einstein and Pauline Koch. He married Mileva Marić and later Elsa Einstein and had two sons, Hans Albert Einstein and Eduard Einstein. He studied at ETH Zurich and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics, and the Institute for Advanced Study in Princeton. His scientific fields included relativity, quantum theory, statistical physics, and cosmology, and his major achievements included special relativity, general relativity, the photoelectric effect, and foundational contributions to early quantum physics. He received the 1921 Nobel Prize in Physics, the Copley Medal, and the Max Planck Medal. Niels Bohr was a human theoretical physicist born on 7 October 1885 in Copenhagen, Denmark, where he also died on 18 November 1962. He held Danish citizenship, was male, and his parents were Christian Bohr and Ellen Adler Bohr. He married Margrethe Nørlund and had children including Aage Bohr, who became a Nobel laureate. Niels Bohr studied at the University of Copenhagen and worked with J. J. Thomson and Ernest Rutherford in Cambridge and Manchester. He founded the Institute for Theoretical Physics (now the Niels Bohr Institute) at the University of Copenhagen. His scientific fields included atomic structure, quantum theory, and nuclear physics. His major achievements included the Bohr model of the atom, foundational contributions to quantum mechanics, and the principle of complementarity. He received the 1922 Nobel Prize in Physics and numerous international honors, and he later contributed as a scientific advisor to the Allied nuclear effort during World War II. Because Dr. Selwyn Hartmere does not exist, no factual properties or collaborations can be attributed to this name.",
#         description="Real scientists mixed with fictional - partial hallucination trap",
#     ),
#     # ─────────────────────────────────────────────────────────────────────────
#     # Comparison questions (tests factual accuracy across entities)
#     # ─────────────────────────────────────────────────────────────────────────
#     TestCase(
#         question="Compare the contributions of Ada Lovelace and Charles Babbage to computing.",
#         ground_truth="Ada Lovelace and Charles Babbage made foundational but distinct contributions to the early history of computing. Charles Babbage, a human mathematician, inventor, and mechanical engineer, conceived the Difference Engine and designed the Analytical Engine, introducing key architectural principles of modern computers including a memory store, a processing unit (the mill), conditional branching, loops, and programmable operations. Ada Lovelace, a human mathematician and writer, studied the Analytical Engine, translated and expanded Luigi Menabrea’s paper on it, and added extensive annotations that included the algorithm for computing Bernoulli numbers—widely regarded as the first published computer program. Lovelace also articulated the theoretical insight that the machine could manipulate symbols beyond numerical calculation, envisioning general-purpose computing and early notions of software. Ada Lovelace was born on 10 December 1815 in London, England, died on 27 November 1852 in Marylebone, London, held British citizenship, was female, and her parents were Lord George Gordon Byron and Anne Isabella Milbanke Byron. She married William King-Noel, later Earl of Lovelace, and had three children: Byron, Anne Isabella, and Ralph. She studied mathematics privately under mentors such as Augustus De Morgan and Mary Somerville and worked primarily as an independent thinker. Her fields included mathematics and early computational theory. Charles Babbage was born on 26 December 1791 in London, England, died on 18 October 1871 in London, held British citizenship, was male, and his parents were Benjamin Babbage and Betsy Plumleigh Teape. He married Georgiana Whitmore and had several children. He studied at Trinity College and Peterhouse, Cambridge, later became Lucasian Professor of Mathematics, and worked on mechanical computation, mathematical tables, and precision engineering. His designs for the Analytical Engine influenced later generations of computer scientists and historians. Babbage provided the conceptual hardware architecture of computing, while Lovelace contributed the earliest software concepts and theoretical understanding of symbolic computation, and they collaborated intellectually on the Analytical Engine project.",
#         description="Comparison of two related historical figures",
#     ),
#     # ─────────────────────────────────────────────────────────────────────────
#     # Specific factual queries (tests precision)
#     # ─────────────────────────────────────────────────────────────────────────
#     TestCase(
#         question="What organization did Alan Turing work for during World War II?",
#         ground_truth="During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park, the British government's cryptanalytic center responsible for signals intelligence and codebreaking. The organization was founded in 1919, operated at Bletchley Park during the war, and later became the foundation of GCHQ. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist born on 23 June 1912 in Maida Vale, London, and he died on 7 June 1954 in Wilmslow, Cheshire. He held British citizenship, was male, and his parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. Turing studied at Sherborne School, King's College Cambridge, and earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park, the National Physical Laboratory, and the University of Manchester. His fields included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His major achievements included formulating the universal Turing machine, foundational contributions to computation theory, breaking key components of the Enigma cipher during World War II, and proposing the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in recognition of his scientific contributions.",
#         description="Specific historical fact requiring precise answer",
#     ),
# ]


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
