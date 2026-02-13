from equivalence_evaluator.evaluator import EquivalenceEvaluator

evaluator = EquivalenceEvaluator()
result = evaluator.evaluate(
    "Mars is known as the Red Planet.",
    "The Red Planet is known as Mars.",
)

print(result)

