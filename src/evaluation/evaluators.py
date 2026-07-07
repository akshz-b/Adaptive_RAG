from src.nodes.critic import critique_answer


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Check whether generated answer overlaps with expected answer or not.
    """
    predicted = outputs.get("answer", "").lower()
    expected = reference_outputs.get("answer", "").lower()

    expected_words = set(expected.split())
    predicted_words = set(predicted.split())

    if not expected_words:
        score = 0.0
    else:
        overlap = expected_words.intersection(predicted_words)
        score = len(overlap) / len(expected_words)

    return {
        "key": "correctness",
        "score": score,
        "comment": f"Word overlap score: {score:.2f}",
    }


def faithfulness_evaluator(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    """
    Check whether generated answer is grounded in final context.
    """
    question = inputs.get("question", "")
    answer = outputs.get("answer", "")
    context = outputs.get("final_context", "")

    if not context.strip():
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "No context available for grounding check.",
        }

    result = critique_answer(
        query=question,
        context=context,
        answer=answer,
    )

    score = 1.0 if result.decision == "grounded" else 0.0

    return {
        "key": "faithfulness",
        "score": score,
        "comment": (
            "Grounded"
            if result.decision == "grounded"
            else f"Unsupported claims: {result.unsupported_claims}"
        ),
    }
