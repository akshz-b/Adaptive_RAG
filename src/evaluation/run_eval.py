from langsmith import Client, evaluate

from src.evaluation.dataset import DATASET_EXAMPLES, DATASET_NAME
from src.evaluation.evaluators import correctness_evaluator, faithfulness_evaluator
from src.graph import rag_graph


def ensure_dataset_exists() -> None:
    """
    Create the LangSmith dataset and upload examples if it does not already exist.
    """
    client = Client()

    existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing_datasets:
        print(f"Dataset already exists: {DATASET_NAME}")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for AdaptiveRAG thesis questions.",
    )

    for example in DATASET_EXAMPLES:
        client.create_example(
            dataset_id=dataset.id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example.get("metadata", {}),
        )

    print(f"Created dataset: {DATASET_NAME}")
    print(f"Uploaded examples: {len(DATASET_EXAMPLES)}")


def adaptive_rag_target(inputs: dict) -> dict:
    """
    Run AdaptiveRAG graph for one evaluation example.
    """
    question = inputs["question"]

    state = rag_graph.invoke(
        {
            "query": question,
            "retry_count": 0,
        },
        config={
            "tags": ["stage-6", "eval"],
            "metadata": {
                "app_stage": "stage-6",
                "pipeline": "adaptive-router-self-rag",
            },
        },
    )

    return {
        "answer": state.get("answer", ""),
        "route": state.get("route", ""),
        "final_context": state.get("final_context", ""),
        "critic_decision": state.get("critic_decision", ""),
        "retry_count": state.get("retry_count", 0),
    }


def run_evaluation() -> None:
    """
    Run LangSmith evaluation on the AdaptiveRAG dataset.
    """
    ensure_dataset_exists()

    results = evaluate(
        adaptive_rag_target,
        data=DATASET_NAME,
        evaluators=[
            correctness_evaluator,
            faithfulness_evaluator,
        ],
        experiment_prefix="adaptive-rag-stage-6",
        metadata={
            "dataset_name": DATASET_NAME,
            "stage": "stage-6",
        },
        max_concurrency=0,
    )

    print(results)


if __name__ == "__main__":
    run_evaluation()
