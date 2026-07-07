from langchain_core.prompts import ChatPromptTemplate

from src.client import get_llm
from src.retrieval.semantic import semantic_retrieval


PROMPT_TEMPLATE = """
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""


def main() -> None:
    """
    Run the Stage 1 naïve RAG question-answering loop.
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    print("AdaptiveRAG Stage 1 is ready.")
    print("Ask a question about your ingested document.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting AdaptiveRAG.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        chunks = semantic_retrieval(question)
        context = "\n\n".join(doc.page_content for doc in chunks)

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        print("\nAnswer:")
        print(response.content)
        print()


if __name__ == "__main__":
    main()
