from typing import List, TypedDict


class RAGState(TypedDict, total=False):
    query: str
    route: str

    rewritten_query: str
    multiple_queries: List[str]

    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]

    chunk_grades: List[str]
    web_results: List[dict]

    final_context: str
    answer: str
    citations: List[str]

    critic_decision: str
    unsupported_claims: List[str]
    retry_count: int
