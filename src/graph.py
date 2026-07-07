from langgraph.graph import END, START, StateGraph

from src.nodes.generator import generator_node
from src.nodes.grader import decide_after_grading, grader_node
from src.nodes.reranker import reranker_node
from src.nodes.retriever import retriever_node
from src.nodes.rewriter import rewriter_node
from src.nodes.web_search import web_search_node
from src.nodes.direct_answer import direct_answer_node
from src.nodes.router import router_node, route_after_router
from src.nodes.critic import critic_node, route_after_critic, finalizer_node
from src.state import RAGState

builder = StateGraph(RAGState)

builder.add_node("router", router_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("rewriter", rewriter_node)
builder.add_node("retriever", retriever_node)
builder.add_node("reranker", reranker_node)
builder.add_node("grader", grader_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generator", generator_node)
builder.add_node("critic", critic_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "direct": "direct_answer",
        "retrieval": "rewriter",
        "web": "web_search",
    },
)

builder.add_edge("direct_answer", END)

builder.add_edge("rewriter", "retriever")
builder.add_edge("retriever", "reranker")
builder.add_edge("reranker", "grader")

builder.add_conditional_edges(
    "grader",
    decide_after_grading,
    {
        "generate": "generator",
        "web_search": "web_search",
    },
)

builder.add_edge("web_search", "generator")
builder.add_edge("generator", "critic")

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "end": END,
        "web_search": "web_search",
        "finalize_unverified": "finalizer",
    },
)

builder.add_edge("finalizer", END)

rag_graph = builder.compile()
