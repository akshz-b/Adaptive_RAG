REWRITE_PROMPT = """You are an expert at reformulating search queries.
Given a user query, rewrite it to be clearer and more specific
for searching a technical document.
Return only the rewritten query, nothing else.

Original question: {query}
Rewritten query:"""


MULTI_QUERY_PROMPT = """You are an expert at generating search queries.
Given a user question, generate 3 different versions of it
to retrieve relevant documents from a vector database.
Return exactly 3 queries, one per line, no numbering, no extra text.

Original question: {query}"""


HYDE_PROMPT = """You are an expert technical writer.

Given a user question, write a short hypothetical answer paragraph
that would likely appear in a technical document.

Do not answer conversationally.
Do not say "I don't know".
Write only the hypothetical document-style paragraph.

User question:
{query}

Hypothetical answer:"""


CHUNK_GRADER_PROMPT = """You are a strict retrieval quality grader.

Your task is to decide whether the retrieved chunk is useful for answering the user query.

Return only one label:
- relevant
- ambiguous
- irrelevant

Definitions:
- relevant: the chunk clearly contains information needed to answer the query.
- ambiguous: the chunk is somewhat related but incomplete, indirect, or uncertain.
- irrelevant: the chunk does not help answer the query.

User query:
{query}

Retrieved chunk:
{chunk}

Grade:"""


GENERATOR_PROMPT = """You are a grounded RAG answer generator.

Use only the provided context to answer the user query.
If the answer is not present in the context, say "I don't know".

When possible, mention the source naturally using available document/page or web source information.

Context:
{context}

User query:
{query}

Answer:"""


ROUTER_PROMPT = """You are an expert query router for a RAG system.

Classify the user query into exactly one route:

direct:
Use this when the question is general knowledge, conceptual, or can be answered without searching the user's documents or the web.
Examples:
- What is RAG?
- What is an embedding?
- Explain BM25 in simple terms.

retrieval:
Use this when the question is about the user's ingested documents, report, thesis, PDF, stored knowledge base, or previously indexed files.
Examples:
- What optimization techniques are proposed in the thesis?
- Who proposed this solution?
- What are the main findings of the report?

web:
Use this when the question needs current, latest, external, or real-time information from the internet.
Examples:
- What are the latest developments in HBM memory in 2026?
- What is today's NVIDIA stock price?
- Find recent papers about NUMA optimization.

Return only one label:
direct
retrieval
web

User query:
{query}

Route:"""


DIRECT_ANSWER_PROMPT = """You are a helpful technical assistant.

Answer the user's question directly and clearly using your own knowledge.

User question:
{query}

Answer:"""


CRITIC_PROMPT = """You are a strict answer grounding critic.

Your task is to check whether the answer is fully supported by the provided context.

Return:
- grounded: if every important claim in the answer is supported by the context
- not_grounded: if the answer contains claims not supported by the context

Also identify unsupported claims if any.

User query:
{query}

Context:
{context}

Answer:
{answer}"""
