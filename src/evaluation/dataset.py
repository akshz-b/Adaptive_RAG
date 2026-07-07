DATASET_NAME = "adaptive-rag-thesis-eval"

DATASET_EXAMPLES = [
    {
        "inputs": {"question": "What is the title of the thesis report?"},
        "outputs": {
            "answer": "The thesis report is titled Optimizing Memory Performance in Multi-Core Systems through Data Placement."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {
            "question": "Who submitted the thesis and under whose guidance was it completed?"
        },
        "outputs": {
            "answer": "The thesis was submitted by Akshay Bhosale under the guidance of Dr. Aryabartta Sahu."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {
            "question": "What are the two memory placement methods proposed in the thesis?"
        },
        "outputs": {
            "answer": "The thesis proposes Local to Requester Mapping and Dynamic Migration."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {"question": "What is Dynamic Migration?"},
        "outputs": {
            "answer": "Dynamic Migration periodically monitors access patterns, identifies hot pages, and migrates them closer to the cores that access them most."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {"question": "What simulation tools were used in the thesis?"},
        "outputs": {
            "answer": "The thesis used SniperSim to generate traces and Ramulator2 to simulate DRAM behavior."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {"question": "Which benchmarks were used in the thesis evaluation?"},
        "outputs": {
            "answer": "The evaluation used Art, Equake, FFT, Matmul, Gauss, and PCNN benchmarks."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {
            "question": "What metrics were used to evaluate the proposed methods?"
        },
        "outputs": {
            "answer": "The thesis used average memory latency and percentage improvement as evaluation metrics."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {"question": "What were the main results of Dynamic Migration?"},
        "outputs": {
            "answer": "Dynamic Migration reduced average memory access latency by about 27% to over 50% across workloads."
        },
        "metadata": {
            "query_type": "retrieval",
            "source": "MTP_Report.pdf",
        },
    },
    {
        "inputs": {"question": "What is an embedding?"},
        "outputs": {
            "answer": "An embedding is a numerical vector representation of data that captures semantic meaning for similarity search."
        },
        "metadata": {
            "query_type": "direct",
            "source": "general",
        },
    },
    {
        "inputs": {
            "question": "What are the latest developments in HBM memory in 2026?"
        },
        "outputs": {
            "answer": "This question requires current web information about HBM memory developments in 2026."
        },
        "metadata": {
            "query_type": "web",
            "source": "web",
        },
    },
]
