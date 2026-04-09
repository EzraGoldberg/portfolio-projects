# Functions for querying the ChromaDB vector store.
# Used by: 3_retrieval.ipynb, 4_rag_pipeline.ipynb
from langchain_community.vectorstores import Chroma

def retrieve(query: str, vectorstore: Chroma, k: int = 3, filter: dict = None) -> list:
    """
    Given a query string, return the top-k most relevant chunks from the vectorstore.
    Returns a list of LangChain Document objects. Optional Metadata filter
    """
    if filter:
        return vectorstore.similarity_search(query, k=k, filter=filter)
    else:
        return vectorstore.similarity_search(query, k=k)

def print_results(query: str, results: list) -> None:
    """
    Pretty-print retrieval results to the notebook output.
    """
    print(f"Query: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content)
        print()


def format_context(results: list) -> str:
    """
    Flatten a list of retrieved Documents into a single string.
    Used to inject context into an LLM prompt in notebook 4.
    """
    return "\n\n".join(doc.page_content for doc in results)
