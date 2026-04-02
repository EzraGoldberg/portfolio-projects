# Functions for loading and interacting with the ChromaDB vector store.
# Used by: 2_embeddings.ipynb, 3_retrieval.ipynb, 4_rag_pipeline.ipynb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_vectorstore(persist_directory: str, model: str = "text-embedding-3-small") -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    Expects the store to have been created by 1_load_chunk.ipynb.
    """
    embeddings = OpenAIEmbeddings(model=model)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(f"Loaded vectorstore: {vectorstore._collection.count()} chunks from '{persist_directory}'")
    return vectorstore
