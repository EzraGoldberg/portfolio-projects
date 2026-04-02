# Functions for building prompts and querying the LLM.
# Used by: 4_rag_pipeline.ipynb

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

##  Builds a prompt that instructs the LLM to answer using only the retrieved context.
##  Returns a list of LangChain message objects.
def build_prompt(query: str, context: str) -> list:
    system = SystemMessage(content=(
        "You are a helpful assistant. "
        "Answer the user's question using only the context provided below. "
        "If the answer is not in the context, say 'I don't have enough information to answer that.'"
    ))
    human = HumanMessage(content=(
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    ))
    return [system, human]

##  Send a query + retrieved context to the LLM and return the answer as a string.
def ask(query: str, context: str, model: str = "gpt-4o-mini") -> str:
    llm = ChatOpenAI(model=model)
    messages = build_prompt(query, context)
    response = llm.invoke(messages)
    return response.content
