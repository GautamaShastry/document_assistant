from typing import List, Dict, Any, Generator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from ai.services.vector_store import get_retriever
from ai.settings import OLLAMA_MODEL, OLLAMA_HOST

MODEL_KWARGS = {
    "temperature": 0.2,}

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_name") or "unknown"
        page = meta.get("page")
        label = f"[{i}] Source: {src}" + (f" . p.{page}" if page is not None else "")
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n".join(parts)

def _model() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        **MODEL_KWARGS,
    )
    
def _prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers strictly from the provided context.
        If the answer is not in the context, say you don't know.

        Question:
        {question}

        Context:
        {context}

        Respond concisely and include bracketed citations like [1], [2] corresponding
        to the numbered context items when you quote or rely on them."""
    )
    
def answer(query: str, store_name: str, k: int = 4, include_sources: bool = True) -> Dict[str, Any]:
    """
    Retrieve top-k chunks from a single store and synthesize an answer.
    """
    retriever = get_retriever(store_name, k)
    docs: List[Document] = retriever.get_relevant_documents(query)
    
    chain = _prompt() | _model() | StrOutputParser()
    text = chain.invoke({
            "question": query,
            "context": _format_docs(docs),
        })
    
    out: Dict[str, Any] = {"answer": text}
    if include_sources:
        out["sources"] = [
            {
                "source": (d.metadata or {}).get("source", "unknown"),
                "page": (d.metadata or {}).get("page"),
                "snippet": d.page_content[:500],
            }
            for d in docs
        ]
    return out

def stream_answer(query: str, store_name: str, k: int = 4) -> Generator[str, None, None]:
    """
    Stream model tokens for a chat-typing UX.
    """
    retriever = get_retriever(store_name, k)
    docs: List[Document] = retriever.get_relevant_documents(query)
    messages = _prompt().format_messages(question=query, context=_format_docs(docs))
    
    for chunk in _model().stream(messages):
        if chunk and getattr(chunk, "content", None):
            yield chunk.content