from typing import List, Dict, Any, Generator, Optional
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
    
def _conversational_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """You are a helpful AI assistant having a conversation about documents. 
        You should answer based on the provided context and maintain awareness of the conversation history.

        Conversation History:
        {conversation_history}

        Current Question:
        {question}

        Relevant Document Context:
        {context}

        Instructions:
        - Answer the current question using the document context provided
        - Consider the conversation history to maintain continuity
        - If the answer is not in the context, say you don't know
        - Include bracketed citations like [1], [2] when referencing specific sources
        - Keep your responses natural and conversational
        - If referring to previous parts of the conversation, be clear about what you're referencing
        Provide a concise and informative answer."""
    )
    
def answer(query: str, store_name: str, k: int = 4, include_sources: bool = True) -> Dict[str, Any]:
    """
    Retrieve top-k chunks from a single store and synthesize an answer.
    This is the original function for backward compatibility
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

def conversational_answer(
        query: str,
        store_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = 4,
        include_sources: bool = True,    
    ) -> Dict[str, Any]:
    """
    Enhanced answer function that considers conversation history for context-aware responses.
    
    Args:
        query: The current user question
        store_name: Vector store name to search
        conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        k: Number of document chunks to retrieve
        include_sources: Whether to include source information
    
    Returns:
        Dict containing answer and sources
    """
    retriever = get_retriever(store_name, k)
    
    # enhanced query construction for better retrieval
    search_query = query
    if conversation_history and len(conversation_history) > 0:
        recent_context = []
        for msg in conversation_history[-4:]:
            role = msg["role"]
            content = msg["content"][:200]
            recent_context.append(f"{role}: {content}") # truncate to 200 chars
            
        contextual_search = f"Recent conversation: {' | '.join(recent_context)}. Current question: {query}"
        search_query = contextual_search
        
    docs: List[Document] = retriever.get_relevant_documents(search_query)
    
    history_text = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-6:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_parts.append(f"{role.title()}: {content}")
        history_text = "\n".join(history_parts)
        
    chain = _conversational_prompt() | _model() | StrOutputParser()
    text = chain.invoke({
            "question": query,
            "context": _format_docs(docs),
            "conversation_history": history_text or "No prior conversation.",
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
            
def stream_conversational_answer(
        query: str,
        store_name: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        k: int = 4,
    ) -> Generator[str, None, None]:
    """
    Stream model tokens for a chat-typing UX with conversational context.
    """
    retriever = get_retriever(store_name, k)
    
    # enhanced query construction for better retrieval
    search_query = query
    if conversation_history and len(conversation_history) > 0:
        recent_context = []
        for msg in conversation_history[-4:]:
            role = msg["role"]
            content = msg["content"][:200]
            recent_context.append(f"{role}: {content}") # truncate to 200 chars
            
        contextual_search = f"Recent conversation: {' | '.join(recent_context)}. Current question: {query}"
        search_query = contextual_search
        
    docs: List[Document] = retriever.get_relevant_documents(search_query)
    
    history_text = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-6:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_parts.append(f"{role.title()}: {content}")
        history_text = "\n".join(history_parts)
        
    messages = _conversational_prompt().format_messages(
        question=query,
        context=_format_docs(docs),
        conversation_history=history_text or "No prior conversation.",
    )
    
    for chunk in _model().stream(messages):
        if chunk and getattr(chunk, "content", None):
            yield chunk.content
            
def get_conversation_summary(conversation_history: List[Dict[str, str]], max_length: int = 500) -> str:
    """
    Generate a summary of conversation history for context.
    Useful when conversation gets very long.
    
    Args:
        conversation_history: List of conversation messages
        max_length: Maximum length of summary
        
    Returns:
        Condensed summary of the conversation
    """
    if not conversation_history or len(conversation_history) < 4:
        return ""
    
    # Create a summary prompt
    summary_prompt = ChatPromptTemplate.from_template(
        """Summarize the following conversation in a concise manner, focusing on key topics discussed and important information exchanged. Keep it under {max_length} characters.

        Conversation:
        {conversation}

        Summary:"""
    )
    
    # Format conversation
    conv_text = []
    for msg in conversation_history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        conv_text.append(f"{role.title()}: {content}")
    
    chain = summary_prompt | _model() | StrOutputParser()
    summary = chain.invoke({
        "conversation": "\n".join(conv_text),
        "max_length": max_length
    })
    
    return summary.strip()

def format_chat_messages_for_context(messages) -> List[Dict[str, str]]:
    """
    Convert database chat messages to conversation history format.
    
    Args:
        messages: List of ChatMessage objects from database
        
    Returns:
        List of formatted conversation messages
    """
    conversation_history = []
    for msg in messages:
        conversation_history.append({
            "role": msg.role,
            "content": msg.content
        })
    return conversation_history