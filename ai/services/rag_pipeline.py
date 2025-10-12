"""
RAG Pipeline Module
Implements Retrieval-Augmented Generation with metrics tracking
"""
import time
import logging
from typing import List, Dict, Any, Generator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from ai.services.vector_store import get_vector_store_manager
from ai.settings import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline with metrics"""
    
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._create_llm()
        self.vector_manager = get_vector_store_manager()
        
        self.metrics = {
            "queries_processed": 0,
            "total_query_time": 0.0,
            "total_generation_time": 0.0,
            "total_tokens_generated": 0,
            "retrieval_failures": 0,
            "generation_failures": 0
        }
        
        logger.info(f"Initialized RAG Pipeline with model: {model_name}")
    
    def _create_llm(self) -> ChatOllama:
        """Create ChatOllama instance"""
        return ChatOllama(
            model=self.model_name,
            base_url=OLLAMA_HOST,
            temperature=self.temperature,
            num_predict=OLLAMA_MAX_TOKENS
        )
    
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Format documents for context
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            source = meta.get("source") or meta.get("file_name", "unknown")
            page = meta.get("page")
            
            # Create citation label
            label = f"[{i+1}] Source: {source}"
            if page is not None:
                label += f", Page: {page}"
            
            parts.append(f"{label}\n{doc.page_content}")
        
        return "\n\n".join(parts)
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        template = """You are a helpful assistant that answers questions based STRICTLY on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY information from the context below
- If the answer is not in the context, say "I cannot answer this question based on the provided documents"
- Include citations like [1], [2] when referencing specific sources
- Be concise and accurate
- Do not make up information or use external knowledge

Question:
{question}

Context:
{context}

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def query(
        self,
        question: str,
        store_name: str,
        k: int = 4,
        include_sources: bool = True,
        score_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            store_name: Vector store to query
            k: Number of documents to retrieve
            include_sources: Whether to include source information
            score_threshold: Minimum similarity score
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Retrieval phase
            retrieval_start = time.time()
            docs = self.vector_manager.similarity_search(
                store_name=store_name,
                query=question,
                k=k,
                score_threshold=score_threshold
            )
            retrieval_time = time.time() - retrieval_start
            
            if not docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I cannot answer this question based on the provided documents.",
                    "sources": [],
                    "metadata": {
                        "retrieval_time": retrieval_time,
                        "generation_time": 0.0,
                        "total_time": time.time() - start_time,
                        "docs_retrieved": 0
                    }
                }
            
            # Generation phase
            generation_start = time.time()
            
            # Create chain
            prompt = self._create_prompt()
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate answer
            answer = chain.invoke({
                "question": question,
                "context": self._format_documents(docs)
            })
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["total_query_time"] += total_time
            self.metrics["total_generation_time"] += generation_time
            # Rough token estimate (4 chars ≈ 1 token)
            self.metrics["total_tokens_generated"] += len(answer) // 4
            
            # Prepare response
            response = {
                "answer": answer,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "docs_retrieved": len(docs),
                    "model": self.model_name,
                    "temperature": self.temperature
                }
            }
            
            # Add sources if requested
            if include_sources:
                response["sources"] = [
                    {
                        "index": i + 1,
                        "source": (d.metadata or {}).get("source", "unknown"),
                        "page": (d.metadata or {}).get("page"),
                        "chunk_index": (d.metadata or {}).get("chunk_index"),
                        "snippet": d.page_content[:500]
                    }
                    for i, d in enumerate(docs)
                ]
            
            logger.info(
                f"Query processed in {total_time:.2f}s "
                f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"
            )
            
            return response
            
        except Exception as e:
            self.metrics["generation_failures"] += 1
            logger.error(f"Error processing query: {e}")
            raise RuntimeError(f"Query processing failed: {e}")
    
    def stream_query(
        self,
        question: str,
        store_name: str,
        k: int = 4
    ) -> Generator[str, None, None]:
        """
        Stream the answer token by token
        
        Args:
            question: User's question
            store_name: Vector store to query
            k: Number of documents to retrieve
            
        Yields:
            Answer tokens
        """
        try:
            # Retrieve documents
            docs = self.vector_manager.similarity_search(
                store_name=store_name,
                query=question,
                k=k
            )
            
            if not docs:
                yield "I cannot answer this question based on the provided documents."
                return
            
            # Create prompt
            prompt = self._create_prompt()
            messages = prompt.format_messages(
                question=question,
                context=self._format_documents(docs)
            )
            
            # Stream response
            full_response = ""
            for chunk in self.llm.stream(messages):
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            # Update metrics after streaming completes
            self.metrics["queries_processed"] += 1
            self.metrics["total_tokens_generated"] += len(full_response) // 4
            
        except Exception as e:
            self.metrics["generation_failures"] += 1
            logger.error(f"Error streaming query: {e}")
            yield f"\n[ERROR] {str(e)}"
    
    def batch_query(
        self,
        questions: List[str],
        store_name: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions
            store_name: Vector store to query
            k: Number of documents to retrieve
            
        Returns:
            List of results
        """
        results = []
        start_time = time.time()
        
        for i, question in enumerate(questions):
            logger.info(f"Processing batch query {i+1}/{len(questions)}")
            try:
                result = self.query(question, store_name, k=k)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch query {i+1}: {e}")
                results.append({
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "metadata": {"error": str(e)}
                })
        
        total_time = time.time() - start_time
        logger.info(
            f"Batch processing completed: {len(questions)} queries in {total_time:.2f}s "
            f"(avg: {total_time/len(questions):.2f}s/query)"
        )
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        avg_query_time = (
            self.metrics["total_query_time"] / 
            self.metrics["queries_processed"]
            if self.metrics["queries_processed"] > 0 else 0
        )
        
        avg_generation_time = (
            self.metrics["total_generation_time"] / 
            self.metrics["queries_processed"]
            if self.metrics["queries_processed"] > 0 else 0
        )
        
        success_rate = (
            (self.metrics["queries_processed"] - self.metrics["generation_failures"]) / 
            self.metrics["queries_processed"] * 100
            if self.metrics["queries_processed"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "avg_query_time": avg_query_time,
            "avg_generation_time": avg_generation_time,
            "success_rate": success_rate
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "queries_processed": 0,
            "total_query_time": 0.0,
            "total_generation_time": 0.0,
            "total_tokens_generated": 0,
            "retrieval_failures": 0,
            "generation_failures": 0
        }


# Global instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# Convenience functions for backward compatibility
def answer(
    query: str,
    store_name: str,
    k: int = 4,
    include_sources: bool = True
) -> Dict[str, Any]:
    """Process a query (convenience function)"""
    pipeline = get_rag_pipeline()
    return pipeline.query(query, store_name, k, include_sources)


def stream_answer(
    query: str,
    store_name: str,
    k: int = 4
) -> Generator[str, None, None]:
    """Stream answer (convenience function)"""
    pipeline = get_rag_pipeline()
    return pipeline.stream_query(query, store_name, k)