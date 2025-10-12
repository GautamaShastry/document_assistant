"""
Comprehensive RAG Pipeline Testing Script
Tests all components and generates performance metrics for resume showcase
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai.services.document import DocumentProcessor
from ai.services.vector_store import VectorStoreManager
from ai.services.rag_pipeline import RAGPipeline


class RAGPipelineTester:
    """Comprehensive testing suite for RAG pipeline"""
    
    def __init__(self, test_docs_dir: str = "tests/test_documents"):
        self.test_docs_dir = Path(test_docs_dir)
        self.results_dir = Path("tests/evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()
        self.rag_pipeline = RAGPipeline()
        
        # Test data
        self.test_questions = []
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
    
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")
    
    def test_document_processing(self, file_path: str) -> Dict[str, Any]:
        """Test document loading and chunking"""
        self.print_section("Testing Document Processing")
        
        test_result = {
            "test_name": "Document Processing",
            "file": str(file_path),
            "status": "pending"
        }
        
        try:
            start_time = time.time()
            
            # Load document
            print(f"Loading document: {Path(file_path).name}")
            docs = self.doc_processor.load_document(file_path)
            print(f"  ✓ Loaded {len(docs)} document(s)")
            
            # Split into chunks
            print("Splitting into chunks...")
            chunks = self.doc_processor.split_documents(docs)
            print(f"  ✓ Created {len(chunks)} chunks")
            
            # Calculate statistics
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            
            processing_time = time.time() - start_time
            
            test_result.update({
                "status": "success",
                "documents_loaded": len(docs),
                "chunks_created": len(chunks),
                "avg_chunk_size": round(avg_chunk_size, 2),
                "min_chunk_size": min(chunk_sizes),
                "max_chunk_size": max(chunk_sizes),
                "processing_time": round(processing_time, 3)
            })
            
            print(f"\n📊 Statistics:")
            print(f"  - Documents loaded: {len(docs)}")
            print(f"  - Chunks created: {len(chunks)}")
            print(f"  - Avg chunk size: {avg_chunk_size:.0f} chars")
            print(f"  - Processing time: {processing_time:.3f}s")
            
            return chunks
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")
            raise
        finally:
            self.results["tests"].append(test_result)
    
    def test_vector_store_creation(self, chunks: List, store_name: str) -> Dict[str, Any]:
        """Test vector store creation and embedding"""
        self.print_section("Testing Vector Store Creation")
        
        test_result = {
            "test_name": "Vector Store Creation",
            "store_name": store_name,
            "status": "pending"
        }
        
        try:
            start_time = time.time()
            
            print(f"Creating vector store: {store_name}")
            print(f"Embedding {len(chunks)} chunks...")
            
            store_path = self.vector_manager.create_vector_store(
                chunks, store_name, overwrite=True
            )
            
            embedding_time = time.time() - start_time
            
            # Get store statistics
            stats = self.vector_manager.get_store_stats(store_name)
            
            test_result.update({
                "status": "success",
                "store_path": store_path,
                "chunks_embedded": len(chunks),
                "embedding_time": round(embedding_time, 3),
                "embedding_speed": round(len(chunks) / embedding_time, 2),
                "vector_count": stats.get("vector_count"),
                "store_size_mb": round(stats.get("total_size_bytes", 0) / 1024 / 1024, 2)
            })
            
            print(f"\n📊 Vector Store Statistics:")
            print(f"  - Chunks embedded: {len(chunks)}")
            print(f"  - Embedding time: {embedding_time:.3f}s")
            print(f"  - Speed: {len(chunks)/embedding_time:.1f} chunks/s")
            print(f"  - Store size: {stats.get('total_size_bytes', 0) / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")
            raise
        finally:
            self.results["tests"].append(test_result)
    
    def test_retrieval(self, store_name: str, queries: List[str], k: int = 4) -> Dict[str, Any]:
        """Test document retrieval"""
        self.print_section("Testing Document Retrieval")
        
        test_result = {
            "test_name": "Document Retrieval",
            "store_name": store_name,
            "status": "pending",
            "queries": []
        }
        
        try:
            print(f"Testing {len(queries)} retrieval queries...")
            
            retrieval_times = []
            
            for i, query in enumerate(queries, 1):
                print(f"\nQuery {i}: {query[:60]}...")
                
                start_time = time.time()
                docs = self.vector_manager.similarity_search(store_name, query, k=k)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                print(f"  ✓ Retrieved {len(docs)} documents in {retrieval_time:.3f}s")
                
                # Show top result snippet
                if docs:
                    print(f"  Top result: {docs[0].page_content[:100]}...")
                
                test_result["queries"].append({
                    "query": query,
                    "docs_retrieved": len(docs),
                    "retrieval_time": round(retrieval_time, 3)
                })
            
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            
            test_result.update({
                "status": "success",
                "total_queries": len(queries),
                "avg_retrieval_time": round(avg_retrieval_time, 3),
                "min_retrieval_time": round(min(retrieval_times), 3),
                "max_retrieval_time": round(max(retrieval_times), 3)
            })
            
            print(f"\n📊 Retrieval Statistics:")
            print(f"  - Queries tested: {len(queries)}")
            print(f"  - Avg retrieval time: {avg_retrieval_time:.3f}s")
            print(f"  - Min/Max time: {min(retrieval_times):.3f}s / {max(retrieval_times):.3f}s")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")
        finally:
            self.results["tests"].append(test_result)
    
    def test_rag_pipeline(self, store_name: str, queries: List[str], k: int = 4) -> Dict[str, Any]:
        """Test end-to-end RAG pipeline"""
        self.print_section("Testing RAG Pipeline (End-to-End)")
        
        test_result = {
            "test_name": "RAG Pipeline",
            "store_name": store_name,
            "status": "pending",
            "queries": []
        }
        
        try:
            print(f"Testing {len(queries)} RAG queries...")
            
            total_times = []
            generation_times = []
            
            for i, query in enumerate(queries, 1):
                print(f"\n{'─' * 80}")
                print(f"Query {i}/{len(queries)}: {query}")
                print(f"{'─' * 80}")
                
                start_time = time.time()
                result = self.rag_pipeline.query(
                    question=query,
                    store_name=store_name,
                    k=k,
                    include_sources=True
                )
                
                metadata = result.get("metadata", {})
                total_time = time.time() - start_time
                total_times.append(total_time)
                generation_times.append(metadata.get("generation_time", 0))
                
                # Display answer
                print(f"\n💡 Answer:")
                print(f"{result['answer']}\n")
                
                # Display sources
                sources = result.get("sources", [])
                if sources:
                    print(f"📚 Sources ({len(sources)}):")
                    for src in sources[:2]:  # Show first 2 sources
                        print(f"  [{src['index']}] {src['source']}")
                
                # Display metrics
                print(f"\n⏱️  Performance:")
                print(f"  - Total time: {total_time:.3f}s")
                print(f"  - Retrieval: {metadata.get('retrieval_time', 0):.3f}s")
                print(f"  - Generation: {metadata.get('generation_time', 0):.3f}s")
                
                test_result["queries"].append({
                    "query": query,
                    "answer_length": len(result["answer"]),
                    "sources_count": len(sources),
                    "total_time": round(total_time, 3),
                    "retrieval_time": round(metadata.get("retrieval_time", 0), 3),
                    "generation_time": round(metadata.get("generation_time", 0), 3)
                })
            
            avg_total_time = sum(total_times) / len(total_times)
            avg_generation_time = sum(generation_times) / len(generation_times)
            
            test_result.update({
                "status": "success",
                "total_queries": len(queries),
                "avg_total_time": round(avg_total_time, 3),
                "avg_generation_time": round(avg_generation_time, 3),
                "min_total_time": round(min(total_times), 3),
                "max_total_time": round(max(total_times), 3)
            })
            
            print(f"\n{'=' * 80}")
            print("📊 RAG Pipeline Statistics:")
            print(f"  - Queries processed: {len(queries)}")
            print(f"  - Avg total time: {avg_total_time:.3f}s")
            print(f"  - Avg generation time: {avg_generation_time:.3f}s")
            print(f"  - Throughput: {len(queries)/sum(total_times):.2f} queries/s")
            
        except Exception as e:
            test_result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ Error: {e}")
        finally:
            self.results["tests"].append(test_result)
    
    def generate_summary(self):
        """Generate test summary"""
        self.print_header("Test Summary")
        
        successful_tests = sum(1 for t in self.results["tests"] if t.get("status") == "success")
        total_tests = len(self.results["tests"])
        
        # Component metrics
        doc_metrics = self.doc_processor.get_metrics()
        vector_metrics = self.vector_manager.get_metrics()
        rag_metrics = self.rag_pipeline.get_metrics()
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0,
            "component_metrics": {
                "document_processor": doc_metrics,
                "vector_store": vector_metrics,
                "rag_pipeline": rag_metrics
            }
        }
        
        self.results["summary"] = summary
        
        # Print summary
        print(f"\n✅ Successful Tests: {successful_tests}/{total_tests}")
        print(f"📈 Success Rate: {summary['success_rate']}%")
        
        print(f"\n📊 Component Metrics:")
        print(f"\n  Document Processor:")
        print(f"    - Documents processed: {doc_metrics.get('documents_processed', 0)}")
        print(f"    - Chunks created: {doc_metrics.get('total_chunks_created', 0)}")
        print(f"    - Avg processing time: {doc_metrics.get('avg_processing_time', 0):.3f}s")
        
        print(f"\n  Vector Store:")
        print(f"    - Stores created: {vector_metrics.get('stores_created', 0)}")
        print(f"    - Documents indexed: {vector_metrics.get('documents_indexed', 0)}")
        print(f"    - Retrieval count: {vector_metrics.get('retrieval_count', 0)}")
        print(f"    - Avg retrieval time: {vector_metrics.get('avg_retrieval_time', 0):.3f}s")
        
        print(f"\n  RAG Pipeline:")
        print(f"    - Queries processed: {rag_metrics.get('queries_processed', 0)}")
        print(f"    - Avg query time: {rag_metrics.get('avg_query_time', 0):.3f}s")
        print(f"    - Avg generation time: {rag_metrics.get('avg_generation_time', 0):.3f}s")
        print(f"    - Success rate: {rag_metrics.get('success_rate', 0):.2f}%")
        
        print("\n" + "=" * 80)
    
    def save_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def run_full_test(self, document_path: str, test_queries: List[str]):
        """Run complete test suite"""
        self.print_header("RAG Pipeline Comprehensive Testing")
        
        print(f"\n📄 Test Document: {Path(document_path).name}")
        print(f"❓ Test Queries: {len(test_queries)}")
        
        try:
            # Test 1: Document Processing
            chunks = self.test_document_processing(document_path)
            
            # Test 2: Vector Store Creation
            store_name = f"test_store_{int(time.time())}"
            self.test_vector_store_creation(chunks, store_name)
            
            # Test 3: Retrieval
            self.test_retrieval(store_name, test_queries[:3])
            
            # Test 4: RAG Pipeline
            self.test_rag_pipeline(store_name, test_queries)
            
            # Generate summary
            self.generate_summary()
            
            # Save results
            self.save_results()
            
            print("\n✨ All tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            self.save_results()


def main():
    """Main test execution"""
    # Sample test queries (customize based on your document)
    test_queries = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important findings?",
        "Are there any recommendations mentioned?",
        "What methodology was used?"
    ]
    
    # Initialize tester
    tester = RAGPipelineTester()
    
    # Check for test document
    test_doc = Path("tests/test_documents/sample.pdf")
    
    if not test_doc.exists():
        print(f"⚠️  Test document not found: {test_doc}")
        print("Please place a test document (PDF, DOCX, or TXT) at:")
        print(f"  {test_doc}")
        print("\nYou can also modify the path in the script.")
        return
    
    # Run tests
    tester.run_full_test(str(test_doc), test_queries)


if __name__ == "__main__":
    main()