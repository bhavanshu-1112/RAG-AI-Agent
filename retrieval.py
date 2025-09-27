import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

class HybridRetriever:
    """
    Enhanced retrieval system for hybrid document sources
    """
    
    def __init__(self, index_name: Optional[str] = None):
        # Initialize pinecone database
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        # Set the pinecone index (using hybrid system index)
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "hybrid-chatbot-docs")
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embeddings model + vector store
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
    
    def get_index_statistics(self) -> Dict:
        """Get comprehensive index statistics"""
        stats = self.index.describe_index_stats()
        return {
            "index_name": self.index_name,
            "total_vectors": stats["total_vector_count"],
            "dimension": stats.get("dimension", "Unknown")
        }
    
    def search_all_documents(self, 
                           query: str, 
                           k: int = 6, 
                           score_threshold: float = 0.5) -> List[Document]:
        """Search across all documents (folder + uploaded)"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        return retriever.invoke(query)
    
    def search_folder_documents(self, 
                              query: str, 
                              k: int = 6, 
                              score_threshold: float = 0.5) -> List[Document]:
        """Search only in folder documents"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k, 
                "score_threshold": score_threshold,
                "filter": {"document_type": "folder_document"}
            },
        )
        return retriever.invoke(query)
    
    def search_uploaded_documents(self, 
                                query: str, 
                                k: int = 6, 
                                score_threshold: float = 0.5) -> List[Document]:
        """Search only in uploaded documents"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k, 
                "score_threshold": score_threshold,
                "filter": {"document_type": "uploaded_document"}
            },
        )
        return retriever.invoke(query)
    
    def search_by_source_file(self, 
                            query: str, 
                            source_file: str,
                            k: int = 6, 
                            score_threshold: float = 0.3) -> List[Document]:
        """Search within a specific source file"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k, 
                "score_threshold": score_threshold,
                "filter": {"source_file": source_file}
            },
        )
        return retriever.invoke(query)
    
    def compare_search_results(self, query: str) -> Dict:
        """Compare search results across different document types"""
        print(f"🔍 Comparing search results for: '{query}'")
        print("=" * 60)
        
        # Search all documents
        all_results = self.search_all_documents(query)
        folder_results = self.search_folder_documents(query)
        uploaded_results = self.search_uploaded_documents(query)
        
        comparison = {
            "query": query,
            "all_documents": {
                "count": len(all_results),
                "results": all_results
            },
            "folder_documents": {
                "count": len(folder_results),
                "results": folder_results
            },
            "uploaded_documents": {
                "count": len(uploaded_results),
                "results": uploaded_results
            }
        }
        
        # Display comparison
        print(f"📊 Results Summary:")
        print(f"  All Documents: {len(all_results)} results")
        print(f"  Folder Documents: {len(folder_results)} results")
        print(f"  Uploaded Documents: {len(uploaded_results)} results")
        print()
        
        return comparison
    
    def analyze_document_coverage(self) -> Dict:
        """Analyze what types of documents are in the index"""
        # This is a simplified analysis since Pinecone doesn't directly support 
        # querying by metadata without a search query
        print("📋 Document Coverage Analysis")
        print("=" * 40)
        
        stats = self.get_index_statistics()
        print(f"Index: {stats['index_name']}")
        print(f"Total documents: {stats['total_vectors']}")
        
        # Sample search to understand document types
        sample_results = self.search_all_documents("document", k=20, score_threshold=0.1)
        
        folder_docs = [doc for doc in sample_results if doc.metadata.get("document_type") == "folder_document"]
        uploaded_docs = [doc for doc in sample_results if doc.metadata.get("document_type") == "uploaded_document"]
        
        folder_files = set(doc.metadata.get("source_file", "Unknown") for doc in folder_docs)
        uploaded_files = set(doc.metadata.get("source_file", "Unknown") for doc in uploaded_docs)
        
        analysis = {
            "total_sampled": len(sample_results),
            "folder_chunks_sampled": len(folder_docs),
            "uploaded_chunks_sampled": len(uploaded_docs),
            "folder_files_found": list(folder_files),
            "uploaded_files_found": list(uploaded_files)
        }
        
        print(f"Sample of {len(sample_results)} chunks:")
        print(f"  Folder document chunks: {len(folder_docs)}")
        print(f"  Uploaded document chunks: {len(uploaded_docs)}")
        print(f"  Folder files found: {len(folder_files)}")
        print(f"  Uploaded files found: {len(uploaded_files)}")
        
        if folder_files:
            print(f"\nFolder files:")
            for file in sorted(folder_files):
                print(f"  📁 {file}")
        
        if uploaded_files:
            print(f"\nUploaded files:")
            for file in sorted(uploaded_files):
                print(f"  ⬆️ {file}")
        
        return analysis

def display_search_results(results: List[Document], title: str):
    """Display search results in a formatted way"""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not results:
        print("❌ No results found")
        return
    
    for i, doc in enumerate(results, 1):
        doc_type = doc.metadata.get("document_type", "unknown")
        source_file = doc.metadata.get("source_file", "Unknown")
        processing_time = doc.metadata.get("processing_time", "Unknown")
        
        # Determine icon based on document type
        if doc_type == "folder_document":
            type_icon = "📁 Folder"
        elif doc_type == "uploaded_document":
            type_icon = "⬆️ Uploaded"
        else:
            type_icon = "❓ Unknown"
        
        print(f"\n📄 Result {i}")
        print(f"Type: {type_icon}")
        print(f"Source: {source_file}")
        print(f"Processed: {processing_time}")
        print(f"Content: {doc.page_content[:150]}...")
        print("-" * 50)

def test_hybrid_retrieval():
    """Run comprehensive tests on the hybrid retrieval system"""
    print("🧪 Testing Hybrid Retrieval System")
    print("=" * 50)
    
    retriever = HybridRetriever()
    
    # Get index statistics
    stats = retriever.get_index_statistics()
    print(f"📊 Index Statistics:")
    print(f"  Name: {stats['index_name']}")
    print(f"  Total Vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")
    
    if stats['total_vectors'] == 0:
        print("\n⚠️ No documents found in the index!")
        print("Please process some documents first using:")
        print("1. The Streamlit app to upload PDFs")
        print("2. The hybrid_initialization.py script")
        return
    
    print(f"\n🔍 Running Test Queries...")
    
    # Define test queries
    test_queries = [
        "What is the main topic?",
        "summarize the key points",
        "tell me about the document",
        "important information"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        
        # Test all document search
        all_results = retriever.search_all_documents(query, k=3)
        display_search_results(all_results, f"🔍 All Documents ({len(all_results)} results)")
        
        # Test folder document search
        folder_results = retriever.search_folder_documents(query, k=3)
        display_search_results(folder_results, f"📁 Folder Documents ({len(folder_results)} results)")
        
        # Test uploaded document search
        uploaded_results = retriever.search_uploaded_documents(query, k=3)
        display_search_results(uploaded_results, f"⬆️ Uploaded Documents ({len(uploaded_results)} results)")
        
        print("\n" + "="*60)

def interactive_hybrid_search():
    """Interactive search with document type selection"""
    print("🎯 Interactive Hybrid Search")
    print("=" * 40)
    
    retriever = HybridRetriever()
    
    # Check if documents exist
    stats = retriever.get_index_statistics()
    if stats['total_vectors'] == 0:
        print("⚠️ No documents in index. Upload some documents first!")
        return
    
    print(f"📚 Ready to search {stats['total_vectors']} document chunks")
    print("\nCommands:")
    print("  'quit' or 'q' - Exit")
    print("  'analysis' - Show document coverage analysis")
    print("  'compare <query>' - Compare results across document types")
    
    while True:
        print("\n" + "-"*40)
        query = input("Enter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if query.lower() == 'analysis':
            retriever.analyze_document_coverage()
            continue
        
        if query.lower().startswith('compare '):
            compare_query = query[8:].strip()
            if compare_query:
                retriever.compare_search_results(compare_query)
            else:
                print("Please provide a query to compare")
            continue
        
        if not query:
            print("Please enter a valid query")
            continue
        
        print("\nChoose search scope:")
        print("1. All documents")
        print("2. Folder documents only")
        print("3. Uploaded documents only")
        print("4. Specific file")
        
        try:
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                results = retriever.search_all_documents(query)
                display_search_results(results, f"🔍 All Documents - '{query}'")
            
            elif choice == "2":
                results = retriever.search_folder_documents(query)
                display_search_results(results, f"📁 Folder Documents - '{query}'")
            
            elif choice == "3":
                results = retriever.search_uploaded_documents(query)
                display_search_results(results, f"⬆️ Uploaded Documents - '{query}'")
            
            elif choice == "4":
                source_file = input("Enter source filename: ").strip()
                if source_file:
                    results = retriever.search_by_source_file(query, source_file)
                    display_search_results(results, f"📄 File '{source_file}' - '{query}'")
                else:
                    print("No filename provided")
            
            else:
                print("Invalid choice")
        
        except KeyboardInterrupt:
            print("\nSearch cancelled")
            continue

def main():
    """Main function with different modes"""
    print("🔧 Hybrid RAG Retrieval Testing Tool")
    print("=" * 50)
    
    # Check environment variables
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found in environment variables")
        return
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    print("Choose a mode:")
    print("1. Run comprehensive tests")
    print("2. Interactive search")
    print("3. Document coverage analysis")
    print("4. Compare search across document types")
    print("5. Show index statistics only")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            test_hybrid_retrieval()
        
        elif choice == "2":
            interactive_hybrid_search()
        
        elif choice == "3":
            retriever = HybridRetriever()
            retriever.analyze_document_coverage()
        
        elif choice == "4":
            query = input("Enter query to compare: ").strip()
            if query:
                retriever = HybridRetriever()
                retriever.compare_search_results(query)
            else:
                print("No query provided")
        
        elif choice == "5":
            retriever = HybridRetriever()
            stats = retriever.get_index_statistics()
            print(f"\n📊 Index Statistics:")
            print(f"Name: {stats['index_name']}")
            print(f"Total Vectors: {stats['total_vectors']}")
            print(f"Dimension: {stats['dimension']}")
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()