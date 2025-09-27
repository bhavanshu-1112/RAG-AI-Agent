import os
import time
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

class HybridDocumentProcessor:
    """
    Enhanced processor that handles both folder documents and individual uploads
    """
    
    def __init__(self, 
                 index_name: Optional[str] = None,
                 documents_folder: str = "documents/"):
        
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "hybrid-chatbot-docs")
        self.documents_folder = documents_folder
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Initialize index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            print(f"Index {self.index_name} created successfully!")
    
    def process_folder_documents(self) -> Tuple[int, List[str], Dict[str, int]]:
        """
        Process all PDF files in the documents folder
        
        Returns:
            Tuple of (total_chunks_added, list_of_errors, file_chunk_counts)
        """
        if not os.path.exists(self.documents_folder):
            return 0, [f"Documents folder {self.documents_folder} does not exist"], {}
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(self.documents_folder) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return 0, [f"No PDF files found in {self.documents_folder}"], {}
        
        print(f"Found {len(pdf_files)} PDF files in folder:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        
        try:
            # Load all PDFs from directory
            loader = PyPDFDirectoryLoader(self.documents_folder)
            documents = loader.load()
            
            if not documents:
                return 0, ["No content found in PDF files"], {}
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to distinguish from uploaded files
            file_chunk_counts = {}
            for chunk in chunks:
                # Extract filename from source path
                source_path = chunk.metadata.get('source', 'unknown')
                source_file = os.path.basename(source_path)
                
                chunk.metadata.update({
                    "document_type": "folder_document",
                    "source_file": source_file,
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "folder_path": self.documents_folder,
                    "original_source": source_path
                })
                
                # Count chunks per file
                file_chunk_counts[source_file] = file_chunk_counts.get(source_file, 0) + 1
            
            # Generate unique IDs
            existing_count = self.index.describe_index_stats()["total_vector_count"]
            uuids = [f"folder_doc_{existing_count + i + 1}" for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_documents(documents=chunks, ids=uuids)
            
            print(f"Successfully processed folder documents:")
            for file, count in file_chunk_counts.items():
                print(f"  - {file}: {count} chunks")
            
            return len(chunks), [], file_chunk_counts
            
        except Exception as e:
            error_msg = f"Error processing folder documents: {str(e)}"
            print(error_msg)
            return 0, [error_msg], {}
    
    def process_single_pdf(self, 
                          pdf_path: str, 
                          source_name: Optional[str] = None,
                          document_type: str = "uploaded_document") -> Tuple[int, Optional[str]]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            source_name: Optional name for the source (defaults to filename)
            document_type: Type of document (uploaded_document or folder_document)
        
        Returns:
            Tuple of (number_of_chunks_added, error_message)
        """
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                return 0, "No content found in PDF"
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            source = source_name or os.path.basename(pdf_path)
            for chunk in chunks:
                metadata_update = {
                    "document_type": document_type,
                    "source_file": source,
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_index": chunks.index(chunk)
                }
                
                if document_type == "uploaded_document":
                    metadata_update["uploaded_file"] = source
                    metadata_update["upload_time"] = metadata_update["processing_time"]
                else:
                    metadata_update["folder_path"] = os.path.dirname(pdf_path)
                    metadata_update["original_source"] = pdf_path
                
                chunk.metadata.update(metadata_update)
            
            # Generate unique IDs
            existing_count = self.index.describe_index_stats()["total_vector_count"]
            prefix = "upload" if document_type == "uploaded_document" else "folder"
            uuids = [f"{prefix}_{existing_count + i + 1}" for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_documents(documents=chunks, ids=uuids)
            
            print(f"Successfully processed {source}: {len(chunks)} chunks added")
            return len(chunks), None
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}"
            print(error_msg)
            return 0, error_msg
    
    def search_documents(self, 
                        query: str, 
                        k: int = 6, 
                        score_threshold: float = 0.5,
                        filter_by_type: Optional[str] = None) -> List[Document]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            filter_by_type: Optional filter by document_type
        
        Returns:
            List of relevant documents
        """
        search_kwargs = {"k": k, "score_threshold": score_threshold}
        
        if filter_by_type:
            search_kwargs["filter"] = {"document_type": filter_by_type}
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs,
        )
        return retriever.invoke(query)
    
    def get_document_stats(self) -> Dict:
        """Get comprehensive statistics about documents"""
        stats = self.index.describe_index_stats()
        
        # Try to get breakdown by document type (this is a simplified approach)
        # In practice, you might want to store this metadata separately
        return {
            "total_chunks": stats["total_vector_count"],
            "index_dimension": stats.get("dimension", 3072),
            "index_name": self.index_name
        }
    
    def clear_all_documents(self):
        """Clear all documents from the index"""
        self.index.delete(delete_all=True)
        print(f"Cleared all documents from index: {self.index_name}")
    
    def clear_documents_by_type(self, document_type: str):
        """
        Clear documents by type (folder_document or uploaded_document)
        Note: This is a simplified implementation. In production, you'd want 
        more sophisticated document tracking.
        """
        print(f"Note: Clearing by document type requires tracking document IDs")
        print(f"For now, this would require a more complex implementation")
    
    def initialize_system(self, process_folder: bool = True) -> Dict:
        """
        Initialize the entire system by processing folder documents
        
        Args:
            process_folder: Whether to process documents in the folder
        
        Returns:
            Dictionary with initialization results
        """
        results = {
            "folder_processed": False,
            "folder_chunks": 0,
            "folder_files": {},
            "errors": []
        }
        
        if process_folder:
            print("=== INITIALIZING HYBRID DOCUMENT SYSTEM ===")
            print(f"Processing documents from folder: {self.documents_folder}")
            
            folder_chunks, folder_errors, file_counts = self.process_folder_documents()
            
            results.update({
                "folder_processed": folder_chunks > 0,
                "folder_chunks": folder_chunks,
                "folder_files": file_counts,
                "errors": folder_errors
            })
            
            if folder_chunks > 0:
                print(f"✅ Folder initialization complete: {folder_chunks} chunks from {len(file_counts)} files")
            else:
                print("⚠️ No documents processed from folder")
        
        # Display final stats
        stats = self.get_document_stats()
        print(f"\n=== SYSTEM READY ===")
        print(f"Index: {stats['index_name']}")
        print(f"Total chunks: {stats['total_chunks']}")
        
        results["final_stats"] = stats
        return results

def main():
    """Main function for standalone execution"""
    print("🔧 Hybrid Document Processor")
    print("=" * 50)
    
    # Check environment variables
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY not found in environment variables")
        return
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize processor
    processor = HybridDocumentProcessor()
    
    print("Choose an option:")
    print("1. Initialize system (process folder documents)")
    print("2. Process a specific PDF file")
    print("3. Search documents")
    print("4. View system statistics")
    print("5. Clear all documents")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            results = processor.initialize_system(process_folder=True)
            if results["errors"]:
                print("Errors encountered:")
                for error in results["errors"]:
                    print(f"  - {error}")
        
        elif choice == "2":
            pdf_path = input("Enter path to PDF file: ").strip()
            if os.path.exists(pdf_path):
                chunks, error = processor.process_single_pdf(pdf_path)
                if error:
                    print(f"❌ Error: {error}")
                else:
                    print(f"✅ Success: Added {chunks} chunks")
            else:
                print("❌ File not found")
        
        elif choice == "3":
            query = input("Enter search query: ").strip()
            if query:
                print("Choose document type to search:")
                print("1. All documents")
                print("2. Folder documents only")
                print("3. Uploaded documents only")
                
                doc_choice = input("Enter choice (1-3): ").strip()
                filter_type = None
                
                if doc_choice == "2":
                    filter_type = "folder_document"
                elif doc_choice == "3":
                    filter_type = "uploaded_document"
                
                results = processor.search_documents(query, filter_by_type=filter_type)
                
                if results:
                    print(f"\n🔍 Found {len(results)} relevant documents:")
                    print("=" * 60)
                    
                    for i, doc in enumerate(results, 1):
                        doc_type = "📁 Folder" if doc.metadata.get("document_type") == "folder_document" else "⬆️ Uploaded"
                        source_file = doc.metadata.get('source_file', 'Unknown')
                        processing_time = doc.metadata.get('processing_time', 'Unknown')
                        
                        print(f"\n📄 Result {i}")
                        print(f"Type: {doc_type}")
                        print(f"Source: {source_file}")
                        print(f"Processed: {processing_time}")
                        print(f"Content Preview:")
                        print(f"{doc.page_content[:200]}...")
                        print("-" * 60)
                else:
                    print("❌ No relevant documents found")
            else:
                print("❌ No query provided")
        
        elif choice == "4":
            stats = processor.get_document_stats()
            print(f"\n📊 System Statistics:")
            print(f"Index Name: {stats['index_name']}")
            print(f"Total Chunks: {stats['total_chunks']}")
            print(f"Dimension: {stats['index_dimension']}")
        
        elif choice == "5":
            confirm = input("Are you sure you want to clear ALL documents? (yes/no): ").strip().lower()
            if confirm == "yes":
                processor.clear_all_documents()
                print("✅ All documents cleared")
            else:
                print("Operation cancelled")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()