import streamlit as st
import os
import tempfile
import time
from typing import List, Tuple
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("Nova AI ☄️")
st.caption("Ask questions about documents from the documents folder AND uploaded docs")

# Configuration
DOCUMENTS_FOLDER = "documents/"
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hybrid-chatbot-docs")

# Initialize session state
if "folder_docs_processed" not in st.session_state:
    st.session_state.folder_docs_processed = False
    
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {}

if "folder_processing_status" not in st.session_state:
    st.session_state.folder_processing_status = {}

# Sidebar
st.sidebar.header("📂 Document Management")

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Check whether index exists, and create if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        st.info(f"Creating new index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        st.success("Index created successfully!")
    
    index = pc.Index(INDEX_NAME)
    return pc, index

pc, index = init_pinecone()

# Initialize embeddings model + vector store
@st.cache_resource
def init_embeddings_and_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return embeddings, vector_store

embeddings, vector_store = init_embeddings_and_vectorstore()

def process_folder_documents():
    """Process all files in the documents folder"""
    if not os.path.exists(DOCUMENTS_FOLDER):
        return 0, ["Documents folder does not exist"]
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return 0, ["No files found in documents folder"]
    
    try:
        # Use PyPDFDirectoryLoader for efficiency
        loader = PyPDFDirectoryLoader(DOCUMENTS_FOLDER)
        documents = loader.load()
        
        if not documents:
            return 0, ["No content found in files"]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to distinguish from uploaded files
        for chunk in chunks:
            # Extract filename from source path
            source_file = os.path.basename(chunk.metadata.get('source', 'unknown'))
            chunk.metadata.update({
                "document_type": "folder_document",
                "source_file": source_file,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "folder_path": DOCUMENTS_FOLDER
            })
        
        # Generate unique IDs
        existing_count = index.describe_index_stats()["total_vector_count"]
        uuids = [f"folder_doc_{existing_count + i + 1}" for i in range(len(chunks))]
        
        # Add to vector store
        vector_store.add_documents(documents=chunks, ids=uuids)
        
        # Update status for each file
        for pdf_file in pdf_files:
            file_chunks = [chunk for chunk in chunks if chunk.metadata.get('source_file') == pdf_file]
            st.session_state.folder_processing_status[pdf_file] = f"Processed: {len(file_chunks)} chunks"
        
        return len(chunks), []
        
    except Exception as e:
        return 0, [f"Error processing folder documents: {str(e)}"]

def process_uploaded_pdf(uploaded_file, vector_store):
    """Process a single uploaded file"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Create chunks
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata about the uploaded file
        for chunk in chunks:
            chunk.metadata.update({
                "document_type": "uploaded_document",
                "uploaded_file": uploaded_file.name,
                "source_file": uploaded_file.name,
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Generate unique IDs for chunks
        existing_docs_count = index.describe_index_stats()["total_vector_count"]
        uuids = [f"upload_{existing_docs_count + i + 1}" for i in range(len(chunks))]
        
        # Add to vector store
        vector_store.add_documents(documents=chunks, ids=uuids)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return len(chunks), None
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        return 0, str(e)

# Sidebar: Process folder documents
st.sidebar.subheader("📁 Documents Folder")

if os.path.exists(DOCUMENTS_FOLDER):
    pdf_files_in_folder = [f for f in os.listdir(DOCUMENTS_FOLDER) if f.lower().endswith('.pdf')]
    
    if pdf_files_in_folder:
        st.sidebar.write(f"Found {len(pdf_files_in_folder)} files in `{DOCUMENTS_FOLDER}`:")
        for pdf_file in pdf_files_in_folder:
            status = st.session_state.folder_processing_status.get(pdf_file, "Not processed")
            st.sidebar.write(f"📄 {pdf_file}")
            st.sidebar.caption(status)
        
        if not st.session_state.folder_docs_processed:
            if st.sidebar.button("🔄 Process Folder Documents", type="primary"):
                with st.sidebar:
                    with st.spinner("Processing folder documents..."):
                        chunks_added, errors = process_folder_documents()
                        
                        if errors:
                            for error in errors:
                                st.error(error)
                        else:
                            st.success(f"✅ Processed folder documents: {chunks_added} chunks added")
                            st.session_state.folder_docs_processed = True
                            time.sleep(1)
                            st.rerun()
        else:
            st.sidebar.success("✅ Folder documents processed")
    else:
        st.sidebar.info(f"No files found in `{DOCUMENTS_FOLDER}`")
else:
    st.sidebar.warning(f"Folder `{DOCUMENTS_FOLDER}` does not exist")

st.sidebar.divider()

# Sidebar: Upload additional documents
st.sidebar.subheader("⬆️ Upload Additional Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files to upload",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader"
)

# Process uploaded files
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
    
    if new_files:
        st.sidebar.write("Processing uploaded files...")
        progress_bar = st.sidebar.progress(0)
        
        for i, uploaded_file in enumerate(new_files):
            st.sidebar.write(f"Processing: {uploaded_file.name}")
            
            # Process the file
            chunks_count, error = process_uploaded_pdf(uploaded_file, vector_store)
            
            if error:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {error}")
                st.session_state.processing_status[uploaded_file.name] = f"Error: {error}"
            else:
                st.sidebar.success(f"✅ Processed {uploaded_file.name} ({chunks_count} chunks)")
                st.session_state.processing_status[uploaded_file.name] = f"Processed: {chunks_count} chunks"
                st.session_state.uploaded_files.append(uploaded_file.name)
            
            # Update progress bar
            progress_bar.progress((i + 1) / len(new_files))
        
        st.sidebar.write("All files processed!")
        time.sleep(1)
        st.rerun()

# Display uploaded files status
if st.session_state.uploaded_files:
    st.sidebar.write("### Uploaded Files:")
    for filename in st.session_state.uploaded_files:
        status = st.session_state.processing_status.get(filename, "Processed")
        st.sidebar.write(f"📄 {filename}")
        st.sidebar.caption(status)

st.sidebar.divider()

# Document statistics
stats = index.describe_index_stats()
total_docs = stats["total_vector_count"]
st.sidebar.metric("Total Document Chunks", total_docs)

# Clear all documents button
if st.sidebar.button("🗑️ Clear All Documents", type="secondary"):
    index.delete(delete_all=True)
    st.session_state.uploaded_files = []
    st.session_state.processing_status = {}
    st.session_state.folder_processing_status = {}
    st.session_state.folder_docs_processed = False
    st.sidebar.success("All documents cleared!")
    time.sleep(1)
    st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Check if any documents are available
    has_folder_docs = st.session_state.folder_docs_processed
    has_uploaded_docs = len(st.session_state.uploaded_files) > 0
    has_any_docs = has_folder_docs or has_uploaded_docs or total_docs > 0

    if not has_any_docs:
        st.info("👈 Please process folder documents and/or upload files using the sidebar to start chatting!")

with col2:
    # Document source summary
    if total_docs > 0:
        # st.subheader("📊 Document Sources")
        
        if has_folder_docs:
            folder_count = len([f for f in st.session_state.folder_processing_status.keys()])
            st.write(f"📁 Folder: {folder_count} files")
        
        if has_uploaded_docs:
            upload_count = len(st.session_state.uploaded_files)
            st.write(f"⬆️ Uploaded: {upload_count} files")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage("You are an assistant for question-answering tasks based on documents from both a documents folder and uploaded files.")
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create the chat input
if has_any_docs:
    prompt_text = "Ask a question about your documents..."
else:
    prompt_text = "Process or upload documents first to start chatting"

prompt = st.chat_input(prompt_text)

# Process user input
if prompt and has_any_docs:
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.5},
    )
    
    # Retrieve relevant documents
    with st.spinner("Searching through your documents..."):
        docs = retriever.invoke(prompt)
    
    # Prepare context from retrieved documents
    if docs:
        # Group documents by source type for better organization
        folder_docs = [doc for doc in docs if doc.metadata.get("document_type") == "folder_document"]
        uploaded_docs = [doc for doc in docs if doc.metadata.get("document_type") == "uploaded_document"]
        
        docs_text_parts = []
        
        if folder_docs:
            docs_text_parts.append("=== FROM DOCUMENTS FOLDER ===")
            for doc in folder_docs:
                source_file = doc.metadata.get('source_file', 'Unknown')
                docs_text_parts.append(f"Source: {source_file} (Folder)\n{doc.page_content}")
        
        if uploaded_docs:
            docs_text_parts.append("\n=== FROM UPLOADED DOCUMENTS ===")
            for doc in uploaded_docs:
                source_file = doc.metadata.get('source_file', 'Unknown')
                docs_text_parts.append(f"Source: {source_file} (Uploaded)\n{doc.page_content}")
        
        docs_text = "\n\n".join(docs_text_parts)
        
        # Show retrieved sources
        with st.expander(f"📚 Found {len(docs)} relevant sections"):
            st.write(f"**Folder documents:** {len(folder_docs)}")
            st.write(f"**Uploaded documents:** {len(uploaded_docs)}")
            
            for doc in docs:
                doc_type = "📁 Folder" if doc.metadata.get("document_type") == "folder_document" else "⬆️ Uploaded"
                source_file = doc.metadata.get('source_file', 'Unknown')
                st.write(f"**{doc_type}:** {source_file}")
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                st.write(content_preview)
                st.divider()
    else:
        docs_text = "No relevant information found in the available documents."
    
    # Create system prompt with context
    system_prompt = """You are an assistant for question-answering tasks based on documents from both a documents folder and uploaded files. 
    Use the following pieces of retrieved context to answer the question. 
    When referencing information, please specify whether it comes from folder documents or uploaded documents.
    If you don't know the answer based on the provided context, just say that you don't know. Please stick with provided context. 
    Keep the answer concise and short.
    
    Context: {context}"""
    
    system_prompt_fmt = system_prompt.format(context=docs_text)
    
    # Add system prompt to messages (temporarily for this query)
    temp_messages = st.session_state.messages + [SystemMessage(system_prompt_fmt)]
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            result = llm.invoke(temp_messages).content
        st.markdown(result)
    
    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(result))

elif prompt and not has_any_docs:

    st.warning("Please process folder documents and/or upload documents first to ask questions!")
