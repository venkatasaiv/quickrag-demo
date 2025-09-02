import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
import tempfile
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError:
    st.error("""
    Missing required dependencies. Please install them using:
    
    ```bash
    pip install langchain openai faiss-cpu pypdf
    ```
    """)
    st.stop()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="QuickRAG Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    margin-top: 1rem;
}
.upload-section {
    border: 2px dashed #cccccc;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.chat-message {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 10px;
    border-left: 4px solid #007ACC;
    background-color: #f8f9fa;
}
.source-citation {
    background-color: #e8f4f8;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def check_api_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("""
        ⚠️ **OpenAI API Key Required**
        
        Please set your OpenAI API key in the environment variables:
        1. Create a `.env` file in your project directory
        2. Add: `OPENAI_API_KEY=your_api_key_here`
        3. Restart the application
        
        You can get your API key from: https://platform.openai.com/api-keys
        """)
        return False
    return True

@st.cache_data
def load_and_process_pdfs(uploaded_files) -> List[Document]:
    """Load and process uploaded PDF files"""
    documents = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            
            # Add source metadata
            for page in pages:
                page.metadata['source'] = uploaded_file.name
                page.metadata['page'] = page.metadata.get('page', 0)
            
            documents.extend(pages)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    return documents

@st.cache_resource
def create_vectorstore(documents: List[Document]):
    """Create FAISS vectorstore from documents"""
    if not documents:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def get_qa_chain(vectorstore):
    """Create QA chain with retrieval"""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=500
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
    )
    
    return qa_chain

def format_sources(source_docs):
    """Format source documents for display"""
    sources = []
    for i, doc in enumerate(source_docs, 1):
        source_info = {
            'number': i,
            'filename': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 'Unknown'),
            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        sources.append(source_info)
    return sources

def main():
    # Header
    st.title("📚 QuickRAG Demo")
    st.markdown(
        "**An interactive RAG application for document Q&A. Upload PDFs and ask questions!**"
    )
    
    # Check API key
    if not check_api_key():
        return
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
            # Process documents button
            if st.button("🚀 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Load and process documents
                    documents = load_and_process_pdfs(uploaded_files)
                    
                    if documents:
                        # Create vectorstore
                        st.session_state.vectorstore = create_vectorstore(documents)
                        st.session_state.documents = documents
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("No documents were processed successfully")
        
        # Settings
        st.header("⚙️ Settings")
        
        chunk_size = st.slider(
            "Chunk Size", 
            min_value=500, 
            max_value=2000, 
            value=1000,
            help="Size of text chunks for processing"
        )
        
        num_results = st.slider(
            "Number of Results", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Number of relevant chunks to retrieve"
        )
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    if not st.session_state.vectorstore:
        # Welcome message
        st.markdown("""
        ### 👋 Welcome to QuickRAG Demo!
        
        To get started:
        1. **Upload PDF files** using the sidebar
        2. **Click "Process Documents"** to analyze them
        3. **Ask questions** about your documents
        
        💡 **Example questions you can ask:**
        - "What is the main topic of these documents?"
        - "Can you summarize the key findings?"
        - "What are the conclusions?"
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📄 Smart Processing
            Automatically extracts and chunks text from your PDFs for optimal retrieval
            """)
        
        with col2:
            st.markdown("""
            #### 🔍 Semantic Search
            Uses advanced embeddings to find the most relevant information
            """)
        
        with col3:
            st.markdown("""
            #### 📝 Source Citations
            Every answer includes references to the original documents
            """)
    
    else:
        # Document stats
        num_docs = len(st.session_state.documents)
        st.success(f"✅ Ready! Processed {num_docs} document(s). Ask your questions below.")
        
        # Chat interface
        st.header("💬 Ask Questions")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'question':
                st.markdown(f"**🙋 You:** {message['content']}")
            else:
                st.markdown(
                    f'<div class="chat-message"><strong>🤖 QuickRAG:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sources
                if 'sources' in message and message['sources']:
                    with st.expander("📚 View Sources", expanded=False):
                        for source in message['sources']:
                            st.markdown(
                                f'<div class="source-citation">' +
                                f'<strong>Source {source["number"]}:</strong> {source["filename"]} ' +
                                f'(Page {source["page"]})<br>' +
                                f'<em>{source["content"]}</em></div>',
                                unsafe_allow_html=True
                            )
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            key="question_input"
        )
        
        # Answer button
        col1, col2 = st.columns([3, 1])
        
        with col2:
            ask_button = st.button("🚀 Get Answer", use_container_width=True)
        
        # Process question
        if (ask_button or question) and question and not st.session_state.processing:
            st.session_state.processing = True
            
            # Add question to chat history
            st.session_state.chat_history.append({
                'type': 'question',
                'content': question
            })
            
            # Get answer
            with st.spinner("Thinking..."):
                try:
                    qa_chain = get_qa_chain(st.session_state.vectorstore)
                    result = qa_chain({"query": question})
                    
                    answer = result['result']
                    sources = format_sources(result['source_documents'])
                    
                    # Add answer to chat history
                    st.session_state.chat_history.append({
                        'type': 'answer',
                        'content': answer,
                        'sources': sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
            
            st.session_state.processing = False
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
            Made with ❤️ using Streamlit, LangChain, and OpenAI<br>
            <a href="https://github.com/venkatasaiv/quickrag-demo" target="_blank">View on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
