import streamlit as st
from pypdf import PdfReader
import os
import time
import warnings
from dotenv import load_dotenv

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 2. Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Try importing from classic, fallback to standard if needed
try:
    from langchain_classic.chains.question_answering import load_qa_chain
except ImportError:
    from langchain.chains.question_answering import load_qa_chain

from langchain_core.prompts import PromptTemplate

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Error: GOOGLE_API_KEY not found. Please check your .env file.")

FAISS_INDEX_PATH = "faiss_index"

# --- PDF Processing Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text if extracted_text else ""
    return text

def get_text_chunks(text):
    if not text.strip():
        st.warning("No text found to split into chunks.")
        return []
    
    # Keep chunk size small (1000) to avoid Free Tier limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        return

    try:
        st.info("Generating embeddings (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success("Index successfully updated!")
        
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")

# --- Conversational Chain with History ---
def get_conversational_chain():
    # UPDATED: Prompt now includes {chat_history}
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "answer is not available in the context".
    
    Chat History:
    {chat_history}
    
    Context: \n {context}\n
    Question: \n{question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Main Logic ---
def main():
    st.set_page_config("Chat with Multiple PDFs", layout="wide")
    st.header("Chat with PDF using Gemini PRO 🤖")

    # 1. Initialize Chat History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Sidebar for Uploads
    with st.sidebar:
        st.title("Menu: 📄")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        # Clear chat history when new files are processed
                        st.session_state.messages = []
            else:
                st.warning("Please upload at least one PDF.")
        
        # Add a "Clear Chat" button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # 3. Display Chat Messages from History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Handle User Input (Using st.chat_input for modern feel)
    if prompt := st.chat_input("Ask a Question from the PDF Files"):
        
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if Index Exists
        if not os.path.exists(FAISS_INDEX_PATH):
            st.error("Index not found. Please upload PDFs and process them first.")
            return

        # Prepare Chat History for the AI (Last 3 turns only to save tokens)
        # Format: "Human: ... \n AI: ..."
        history_text = ""
        for msg in st.session_state.messages[-6:]: # Last 6 messages = 3 turns
            role = "Human" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        # RAG Pipeline
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            
            # Retrieve Docs (k=5)
            docs = new_db.similarity_search(prompt, k=5)
            
            chain = get_conversational_chain()
            
            # Auto-Retry Logic
            max_retries = 3
            response_text = ""
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    for attempt in range(max_retries):
                        try:
                            response = chain.invoke(
                                {
                                    "input_documents": docs, 
                                    "question": prompt,
                                    "chat_history": history_text # Pass history here
                                },
                                return_only_outputs=True
                            )
                            response_text = response["output_text"]
                            st.markdown(response_text)
                            break 
                        except Exception as e:
                            if "429" in str(e) and attempt < max_retries - 1:
                                time.sleep(20) # Wait for Free Tier
                            else:
                                st.error(f"Error: {e}")
                                break
            
            # Add Assistant Response to History
            if response_text:
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
        except Exception as e:
            st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()