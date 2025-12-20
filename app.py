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
    
    # --- CRITICAL FIX: Smaller Chunks ---
    # Reduced from 10,000 to 1,000 to save token usage
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
        st.info("Generating embeddings locally (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success("Index successfully updated! (Optimized for Free Tier)")
        
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "answer is not available in the context".
    
    Context: \n {context}\n
    Question: \n{question}\n
    
    Answer:
    """

    # Using 'gemini-flash-latest' (1.5 Flash) which usually has the highest RPM limit
    # If this fails, try "gemini-1.5-pro-latest"
    model = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("Index not found. Please upload PDFs and process them first.")
        return

    try:
        new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # --- CRITICAL FIX: Retrieve fewer docs ---
        # Reduced from k=10 to k=5 to keep total tokens low
        docs = new_db.similarity_search(user_question, k=5)
        
        chain = get_conversational_chain()
        
        # --- RETRY LOGIC ---
        # If Google says "Wait 20s", we wait 20s automatically.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner("Asking Gemini..."):
                    response = chain.invoke(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                st.write("Reply: ", response["output_text"])
                break # Success
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait_time = 20 # The error explicitly asked for ~19s
                    st.warning(f"Free Tier Limit Hit. Waiting {wait_time}s before retrying... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error(f"Error: {e}")
                    break
        
    except Exception as e:
        st.error(f"System Error: {e}")

def main():
    st.set_page_config("Chat with Multiple PDFs", layout="wide")
    st.header("Chat with PDF using Gemini (Free Tier Optimized) 🤖")

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
            else:
                st.warning("Please upload at least one PDF.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()