import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the path where the FAISS index will be saved
FAISS_INDEX_PATH = "faiss_index"

# --- PDF Processing Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        # st.file_uploader uses a temporary file, we need to ensure the pointer is at the start
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text if extracted_text else ""
    return text

def get_text_chunks(text):
    """Splits the raw text into manageable chunks for embedding."""
    if text.strip() == "":
        st.warning("No text found to split into chunks.")
        return []
    # Using a large chunk size/overlap for large documents, adjust as needed.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# --- CORE FIX: Caching Function ---

def get_vector_store(text_chunks):
    """
    Creates and saves the FAISS vector store.
    CRITICALLY: It only calls the Gemini Embeddings API if the index does not exist.
    """
    # 1. Check if the index already exists locally
    if os.path.exists(FAISS_INDEX_PATH):
        st.success("FAISS index found locally. Skipping API call.")
        return

    # 2. If it doesn't exist, proceed with API call
    if text_chunks:
        st.info("No existing index found. Generating embeddings and saving vector store (one-time operation)...")
        
        # Initialize Embeddings model (API call starts here)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create Vector Store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Ensure the directory exists
        if not os.path.exists(FAISS_INDEX_PATH):
            os.makedirs(FAISS_INDEX_PATH)
            
        # Save the index locally
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success(f"Successfully created and saved new FAISS index to {FAISS_INDEX_PATH}!")
    else:
        st.warning("Cannot create vector store: No text chunks provided.")


# --- Conversational Chain Functions ---

def get_conversational_chain():
    """Sets up the LangChain QA chain using Gemini 1.5 Pro."""
    prompt_template = """Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide wrong information\n\n
    Context: \n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.6)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """Retrieves context from the vector store and generates the response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if the FAISS index exists before attempting to load
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("FAISS index not found. Please upload and process the PDFs first.")
        return

    try:
        # Load the local index
        new_db = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        
        # Search for relevant documents
        docs = new_db.similarity_search(user_question)
        
        # Get the conversational chain
        chain = get_conversational_chain()
        
        # Run the chain
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        
        st.session_state["last_response"] = response["output_text"]
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        # Catch any errors during loading or chain execution
        st.error(f"An error occurred during query processing: {e}")
        print(f"Error details: {e}")


# --- Streamlit Main App ---

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with PDF using Gemini 🤖")

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = ""

    # Question/Chat form
    with st.form(key="question_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_question = st.text_input("Ask a Question from the PDF Files")
        with col2:
            reset_button = st.form_submit_button(label="Reset")

        col3, col4 = st.columns([1, 1])
        with col3:
            submit_button = st.form_submit_button(label="Submit")

        if submit_button and user_question:
            user_input(user_question)
        elif reset_button:
            st.session_state["last_response"] = ""
            st.rerun()

    # Sidebar for File Upload and Processing
    with st.sidebar:
        st.title("Menu: 📄")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process!",
            accept_multiple_files=True,
            type="pdf",
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # 1. Extract Text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Split Text into Chunks
                    text_chunks = get_text_chunks(raw_text)

                    # 3. Create/Load Vector Store (Quota-friendly)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Indexing complete! You can now ask questions.")
                    else:
                        st.warning("Could not extract any meaningful text.")
            else:
                 st.warning("Please upload at least one PDF document.")


if __name__ == "__main__":
    main()