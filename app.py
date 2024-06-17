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

load_dotenv()

flag = True

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except:
    flag = False
    genai.configure(api_key=st.secrets["api_key"])

FAISS_INDEX_PATH = "faiss_index"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)  # Ensure the file pointer is at the start
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            print(f"Extracted text from page: {extracted_text}")
            text += extracted_text if extracted_text else ""
    print(f"Final extracted text: {text}")
    return text


def get_text_chunks(text):
    if text.strip() == "":
        print("No text found to split into chunks.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(f"Generated text chunks: {chunks}")
    return chunks


def get_vector_store(text_chunks):
    if flag:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["api_key"])
    print(f"Generating embeddings for text chunks: {text_chunks}")
    if text_chunks:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)
    vector_store.save_local(FAISS_INDEX_PATH)


def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context,
    make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide wrong information\n\n
    Context: \n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    if flag:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.6)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.6, google_api_key=st.secrets["api_key"])

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    if flag:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["api_key"])
    try:
        new_db = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        print(response)
        st.session_state["last_response"] = response["output_text"]
        st.write("Reply: ", response["output_text"])
    except FileNotFoundError as e:
        print(f"Error loading FAISS index: {e}")
        st.error("FAISS index not found. Please process the PDFs first.")


def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with PDF using Gemini")

    if "last_response" not in st.session_state:
        st.session_state["last_response"] = ""

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    else:
        st.write("Reply: ", st.session_state["last_response"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process!",
            accept_multiple_files=True,
            type="pdf",
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.session_state["last_response"] = ""  # Reset the response on new file upload
                user_question = ""  # Reset the question input box
                raw_text = get_pdf_text(pdf_docs)

                # Debug: Print raw text to ensure it is extracted correctly
                print("Raw Text:", raw_text)

                text_chunks = get_text_chunks(raw_text)

                # Debug: Print text chunks to ensure they are created correctly
                print("Text Chunks:", text_chunks)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.success("Done!")
                else:
                    st.warning("Please select a valid PDF document!")


if __name__ == "__main__":
    main()
