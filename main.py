import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if PDFs are uploaded and processed
    if "pdf_docs" in st.session_state and st.session_state.pdf_docs:
        raw_text = get_pdf_text(st.session_state.pdf_docs)
    else:
        st.error("No PDF files uploaded.")
        return

    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Load the FAISS index with the deserialization allowance
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Generate the response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Save the question and response to session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"user": user_question, "assistant": response["output_text"]})

    # Display the response
    st.write("Reply: ", response["output_text"])
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat-Mate... I can read any PDF file")
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Initialize PDF documents in session state if not present
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**User:** {chat['user']}")
                st.markdown(f"**Assistant:** {chat['assistant']}")
    # File uploader
    uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")
    # Save uploaded files to session state
    if uploaded_files:
        st.session_state.pdf_docs = uploaded_files
        st.session_state.is_new_pdf = True
    else:
        st.session_state.is_new_pdf = False
    # Text input for user question
    user_question = st.text_input("Ask a Question from the PDF File")
    if st.button("Process and Get Answer"):
        if st.session_state.pdf_docs:
            with st.spinner("Processing..."):
                if st.session_state.is_new_pdf:
                    # Process only if new files are uploaded
                    process_user_question(user_question)
                    st.session_state.is_new_pdf = False
                else:
                    # Use previously processed files
                    process_user_question(user_question)
                st.success("Processing Done")
        else:
            st.error("No PDF file available for processing.")
if __name__ == "__main__":
    main()
