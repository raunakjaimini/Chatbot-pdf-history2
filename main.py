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

# Set the page configuration
st.set_page_config(page_title="Chatbot", page_icon="", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CSS for custom font colors and layout
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212;
        color: #e0e0e0;
    }
    .title {
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        color: #bb86fc;
        text-align: center;
        margin-bottom: 20px;
    }
    .text-input, .button {
        background-color: #333333;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 4px;
    }
    .text-input::placeholder {
        color: #888;
    }
    .button:hover {
        background-color: #444;
    }
    .response {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.3);
        color: #e0e0e0;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 20px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store for the text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain using the Gemini LLM
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context."; don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to process user input, search for similar content, and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Save the question and response to session state
    st.session_state.chat_history.append({"user": user_question, "assistant": response["output_text"]})
    
    # Display the response
    st.markdown(f"<div class='response'><strong>Reply:</strong> {response['output_text']}</div>", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.markdown("<h1 class='title'>Chat-Mate...I can read any pdf file and maintain history</h1>", unsafe_allow_html=True)

    
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])

    # Input field for user's message
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", placeholder="Type your question here...")

    if st.button("Submit", key="submit_button"):
        if 'pdf_docs' in st.session_state:
            with st.spinner("Processing..."):
                user_input(user_question)
                st.success("Processing Done")
        else:
            st.error("No PDF file uploaded. Please upload a file in the sidebar.")

    with st.sidebar:
        st.markdown("<h3 class='header'>Upload Your PDF Files</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, key="pdf_uploader")
        
        if pdf_docs:
            st.session_state['pdf_docs'] = pdf_docs  # Save uploaded files to session state
        
        if st.button("Process and Get Answer"):
            if 'pdf_docs' in st.session_state and st.session_state['pdf_docs']:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(st.session_state['pdf_docs'])
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Your file has been processed. You can now ask questions.")
            else:
                st.error("Please upload your PDF files first.")

if __name__ == "__main__":
    main()

