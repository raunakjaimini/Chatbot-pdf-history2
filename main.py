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

# Set the page configuration at the top
st.set_page_config(page_title="SkyChat 3.0.0", page_icon="👽", layout="wide")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CSS for modern and clean design
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #1e90ff;  /* Dodger Blue */
        font-size: 2.5em;
        text-align: center;
        margin-top: 20px;
    }
    .header {
        color: #ff6347;  /* Tomato */
        font-size: 1.5em;
        text-align: center;
    }
    .sidebar {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .success {
        color: #28a745;  /* Success Green */
    }
    .text-input {
        color: #333;
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .chat-container {
        margin: 0 auto;
        max-width: 800px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fff;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #d1e7dd;  /* Light Green */
        text-align: left;
    }
    .assistant-message {
        background-color: #e2e3e5;  /* Light Gray */
        text-align: right;
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
    the provided context, just say, "Answer is not available in the context"; don't provide the wrong answer.\n\n
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
    st.write("Reply: ", response["output_text"])

# Main function for the Streamlit app
def main():
    st.markdown("<h1 class='title'>👽SkyChat 3.0.0</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='header'>Chat with PDF - Gemini LLM App</h2>", unsafe_allow_html=True)
    
    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history in a clean and styled manner
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.markdown(f"<div class='chat-message user-message'>{chat['user']}</div>", unsafe_allow_html=True):
            pass
        with st.markdown(f"<div class='chat-message assistant-message'>{chat['assistant']}</div>", unsafe_allow_html=True):
            pass
    st.markdown("</div>", unsafe_allow_html=True)

    # Input field for user's message
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_input", placeholder="Type your question here...", help="Enter your question related to the PDF content.")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
        st.markdown("<h3 class='menu-title'>Upload PDF Files</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.balloons()
                    st.success("Your file has been processed, you can ask questions now!")
            else:
                st.error("Please upload at least one PDF file.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

