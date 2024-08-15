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

# Hard-coded path to your PDF file (for fallback use)
PDF_PATH = "path/to/your/pdf_file.pdf"

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

def user_input(user_question, pdf_docs=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use uploaded PDF or fallback to hard-coded PDF
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
    else:
        raw_text = get_pdf_text([PDF_PATH])  # Wrap in a list for compatibility

    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Load the FAISS index with the deserialization allowance
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Generate the response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Save the question and response to session state
    st.session_state.chat_history.append({"user": user_question, "assistant": response["output_text"]})
    
    # Display the response
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat-Mate... I can read any PDF file")

    # Initialize chat history in session state if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**User:** {chat['user']}")
                st.markdown(f"**Assistant:** {chat['assistant']}")

    # Upload PDF files
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    # Text input for user question
    user_question = st.text_input("Ask a Question from the PDF File")

    if st.button("Process and Get Answer"):
        if pdf_docs or os.path.isfile(PDF_PATH):
            with st.spinner("Processing..."):
                user_input(user_question, pdf_docs if pdf_docs else None)
                st.success("Processing Done")
        else:
            st.error("No PDF file available for processing.")

    # Sidebar for PDF upload
    with st.sidebar:
        st.markdown('### Upload PDF Files')
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        # Store uploaded files in session state
        if pdf_docs:
            st.session_state['pdf_docs'] = pdf_docs
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.balloons()
                    st.success("Your file has been processed, you can ask questions now!")
            else:
                st.error("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
