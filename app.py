import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="📘 PDF Chatbot", page_icon="📄")
st.title("📘 PDF Chatbot with Continuous Conversation")
st.write("Upload PDFs and chat with them continuously using RAG + memory.")

# ------------------------------------------------------
# Upload PDF
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = None

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def chunk_documents(documents, chunk_size=800, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectordb.persist()
    return vectordb

def setup_chat(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(
        model="chat-bison-001",
        temperature=0.2,
        google_api_key=API_KEY
    )
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conv_chain

# ------------------------------------------------------
# Process PDF and create conversation chain
# ------------------------------------------------------
if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    with st.spinner("📖 Reading PDF..."):
        documents = read_pdf(pdf_path)

    with st.spinner("✂️ Splitting PDF into chunks..."):
        chunks = chunk_documents(documents)
    st.success(f"✅ Created {len(chunks)} chunks.")

    with st.spinner("🧠 Creating vector store and setting up conversation..."):
        vectorstore = create_vectorstore(chunks)
        st.session_state.conv_chain = setup_chat(vectorstore)
    st.success("🎉 Chatbot ready! Ask questions below.")

# ------------------------------------------------------
# Chat Interface
# ------------------------------------------------------
if st.session_state.conv_chain:
    user_question = st.text_input("💬 Ask a question about the PDF:")

    if user_question:
        with st.spinner("🤖 Thinking..."):
            result = st.session_state.conv_chain({"question": user_question})
            answer = result["answer"]
            # Save messages in session state for continuous chat
            st.session_state.messages.append({"user": user_question, "bot": answer})

    # Display chat history
    if st.session_state.messages:
        st.subheader("🗨️ Conversation")
        for msg in st.session_state.messages:
            st.markdown(f"**You:** {msg['user']}")
            st.markdown(f"**Bot:** {msg['bot']}")
            st.markdown("---")
