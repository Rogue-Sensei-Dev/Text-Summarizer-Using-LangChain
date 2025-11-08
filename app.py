import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ------------------------------------------------------
# Streamlit App UI
# ------------------------------------------------------
st.title("📘 PDF Embedding App with LangChain + Google GenAI + Chroma")
st.write("Upload a PDF to generate embeddings and store them locally using ChromaDB.")

# ------------------------------------------------------
# PDF Upload Section
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

def read_document(file_path):
    """Reads a PDF file using LangChain's PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def form_chunks(document, chunk_size=800, overlap=50):
    """Splits text into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_documents(document)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # Read and display PDF
    with st.spinner("📖 Reading the PDF..."):
        document = read_document("temp.pdf")
    st.write(f"Loaded **{len(document)}** pages.")

    # Preview the first page
    st.text_area("📄 First Page Preview", document[0].page_content[:1000])

    # Split into chunks
    with st.spinner("✂️ Splitting document into chunks..."):
        chunks = form_chunks(document)
    st.success(f"✅ Created {len(chunks)} chunks.")

    # Create embeddings
    with st.spinner("🧠 Generating embeddings using Google GenAI..."):
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=API_KEY
        )

    # Store embeddings locally in Chroma
    with st.spinner("💾 Storing embeddings in local Chroma database..."):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory="chroma_db"  # Local folder for Chroma storage
        )
        vectorstore.persist()  # Save to disk

    st.success("🎉 Embeddings created and stored locally using Chroma!")

    # Optional: Show a few example chunks
    if st.checkbox("Show sample chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk.page_content[:500] + "...")
else:
    st.info("Please upload a PDF to begin.")
