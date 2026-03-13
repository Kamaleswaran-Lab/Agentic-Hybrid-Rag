import os
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from Functions.auxiliary import is_reference_chunk, load_and_clean_pdfs


def create_chunks():
    """
    Processes academic papers in PDF format and creates a vector store using FAISS.

    Functionality:
    - Loads all PDF files from the specified folder.
    - Splits the text from the PDFs into smaller chunks.
    - Saves the text chunks as a pickle file for later use.
    - Generates vector embeddings from the text chunks using the Ollama model.
    - Saves the FAISS vector index locally.

    Returns:
    - None (saves processed chunks and vector store to disk).
    """
    # Load the documents from the PDF files
    docs = load_and_clean_pdfs("Papers")

    # Initialize a text splitter to divide the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2024, chunk_overlap=50)

    # Split the documents into manageable text chunks
    text_splits = text_splitter.split_documents(docs)

    # Keep only non-reference chunks
    text_splits = [chunk for chunk in text_splits if not is_reference_chunk(chunk.page_content)]

    # Save the split text chunks to a pickle file for later use
    with open("Database/text_splits.pkl", "wb") as f:
        pickle.dump(text_splits, f)

    print("text_splits was created with success!")

    # Initialize the embedding model to generate vector representations
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS index from the text chunks using the embeddings
    index = FAISS.from_documents(text_splits, embedding=embedding)

    # Save the FAISS index locally
    index.save_local("Database/faiss_index")

    print("vector store was created with success!")

    return
