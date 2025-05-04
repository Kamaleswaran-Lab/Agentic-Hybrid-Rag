import os
import pickle

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


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

    # Define the path to the folder containing the PDFs
    pdf_folder_path = '../Papers/'

    # Load all PDF files in the folder using PDFPlumberLoader
    loaders = [
        PDFPlumberLoader(os.path.join(pdf_folder_path, fn))
        for fn in os.listdir(pdf_folder_path)
        if fn.lower().endswith(".pdf")  # Filter for PDF files only
    ]

    # Load the documents from the PDF files
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Initialize a text splitter to divide the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Split the documents into manageable text chunks
    text_splits = text_splitter.split_documents(docs)

    # Save the split text chunks to a pickle file for later use
    with open("../Database/text_splits.pkl", "wb") as f:
        pickle.dump(text_splits, f)

    print("text_splits was created with success!")

    # Initialize the embedding model to generate vector representations
    embedding = OllamaEmbeddings(model="mistral")

    # Create a FAISS index from the text chunks using the embeddings
    index = FAISS.from_documents(text_splits, embedding=embedding)

    # Save the FAISS index locally
    index.save_local("../Database/faiss_index")

    print("vector store was created with success!")

    return
