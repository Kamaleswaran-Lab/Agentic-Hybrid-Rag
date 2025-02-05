import os
import base64
import gc
import tempfile
import uuid
import pickle
import pandas as pd
import torch
import streamlit as st

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core import Document

# ---------------------------------------
# Set up the Ollama, LLM and embedding settings
# ---------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

#my_llm = OllamaLLM(model_name="llama2")

my_llm = Ollama(model="llama3.2")  # or your preferred model
# ---------------------------------------
# Streamlit App
# ---------------------------------------
st.title("Chat with Your Documents")

# Session state for conversation + index caching
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def reset_chat():
    """Clear the chat messages and force garbage collection."""
    st.session_state.messages = []
    gc.collect()

def display_pdf(file):
    """Display PDF in an iframe preview (optional)."""
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}"
            width="400" height="100%"
            style="height:100vh; width:100%">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_vectorized_index(file_path):
    """Load a previously saved VectorStoreIndex from disk."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_vectorized_index(index, file_path):
    """Save a VectorStoreIndex to disk."""
    with open(file_path, "wb") as f:
        pickle.dump(index, f)

# Session state for conversation + index caching
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def reset_chat():
    """Clear the chat messages and force garbage collection."""
    st.session_state.messages = []
    gc.collect()

def save_embeddings(index, file_path):
    """Save the embeddings of the indexed documents to disk."""
    with open(file_path, "wb") as f:
        pickle.dump(index, f)

def load_embeddings(file_path):
    """Load previously saved embeddings from disk."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def process_pdf(temp_dir, uploaded_pdf):
    """Process a PDF file and return documents."""
    file_path = os.path.join(temp_dir, uploaded_pdf.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_pdf.getvalue())

    loader = SimpleDirectoryReader(input_dir=temp_dir, required_exts=[".pdf"], recursive=True)
    return loader.load_data()

def process_csv(file):
    """Convert each CSV row into a LlamaIndex Document object."""
    df = pd.read_csv(file)
    documents = [Document(text=" ".join(map(str, row))) for _, row in df.iterrows()]
    return documents

with st.sidebar:
    st.header("Add your documents!")

    uploaded_pdf = st.file_uploader("Choose a `.pdf` file", type="pdf")
    uploaded_csv = st.file_uploader("Or upload a `.csv` file", type="csv")
    uploaded_vectorized_file = st.file_uploader("Or load a vectorized `.pkl` file", type="pkl")

    # 1) Load a pre-vectorized index if provided
    if uploaded_vectorized_file:
        try:
            vectorized_file_key = f"{session_id}-{uploaded_vectorized_file.name}"
            st.write("Loading your vectorized data...")
            vectorized_index = load_vectorized_index(uploaded_vectorized_file)
            st.session_state.file_cache[vectorized_file_key] = vectorized_index
            st.success("Vectorized data loaded successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # 2) Otherwise, if user uploads a PDF or CSV, build a new index
    # Updated File Processing in Sidebar
    if uploaded_pdf or uploaded_csv:
        try:
            docs = []
            file_name = uploaded_pdf.name if uploaded_pdf else uploaded_csv.name
            vectorized_index_file = f"{session_id}-{file_name}.pkl"

            if uploaded_pdf:
                with tempfile.TemporaryDirectory() as temp_dir:
                    docs = process_pdf(temp_dir, uploaded_pdf)

            elif uploaded_csv:
                docs = process_csv(uploaded_csv)

            # Check if embeddings already exist
            if os.path.exists(vectorized_index_file):
                st.write("Loading existing embeddings...")
                vectorized_index = load_embeddings(vectorized_index_file)
            else:
                # Build new index and save embeddings
                vectorized_index = VectorStoreIndex.from_documents(
                    docs, llm=my_llm, embed_model=Settings.embed_model, show_progress=True
                )
                save_embeddings(vectorized_index, vectorized_index_file)

            st.session_state.file_cache[vectorized_index_file] = vectorized_index
            st.success("Documents indexed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()


            # Build the VectorStoreIndex
            #Settings.embed_model = service_context.embed_model  # optional
            index = VectorStoreIndex.from_documents(
                docs,
                llm = my_llm,
                embed_model = Settings.embed_model,
                show_progress=True
            )

            # Cache it in a .pkl file
            vectorized_index_file = f"{session_id}-vectorized.pkl"
            save_vectorized_index(index, vectorized_index_file)
            st.session_state.file_cache[f"{session_id}-vectorized"] = index
            st.success("Documents indexed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Layout: main chat area + a clear button
col1, col2 = st.columns([6, 1])

with col1:
    st.header("")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat messages in session state if not present
if "messages" not in st.session_state:
    reset_chat()

# Display existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt input at the bottom
if prompt := st.chat_input("Ask something!"):
    # 1) Show user message in chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Prepare a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Retrieve the loaded or newly built index
        vectorized_index = None
        # If you uploaded a .pkl file, we store it as session_id-filename
        # If you built a new one, we store it as session_id-vectorized
        for key in st.session_state.file_cache:
            if key.startswith(str(session_id)):
                vectorized_index = st.session_state.file_cache[key]
                break

        # 3) If we have an index, do retrieval, then feed context + question to Ollama
        if vectorized_index:
            query_engine = vectorized_index.as_query_engine(
                llm=my_llm,
                embed_model=Settings.embed_model#service_context=service_context
            )
            retrieval_response = query_engine.query(prompt)
            retrieved_context = retrieval_response.response  # The retrieved chunks summary

            # Now do a final LLM call with the retrieved context
            # We'll combine it with the user prompt in a single user message.
            combined_prompt = (
                f"Context from documents:\n{retrieved_context}\n\n"
                f"User question: {prompt}"
            )
            # Use the new Ollama pipeline
            full_response = my_llm.chat([ChatMessage(role="user", content=combined_prompt)])
        else:
            # If no index is loaded, just pass the user prompt directly to Ollama
            full_response = my_llm.chat([ChatMessage(role="user", content=prompt)])

        # 4) Display the final answer
        message_placeholder.markdown(full_response)

    # 5) Save the assistant message to session
    st.session_state.messages.append({"role": "assistant", "content": full_response})
