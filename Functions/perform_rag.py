import json
import os
import pickle

from Functions.functions import cypher_search, similarity_search
from Functions.tool import tool
from Functions.tool_agent import ToolAgent
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_neo4j import Neo4jGraph
from groq import Groq
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS


def agent_rag():
    """
    Creates and returns a Retrieval-Augmented Generation (RAG) agent that combines Cypher and similarity-based tools.

    Parameters:
    - None

    Functionality:
    - Wraps the Cypher-based graph query function and similarity-based PDF search function as tools.
    - Constructs an agent capable of choosing between these tools to answer user queries effectively.
      The agent can decide whether to query the Neo4j graph or retrieve context from PDFs based on the question type.

    Returns:
    - ToolAgent: An agent instance capable of performing tool-augmented question answering.
    """

    # Convert the cypher and similarity retrieval functions into usable tools for the agent
    cypher_tool = tool(cypher_search)
    similarity_tool = tool(similarity_search)

    # Create a ToolAgent with both tools registered
    agent = ToolAgent(tools=[cypher_tool, similarity_tool])

    # Return the configured agent
    return agent


def baseline_RAG(question):
    # Get credentials
    uri = os.getenv("KG_URI")
    username = os.getenv("KG_USERNAME")
    password = os.getenv("KG_PASSWORD")

    llm = OllamaLLM(model="mistral")

    with open("Database/text_splits.pkl", "rb") as f:
        text_splits = pickle.load(f)

    # Initialize a keyword-based retriever using BM25
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 5  # Retrieve top 5 most relevant documents

    # Initialize embedding model for vector similarity search
    embedding = OllamaEmbeddings(model="mistral")

    # Load the FAISS vector index from local storage
    vectorstore = FAISS.load_local("Database/faiss_index", embedding, allow_dangerous_deserialization=True)

    # Create a retriever using FAISS with top-5 search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Combine both retrievers with equal weight
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    # Set API key for Cohere's reranker model

    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    os.environ["COHERE_API_KEY"] = cohere_api_key

    # Apply Cohere's reranking model to compress and filter context
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # Retrieve the most relevant context for the input question
    context = compression_retriever.invoke(question)

    vs_context = [
        {
            "source": doc.metadata.get('source', 'unknown'),
            "content": doc.page_content
        }
        for doc in context
    ]

    print(f"Vs context: {vs_context}")

    kg_index = Neo4jVector.from_existing_graph(
        embedding,
        search_type="hybrid",
        url=uri,
        username=username,
        password=password,
        index_name='kg_search',
        node_label="document",
        text_node_properties=['info'],
        embedding_node_property='embedding',
    )

    kg_retriever = kg_index.as_retriever()

    kg_data = [el.page_content for el in kg_retriever.invoke(question)]

    print(f"kg data: {kg_data}")

    final_context = vs_context + kg_data

    kg_prompt = f"""Provide a concise answer to the question based only on the given context.
            If the context is not related to the question, just output 'no comment'. No extra text.
            The question and context are provided within the xml tags.
            <question> 
            {question}
            </question>
            <context>
            {kg_data}
            </context>
            
            Just answer the question. Do not add any information that is not related to the question. Do not deviate from the specified format.
            Answer:
            """

    #kg_context = llm.invoke(kg_prompt)

    print(f"kg context: {kg_data}")

    RAG_PROMPT = f"""Provide a concise answer to the question based only on the given context.
        If the context is not related to the question, state that you cannot answer the question. 
        The question and context are provided within the xml tags.
        <question> 
        {question}
        </question>
        <context>
        {final_context}
        </context>
        
        Just answer the question. Do not add any information that is not related to the question. Do not deviate from the specified format.
        Answer:
        """
    final_answer = llm.invoke(RAG_PROMPT)
    final_answer = final_answer.replace(r"\n", "")

    return final_context, final_answer
