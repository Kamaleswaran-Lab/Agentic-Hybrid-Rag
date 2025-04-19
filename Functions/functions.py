import json
import pickle
import os

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


def cypher_search(question: str):
    """
    Answers a natural language question by converting it into a Cypher query and querying a Neo4j graph database.

    Parameters:
    - question (str): A natural language question related to the academic paper graph database.
                      Examples include inquiries about publication year, number of papers,
                      authors, references, frequent keywords, or specific abstracts from a given paper.

    Functionality:
    - Connects to a remote Neo4j graph database.
    - Uses a language model to translate the question into a Cypher query based on the graph's schema.
    - Executes the query and retrieves results.
    - Formats and returns a concise answer in JSON format.

    Returns:
    - str: JSON string containing a concise answer to the input question.
    """

    # Initialize the language model to generate Cypher queries
    llm = OllamaLLM(model="mistral")

    # Connection credentials for the Neo4j database
    uri = "neo4j+s://91f991ec.databases.neo4j.io"
    username = "neo4j"
    password = "COeHGYRiC2H4YzRFer_o11lHQDEsuBBfr8Ules7G1PQ"

    # Connect to the Neo4j graph database
    graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password
    )

    # Template for prompting the LLM to create a Cypher query based on the database schema and user question
    cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question. START sintax was deprecated, use MATCH instead, if necessary. Only return the query, with no additional text:
    Node Properties: {node_props}

    Question: {question}
    Cypher query:"""

    # Set up the prompt for the language model
    cypher_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to a Cypher query. No pre-amble."),
            ("human", cypher_template),
        ]
    )

    # Create the LLM pipeline with the prompt and model
    llm_chain = cypher_prompt | llm

    # Invoke the chain to get the Cypher query from the question and graph schema
    retrieval = llm_chain.invoke({"question": question, "node_props": graph.structured_schema})

    # Print the Cypher query being executed
    print("Query: ", retrieval, sep="\n")

    # Attempt to run the Cypher query on the graph
    try:
        result = graph.query(retrieval)
    except:
        # Handle query failure (e.g. syntax error or no result)
        result = "no data was found"

    # Prompt the LLM again to generate a natural language answer from the Cypher result
    prompt = f"""Provide a concise answer based only on the given response.
    If no data was found, state that you cannot answer since no response was given.
    Question: {question}
    Response: {result}
    Answer:"""

    # Return the final answer as a JSON string
    return json.dumps({"Answer": llm.invoke(prompt)})


def similarity_search(question: str):
    """
    Answers a question using similarity-based retrieval from PDF documents and returns a concise response in JSON format.

    Parameters:
    - question (str): A natural language question intended to retrieve conceptual or broad information from PDF content.

    Functionality:
    - Loads previously split and embedded text chunks from disk.
    - Uses BM25 and vector-based retrieval (via FAISS) to find relevant chunks.
    - Combines the two methods using ensemble retrieval for improved accuracy.
    - Applies a reranking model to compress and refine the context using Cohere's re-ranker.
    - Uses a language model to generate a concise, natural language answer based on the retrieved context.

    Returns:
    - str: JSON string containing the generated answer.
    """

    # Load preprocessed text chunks from pickle file
    with open("Database/text_splits.pkl", "rb") as f:
        text_splits = pickle.load(f)

    # Initialize a keyword-based retriever using BM25
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 5  # Retrieve top 5 most relevant documents

    # Initialize embedding model for vector similarity search
    embedding = OllamaEmbeddings(model="llama3")

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
    os.environ["COHERE_API_KEY"] = "Ni2SuJm5hKdJict4OAblCsQ3l08tA3AYZwbQa2CL"

    # Apply Cohere's reranking model to compress and filter context
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # Retrieve the most relevant context for the input question
    context = compression_retriever.invoke(question)

    # Initialize the language model for answer generation
    llm = OllamaLLM(model="mistral")

    # Format prompt to instruct LLM to answer concisely based on the retrieved context
    prompt = f"""Provide a concise answer based only on the given context.
        If the context is not related to the question, state that you cannot answer the question.
        Question: {question}
        Context: {context}
        Answer:"""

    # Return the final answer as a JSON string
    return json.dumps({"Answer": llm.invoke(prompt)})
