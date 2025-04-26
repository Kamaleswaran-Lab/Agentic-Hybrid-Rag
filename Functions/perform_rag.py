import json

from Functions.functions import cypher_search, similarity_search
from Functions.tool import tool
from Functions.tool_agent import ToolAgent
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_neo4j import Neo4jGraph
from groq import Groq


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
    uri = "neo4j+s://91f991ec.databases.neo4j.io"
    username = "neo4j"
    password = "COeHGYRiC2H4YzRFer_o11lHQDEsuBBfr8Ules7G1PQ"

    embedding_model = OllamaEmbeddings(model="llama3")

    llm = OllamaLLM(model="llama3")

    vs_answer = similarity_search(question)

    # Safely extract the Answer from the JSON response
    retrieved_context = json.loads(vs_answer)

    # Treat context as a single retrieved chunk
    vs_context = retrieved_context["Answer"]

    print(f"Vs context: {vs_context}")

    kg_index = Neo4jVector.from_existing_graph(
        embedding_model,
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

    kg_prompt = f"""Provide a concise answer to the question based only on the given context.
            If the context is not related to the question, state that you cannot answer the question. 
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

    kg_context = llm.invoke(kg_prompt)

    print(f"kg context: {kg_context}")

    RAG_PROMPT = f"""Provide a concise answer to the question based only on the given context. 
                You will receive the responses of both a vector store and a knowlegde graph search as context.
                Interpret them carefully to extract insights that best answer the question.
                If the context is not related to the question, state that you cannot answer the question. 
                The question and responses for both strategies are provided within the xml tags.
                
                <question> 
                {question}
                </question>
                <response_vector_store>
                {vs_context}
                </response_vector_store>
                <response_knowledge_graph>
                {kg_context}
                </response_knowledge_graph>
                
                Just answer the question. Do not add any information that is not related to the question. Do not deviate from the specified format.
                Answer:
                """

    client = Groq(api_key="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")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": RAG_PROMPT}
        ]
    )

    final_context = [vs_context, kg_context]
    final_answer = response.choices[0].message.content

    return final_context, final_answer
