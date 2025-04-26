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
    Answers a natural language question by converting it into a Cypher query and querying a Neo4j graph database. Usage includes inquiries about publication year, number of papers,
                      authors, references, frequent keywords, or specific abstracts from a given paper.

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
    cypher_template = """Based on the Neo4j graph schema within the xml tags, write a Cypher query that would answer the user's question. START sintax was deprecated, use MATCH instead, if necessary. 
    Return only the Cypher query, with no additional text.
    
    Here are some examples:
        user_question: abstract of the paper 'deep learning for medical image analysis'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'deep learning for medical image analysis' RETURN p.abstract as abstract

        user_question: authors of the paper 'an ai-enabled nursing future with no documentation burden: a vision for a new reality'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'an ai-enabled nursing future with no documentation burden: a vision for a new reality' MATCH (p)-[:authored_by]->(a:author) RETURN a.author as list_authors

        user_question: publication year of the paper 'from text to multimodality: exploring the evolution and impact of large language models in medical practice'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'from text to multimodality: exploring the evolution and impact of large language models in medical practice' MATCH (p)-[:published_in]->(y:year) RETURN y.year as published_in

        user_question: database where paper 'title' is indexed
        Cypher query: MATCH (p:paper) WHERE p.paper = 'title' MATCH (p)-[:indexed_at]->(d:database) RETURN d.database as published_at
        
        user_question: is paper 'from text to multimodality: exploring the evolution and impact of large language models in medical practice' indexed in the database 'PubMed'?
        Cypher query: MATCH (p:paper) WHERE p.paper = 'from text to multimodality: exploring the evolution and impact of large language models in medical practice' MATCH (p)-[:indexed_at]->(d:database) RETURN d.database as published_at
        
        user_question: did the author 'yin ch' write any paper that contains the keyword 'mllm'?
        Cypher query: MATCH (p:paper)-[:authored_by]->(a:author), (p)-[:has_keyword]->(k:keyword) WHERE a.author = 'yin ch' RETURN DISTINCT k.keyword AS author_keywords
        
        user_question relationship of the keyword 'medical' to the paper 'from text to multimodality: exploring the evolution and impact of large language models in medical practice'?
        Cypher query: MATCH (p:paper)-[r]->(k:keyword) WHERE p.paper = 'from text to multimodality: exploring the evolution and impact of large language models in medical practice' AND k.keyword = 'medical' RETURN type(r) AS relationship
        
        user_question: keywords associated with the paper 'advancing precision medicine through AI-driven research'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'advancing precision medicine through AI-driven research' MATCH (p)-[:has_keyword]->(k:keyword) RETURN k.keyword as keywords

        user_question: which database indexes the paper 'deep transfer learning in radiology'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'deep transfer learning in radiology' MATCH (p)-[:indexed_at]->(d:database) RETURN d.database as published_at

        user_question: all papers written by the author 'andrew ng'
        Cypher query: MATCH (p:paper)-[:authored_by]->(a:author) WHERE a.author = 'andrew ng' RETURN p.paper as papers

        user_question: what is the abstract of the paper 'automated diagnosis systems in healthcare'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'automated diagnosis systems in healthcare' RETURN p.abstract as abstract

        user_question: in which year was the paper 'emerging trends in health informatics' published?
        Cypher query: MATCH (p:paper) WHERE p.paper = 'emerging trends in health informatics' MATCH (p)-[:published_in]->(y:year) RETURN y.year as published_in

        user_question: author of the paper 'integrating AI in public health surveillance'
        Cypher query: MATCH (p:paper)-[:authored_by]->(a:author) WHERE p.paper = 'integrating AI in public health surveillance' RETURN a.author as list_authors

        user_question: what are the keywords of the paper 'clinical decision support systems'
        Cypher query: MATCH (p:paper)-[:has_keyword]->(k:keyword) WHERE p.paper = 'clinical decision support systems' RETURN k.keyword as keywords

        user_question: papers indexed in the database 'Scopus'
        Cypher query: MATCH (p:paper)-[:indexed_at]->(d:database) WHERE d.database = 'Scopus' RETURN p.paper as papers

        user_question: papers published in the year '2021'
        Cypher query: MATCH (p:paper)-[:published_in]->(y:year) WHERE y.year = '2021' RETURN p.paper as papers

        user_question: papers published in the year '2020'
        Cypher query: MATCH (p:paper)-[:published_in]->(y:year) WHERE y.year = '2020' RETURN p.paper as papers

        user_question: relationship of the keyword 'covid-19' to the paper 'pandemic response using AI'
        Cypher query: MATCH (p:paper)-[r]->(k:keyword) WHERE p.paper = 'pandemic response using AI' AND k.keyword = 'covid-19' RETURN type(r) AS relationship

        user_question: find all authors who wrote about 'telemedicine'
        Cypher query: MATCH (p:paper)-[:has_keyword]->(k:keyword), (p)-[:authored_by]->(a:author) WHERE k.keyword = 'telemedicine' RETURN DISTINCT a.author as authors

        user_question: find all keywords related to papers authored by 'li wei'
        Cypher query: MATCH (p:paper)-[:authored_by]->(a:author), (p)-[:has_keyword]->(k:keyword) WHERE a.author = 'li wei' RETURN DISTINCT k.keyword as keywords

        user_question: DOI of the paper 'machine learning for diabetic retinopathy'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'machine learning for diabetic retinopathy' RETURN p.doi as doi

        user_question: is the keyword 'deep learning' used in the paper 'medical image segmentation techniques'?
        Cypher query: MATCH (p:paper)-[:has_keyword]->(k:keyword) WHERE p.paper = 'medical image segmentation techniques' RETURN k.keyword as used_keywords

        user_question: papers co-authored by 'michael smith' and 'jessica lee'
        Cypher query: MATCH (p:paper)-[:authored_by]->(a1:author), (p)-[:authored_by]->(a2:author) WHERE a1.author = 'michael smith' AND a2.author = 'jessica lee' RETURN p.paper as papers

        user_question: database for the paper 'AI for cardiovascular diagnosis'
        Cypher query: MATCH (p:paper)-[:indexed_at]->(d:database) WHERE p.paper = 'AI for cardiovascular diagnosis' RETURN d.database as database

        user_question: papers that share a keyword with the document 'AI in oncology'
        Cypher query: MATCH (p1:paper)-[:has_keyword]->(k:keyword)<-[:has_keyword]-(p2:paper) WHERE p1.paper = 'AI in oncology' AND p1 <> d2 RETURN DISTINCT p2.paper as related_papers

        user_question: papers written by authors who also wrote 'neural networks in healthcare'
        Cypher query: MATCH (p1:paper)-[:authored_by]->(a:author)<-[:authored_by]-(p2:paper) WHERE p1.paper = 'neural networks in healthcare' RETURN DISTINCT p2.paper as related_papers

        user_question: all keywords used in papers published in 2022
        Cypher query: MATCH (p:paper)-[:published_in]->(y:year), (p)-[:has_keyword]->(k:keyword) WHERE y.year = '2022' RETURN DISTINCT k.keyword as keywords

        user_question: abstract and doi of the paper 'AI-powered clinical triage'
        Cypher query: MATCH (p:paper) WHERE p.paper = 'AI-powered clinical triage' RETURN p.abstract as abstract, p.doi as doi

        user_question: name of the database and year of publication for the paper 'healthcare big data analytics'
        Cypher query: MATCH (p:paper)-[:indexed_at]->(d:database), (p)-[:published_in]->(y:year) WHERE p.paper = 'healthcare big data analytics' RETURN d.database as database, y.year as year

        user_question: which authors have published in the database 'IEEE Xplore'
        Cypher query: MATCH (p:paper)-[:indexed_at]->(d:database), (p)-[:authored_by]->(a:author) WHERE d.database = 'IEEE Xplore' RETURN DISTINCT a.author as authors

    <user_question>
    {question}
    </user_question>
    
    <schema>
    {node_props}
    </schema>
    
    Return only the cypher query. No pre-amble or any other explanations. Do not deviate from that specified format.
    
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
    retrieval = llm_chain.invoke({"question": question.lower(), "node_props": graph.structured_schema})

    # Print the Cypher query being executed
    print("Query: ", retrieval.lower(), sep="\n")

    # Attempt to run the Cypher query on the graph
    try:
        result = graph.query(retrieval.lower())
    except:
        # Handle query failure (e.g. syntax error or no result)
        result = "no data was found"

    # Prompt the LLM again to generate a natural language answer from the Cypher result
    prompt = f"""Provide a concise Answer to the question based only on the given response within the xml tags.
    Provide a response solely on the response given.
    
    <question> 
    {question}
    </question>
    <response>
    {result}
    </response>
    
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

    context = [
        {
            "source": doc.metadata.get('source', 'unknown'),
            "content": doc.page_content
        }
        for doc in context
    ]

    print(context)

    # Initialize the language model for answer generation
    llm = OllamaLLM(model="mistral")

    # Format prompt to instruct LLM to answer concisely based on the retrieved context
    prompt = f"""Provide a concise answer to the question based only on the given context.
        If the context is not related to the question, state that you cannot answer the question. 
        The question and context are provided within the xml tags.
        <question> 
        {question}
        </question>
        <context>
        {context}
        </context>
        Answer:
        """

    # Return the final answer as a JSON string
    return json.dumps({"Answer": llm.invoke(prompt)})
