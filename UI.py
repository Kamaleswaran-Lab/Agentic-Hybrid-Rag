import os
import base64
import gc
import tempfile
import uuid
import streamlit as st
import pandas as pd
import cohere
import json
import ast

from py2neo import Graph, Node, Relationship
from datetime import datetime
from pathlib import Path
from Functions.search_papers import search_papers
from Functions.select_papers import select_papers
from Functions.download_papers import download_pubmed, download_arxiv_googleScholar
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from Functions.tool import tool
from Functions.tool_agent import ToolAgent
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_neo4j import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from Functions.auxiliary import sanitize_filename
from metapub import FindIt


# ---------------------------------------
# Streamlit App
# ---------------------------------------
st.title("Literature Review Agent")

# Session state for conversation + index caching
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

# Initialize storage for the DataFrame
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
    st.session_state.tempdir = None
    st.session_state.tempdirname = None

# Initialize session state variable
if "search" not in st.session_state:
    st.session_state.search = False

# Initialize authentication state if it doesn't exist
if "auth_success" not in st.session_state:
    st.session_state.auth_success = False
    st.session_state.graph = None
    st.session_state.cohere = None

# Initialize authentication state if it doesn't exist
if "neo_uri" not in st.session_state:
    st.session_state.neo_uri = os.getenv("")
    st.session_state.neo_username = None
    st.session_state.neo_password = None
    st.session_state.index = None
    st.session_state.splits = None


@st.cache_resource
def load_embedding_model():
    return OllamaEmbeddings(model="llama3")


@st.cache_resource
def create_faiss_index_from_splits(splits):
    embedding = load_embedding_model()
    index = FAISS.from_documents(splits, embedding=embedding)
    return index


def download_papers_tempdir(papers, url_column="PDF_URL", title_column="Title", database_column="Database"):
    """
    Downloads academic papers from different databases (PubMed, ArXiv, Google Scholar).

    Parameters:
    - papers (DataFrame): A pandas DataFrame containing the paper information.
    - path (str): The directory where PDFs will be saved (default is "Papers").
    - url_column (str): The column name containing the paper download links or PubMed IDs (default is "PDF_URL").
    - title_column (str): The column name containing paper titles (default is "Title").
    - database_column (str): The column indicating the database source (e.g., "PubMed", "ArXiv", "Google Scholar").

    How to use:
    - Call `download_papers(papers)` with a DataFrame containing paper details.
    - The function separates papers based on their database source.
    - Papers from PubMed are downloaded using `download_pubmed()`.
    - Papers from other sources (ArXiv, Google Scholar) are downloaded using `download_arxiv_googleScholar()`.

    Output:
    - Downloads the available PDFs into the specified directory.
    - Prints the total number of successfully downloaded papers.
    """

    # Create temporary directory to store the retrieved PDFs
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = temp_dir.name

    # Select relevant columns from the DataFrame (database, title, and URL)
    papers = papers[[database_column, title_column, url_column]]

    # Separate papers from PubMed database
    pubmed = papers[papers[database_column] == "PubMed"]

    # Separate papers from other databases (ArXiv, Google Scholar, etc.)
    others = papers[papers[database_column] != "PubMed"]

    # Download PubMed papers using the download_pubmed function
    pubmed_count = download_pubmed(pubmed, path=temp_path, pmid_column=url_column, title_column=title_column)

    # Download papers from other databases (ArXiv, Google Scholar) using the download_arxiv_googleScholar function
    others_count = download_arxiv_googleScholar(others, path=temp_path, url_column=url_column, title_column=title_column, verbose=True)

    # Print the total number of successfully downloaded papers
    print(f"Total papers with successfull downloading process: {pubmed_count + others_count}")

    path = Path(temp_path)

    if len(list(path.glob('*.pdf'))) == 0:
        return None, None

    return temp_dir, temp_dir.name


def create_knowledge_graph_st(papers, uri, username, password):
    """
    Creates a knowledge graph in Neo4j using information from a DataFrame containing academic papers.

    Parameters:
    - papers (DataFrame): A pandas DataFrame containing columns such as 'Year', 'Database', 'Authors',
                          'Keywords', 'Summary', 'DOI', 'Title', and 'Abstract'.

    Functionality:
    - Extracts unique values for years, databases, authors, keywords, and citations.
    - Connects to a Neo4j database and clears all existing nodes.
    - Creates nodes for years, databases, authors, keywords, citations, and papers.
    - Establishes relationships between papers and other entities (year, database, authors, keywords, citations, sections).

    Returns:
    - None (modifies the Neo4j database).
    """

    # Extract unique values, ensuring no empty nodes are created
    years = [str(y).strip() for y in papers["Year"].unique() if pd.notna(y)]
    databases = [str(db).strip() for db in papers["Database"].unique() if pd.notna(db)]
    authors = [a.strip() for a in papers["Authors"].explode().unique() if pd.notna(a) and a.strip()]
    keywords = set(word.strip() for item in papers.Keywords.dropna() for word in item.split(','))

    if "References" in papers.columns:
        citations = [a.strip() for a in papers["References"].explode().unique() if pd.notna(a) and a.strip()]

    # Connect to the Neo4j database
    graph = Graph(uri,
                  auth=(username, password))  # Cloud instance

    # Clear the database (delete all existing nodes and relationships)
    graph.delete_all()

    # Create nodes for years
    for year in years:
        if year:
            info = str(year).strip()
            graph.merge(Node("year", year=info), "year", "year")


    # Create nodes for databases
    for database in databases:
        if database:
            info = database.lower()
            graph.merge(Node("database", database=info), "database", "database")


    # Create nodes for authors
    for author in authors:
        if author:
            info = author.lower()
            graph.merge(Node("author", author=info), "author", "author")

    # Create nodes for keywords
    for keyword in keywords:
        if keyword:
            info = keyword.lower()
            graph.merge(Node("keyword", keyword=info), "keyword", "keyword")

    # Create nodes for citations
    if "References" in papers.columns:
        for citation in citations:
            if citation:
                info = citation.lower()
                graph.merge(Node("citation", citation=info), "citation", "citation")

    # Iterate through each paper and create nodes and relationships
    for _, row in papers.iterrows():
        doi = row["DOI"]  # Digital Object Identifier
        title = row["Title"]  # Paper title
        abstract = str(row["Abstract"]).strip() if pd.notna(row["Abstract"]) else ""  # Paper abstract
        year = str(row["Year"]).strip() if pd.notna(row["Year"]) else None  # Publication year
        database = row["Database"]  # Source database
        keywords = row["Keywords"]  # Keywords associated with the paper
        if "References" in papers.columns:
            references = row["References"]  # Paper's extracted references
        authors = row["Authors"]  # List of authors

        title = title.replace(".", "")

        # Create a node for the paper
        paper = Node("paper", doi=doi, paper=title.lower(), abstract=abstract)
        graph.merge(paper, "paper", "doi")

        # Link the paper to its publication year
        if year:
            year_node = graph.nodes.match("year", year=str(year).strip()).first()
            if year_node:
                graph.merge(Relationship(paper, "published_in", year_node))

        # Link the paper to its source database
        database_node = graph.nodes.match("database", database=database.lower()).first()
        if database_node:
            graph.merge(Relationship(paper, "indexed_at", database_node))

        # Ensure keywords are processed correctly (convert to list if still a string)
        if isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.split(',')]

        # Link the paper to its associated keywords
        for k in keywords:
            keyword_node = graph.nodes.match("keyword", keyword=k.lower()).first()
            if keyword_node:
                graph.merge(Relationship(paper, "has_keyword", keyword_node))

        # Link the paper to its authors
        if isinstance(authors, str):
            # Convert string into a list
            authors = ast.literal_eval(authors)

        for author in authors:
            if author:
                author_node = graph.nodes.match("author", author=author.lower()).first()
                if author_node:
                    graph.merge(Relationship(paper, "authored_by", author_node))

        # Link the paper to its references
        if "References" in papers.columns:
            if isinstance(references, str):
                # Convert string into a list
                references = ast.literal_eval(references)

            for reference in references:
                if reference:
                    reference_node = graph.nodes.match("citation", citation=reference.lower()).first()
                    if reference_node:
                        graph.merge(Relationship(paper, "cites", reference_node))

    # Refresh all nodes
    graph.run("MATCH (n) SET n = n RETURN count(n);")

    # Refresh all relationships
    graph.run("MATCH ()-[r]->() SET r = r RETURN count(r);")

    print("Graph created with success!")

    return


def is_valid_pdf(filepath: str) -> bool:
    """Check if the file starts with %PDF- to ensure it's a valid PDF."""

    try:
        with open(filepath, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def create_chunks_st(tempfolder_path):
    """Loads all valid PDF files from a folder, splits text, creates embeddings and a FAISS index."""

    docs = []

    for fn in os.listdir(tempfolder_path):
        if not fn.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(tempfolder_path, fn)

        if not is_valid_pdf(file_path):
            print(f"Skipping invalid PDF: {fn}")
            continue

        try:
            loader = PDFPlumberLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                page = doc.metadata.get('page', '')
                source = doc.metadata.get("source", "")
                page_content = doc.page_content
                docs.append(Document(page_content=page_content, metadata={"source": f"{source} (page {page})"}))
        except Exception as e:
            print(f"Error loading {fn}: {e}")

    if not docs:
        raise ValueError("No valid PDFs found or none could be loaded.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2024, chunk_overlap=0)
    text_splits = text_splitter.split_documents(docs)

    print("text_splits was created with success!")

    embedding = OllamaEmbeddings(model="llama3")
    index = FAISS.from_documents(text_splits, embedding=embedding)

    print("vector store was created with success!")

    return text_splits, index


@st.dialog("Provide Neo4j Aura account")
def get_account():
    """Get necessary credentials to run the agent"""

    st.markdown("Don't have it yet? [Create an account](https://neo4j.com/product/auradb/)", unsafe_allow_html=True)

    # Neo4j credentials
    uri = st.text_input("URI").strip()
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()

    # Cohere credentials
    st.markdown("Please also provide a Cohere API key. [Create an account if needed](https://dashboard.cohere.com/welcome/login)", unsafe_allow_html=True)
    cohere_api = st.text_input("Cohere API KEY").strip()

    if st.button("Authenticate"):
        if not uri or not username or not password or not cohere_api:
            st.warning("Please fill in all fields.")
        else:

            with st.spinner("Checking connection..."):
                try:
                    # check neo4j instance
                    graph = Graph(uri, auth=(username, password))
                    graph.run("RETURN 1")  # Test query to verify auth

                    # check cohere
                    co = cohere.Client(cohere_api)

                    response = co.generate(
                        model='command',
                        prompt='Hello, Cohere!',
                        max_tokens=10
                    )

                    st.success("Authentication successful.")
                    st.session_state.auth_success = True
                    st.session_state.neo_uri = uri
                    st.session_state.neo_username = username
                    st.session_state.neo_password = password
                    st.session_state.cohere = cohere_api

                # If not successful
                except:
                    st.session_state.auth_success = False
                    st.warning("Failed authentication. Please check your credentials")

    # Show "Start agent" button only if authentication succeeded
    if st.session_state.auth_success:
        if st.button("Start agent"):
            with st.spinner("Creating permanent knowledge (this may take a while)..."):

                # Create knowledge graph
                create_knowledge_graph_st(st.session_state.results_df, st.session_state.neo_uri, st.session_state.neo_username, st.session_state.neo_password)

                # Create vector space
                splits, index = create_chunks_st(st.session_state.tempdirname)

                # Save results in cache
                st.session_state.index = index
                st.session_state.splits = splits

                st.success("All set! You may now chat with your personalized agent")
                st.rerun()


def cypher_search_st(question: str):
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

    # Connect to the Neo4j graph database
    graph = Neo4jGraph(
        url=st.session_state.neo_uri,
        username=st.session_state.neo_username,
        password=st.session_state.neo_password
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

        user_question: amount of papers published in 2024
        Cypher query: MATCH (p:paper)-[:PUBLISHED_IN]->(y:year) WHERE y.year = '2024' RETURN count(p) AS papers_published_in_2024

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
    return json.dumps({"Answer": llm.invoke(prompt), "Context": [f"{result}"]})


def similarity_search_st(question: str):
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

    splits = st.session_state.splits

    # Initialize a keyword-based retriever using BM25
    keyword_retriever = BM25Retriever.from_documents(splits)
    keyword_retriever.k = 5  # Retrieve top 5 most relevant documents

    index = st.session_state.index

    # Create a retriever using FAISS with top-5 search
    vector_retriever = index.as_retriever(search_kwargs={"k": 5})

    # Combine both retrievers with equal weight
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    # Set API key for Cohere's reranker model
    os.environ["COHERE_API_KEY"] = st.session_state.cohere

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

        Just answer the question. Do not add any information that is not related to the question. Do not deviate from the specified format.
        Answer:
        """

    # Return the final answer as a JSON string
    return json.dumps({"Answer": llm.invoke(prompt), "Context": context})


def agent_rag_st():
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

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # Convert the cypher and similarity retrieval functions into usable tools for the agent
    cypher_tool = tool(cypher_search_st)
    similarity_tool = tool(similarity_search_st)

    # Create a ToolAgent with both tools registered
    agent = ToolAgent(tools=[cypher_tool, similarity_tool])

    # Return the configured agent
    return agent


def is_real_pdf(data: bytes) -> bool:
    """Check if pdf data is valid and able to be previewed through iframe"""

    if data is None or isinstance(data, float) and pd.isna(data):
        return False
    if not isinstance(data, (bytes, bytearray)):
        return False
    return data[:5] == b"%PDF-"


def display_pdf_from_temp(filename: str):
    """Display PDF from temp directory by filename."""

    # Ensure tempdir is a Path object
    tempdirname = Path(st.session_state.tempdirname)
    file_path = tempdirname / filename

    if not file_path.exists():
        st.warning(f"File not found: {filename}")
        return

    # Read the file bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Check if it's a valid PDF (optional)
    if is_real_pdf(file_bytes):
        st.markdown("### PDF Preview")
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}"
                    width="100%" height="800px" frameborder="0">
            </iframe>
            """
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Optional download
        st.download_button(
            label="📥 Download PDF",
            data=file_bytes,
            file_name=filename,
            mime="application/pdf"
        )
    # If PDF content is not valid
    else:
        # Try to locate url
        file = filename.split(".")[0]
        row = st.session_state.results_df[st.session_state.results_df["sanitizedTitle"] == file].iloc[0]
        url = None

        if row["Database"] == "PubMed":
            src = FindIt(row["PDF_URL"])
            if src.url:
                url = src.url
        else:
            url = row["PDF_URL"]

        if url:
            st.warning("PDF unavailable.")
            st.markdown(f"[🔗 Open in browser]({url})")

        # If there is no url
        else:
            st.warning("PDF unavailable or corrupted.")


def perform_search():
    with st.spinner("Searching papers..."):
        # Search papers
        papers = search_papers(query, max_results, y_min, y_max)
        if papers.empty:
            st.error("The search retrieved no results")
            st.session_state.results_df = None
            return
        else:
            # Filter papers
            papers = select_papers(papers, query)
            if papers.empty:
                st.error("The search retrieved no results")
                st.session_state.results_df = None
                return
            else:
                # Download PDFs to temporary directory
                tempdir, temp_name = download_papers_tempdir(papers)
                print(temp_name)
                path = Path(temp_name)
                p = len(list(path.glob('*.pdf')))
                print(list(path.glob('*.pdf')))
                if tempdir is None or p == 0:
                    st.error("The search retrieved no results")
                    st.session_state.results_df = None
                    return
                papers["sanitizedTitle"] = papers["Title"].apply(sanitize_filename)
                st.session_state.results_df = papers
                st.session_state.tempdir = tempdir
                st.session_state.tempdirname = temp_name
                st.success(f"{p} relevant papers were found")
                return


def reset_chat():
    """Clear the chat messages and force garbage collection."""

    st.session_state.messages = []
    gc.collect()


with st.sidebar:

    # Get initial parameters to search available literature databases
    current = datetime.now().year

    st.header("Start your Review!")

    query = st.text_area("Share your search query...", height=68)

    y_min_max = st.slider(
            "Select the date range",
            2000, int(current), (2010, 2020)
        )

    max_results = st.number_input("Max results per database", min_value=1, step=1)

    if query and y_min_max and max_results:

        y_min, y_max = y_min_max

        if st.session_state.tempdir and not st.session_state.results_df.empty and len(st.session_state.results_df) > 0:
            # Search again and results UI
            left, right = st.columns(2)

            if left.button("Search again"):
                perform_search()

            path = Path(st.session_state.tempdirname)
            p = len(list(path.glob('*.pdf')))
            if p > 0:
                if right.button("Start chatting"):
                    with st.spinner("Creating permanent knowledge (this may take a while)..."):

                        # Create knowledge graph
                        create_knowledge_graph_st(st.session_state.results_df, st.session_state.neo_uri, st.session_state.neo_username, st.session_state.neo_password)

                        # Create vector space
                        splits, index = create_chunks_st(st.session_state.tempdirname)

                        # Save results in cache
                        st.session_state.index = index
                        st.session_state.splits = splits

                        st.success("All set! You may now chat with your personalized agent")
                        st.rerun()

            # display PDFs preview
            titles = list(path.glob('*.pdf'))
            titles = [pdf.stem for pdf in titles]
            selected_title = st.selectbox("Available papers", titles)

            if selected_title:
                st.subheader(selected_title)
                display_pdf_from_temp(f"{selected_title}.pdf")

        else:
            # Initial search UI
            if st.button("Search"):
                perform_search()
                st.rerun()

            # If results exist (possibly from previous runs), show selectbox
            if st.session_state.tempdir and not st.session_state.results_df.empty and len(st.session_state.results_df) > 0:
                path = Path(st.session_state.tempdirname)
                titles = list(path.glob('*.pdf'))
                titles = [pdf.stem for pdf in titles]
                selected_title = st.selectbox("Available papers", titles)

                if selected_title:
                    st.subheader(selected_title)
                    display_pdf_from_temp(f"{selected_title}.pdf")


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
    # Show user message in chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get QA system working
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # If databases haven't been searched yet, provide a placeholder llm
        if not st.session_state.auth_success:

            llm = OllamaLLM(model="mistral")

            combined_prompt = (
                "You are an assistant that appears before the user has entered the required information for the full literature search agent to activate."
                "Your job is to politely inform the user that they need to complete the sidebar fields (search query, date range, max results) before continuing."
                "Be brief, friendly, and supportive. Do not attempt to answer any literature-related questions, generate summaries, or engage in advanced reasoning. Simply prompt the user to fill in the required information."
                "If the user tries to ask a research-related question, kindly redirect them to the sidebar and let them know you'll be a more capable assistant once they complete that step.\n\n"
                f"History: {st.session_state.messages}\n"
                f"User question: {prompt}"
            )

            print(combined_prompt)

            # Use placeholder assistant
            full_response = llm.invoke([HumanMessage(content=combined_prompt)])

        # If search was conducted and permanent knowledge is available to the agent, call the agent
        else:

            agent = agent_rag_st()

            # Use agent to answer
            full_response = agent.run(user_msg=prompt)["final_response"]

            if full_response is None:
                full_response = "If you have any questions regarding your literature search, feel free to share them with me :)"

        # Display the final answer
        message_placeholder.markdown(full_response)

    # Save the assistant message to session
    st.session_state.messages.append({"role": "assistant", "content": full_response})
