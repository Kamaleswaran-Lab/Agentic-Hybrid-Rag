import pandas as pd
import ast

from py2neo import Graph, Node, Relationship
from langchain_ollama import OllamaEmbeddings


def create_knowledge_graph(papers):
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

    embedding_model = OllamaEmbeddings(model="llama3")

    # Extract unique values, ensuring no empty nodes are created
    years = [str(y).strip() for y in papers["Year"].unique() if pd.notna(y)]
    databases = [str(db).strip() for db in papers["Database"].unique() if pd.notna(db)]
    authors = [a.strip() for a in papers["Authors"].explode().unique() if pd.notna(a) and a.strip()]
    keywords = set(word.strip() for item in papers.Keywords.dropna() for word in item.split(','))
    citations = [a.strip() for a in papers["References"].explode().unique() if pd.notna(a) and a.strip()]

    # Get credentials
    uri = "neo4j+s://91f991ec.databases.neo4j.io"
    username = "neo4j"
    password = "COeHGYRiC2H4YzRFer_o11lHQDEsuBBfr8Ules7G1PQ"

    # Connect to the Neo4j database
    graph = Graph(uri,
                  auth=(username, password))  # Cloud instance

    # Clear the database (delete all existing nodes and relationships)
    graph.delete_all()

    # Create nodes for years
    for year in years:
        if year:
            info = str(year).strip()
            embeddings = embedding_model.embed_documents([info])[0]
            graph.merge(Node("year", "document", info=f"Year: {info}", year=info, embedding=embeddings), "year", "year")

    # Create nodes for databases
    for database in databases:
        if database:
            info = database.lower()
            embeddings = embedding_model.embed_documents([info])[0]
            graph.merge(Node("database", "document", info=f"Database: {info}", database=info, embedding=embeddings), "database", "database")

    # Create nodes for authors
    for author in authors:
        if author:
            info = author.lower()
            embeddings = embedding_model.embed_documents([info])[0]
            graph.merge(Node("author", "document", info=f"Author: {info}", author=info, embedding=embeddings), "author", "author")

    # Create nodes for keywords
    for keyword in keywords:
        if keyword:
            info = keyword.lower()
            embeddings = embedding_model.embed_documents([info])[0]
            graph.merge(Node("keyword", "document", info=f"Keyword: {info}", keyword=keyword, embedding=embeddings), "keyword", "keyword")

    # Create nodes for citations
    for citation in citations:
        if citation:
            info = citation.lower()
            embeddings = embedding_model.embed_documents([info])[0]
            graph.merge(Node("citation", "document", info=f"Citation: {info}", citation=info, embedding=embeddings), "citation", "citation")

    # Iterate through each paper and create nodes and relationships
    for _, row in papers.iterrows():
        doi = row["DOI"]  # Digital Object Identifier
        title = row["Title"]  # Paper title
        abstract = str(row["Abstract"]).strip() if pd.notna(row["Abstract"]) else ""  # Paper abstract
        year = str(row["Year"]).strip() if pd.notna(row["Year"]) else None  # Publication year
        database = row["Database"]  # Source database
        keywords = row["Keywords"]  # Keywords associated with the paper
        references = row["References"]  # Paper's extracted references
        authors = row["Authors"]  # List of authors

        title = title.replace(".", "")

        info = f"DOI: {doi}. Title: {title.lower()}. Abstract: {abstract}"
        embeddings = embedding_model.embed_documents([info])[0]

        # Create a node for the paper
        paper = Node("paper", "document", doi=doi, paper=title.lower(), abstract=abstract, info=info, embedding=embeddings)
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
            authors = ast.literal_eval(authors) # Convert string into a list

        for author in authors:
            if author:
                author_node = graph.nodes.match("author", author=author.lower()).first()
                if author_node:
                    graph.merge(Relationship(paper, "authored_by", author_node))

        # Link the paper to its references
        if isinstance(references, str):
            # Convert string into a list
            references = ast.literal_eval(references)  # Convert string into a list

        for reference in references:
            if reference:
                reference_node = graph.nodes.match("citation", citation=reference.lower()).first()
                if reference_node:
                    graph.merge(Relationship(paper, "cites", reference_node))

    # Refresh all nodes
    graph.run("MATCH (n) SET n = n RETURN count(n);")

    # Refresh all relationships
    graph.run("MATCH ()-[r]->() SET r = r RETURN count(r);")

    # Create full-text index with all relevant nodes
    #indexes = graph.run("SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT'")
    #indexes = [index["name"] for index in indexes.data()]

    # Check for the presence of the desired index by name
    #if "fullIndex" not in indexes:
    #    graph.run(
    #        "CREATE FULLTEXT INDEX fullIndex FOR (n:Paper|Keyword|Year|Author|Database) ON EACH [n.year, n.citation, n.author, n.keyword, n.database, n.doi, n.paper, n.abstract]")

    print("Graph created with success!")

    return  # The function modifies the Neo4j database, so no return value is needed
