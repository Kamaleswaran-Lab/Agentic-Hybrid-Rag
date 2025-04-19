import pandas as pd
import ast

from py2neo import Graph, Node, Relationship


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
            graph.merge(Node("Year", Year=str(year).strip()), "Year", "Year")

    # Create nodes for databases
    for database in databases:
        if database:
            graph.merge(Node("Database", Database=database), "Database", "Database")

    # Create nodes for authors
    for author in authors:
        if author:
            graph.merge(Node("Author", Author=author), "Author", "Author")

    # Create nodes for keywords
    for keyword in keywords:
        if keyword:
            graph.merge(Node("Keyword", Keyword=keyword), "Keyword", "Keyword")

    # Create nodes for citations
    for citation in citations:
        if citation:
            graph.merge(Node("Citation", Citation=citation), "Citation", "Citation")

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

        # Create a node for the paper
        paper = Node("Paper", DOI=doi, Paper=title, Abstract=abstract)
        graph.merge(paper, "Paper", "DOI")

        # Link the paper to its publication year
        if year:
            year_node = graph.nodes.match("Year", Year=str(year).strip()).first()
            if year_node:
                graph.merge(Relationship(paper, "PUBLISHED_IN", year_node))

        # Link the paper to its source database
        database_node = graph.nodes.match("Database", Database=database).first()
        if database_node:
            graph.merge(Relationship(paper, "PUBLISHER", database_node))

        # Ensure keywords are processed correctly (convert to list if still a string)
        if isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.split(',')]

        # Link the paper to its associated keywords
        for k in keywords:
            keyword_node = graph.nodes.match("Keyword", Keyword=k).first()
            if keyword_node:
                graph.merge(Relationship(paper, "KEYWORDS", keyword_node))

        # Link the paper to its authors
        if isinstance(authors, str):
            # Convert string into a list
            authors = ast.literal_eval(authors) # Convert string into a list

        for author in authors:
            if author:
                author_node = graph.nodes.match("Author", Author=author).first()
                if author_node:
                    graph.merge(Relationship(paper, "AUTHORED_BY", author_node))

        # Link the paper to its references
        if isinstance(references, str):
            # Convert string into a list
            references = ast.literal_eval(references)  # Convert string into a list

        for reference in references:
            if reference:
                reference_node = graph.nodes.match("Citation", Citation=reference).first()
                if reference_node:
                    graph.merge(Relationship(paper, "CITES", reference_node))

    # Refresh all nodes
    graph.run("MATCH (n) SET n = n RETURN count(n);")

    # Refresh all relationships
    graph.run("MATCH ()-[r]->() SET r = r RETURN count(r);")

    print("Graph created with success!")

    return  # The function modifies the Neo4j database, so no return value is needed
