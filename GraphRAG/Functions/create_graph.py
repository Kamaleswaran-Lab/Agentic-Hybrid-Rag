import pandas as pd
import ast
import re

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
    citations = set()

    # Extract citations from "Summary" column
    for row in papers["Summary"].dropna():  # Ensure no NaN values
        if isinstance(row, dict) and "References" in row:
            citations.update(row["References"])

    # Connect to the Neo4j database
    # graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))  # Local instance
    graph = Graph("neo4j+s://91f991ec.databases.neo4j.io",
                  auth=("neo4j", "COeHGYRiC2H4YzRFer_o11lHQDEsuBBfr8Ules7G1PQ"))  # Cloud instance

    # Clear the database (delete all existing nodes and relationships)
    graph.delete_all()

    # Create nodes for years
    for year in years:
        if year:
            graph.merge(Node("Year", year=str(year).strip()), "Year", "year")

    # Create nodes for databases
    for database in databases:
        if database:
            graph.merge(Node("Database", name=database), "Database", "name")

    # Create nodes for authors
    for author in authors:
        if author:
            graph.merge(Node("Author", name=author), "Author", "name")

    # Create nodes for keywords
    for keyword in keywords:
        if keyword:
            graph.merge(Node("Keyword", keyword=keyword), "Keyword", "keyword")

    # Create nodes for citations
    for citation in citations:
        if citation:
            graph.merge(Node("Citation", doi=citation), "Citation", "doi")

    # Iterate through each paper and create nodes and relationships
    for _, row in papers.iterrows():
        doi = row["DOI"]  # Digital Object Identifier
        title = row["Title"]  # Paper title
        abstract = str(row["Abstract"]).strip() if pd.notna(row["Abstract"]) else ""  # Paper abstract
        year = str(row["Year"]).strip() if pd.notna(row["Year"]) else None  # Publication year
        database = row["Database"]  # Source database
        keywords = row["Keywords"]  # Keywords associated with the paper
        summary = row["Summary"]  # Summary containing sections and references
        authors = row["Authors"]  # List of authors

        # Create a node for the paper
        paper = Node("Paper", doi=doi, title=title, abstract=abstract)
        graph.merge(paper, "Paper", "doi")

        # Link the paper to its publication year
        if year:
            year_node = graph.nodes.match("Year", year=str(year).strip()).first()
            if year_node:
                graph.merge(Relationship(paper, "PUBLISHED_IN", year_node))

        # Link the paper to its source database
        database_node = graph.nodes.match("Database", name=database).first()
        if database_node:
            graph.merge(Relationship(paper, "PUBLISHER", database_node))

        # Ensure keywords are processed correctly (convert to list if still a string)
        if isinstance(keywords, str):
            keywords = [word.strip() for word in keywords.split(',')]

        # Link the paper to its associated keywords
        for k in keywords:
            keyword_node = graph.nodes.match("Keyword", keyword=k).first()
            if keyword_node:
                graph.merge(Relationship(paper, "KEYWORDS", keyword_node))

        # Link the paper to its authors
        if isinstance(authors, str):
            # Convert string into a list
            authors = ast.literal_eval(authors) # Convert string into a list

        for author in authors:
            if author:
                author_node = graph.nodes.match("Author", name=author).first()
                if author_node:
                    graph.merge(Relationship(paper, "AUTHORED_BY", author_node))

        if summary:
            # Link the paper to the citations it references
            for citation in summary.get("References", []):  # Avoid errors if "References" key is missing
                citation_node = graph.nodes.match("Citation", doi=citation).first()
                if citation_node:
                    graph.merge(Relationship(paper, "CITES", citation_node))

            # Link the paper to its sections
            for section, content in summary.items():
                if section != "References":  # Ignore the "References" section
                    section_node = Node("Section", id=f"{section} ({doi})", name=section, content=content)
                    graph.merge(section_node, "Section", "id")
                    graph.merge(Relationship(paper, "HAS_SECTION", section_node))

    # Refresh all nodes
    graph.run("MATCH (n) SET n = n RETURN count(n);")

    # Refresh all relationships
    graph.run("MATCH ()-[r]->() SET r = r RETURN count(r);")

    print("Graph created with success!")

    return  # The function modifies the Neo4j database, so no return value is needed
