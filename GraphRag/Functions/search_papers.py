import arxiv
import pandas as pd

from metapub import PubMedFetcher
from tqdm import tqdm
from scholarly import scholarly, ProxyGenerator
from Functions.auxiliary import format_query, generate_final_df


# Function: Fetch ArXiv Papers
def arxiv_search(query, max_results=10):
    """
    This function performs a search on ArXiv based on the provided query and returns a DataFrame with paper details.

    Parameters:
    query (str): The query string to search on ArXiv.
    max_results (int): The maximum number of results to retrieve (default is 10).
    general_query (bool): If True, the query will be formatted (default is True).

    How to use:
    - Call `arxiv_search(query, max_results, general_query)` with the desired query.
    - The function returns a DataFrame with details of the retrieved papers.

    Output:
    - Returns a DataFrame with the following columns:
      - 'DOI': The Digital Object Identifier of the paper.
      - 'Title': The title of the paper.
      - 'Abstract': The summary of the paper.
      - 'Year': The publication year.
      - 'Authors': The list of authors.
      - 'PDF_URL': The URL to the PDF of the paper.
      - 'Database': The database source (ArXiv).
    """
    # format query if needed
    if "AND" in query:
        query = format_query(query)

    # Create a client to connect to ArXiv
    client = arxiv.Client()

    # Define the search parameters
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance  # Sort by relevance
    )

    # Perform the search and get the results
    results = list(client.results(search))

    papers = []  # List to store paper details
    for result in tqdm(results, desc="Fetching ArXiv articles"):  # Progress bar while fetching results
        papers.append({
            "DOI": result.doi,
            "Title": result.title,
            "Abstract": result.summary,
            "Year": result.published,
            "Authors": ", ".join([author.name for author in result.authors]),  # Combine author names
            "PDF_URL": result.pdf_url,
            "Database": "ArXiv"  # Specify the source as ArXiv
        })

    # Convert the list of papers into a DataFrame
    papers = pd.DataFrame(papers)

    # Convert 'Year' column to datetime and handle errors
    papers['Year'] = pd.to_datetime(papers['Year'], errors='coerce')

    # Extract only the year from the datetime object
    papers['Year'] = papers['Year'].dt.year

    # Print the total number of papers retrieved
    print(f"Total papers retrieved from Arxiv: {len(papers)}")

    # Return the DataFrame containing the paper details
    return papers


# search pubmed database and get metadata
def pubmed_search(query, max_results=10):
    """
    This function performs a search on PubMed based on the provided query and returns a DataFrame with paper details.

    Parameters:
    query (str): The query string to search on PubMed.
    max_results (int): The maximum number of results to retrieve (default is 10).

    How to use:
    - Call `pubmed_search(query, max_results)` with the desired query.
    - The function returns a DataFrame with details of the retrieved papers.

    Output:
    - Returns a DataFrame with the following columns:
      - 'DOI': The Digital Object Identifier of the paper.
      - 'Title': The title of the paper.
      - 'Abstract': The summary of the paper.
      - 'Year': The publication year.
      - 'Authors': The list of authors.
      - 'PDF_URL': The PMID of the paper.
      - 'Database': The database source (PubMed).
    """

    # Initialize the PubMedFetcher to fetch data
    fetch = PubMedFetcher()

    # Get the PMIDs (PubMed IDs) for the query
    pmids = fetch.pmids_for_query(query, retmax=max_results)

    # Initialize dictionaries to store data for articles
    articles = {}
    titles = {}
    abstracts = {}
    authors = {}
    years = {}
    dois = {}

    # Loop through PMIDs and fetch article details, showing a progress bar
    for pmid in tqdm(pmids, desc="Fetching PubMed articles"):
        if pmid is not None:
            # Fetch the article by PMID
            article = fetch.article_by_pmid(pmid)
            # Store the article details in dictionaries
            articles[pmid] = article
            titles[pmid] = article.title
            abstracts[pmid] = article.abstract
            authors[pmid] = article.authors
            years[pmid] = article.year
            dois[pmid] = article.doi
        else:
            print(f"Couldn't retrieve pmid {pmid}")

    # Create a DataFrame from the retrieved article data
    papers = pd.DataFrame({
        'DOI': dois.values(),
        'Title': titles.values(),
        'Abstract': abstracts.values(),
        'Year': years.values(),
        'Authors': authors.values(),
        'PDF_URL': pmids,
    })

    # Add a column specifying the database source
    papers["Database"] = "PubMed"

    # Print the total number of papers retrieved from PubMed
    print(f"Total papers retrieved from PubMed: {len(papers)}")

    # Return the DataFrame containing the paper details
    return papers


# function to google scholar papers
def googleScholar_search(query, max_results=10):
    """
    This function performs a search on Google Scholar based on the provided query and returns a DataFrame with paper details.

    Parameters:
    query (str): The query string to search on Google Scholar.
    max_results (int): The maximum number of results to retrieve (default is 10).

    How to use:
    - Call `googleScholar_search(query, max_results)` with the desired query.
    - The function returns a DataFrame with details of the retrieved papers.

    Output:
    - Returns a DataFrame with the following columns:
      - 'DOI': The Digital Object Identifier of the paper (not available from Google Scholar).
      - 'Title': The title of the paper.
      - 'Abstract': The summary of the paper.
      - 'Year': The publication year.
      - 'Authors': The list of authors.
      - 'PDF_URL': The URL to the paper's PDF (if available).
      - 'Database': The database source (Google Scholar).
    """

    # Initialize the proxy generator for Google Scholar access (this line can be uncommented to use free proxies)
    pg = ProxyGenerator()

    scholarly.use_proxy(pg, pg)  # Set up the proxy for scholarly

    # Perform the search on Google Scholar
    results = scholarly.search_pubs(query)

    # Create an empty DataFrame to store paper details
    papers = pd.DataFrame(columns=["DOI", "Title", "Abstract", "Year", "Authors", "PDF_URL", "Database"])

    # Loop through the search results with a progress bar
    for i, result in tqdm(enumerate(results), desc="Fetching Google Scholar articles"):

        # Stop if the max results limit is reached
        if i >= max_results:
            break

        # Extract relevant data from the search result and add it to the DataFrame
        paper = pd.DataFrame([{
            "DOI": "",  # DOI is not retrieved from Google Scholar
            "Title": result.get("bib", {}).get("title", ""),
            "Abstract": result.get("bib", {}).get("abstract", ""),
            "Year": result.get("bib", {}).get("pub_year", ""),
            "Authors": ", ".join(result.get("bib", {}).get("author", [])),
            "PDF_URL": result.get("eprint_url", ""),
            "Database": "Google Scholar"  # Set the database source as Google Scholar
        }])

        # Concatenate the new paper data to the DataFrame
        papers = pd.concat([papers, paper], ignore_index=True)

    # Print the total number of papers retrieved
    print(f"Total papers retrieved from Google Scholar: {len(papers)}")

    # Return the DataFrame containing the paper details
    return papers


def search_papers(query, max_results=10):
    """
    This function searches for academic papers across multiple databases (PubMed, Google Scholar, and ArXiv)
    based on the provided query and returns a combined DataFrame with paper details.

    Parameters:
    query (str): The query string to search across the databases.
    max_results (int): The maximum number of results to retrieve from each database (default is 10).
    adapt_arxiv (bool): Whether to adapt the ArXiv search query for a general or specific search (default is True).

    How to use:
    - Call `search_papers(query, max_results, adapt_arxiv)` with the desired query.
    - The function returns a combined DataFrame with details of the retrieved papers from all databases.

    Output:
    - Returns a DataFrame with the following columns:
      - 'DOI': The Digital Object Identifier of the paper.
      - 'Title': The title of the paper.
      - 'Abstract': The summary of the paper.
      - 'Year': The publication year.
      - 'Authors': The list of authors.
      - 'PDF_URL': The URL to the paper's PDF (if available).
      - 'Database': The database source (PubMed, Google Scholar, or ArXiv).
    """

    # Perform search on PubMed
    pubmed = pubmed_search(query=query, max_results=max_results)

    # Perform search on Google Scholar
    google_scholar = googleScholar_search(query, max_results=max_results)

    # Perform search on ArXiv
    arxiv = arxiv_search(query=query, max_results=max_results)

    # Combine results from all databases into a single DataFrame
    return generate_final_df(arxiv, pubmed, google_scholar)
