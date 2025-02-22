import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Functions.auxiliary import extract_keywords
from habanero import Crossref


def compute_cosine_similarity(query, abstract, threshold=0.7):
    """
    This function computes the cosine similarity between a query and an abstract, and checks if it meets a given threshold.

    Parameters:
    query (str): The query string to compare against the abstract.
    abstract (str): The abstract text to compare with the query.
    threshold (float): The minimum similarity score (between 0 and 1) to consider the query and abstract as similar (default is 0.7).

    How to use:
    - Call `compute_cosine_similarity(query, abstract, threshold)` with the desired query and abstract.
    - The function returns True if the cosine similarity between the query and the abstract is greater than or equal to the threshold, and False otherwise.

    Output:
    - Returns a boolean value indicating if the similarity between the query and abstract meets or exceeds the threshold.
    """

    # Process the query to extract relevant keywords
    processed_query = extract_keywords(query)

    # Initialize a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform both the query and the abstract into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([processed_query, abstract])

    # Calculate the cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Return True if the similarity is above the threshold, else return False
    return similarity >= threshold


# Retorn based on Title
def get_dois(title, cr):
    """
    This function retrieves the DOI (Digital Object Identifier) of a paper based on its title by querying an external service.

    Parameters:
    title (str): The title of the paper to search for.
    cr (Crossref): The Crossref client object used to query the Crossref database for DOIs.

    How to use:
    - Call `get_dois(title, cr)` with the paper title and a Crossref client object.
    - The function returns the DOI of the paper if found, or None if not found or if an error occurs.

    Output:
    - Returns the DOI of the paper (str) or None if not found or an error occurs.
    """
    try:
        time.sleep(1.21)  # To avoid rate limits (sleep for 1.21 seconds)

        # Query the Crossref API using the provided title
        result = cr.works(query=title)

        # Extract items from the response
        items = result.get('message', {}).get('items', [])

        # Return the DOI if available, otherwise return None
        if items:
            return items[0].get('DOI', None)
        return None  # Return None if no items are found

    except Exception as e:
        # Print an error message if an exception occurs during the query
        print(f"Error while retrieving DOI for '{title}': {e}")
        return None


# get paper citations
def get_references(doi, cr):
    """
    This function retrieves the references (DOIs) of a paper based on its DOI from the Crossref database.

    Parameters:
    doi (str): The DOI of the paper for which references are to be fetched.
    cr (Crossref): The Crossref client object used to query the Crossref database for references.

    How to use:
    - Call `get_references(doi, cr)` with the DOI of the paper and a Crossref client object.
    - The function returns a list of DOIs for the references of the paper if found, or an empty list if no references are found or if an error occurs.

    Output:
    - Returns a list of DOIs (str) of the references or an empty list if no references are found or an error occurs.
    """
    try:
        # If the DOI is None, return an empty list
        if doi is None:
            return []  # Return an empty list if no DOI is provided

        time.sleep(1.21)  # To avoid rate limits (sleep for 1.21 seconds)

        # Query the Crossref API using the provided DOI
        result = cr.works(doi)

        # Extract the references from the response
        references = result.get("message", {}).get("reference", [])

        # Return the DOIs of references, ensuring they exist
        return [ref['DOI'] for ref in references if 'DOI' in ref]

    except Exception as e:
        # Print an error message if an exception occurs during the query
        print(f"Error while retrieving references for DOI '{doi}': {e}")
        return []  # Return an empty list if an error occurs


def select_papers(papers, query, threshold=0.7):
    """
    This function filters papers based on their similarity to a given query, retrieves DOIs if missing,
    and ensures that the papers have citations. Papers without a DOI or citations are removed.

    Parameters:
    papers (DataFrame): A pandas DataFrame containing paper details such as DOI, Title, and Abstract.
    query (str): The query to compare against the abstracts of the papers.
    threshold (float): The cosine similarity threshold (between 0 and 1) used to filter papers based on their abstract similarity to the query.

    How to use:
    - Call `select_papers(papers, query, threshold)` with a DataFrame of papers, a query string, and a threshold for similarity.
    - The function returns a filtered DataFrame containing only papers that match the query, have a DOI, and have citations.

    Output:
    - Returns a DataFrame with selected papers that meet the specified criteria.
    """
    # Apply the cosine similarity function to compute a score for each paper based on its abstract
    papers["Score"] = papers["Abstract"].apply(lambda x: compute_cosine_similarity(query, x, threshold=threshold))

    # Keep only the papers with a score above the threshold (True)
    papers = papers[papers.Score == True]

    # Drop the Score column as it's no longer needed
    papers.drop("Score", axis=1, inplace=True)

    print(f"Total papers after computing similarity scores: {len(papers)}")

    cr = Crossref(mailto="ric.accorsi@gmail.com")  # Initialize Crossref client

    # For papers without a DOI, try to retrieve it using the title
    papers.loc[papers["DOI"].isna() | (papers["DOI"] == ""), "DOI"] = papers.loc[
        papers["DOI"].isna() | (papers["DOI"] == ""), "Title"
    ].apply(lambda x: get_dois(x, cr))

    # Remove papers that still don't have a DOI
    papers = papers[~papers["DOI"].isna() & (papers["DOI"] != "")]

    print(f"Total papers after removing papers without DOI: {len(papers)}")

    # Retrieve references (citations) for each paper based on its DOI
    papers["Citations"] = papers["DOI"].apply(lambda x: get_references(x, cr))

    # Keep only the papers that have at least one citation
    papers = papers[papers["Citations"].apply(len) > 0]

    print(f"Total papers after removing papers without citations: {len(papers)}")

    return papers
