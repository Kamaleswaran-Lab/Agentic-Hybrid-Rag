from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Functions.auxiliary import extract_keywords_query, extract_keywords_text


def compute_cosine_similarity(query, text):
    """
    This function computes the cosine similarity between core keywords from the query and a given text, thought to be a combination of the abstract and title.

    Parameters:
    query (str): The query string to compare against the text.
    text (str): The text to compare with the query.

    How to use:
    - Call `compute_cosine_similarity(query, text)` with the desired query and text.
    - The function returns the cosine similarity score between the query and the text.

    Output:
    - Returns float value indicating the similarity between the query and text.
    """

    # Process the query to extract relevant keywords
    processed_query = extract_keywords_query(query)

    # Initialize a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform both the query and the abstract into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([processed_query, text])

    # Calculate the cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Return True if the similarity is above the threshold, else return False
    return similarity


def select_papers(papers, query):
    """
    This function filters papers based on their similarity to a given query, retrieves DOIs if missing,
    and ensures that the papers have citations. Papers without a DOI or citations are removed.

    Parameters:
    papers (DataFrame): A pandas DataFrame containing paper details such as DOI, Title, and Abstract.
    query (str): The query to compare against the abstracts of the papers.

    How to use:
    - Call `select_papers(papers, query)` with a DataFrame of papers and a query string.
    - The function returns a filtered DataFrame containing only papers that match the query, have a DOI, and have citations.

    Output:
    - Returns a DataFrame with selected papers that meet the specified criteria.
    """

    # merge text from abstract and title
    papers["Text"] = papers["Title"] + " " + papers["Abstract"]

    # extract keywords from text
    papers["Keywords"] = papers["Text"].apply(lambda x: ", ".join(extract_keywords_text(x)))

    # delete Text column
    papers.drop("Text", axis=1, inplace=True)

    # Apply the cosine similarity function to compute a score for each paper based on its abstract
    papers["Score"] = papers["Keywords"].apply(lambda x: compute_cosine_similarity(query, x))

    # Keep only the papers with a score above the third quantile
    third_quantile = papers['Score'].quantile(0.75)  # Get the 75th percentile value

    # Filter papers with a score greater than or equal to the third quantile
    papers = papers[papers['Score'] >= third_quantile]

    # Drop the Score column as it's no longer needed
    papers.drop("Score", axis=1, inplace=True)

    print(f"Total papers after computing similarity scores: {len(papers)}")

    return papers
