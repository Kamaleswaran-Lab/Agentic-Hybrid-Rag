import os
import requests
import time

from Functions.auxiliary import sanitize_filename
from metapub import FindIt


# function to download papers from ArXiv and Google Scholar
def download_arxiv_googleScholar(papers, path="Papers", url_column="PDF_URL", title_column="Title", verbose=False):
    """
    Downloads PDFs from a given dataframe containing URLs and titles, saving them to a specified directory.

    Parameters:
    papers (pd.DataFrame): A dataframe containing at least two columns: one for PDF URLs and another for titles.
    path (str): The directory where the downloaded PDFs will be saved (default is "Papers").
    url_column (str): The name of the column in `papers` that contains the PDF URLs (default is "PDF_URL").
    title_column (str): The name of the column in `papers` that contains the titles (default is "Title").
    verbose (bool): If True, prints messages about the download progress (default is False).

    How to use:
    - Call `download_pdfs(papers, path, url_column, title_column, verbose)` with a dataframe as input.
      The function will download PDFs and save them in the specified directory.

    Output:
    - Saves the downloaded PDFs to the specified directory.
    - Prints the total number of successfully downloaded papers.
    """

    # Select only the necessary columns
    papers = papers[[url_column, title_column]]


    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    s = 0  # Counter for successful downloads

    for i, row in enumerate(papers.itertuples(index=False)):
        url = row[0]
        title = sanitize_filename(row[1])

        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    s += 1
                    # Define the file name, using index if title is missing
                    file_pdf = os.path.join(path, f"{title}.pdf") if title else os.path.join(path, f"{i}.pdf")

                    # Save the PDF file
                    with open(file_pdf, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)

                    if verbose:
                        print(f"Downloaded PDF: {title}")
                else:
                    if verbose:
                        print(f"Code {response.status_code} for paper: {title}")

            except Exception as e:
                if verbose:
                    print(f"Error downloading PDF: {e}")
        else:
            if verbose:
                print(f"Could not get link for: {title}")

        # avoid request limit barriers
        time.sleep(0.5)

    return s


# download papers from pubmed
def download_pubmed(papers, path="Papers", pmid_column="PDF_URL", title_column="Title"):
    """
    Downloads PubMed papers in PDF format using their PubMed ID (PMID).

    Parameters:
    - papers (DataFrame): A pandas DataFrame containing the paper information.
    - path (str): The directory where PDFs will be saved (default is "Papers").
    - pmid_column (str): The column name containing the PubMed IDs (default is "PDF_URL").
    - title_column (str): The column name containing paper titles (default is "Title").

    How to use:
    - Call `download_pubmed(papers)` with a DataFrame containing PMIDs and titles.
    - The function will attempt to download full-text PDFs using `FindIt(pmid)`.
    - If a PDF is found, it will be saved using the title as the filename.
    - If no title is available, a numerical index will be used.

    Output:
    - Downloads the available PDFs into the specified directory.
    - Prints errors if a paper is behind a paywall or unavailable.
    - Displays the total number of successfully downloaded papers.
    """

    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    papers = papers[[pmid_column, title_column]]
    s = 0  # Counter for successful downloads

    for i, row in enumerate(papers.itertuples(index=False)):
        pmid = row[0]
        title = sanitize_filename(row[1])

        if pmid:
            src = FindIt(pmid)

            # If a full-text PDF URL is found, attempt to download
            if src.url:
                response = requests.get(src.url)
                pdf_name = title if title else str(i)
                with open(f"{path}/{pdf_name}.pdf", "wb") as f:
                    f.write(response.content)
                    s += 1
            else:
                # If no URL, reason is one of "PAYWALL", "TXERROR", or "NOFORMAT"
                print(f"Error downloading {pmid}: {src.reason}")

        time.sleep(0.5)  # Delay to avoid rate limiting

    return s


# final function to download papers
def download_papers(papers, path="Papers", url_column="PDF_URL", title_column="Title", database_column="Database"):
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

    # Select relevant columns from the DataFrame (database, title, and URL)
    papers = papers[[database_column, title_column, url_column]]

    # Separate papers from PubMed database
    pubmed = papers[papers[database_column] == "PubMed"]

    # Separate papers from other databases (ArXiv, Google Scholar, etc.)
    others = papers[papers[database_column] != "PubMed"]

    # Download PubMed papers using the download_pubmed function
    pubmed_count = download_pubmed(pubmed, path=path, pmid_column=url_column, title_column=title_column)

    # Download papers from other databases (ArXiv, Google Scholar) using the download_arxiv_googleScholar function
    others_count = download_arxiv_googleScholar(others, path=path, url_column=url_column, title_column=title_column, verbose=True)

    # Print the total number of successfully downloaded papers
    print(f"Total papers after downloading process: {pubmed_count + others_count}")

    return
