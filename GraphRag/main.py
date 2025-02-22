import warnings

from Functions.search_papers import search_papers
from Functions.select_papers import select_papers
from Functions.auxiliary import get_validated_input
from Functions.download_papers import download_papers


def main():

    # Deactivate warnings
    warnings.filterwarnings("ignore")

    # Prompt user for necessary info
    query, max_results, threshold = get_validated_input()

    # search papers in 3 different databases
    papers = search_papers(query=query, max_results=max_results)

    # select papers after cleaning processes and data aquisition
    papers = select_papers(papers, query, threshold=threshold)

    # download papers
    download_papers(papers)

    # create graph


    # perform GraphRag

if __name__ == "__main__":
    main()