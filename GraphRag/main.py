import warnings

from Functions.auxiliary import get_validated_input
from Functions.search_papers import search_papers
from Functions.select_papers import select_papers
from Functions.download_papers import download_papers


def main():

    # Deactivate warnings
    warnings.filterwarnings("ignore")

    # Prompt user for necessary info
    query, year_low, year_high, max_results, threshold = get_validated_input()

    # search papers in 3 different databases
    papers = search_papers(query=query, max_results=max_results, year_low=year_low, year_high=year_high)

    # stop code if the query didn't retrieve any papers
    if len(papers) == 0:
        print("The search didn't retrieve any papers")
        return

    # select papers after cleaning processes and data aquisition
    papers = select_papers(papers, query, threshold=threshold)

    # download papers
    download_papers(papers)

    # create graph

    # perform GraphRag


if __name__ == "__main__":
    main()
