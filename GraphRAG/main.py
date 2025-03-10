import warnings
import os
import pandas as pd

from Functions.search_papers import search_papers
from Functions.select_papers import select_papers
from Functions.auxiliary import get_validated_input, clean_folder
from Functions.download_papers import download_papers
from Functions.summarize_papers import summarize_papers
from Functions.create_graph import create_knowledge_graph


def main():

    # Deactivate warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # remove papers from last query
    clean_folder("Papers")

    # Prompt user for necessary info
    query, year_low, year_high, max_results = get_validated_input()

    # search papers in 3 different databases
    papers = search_papers(query=query, max_results=max_results, year_low=year_low, year_high=year_high)

    # stop code if the query didn't retrieve any papers
    if len(papers) == 0:
        print("The search didn't retrieve any papers")
        return

    # select papers after cleaning processes and data aquisition
    papers = select_papers(papers, query)

    # download papers
    download_papers(papers)

    # summarize papers
    summarize_papers(papers)

    # save intermediate results
    papers.to_json("Papers.json", orient="table", force_ascii=False)
    print("Results were successfully saved as a json file!")

    # create knowledge graph
    create_knowledge_graph(papers)

    # perform GraphRag


if __name__ == "__main__":
    main()
