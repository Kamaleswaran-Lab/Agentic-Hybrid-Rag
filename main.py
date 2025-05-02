import warnings
import os

from Functions.search_papers import search_papers
from Functions.select_papers import select_papers
from Functions.auxiliary import get_validated_input, clean_folder
from Functions.download_papers import download_papers
from Functions.extract_info import get_citations
from Functions.create_graph import create_knowledge_graph
from Functions.perform_rag import agent_rag
from Functions.create_vector import create_chunks


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

    # extract citations
    get_citations(papers)

    # save intermediate results
    papers.to_json("Papers.json", orient="table", force_ascii=False)
    print("Results were successfully saved as a json file!")

    # create knowledge graph
    create_knowledge_graph(papers)

    # create vector space
    create_chunks()

    # create agent
    agent = agent_rag()

    while True:
        query = input("Enter your question (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        try:
            response = agent.run(user_msg=query)["final_response"]

            print(response)
        except Exception as e:
            print(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
