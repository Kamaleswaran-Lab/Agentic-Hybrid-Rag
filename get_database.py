import pandas as pd

from tqdm import tqdm
from pathlib import Path
from functions import summarizer, find_sections, extract_text_by_section, create_embeddings


def format_data():
    df = pd.read_excel("../Sheets/papers.xlsx")
    df["Doi"] = df.Doi.str.split("/").str[1]

    # Caminho do diretório
    caminho_pasta = Path(r"C:\Users\User\UnB\TCM\Trabalho Final\Papers")

    # Listar apenas pastas
    artigos = [f.name for f in caminho_pasta.iterdir() if f.is_file() and f.name != ".ipynb_checkpoints"]

    chunks = []
    for artigo in tqdm(artigos, desc="Processing papers", unit="paper"):
        pdf_path = f"../Papers/{artigo}"

        doi = artigo.split(".pd")[0]

        # get text divided by sections
        sections = extract_text_by_section(pdf_path)

        # find related important sections
        r = find_sections(sections)

        # summarize introduction
        if r["Introduction"]["content"] != "missing":
            intro = summarizer(r["Introduction"]["content"])
        else:
            print(f"paper {doi} is missing the intro")
            intro = ""

        if r["Methods"]["content"] != "missing":
            methods = summarizer(r["Methods"]["content"])
        else:
            print(f"paper {doi} is missing the methods")
            methods = ""

        if r["Results"]["content"] != "missing":
            results = summarizer(r["Results"]["content"])
        else:
            print(f"paper {doi} is missing the results")
            results = ""

        if r["Conclusion"]["content"] != "missing":
            conclusion = summarizer(r["Conclusion"]["content"])
        else:
            print(f"paper {doi} is missing the conclusion")
            conclusion = ""

        # get right format for each paper
        chunk = f'''
        Title: {df.loc[df.Doi == doi, "Title"].values[0]}

        Abstract: {df.loc[df.Doi == doi, "Abstract"].values[0]}

        Year: {df.loc[df.Doi == doi, "Year"].values[0]}

        Keywords: {df.loc[df.Doi == doi, "Keywords"].values[0]}

        ### Introduction
        {intro}

        ### Methods
        {methods}

        ### Results
        {results}

        ### Conclusion
        {conclusion}    
        '''

        chunks.append(chunk)

    print(chunks)

    create_embeddings(chunks, "../Database")

    return 0
