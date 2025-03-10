from transformers import AutoTokenizer, pipeline
import re
import os
import torch
#import ollama

from PyPDF2 import PdfReader
from pathlib import Path
from tqdm import tqdm
from Functions.auxiliary import sanitize_filename


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizador from pretrained BART model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# load summarizer from pretrained BART model
summarizer = pipeline("summarization", model="facebook/bart-base", device=0 if device == "cuda" else -1)


def clean_text(line):
    """
    Cleans the input text by removing non-ASCII characters and excessive spaces.

    Parameters:
    line (str): The input text that needs to be cleaned.

    How to use:
    - Call `clean_text(line)` with a string as input.
      The function will return a cleaned version of the text.

    Output:
    - Returns a string with:
      - Non-ASCII characters replaced by a space.
      - Consecutive spaces reduced to a single space.
      - Leading and trailing whitespace removed.
    """

    # Remove non-ASCII characters by replacing them with a single space
    line = re.sub(r"[^\x20-\x7E]+", " ", line)

    # Replace consecutive spaces with a single space
    line = re.sub(r"\s{2,}", " ", line)

    # Return the cleaned line after stripping leading/trailing whitespace
    return line.strip()


def extract_text_by_section(pdf_path):
    """
    Extracts text from a PDF and organizes it into sections based on main and subsection headers.
    Sections are identified using regex patterns for main and subsection headers.

    Parameters:
    pdf_path (str): The path to the PDF file from which text will be extracted.

    How to use:
    - Call `extract_text_by_section(pdf_path)` with the path to a PDF as input.
      The function will return a dictionary with sections as keys and their corresponding text as values.

    Output:
    - Returns a dictionary where the keys are section titles, and the values are the text associated with those sections.
      Sections without content will be marked as "Missing content" and omitted from the final output.
    """

    # Initialize dictionary to store sections and the current section being processed
    text_by_section = {"Unlabeled Section": ""}
    current_section = "Unlabeled Section"

    # Regex patterns for identifying main section and subsection headers
    main_section_pattern = re.compile(r"^\d{1,3}(\.|\s|\.\s)[A-Z][A-Za-z\-]+(,?\s[A-Za-z\-]+)*(:)?$")
    subsection_pattern = re.compile(r"^\d{1,3}\.\d{1,2}(\.\s|\s|\.)[A-Z][A-Za-z\-]+(,?\s[A-Za-z\-]+)*(:)?$")

    # Open the PDF and extract text from each page
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)  # Read the PDF using PdfReader
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()  # Extract text from the current page
            if not text:  # Skip pages with no extractable text
                continue

            lines = text.split("\n")  # Split the extracted text into lines
            cleaned_lines = [clean_text(line) for line in lines]  # Clean each line using the clean_text function
            for line in cleaned_lines:
                stripped_line = line.strip()  # Strip leading/trailing spaces from the line

                # Check if the line matches the pattern for main section headers
                if main_section_pattern.match(stripped_line):
                    current_section = stripped_line  # Set current_section to the new main section
                    if current_section not in text_by_section:
                        text_by_section[current_section] = ""  # Initialize the section in the dictionary

                # Check if the line matches the pattern for subsection headers
                elif subsection_pattern.match(stripped_line):
                    text_by_section[current_section] += f"\n{stripped_line}"  # Append subsection to the current section

                # Add any other text to the current section
                else:
                    text_by_section[current_section] += " " + stripped_line

    # Clean the sections: remove leading/trailing spaces and handle missing content
    text_by_section = {
        section: content.strip() or "Missing content"
        for section, content in text_by_section.items()
    }

    # Remove the default "Unlabeled Section" if it exists
    if "Unlabeled Section" in text_by_section:
        del text_by_section["Unlabeled Section"]

    # Remove sections with no content ("Missing content") and format section to leave just its actual name
    text_by_section = {re.sub(r'\.+$', '', re.sub(r'\d+', '', k)): v for k, v in text_by_section.items() if v != "Missing content"}

    return text_by_section


def split_text_into_chunks(text, max_tokens=512):
    """
    Divides the input text into chunks, each containing a maximum of `max_tokens` tokens,
    ensuring sentence boundaries are respected.

    Parameters:
    text (str): The input text that will be split into chunks.
    max_tokens (int): The maximum number of tokens allowed per chunk. Default is 1024.

    How to use:
    - Call `split_text_into_chunks(text, max_tokens)` with the input text and optional token limit.
    - The function will return a list of text chunks, each having a token count of `max_tokens` or less.

    Output:
    - Returns a list of strings, where each string is a chunk of the original text.
      Each chunk will contain sentences that do not exceed the token limit.
    """

    # Split the input text into sentences based on punctuation marks (., !, ?)
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Variables to store the current chunk, token count, and the list of chunks
    current_chunk = ""  # Initialize the current chunk as an empty string
    current_tokens = 0  # Initialize the token counter for the current chunk
    chunks = []  # Initialize the list to store the resulting chunks

    # Loop through each sentence to build chunks
    for sentence in sentences:
        # Encode the sentence into tokens (without truncation or padding)
        sentence_tokens = tokenizer.encode(sentence, truncation=True, padding=False, max_length=max_tokens)

        # Check if adding this sentence would exceed the max token limit
        if current_tokens + len(sentence_tokens) <= max_tokens:
            # If it fits, add the sentence to the current chunk
            current_chunk += sentence + " "
            current_tokens += len(sentence_tokens)
        else:
            # If it exceeds, add the current chunk to the list and start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())  # Add the current chunk to the list
            current_chunk = sentence + " "  # Start a new chunk with the current sentence
            current_tokens = len(sentence_tokens)  # Reset the token count for the new chunk

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())  # Add the final chunk to the list

    return chunks


def summarize_text(text, max_tokens=512):
    """
    Summarizes the input text by splitting it into chunks of up to `max_tokens` tokens,
    respecting sentence boundaries. Then, it generates a summary for each chunk and
    combines them into a final summary.

    Parameters:
    text (str): The input text that will be summarized.
    max_tokens (int): The maximum number of tokens allowed per chunk. Default is 1024.

    Returns:
    - A summarized version of the input text.
    """

    # Split the text into chunks of up to `max_tokens` tokens
    chunks = split_text_into_chunks(text, max_tokens)

    # List to store the final summarized text
    summarized_chunks = []

    # Iterate through the chunks to summarize those that need it and add the rest as is
    for chunk in chunks:

        # summarize chunks
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
        summarized_chunks.append(summary[0]["summary_text"])

    # Join all the chunks back together while preserving the order
    return " ".join(summarized_chunks)  # Return the concatenated summaries with preserved order


def extract_dois(text):
    """
    Extracts all DOIs from the given text and returns them as a list.

    Parameters:
    text (str): The text from which DOIs will be extracted.

    How to use:
    - Call `extract_dois(text)` with a string `text` that contains references.
    - The function will return a list of DOIs found in the text.

    Output:
    - Returns a list of DOIs extracted from the text.
    """
    # Regular expression to match DOIs
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+(?:[A-Z0-9])\b"

    # Find all matching DOIs in the text
    dois = re.findall(doi_pattern, text, re.IGNORECASE)

    return dois


def summarize_papers(papers, folder_name="Papers"):
    """
    Summarizes the papers found in a specified folder and stores the summaries in a DataFrame.

    Parameters:
    papers (DataFrame): A pandas DataFrame containing the papers, where each paper has a "Title" column.
    folder_name (str): The name of the folder where the PDF papers are stored. Default is "Papers".

    How to use:
    - Call `summarize_papers(papers, folder_name)` with the DataFrame `papers` and the folder name.
    - The function will iterate over all PDF files in the folder, extract sections, summarize them,
      and update the "Summary" column in the DataFrame with the summaries of each document.

    Output:
    - Updates the "Summary" column in the `papers` DataFrame with summaries for each document.
    """

    # Get the current working directory
    cwd = os.getcwd()

    # Define the full path to the folder where the papers are stored
    path = Path(rf"{cwd}\\{folder_name}")

    # List all PDF files in the folder (ignoring subdirectories)
    documents = [f.name for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']

    # Initialize the "Summary" column in the papers DataFrame
    papers["Summary"] = ""

    # Loop over each document (PDF) in the folder
    for document in tqdm(documents, desc="Summarizing documents", unit="document"):

        # Extract text from each section of the paper
        text_by_section = extract_text_by_section(rf"{path}\\{document}")

        # Dictionary to store summaries for each section
        summaries = {}

        cont = 0
        # Loop through each section of the paper and summarize
        for k, v in text_by_section.items():

            # if section is not the references section, summarize it
            if "references" not in k.lower():

                # Generate summary for each section of the paper
                try:
                    summary = summarize_text(v)
                    # treat section name
                    k = re.sub(r'\d+', '', k)
                    k = k.replace(".", " ").strip()

                    # drop citations within the text to make it more fluid
                    summary = re.sub(r'\[.*?\]', '', summary)

                    # Store the summary for the section
                    summaries[k] = summary

                except Exception as e:
                    print(f"Couldn't summarize section '{k}' from {document}: {e}")
                    pass

            # if references section is found, extract dois
            else:
                summaries["References"] = extract_dois(v)
                print(summaries["References"])

            cont += 1

            print(f"{cont}/{len(text_by_section)} sections were summarized from {document}")

        # Get Title column in equivalent format with documents name
        papers["Title_sanitized"] = papers["Title"].apply(lambda x: sanitize_filename(x))

        # Update the "Summary" column in the DataFrame for the corresponding paper title
        papers.at[papers[papers["Title_sanitized"] == document.split(".")[0]].index[0], "Summary"] = summaries

        # Drop column, since it will no longer be needed
        papers.drop("Title_sanitized", axis=1, inplace=True)

    return
