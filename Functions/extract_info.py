import re
import os

from PyPDF2 import PdfReader
from pathlib import Path
from tqdm import tqdm
from Functions.auxiliary import sanitize_filename


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


def get_citations(papers, folder_name="Papers"):
    # Get the current working directory
    cwd = os.getcwd()

    # Define the full path to the folder where the papers are stored
    path = Path(rf"{cwd}\\{folder_name}")

    # List all PDF files in the folder (ignoring subdirectories)
    documents = [f.name for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']

    papers["References"] = papers.apply(lambda _: [], axis=1)

    # Get Title column in equivalent format with documents name
    papers["Title_sanitized"] = papers["Title"].apply(lambda x: sanitize_filename(x))

    # Loop over each document (PDF) in the folder
    for document in tqdm(documents, desc="Extracting references", unit="document"):

        # Extract text from each section of the paper
        text_by_section = extract_text_by_section(rf"{path}\\{document}")

        ref_count = 0

        # Loop through each section of the paper and summarize
        for k, v in text_by_section.items():
            if "references" in k.lower():

                references = extract_dois(v)

                # Update the "References" column in the DataFrame for the corresponding paper title
                papers.at[papers[papers["Title_sanitized"] == document.split(".")[0]].index[0], "References"] = references

                ref_count = len(references)

        print(f"{ref_count} references were found from {document}")

    # Drop column, since it will no longer be needed
    papers.drop("Title_sanitized", axis=1, inplace=True)

    return
