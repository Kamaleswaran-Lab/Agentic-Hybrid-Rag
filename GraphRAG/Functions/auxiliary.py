import pandas as pd
import re
import nltk
import spacy
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def get_validated_input():
    """
    Prompts the user for a query, date range in YYYY-YYYY format, max results and a threshold, ensuring all inputs are valid.

    Output:
    - Returns a tuple containing the validated values:
      - `query` (str)
      - `date_range` (tuple of str or None)
      - `max_results` (int)
      - `threshold` (float)
    """
    # Ensure the query is not empty
    while True:
        query = input("Insert query: ").strip()
        if query:
            break
        print("Query cannot be empty. Please enter a valid query.")

    # Validate date range (YYYY-YYYY) and ensure the start year is less than or equal to the end year
    while True:
        date_range = input("Enter the date range (YYYY-YYYY): ").strip()
        if "-" in date_range:
            years = date_range.split("-")
            if len(years) == 2 and years[0].isdigit() and years[1].isdigit():
                year_low, year_high = int(years[0]), int(years[1])
                if year_low <= year_high:
                    break

        print("Invalid format. Please enter the date range in YYYY-YYYY format with a valid range (e.g., 2000-2023).")

    # Validate max_results (integer)
    while True:
        try:
            max_results = int(input("How many should each database retrieve: ").strip())
            if max_results > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    return query, year_low, year_high, max_results


def format_query(query, year_low=2020, year_high=2025):
    """
    This function takes a query string, splits it into parts based on "AND",
    removes unnecessary spaces, and formats each part by enclosing it in 'all:()'.
    It adapts general queries to  fit into ArXiv standard format

    Parameters:
    query (str): The input query string that contains multiple parts separated by "AND".

    How to use:
    - Pass a query string to the function, for example:
      formatted_query = format_query("condition1 AND condition2 AND condition3")

    Output:
    - The function will return a formatted string where each part is wrapped in 'all:()',
      and the parts are joined by "AND". Example output:
      "all:(condition1) AND all:(condition2) AND all:(condition3)"
    """

    # Split the query based on the appearance of "AND" and remove unnecessary spaces
    parts = [part.strip() for part in query.split("AND")]

    if query.startswith("(") and query.endswith(")"):
        # Format each part by removing the leading and trailing spaces and enclosing the part in parentheses
        formatted_parts = [f"all:({part[1:-1].strip()})" for part in parts]

    else:
        formatted_parts = [f"all:({part.strip()})" for part in parts]

    formatted_parts.append(f"all:(submittedDate:[{year_low}0101 TO {year_high}1231])")

    # Join all the formatted parts with "AND" between them
    return " AND ".join(formatted_parts)


def generate_final_df(*dfs):
    """
    This function combines multiple DataFrames into a single DataFrame, removes duplicates,
    and sorts the data based on the number of non-null values for each row.

    Parameters:
    *dfs (DataFrame): One or more DataFrames to be concatenated.

    How to use:
    - Call `generate_final_df()` with the DataFrames you wish to combine. The function will merge them,
      remove duplicates based on DOI and Title, and sort them by the number of non-null values.

    Output:
    - Returns the final cleaned and sorted DataFrame.
    """

    # Concatenate all input DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    print(f"Total papers retrieved: {len(final_df)}")  # Print the total number of papers retrieved

    # remove papers without DOI or Title missing information
    final_df.dropna(subset=["DOI", "Title"], how="any", inplace=True)

    print(f"Total papers after removing papers without DOI or Title: {len(final_df)}")

    # Calculate the number of non-null values for each row
    final_df['non_null_count'] = final_df.notnull().sum(axis=1)

    # Sort the DataFrame by the count of non-null values and by ['doi', 'title']
    final_df = final_df.sort_values(by='non_null_count', ascending=True)

    # Drop duplicates, keeping the row with the most non-null values
    final_df = final_df.drop_duplicates(subset=['DOI', 'Title'], keep='first')

    # Drop the helper column 'non_null_count' as it is no longer needed
    final_df = final_df.drop(columns=['non_null_count'])

    print(
        f"Total papers after removing duplicates: {len(final_df)}")  # Print the total number of papers after removing duplicates

    return final_df  # Return the final cleaned DataFrame


def preprocess_text(text):
    """
    Preprocesses the input text by performing the following operations:
    - Converts the text to lowercase.
    - Removes punctuation and numbers.
    - Tokenizes the text into individual words.
    - Removes stopwords (common words that do not add much meaning, e.g., 'the', 'is', etc.).
    - Lemmatizes the words (reduces them to their base form).

    Parameters:
    text (str): The input text to be preprocessed.

    How to use:
    - Call `preprocess_text(text)` with a text string as input.
      The function will return the preprocessed text as a string.

    Output:
    - Returns a string with the preprocessed text (lowercased, no punctuation or numbers, stopwords removed, and lemmatized).
    """

    # Step 1: Convert the text to lowercase
    text = text.lower()

    # Step 2: Remove punctuation & numbers using regex (only keeps lowercase alphabetic characters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 3: Tokenize the text (split it into individual words)
    nlp = spacy.load("en_core_web_sm")
    tokens = [token.text for token in nlp(text)]

    # Step 4: Remove stopwords using NLTK's predefined list of English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Lemmatize each token (convert words to their base form)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Return the preprocessed text as a single string
    return ' '.join(lemmatized_tokens)


def extract_keywords_text(text, num_keywords=5):
    """
    Extracts the top N keywords from the input text using TF-IDF (Term Frequency - Inverse Document Frequency) method.
    The TF-IDF method highlights important words based on their frequency in the text and their rarity across a corpus.

    Parameters:
    text (str): The input text from which keywords will be extracted.
    num_keywords (int): The number of top keywords to extract. Default is 5.

    How to use:
    - Call `extract_keywords_text(text, num_keywords)` with a text string and the desired number of keywords.
      The function will return a list of the most important keywords based on TF-IDF.

    Output:
    - Returns a list of strings representing the most important keywords in the input text.
    """

    # Step 1: Preprocess the input text using the `preprocess_text` function
    text = preprocess_text(text)

    # Step 2: Initialize the TF-IDF vectorizer with the desired number of keywords (n-grams)
    vectorizer = TfidfVectorizer(max_features=num_keywords, ngram_range=(1, 2))  # Unigrams and bigrams

    # Step 3: Compute the TF-IDF matrix for the input text
    tfidf_matrix = vectorizer.fit_transform([text])

    # Step 4: Get the feature names (keywords)
    keywords = vectorizer.get_feature_names_out()

    # Return the list of extracted keywords
    return keywords


def extract_keywords_query(query):
    """
    Extracts keywords from a boolean query, which may include phrases in quotes, and removes logical operators.

    Parameters:
    query (str): The boolean query string from which keywords will be extracted.

    How to use:
    - Call `extract_keywords_query(query)` with a boolean query as input.
      The function will return a string of keywords separated by commas.

    Output:
    - Returns a string of extracted keywords, where each keyword is separated by a comma.
      The keywords are in lowercase and any logical operators (such as 'AND', 'OR', 'all') are excluded.
      Phrases enclosed in quotes are treated as single keywords.
    """

    # Remove parentheses, colons, and asterisks from the query to focus on the keywords
    query = query.replace("(", "").replace(")", "").replace(":", "").replace("*", "").replace("-", " ")

    # Step 1: Split the query into individual words
    words = query.split()

    # Step 2: Initialize an empty list to store the extracted keywords
    keywords = []

    i = 0
    while i < len(words):
        word = words[i]

        # If the word is a quoted phrase (starts with a quote)
        if word.startswith("'") or word.startswith('"'):
            phrase = word.strip("'\"")  # Remove the surrounding quotes
            c = i + 1  # Start after the initial quoted word
            # Continue concatenating words until another quote is encountered
            while not words[c].endswith("'") and not words[c].endswith('"'):
                phrase += " " + words[c]
                c += 1
            phrase += " " + words[c].strip("'\"")  # Add the last word, removing the quote
            keywords.append(phrase.lower())  # Append the complete phrase to the list
            i = c + 1  # Move the index to the word after the quoted phrase
        else:
            # If the word is not a logical operator ('AND', 'OR', 'all'), add it to the keywords list
            if word.lower() not in {'and', 'or', 'all'}:
                keywords.append(word.lower())
            i += 1

    # Step 3: Remove any duplicate keywords by converting the list to a set and back to a list
    keywords = list(set(keywords))

    # Step 4: Sort the keywords alphabetically
    keywords.sort()

    # Return the keywords as a single string, separated by commas
    return ", ".join(keywords)


def sanitize_filename(filename):
    """
    Removes invalid characters from a filename and replaces them with '_',
    ensuring the filename is safe for saving.

    Parameters:
    filename (str): The original filename that may contain invalid characters.

    How to use:
    - Call `sanitize_filename(filename)` with a filename as input.
      The function will return a sanitized version of the filename.

    Output:
    - Returns a string with invalid characters (<, >, :, ", /, \, |, ?, *) replaced by '_'.
    """
    filename = filename.replace(".", "") # Remove final '.' if there is one

    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def clean_folder(path):
    """
    Deletes all files inside the specified folder while keeping the folder intact.

    Parameters:
    path (str): The path to the folder where files should be deleted.

    Usage:
    clean_folder("C:/Users/User/Documents/temp_folder")

    Output:
    - Deletes all files in the specified folder.
    - Does not delete subdirectories.
    - Returns None.
    """

    # Iterate through all files in the given folder
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)  # Get the full file path

        # Check if the current item is a file (not a folder)
        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file

    return  # Function does not return anything


def format_author_arxiv(authors):
    """
    Formats author names from arXiv-style to the format "LastName Initials".

    Parameters:
    - authors (list of str): A list of author names, where each name is a full name string.

    How to Use:
    - Call this function with a list of author names.
    - Example: format_author_arxiv(["John Doe", "Alice B. Smith"]) → ["Doe J", "Smith AB"]

    Output:
    - Returns a list of formatted author names where:
      - The last name is preserved.
      - The first and middle names are converted to initials.

    """

    formatted_authors = []  # Initialize an empty list to store formatted author names.

    for author in authors:  # Iterate over each author's full name.
        parts = author.split()  # Split the full name into parts (first name, middle name, last name).

        if len(parts) > 1:  # Ensure there is more than one name part.
            last_name = parts[-1]  # Extract the last name (last element in the list).
            initials = "".join([name[0] for name in parts[:-1]])  # Get initials of first and middle names.
            formatted_authors.append(f"{last_name} {initials}")  # Format as "LastName Initials" and add to the list.
        else:
            formatted_authors.append(parts[0])  # If there is only one name, keep it as is.

    return formatted_authors  # Return the list of formatted author names.


def format_author_googlescholar(authors):
    """
    This function takes a string of author names and formats them into a new structure where
    the last name comes first, followed by the initials of the first and middle names (if present).

    Parameters:
    names_str (str): A string containing author names in the format "First Initial Last Name, First Initial Last Name, ..."

    How to use:
    - Pass a string of author names separated by commas in the format: "First Initial Last Name".
    - The function will return the names in the format: "Last Name First Initial".

    Output:
    str: A formatted string where the names are in the format: [Last Name First Initial(s), Last Name First Initial(s), ...]
    """

    # Split the input string into individual author names by comma separator
    formatted_authors = []

    # Iterate over each author in the list
    for author in authors:
        # Split the author's name into parts (initials and last name)
        parts = author.split()

        # Extract the last name (last part) and the initials (first letter of the first name and middle name if available)
        last_name = parts[-1]
        initials = "".join([part[0] for part in parts[:-1]])

        # Format the name as Last Name First Initial(s) and append to the list
        formatted_authors.append(f"{last_name} {initials}")

    # return formatted_authors
    return formatted_authors
