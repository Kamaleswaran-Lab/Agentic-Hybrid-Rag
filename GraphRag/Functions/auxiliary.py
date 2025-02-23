import pandas as pd
import re


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

    # Validate threshold (float between 0 and 1)
    while True:
        try:
            threshold = float(input("What should be the threshold similarity score (0-1): ").strip())
            if 0 <= threshold <= 1:
                break
            else:
                print("Threshold must be between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid decimal number between 0 and 1.")

    return query, year_low, year_high, max_results, threshold


def format_query(query):
    """
    This function takes a query string, splits it into parts based on "AND",
    removes unnecessary spaces, and formats each part by enclosing it in 'all:()'.

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


def extract_keywords(query):
    """
    Extracts keywords from a boolean query, regardless of its format,
    and returns a list of these keywords.

    Parameters:
    query (str): The boolean query string from which keywords will be extracted.

    How to use:
    - Call `extract_keywords(query)` with a boolean query as input.
      The function will return a string with extracted keywords separated by spaces.

    Output:
    - Returns a string of extracted keywords, where each keyword is separated by a space.
    """

    # Find all words or quoted phrases using a regular expression
    keywords = re.findall(r'"([^"]+)"|(\w+\*?)', query)

    # 'keywords' returns a list of tuples, so we extract the first non-null value from each tuple
    extracted_keywords = [kw[0] if kw[0] else kw[1] for kw in keywords]

    # Remove logical operators like 'AND', 'OR', and 'all' from the list of keywords
    extracted_keywords = [kw for kw in extracted_keywords if kw not in ['AND', 'OR', 'all']]

    # Remove duplicate keywords
    extracted_keywords = list(set(extracted_keywords))

    # Return a single string with keywords separated by space
    return " ".join(extracted_keywords)  # Return the keywords as a space-separated string


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

    return re.sub(r'[<>:"/\\|?*]', '_', filename)
