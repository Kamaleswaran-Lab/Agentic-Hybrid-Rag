import re
import faiss
import pickle

from rapidfuzz import process
from PyPDF2 import PdfReader
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama, OllamaLLM


def clean_text(line):
    # Remove excessive spaces, special characters, or invalid sequences
    line = re.sub(r"[^\x20-\x7E]+", " ", line)  # Remove non-ASCII
    line = re.sub(r"\s{2,}", " ", line)  # Replace multiple spaces with one
    return line.strip()


def extract_text_by_section(pdf_path):
    text_by_section = {"Unlabeled Section": ""}
    current_section = "Unlabeled Section"

    # Regex patterns for main and subsection headers
    main_section_pattern = re.compile(r"^\d{1,3}(\.|\s|\.\s)[A-Z][A-Za-z\-]+(,?\s[A-Za-z\-]+)*(:)?$")
    subsection_pattern = re.compile(r"^\d{1,3}\.\d{1,2}(\.\s|\s|\.)[A-Z][A-Za-z\-]+(,?\s[A-Za-z\-]+)*(:)?$")

    # define stop terms
    stop_terms = ["Appendix", "Declaration of Competing Interest", "Acknowledgements", "Acknowledgement", "References", "Credit authorship contribution statement", "CRediT authorship contribution statement"]

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            cleaned_lines = [clean_text(line) for line in lines]
            for line in cleaned_lines:
                stripped_line = line.strip()

                # Check if conclusion is over
                if any(stripped_line == term for term in stop_terms):
                    return text_by_section  # stop reading current pdf

                # Check for main section headers
                if main_section_pattern.match(stripped_line):
                    current_section = stripped_line
                    if current_section not in text_by_section:
                        text_by_section[current_section] = ""

                # Check for subsection headers and append to the current main section
                elif subsection_pattern.match(stripped_line):
                    text_by_section[current_section] += f"\n{stripped_line}"

                # Add other text to the current section
                else:
                    text_by_section[current_section] += " " + stripped_line

    # Remove leading/trailing spaces and ensure no empty sections
    text_by_section = {
        section: content.strip() or "Missing content"
        for section, content in text_by_section.items()
    }

    return text_by_section


# Keywords for each target section
SECTION_KEYWORDS = {
    "Introduction": ["Introduction", "Background", "Overview"],
    "Methods": ["Methods", "Method", "Methodology", "System Description", "Approach", "Modelling", "Modeling", "Model", "Assumptions", "Experimental", "Process", "System Model", "System Modeling", "System Modelling", "System", "Setup", "Configuration"],
    "Results": ["Results", "Results and discussions", "Findings", "Discussion", "Discussions", "Analysis", "Proposed", "Proposition", "Proposal", "Implications"],
    "Conclusion": ["Conclusions", "Conclusion", "Summary", "Final Remarks", "Final Considerations", "Remarks", "Recommendations",  "Concluding", "Challenges"]
}


def find_sections(text_by_section, section_keywords=SECTION_KEYWORDS, threshold=80):
    matched_sections = {}

    for target_section, keywords in section_keywords.items():
        best_match = None
        best_score = 0
        content = "missing"  # Default content for unmatched sections

        for section_title, section_content in text_by_section.items():
            # Match section titles to keywords using fuzzy matching
            result = process.extractOne(section_title, keywords)
            if result:
                match, score, *_ = result
                if score > best_score and score >= threshold:
                    best_match = section_title
                    best_score = score
                    content = section_content

        # Store the result for the target section
        matched_sections[target_section] = {
            "original_title": best_match or "unmatched",
            "content": content,
        }

    return matched_sections


def split_text_into_chunks(text, max_length):
    words = text.split()
    chunks = []
    while words:
        chunk = words[:max_length]
        chunks.append(" ".join(chunk))
        words = words[max_length:]
    return chunks


def summarizer(text, max_length=150, min_length=50):
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Split input text if too long
    if len(text.split()) > 512:
        text_chunks = split_text_into_chunks(text, max_length=500)
        summaries = []
        for chunk in text_chunks:
            tokenized = tokenizer(chunk, truncation=True, max_length=512, return_tensors="pt")
            summary_ids = model.generate(tokenized["input_ids"], max_length=max_length, min_length=min_length, repetition_penalty=2.5)
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        return " ".join(summaries)
    else:
        tokenized = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        summary_ids = model.generate(tokenized["input_ids"], max_length=max_length, min_length=min_length, repetition_penalty=2.5)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def create_embeddings(chunk, path):

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or any other suitable model

    # Create embeddings
    embeddings = embedder.encode(chunk, convert_to_tensor=False)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save the index and documents
    with open(f"{path}/chunk.pkl", "wb") as f:
        pickle.dump(chunk, f)
    faiss.write_index(index, "faiss_index")

    print(f"Chunk file saved with success at {path}")

    return 0


def RAG(query, path, n_chunks=5):
    # Load FAISS index and documents
    index = faiss.read_index("faiss_index")
    with open(path, "rb") as f:
        documents = pickle.load(f)

    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Query
    query_embedding = embedder.encode([query])

    # Retrieve top-k relevant chunks (n_chunks)
    distances, indices = index.search(query_embedding, n_chunks)

    retrieved_chunks = [documents[idx] for idx in indices[0]]

    return retrieved_chunks


def get_response(query, top_chunks, topic):
    # Initialize the ChatOllama or OllamaLLM model
    llm = ChatOllama(
        model="llama3",
        temperature=0,
    )

    # Combine the context from retrieved chunks
    context = " ".join(top_chunks)

    # Create the prompt with context and query
    prompt = f"You are a expert researcher in {topic}. Consider the following context retrieved from relevant papers about the topic and the given query to generate the most accurate answer you can. Don't produce answers you are not sure. Context: {context}\n\nQuery: {query}\n\nAnswer:"

    # Use the LLM to generate a response
    response = llm.invoke(input=prompt)

    return response.content, context
