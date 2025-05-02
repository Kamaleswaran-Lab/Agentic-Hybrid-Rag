import pickle
import pandas as pd
from tqdm import tqdm
import os
import re
import json
import ast
import numpy as np
import nltk
from scipy import stats

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jGraph
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.prompts import ChatPromptTemplate
from Functions.perform_rag import agent_rag, baseline_RAG
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from nltk.tokenize import sent_tokenize
from Functions.functions import cypher_search, similarity_search
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextPrecision
from collections import defaultdict


nltk.download('punkt', quiet=True)


def get_questions_similarity_tool():

    with open("Database/text_splits.pkl", "rb") as f:
        documents = pickle.load(f)

    context = [
        {
            "source": doc.metadata.get('source', 'unknown'),
            "content": doc.page_content
        }
        for doc in documents
    ]

    testset = pd.DataFrame({"Reference Chunk": context})

    if len(testset) >= 20:
        testset = testset.sample(20).reset_index(drop=True)

    llm = OllamaLLM(model="llama3")

    questions = []
    ground_truths = []
    for _, row in tqdm(testset.iterrows(), total=len(testset), desc="Generating Similarity Retrieval Questions"):

        c = row["Reference Chunk"]["content"]

        prompt = f"""Generate a question and its corresponding answer based strictly on the content inside the <context> XML tags. 

                Instructions for the question: 
                    - Create a question that can only be answered using the provided context.
                    - Avoid questions that refer to specific items such as surveys, figures, tables, sections, citations, or other document markers.
                    - Focus on the main ideas and content, ensuring the question is specific to what is described.

                Instructions for the answer:
                    - Provide a concise, complete, and detailed answer, between 1 and 3 lines, based solely on the context. Do not add extra information.

                <context> 
                {c}
                </context>

                No pre-amble. Do not deviate from the specified format.

                Question:
                Answer:
                """

        response = llm.invoke(prompt)

        try:
            question = re.search(r"Question:\s*(.*)", response).group(1)
            truth = re.search(r"Answer:\s*(.*)", response).group(1)

        except:
            question = np.nan
            truth = np.nan

        questions.append(question)
        ground_truths.append(truth)

    testset["Ground Truth"] = ground_truths

    testset["Question"] = questions

    testset["Tool"] = "similarity_search"

    return testset


def get_questions_cypher_tool():

    # Connection credentials for the Neo4j database
    uri = "neo4j+s://91f991ec.databases.neo4j.io"
    username = "neo4j"
    password = "COeHGYRiC2H4YzRFer_o11lHQDEsuBBfr8Ules7G1PQ"

    # Connect to the Neo4j graph database
    graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password
    )

    questions = []
    answers = []

    # Entity centered questions
    """Questions about an entity (the subject) with no specific relation constraints
    (the predicate) and target entity (the object). The task is to answer general
    question about the entity. Example: “Who is Donald Trump?”
    """

    # 1. What's the abstract of the paper <paper.title>?
    query = """
    MATCH (p:paper)
    WITH p ORDER BY rand()
    LIMIT 1
    RETURN p.paper AS title, p.abstract AS abstract
    """

    result = graph.query(query)

    questions.append(f"What's the abstract of the paper {result[0]['title']}?")
    answers.append(result[0]["abstract"])

    # 2. What's the doi of the paper <paper.title>?
    query = """
        MATCH (p:paper)
        WITH p ORDER BY rand()
        LIMIT 1
        RETURN p.paper AS title, p.doi AS doi
        """

    result = graph.query(query)

    questions.append(f"What's the DOI of the paper {result[0]['title']}?")
    answers.append(result[0]["doi"])

    # 3. How many keywords were used?
    query = """
            MATCH (k:keyword)
            RETURN count(k) AS total_keywords
            """

    result = graph.query(query)

    questions.append("How many keywords were used?")
    answers.append(result[0]["total_keywords"])

    # 4. Which are the databases available?
    query = """
            MATCH (d:database)
            RETURN DISTINCT d.database AS database_name
            ORDER BY database_name
            """

    result = graph.query(query)

    questions.append("Which are the databases available?")
    answers.append([db['database_name'] for db in result])

    # Relationship focused questions
    """Questions about an entity (the subject) with specific relation constraints (the
    predicate) but misses the target entity (the object). The task is to answer
    one specific aspect of the entity. Example: “Which college did the current
    president of United States graduate from?”
    """

    # 6. Which year was the paper titled <Paper> published in?
    query = """
            MATCH (p:paper)-[:published_in]->(y:year)
            WITH p, y, rand() AS r
            ORDER BY r
            LIMIT 1
            RETURN p.paper AS title, y.year AS publication_year
            """

    result = graph.query(query)

    questions.append(f"Which year was the paper titled {result[0]['title']} published in?")
    answers.append(result[0]["publication_year"])

    # 7. Which database indexed the paper <Paper>?
    query = """
            MATCH (p:paper)-[:indexed_at]->(d:database)
            WITH p, d, rand() AS r
            ORDER BY r
            LIMIT 1
            RETURN p.paper AS title, d.database AS database
            """

    result = graph.query(query)

    questions.append(f"Which database indexed the paper {result[0]['title']}?")
    answers.append(result[0]["database"])

    # 8. Who are the authors of the research paper <Paper>?
    query = """
            MATCH (p:paper)
            WITH p
            ORDER BY rand()
            LIMIT 1
            MATCH (p)-[:authored_by]->(a:author)
            RETURN p.paper AS title, collect(DISTINCT a.author) AS authors
            """

    result = graph.query(query)

    questions.append(f"Who are the authors of the research paper {result[0]['title']}?")
    answers.append(result[0]["authors"])

    # 9. What keywords are associated with the paper <Paper>?
    query = """
            MATCH (p:paper)
            WITH p
            ORDER BY rand()
            LIMIT 1
            MATCH (p)-[:has_keyword]->(k:keyword)
            RETURN p.paper AS title, collect(DISTINCT k.keyword) AS keywords
            """

    result = graph.query(query)

    questions.append(f"What keywords are associated with the paper {result[0]['title']}?")
    answers.append(result[0]["keywords"])

    # Relation discovery questions
    """Questions about any relations (the predicate) between two entities (the
    subject and object). The task is to provide the relations between two entities.
    Example: “What is the relationship between Donald Trump and Joe Biden?”
    """

    # 11. What is the relationship between the author <Author> and the paper <Paper>?
    query = """
            MATCH (p:paper)-[r]-(a:author)
            WITH p, a, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.paper AS paper, a.author AS author, relationship
            """

    result = graph.query(query)

    questions.append(f"What is the relationship between the author {result[0]['author']} and the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # 12. How is the paper <Paper> linked to the year <Year>?
    query = """
            MATCH (p:paper)-[r]-(y:year)
            WITH p, y, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.paper AS paper, y.year AS year, relationship
            """

    result = graph.query(query)

    questions.append(f"How is the paper {result[0]['paper']} linked to the year {result[0]['year']}?")
    answers.append(result[0]["relationship"])

    # 13. What connection exists between the keyword <Keyword> and the paper <Paper>?
    query = """
            MATCH (p:paper)-[r]-(k:keyword)
            WITH p, k, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.paper AS paper, k.keyword AS keyword, relationship
            """

    result = graph.query(query)

    questions.append(f"What connection exists between the keyword {result[0]['keyword']} and the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # 14. In what way is the database <Database> related to the paper <Paper>?
    query = """
            MATCH (p:paper)-[r]-(d:database)
            WITH p, d, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.paper AS paper, d.database AS database, relationship
            """

    result = graph.query(query)

    questions.append(f"In what way is the database {result[0]['database']} related to the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # Fact-check questions
    """Questions about specific relations (the predicate) between two entities (the
    subject and object). The task is to check the existence of a specific relationship between the two entities. Example: “Have Donald Trump and Joe
    Biden ever run for the same presidential term, and if so, when is that?”
    """

    # 16. Was the paper <Paper> authored by <Author>?
    query = """
            MATCH (p:paper), (a:author)
            WITH p, a
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:authored_by]->(a)
            OPTIONAL MATCH (p)-[:authored_by]->(actualAuthor:author)
            RETURN 
            p.paper AS paper, 
            a.author AS author, 
            CASE WHEN r IS NULL THEN false ELSE true END AS wasAuthored,
            collect(actualAuthor.author) AS actualAuthors
            """

    result = graph.query(query)

    questions.append(f"Was the paper {result[0]['paper']} authored by {result[0]['author']}")

    answers.append(f"{result[0]['wasAuthored']}. Authors that published that paper are: {result[0]['actualAuthors']}")

    # 17. Is the keyword <Keyword> used to describe the paper <Paper>?
    query = """
            MATCH (p:paper), (k:keyword)
            WITH p, k
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:has_keyword]->(k)
            OPTIONAL MATCH (p)-[:has_keyword]->(actualKeywords:keyword)
            RETURN 
            p.paper AS paper, 
            k.keyword AS keyword, 
            CASE WHEN r IS NULL THEN false ELSE true END AS hasKeyword,
            collect(actualKeywords.keyword) AS keywordsUsed
            """

    result = graph.query(query)

    questions.append(f"Is the keyword {result[0]['keyword']} used to describe the paper {result[0]['paper']}?")
    answers.append(f"{result[0]['hasKeyword']}. The following keywords are used to describe that paper: {result[0]['keywordsUsed']}")

    # 18. Was the paper <Paper> published in the year <Year>?
    query = """
            MATCH (p:paper), (y:year)
            WITH p, y
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:published_in]->(k)
            OPTIONAL MATCH (p)-[:published_in]->(actualYear:year)
            RETURN 
            p.paper AS paper, 
            y.year AS year, 
            CASE WHEN r IS NULL THEN false ELSE true END AS published,
            collect(actualYear.year) AS rightYear
            """

    result = graph.query(query)

    questions.append(f"Was the paper {result[0]['paper']} published in the year {result[0]['year']}?")
    answers.append(f"{result[0]['published']}. The paper was published in {result[0]['rightYear']}")

    # 19. Is the paper <Paper> indexed in the database <Database>?
    query = """
            MATCH (p:paper), (d:database)
            WITH p, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:indexed_at]->(d)
            OPTIONAL MATCH (p)-[:indexed_at]->(actualDatabase:database)
            RETURN 
            p.paper AS paper, 
            d.database AS database, 
            CASE WHEN r IS NULL THEN false ELSE true END AS indexed,
            collect(actualDatabase.database) AS rightDatabase
            """

    result = graph.query(query)

    questions.append(f"Is the paper {result[0]['paper']} indexed in the database {result[0]['database']}?")
    answers.append(f"{result[0]['indexed']}. The paper was indexed at {result[0]['rightDatabase']}")

    # Indirect relationship questions (added)
    """
    These questions involve connecting two entities that are not directly related in the schema but are 
    linked through an intermediate node, typically a Paper.
    """

    # 17. Did the author <Author> write any paper that contains the keyword <Keyword>?
    query = """
            MATCH (a:author)<-[:authored_by]-(p:paper)-[:has_keyword]->(k:keyword)
            WITH a, k, COLLECT(p) AS papers
            WHERE SIZE(papers) > 0
            WITH a, k, papers
            ORDER BY rand()
            LIMIT 1
            RETURN 
            a.author AS author,
            k.keyword AS keyword,
            [paper IN papers | paper.paper] AS papers,
            true AS wroteWithKeyword
            """

    result = graph.query(query)

    questions.append(f"Did the author {result[0]['author']} write any paper that contains the keyword {result[0]['keyword']}?")
    answers.append(f"{result[0]['wroteWithKeyword']}. The papers from the author {result[0]['author']} that contain the keyword {result[0]['keyword']} are: {result[0]['papers']}")

    # 18. Was the keyword <Keyword> associated with any paper published in the year <Year>?
    query = """
            MATCH (y:year), (k:keyword)
            WITH y, k
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p:paper)-[:published_in]->(y)
            WITH y, k, COLLECT(p) AS papersInYear
            UNWIND papersInYear AS paper
            OPTIONAL MATCH (paper)-[:has_keyword]->(k)
            WITH y.year AS year, k.keyword AS keyword, paper
            WHERE paper IS NOT NULL
            WITH year, keyword, COLLECT(DISTINCT paper.paper) AS papersWithKeyword
            RETURN 
            year,
            keyword,
            papersWithKeyword,
            CASE WHEN SIZE(papersWithKeyword) > 0 THEN true ELSE false END AS isAssociated
            """

    result = graph.query(query)

    questions.append(f"Was the keyword {result[0]['keyword']} associated with any paper published in the year {result[0]['year']}?")
    answers.append(f"{result[0]['isAssociated']}. The keyword {result[0]['keyword']} is associated with the following papers published in {result[0]['year']}: {result[0]['papersWithKeyword']}")

    # 19. Has the author <Author> published any paper indexed in the database <Database>?
    query = """
            MATCH (a:author), (d:database)
            WITH a, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (a)<-[:authored_by]-(p:paper)-[:indexed_at]->(d)
            WHERE p IS NOT NULL
            WITH a.author AS author, d.database AS database, COLLECT(p.paper) AS papers
            RETURN 
            author,
            database,
            papers,
            CASE WHEN SIZE(papers) > 0 THEN true ELSE false END AS hasPublishedInDatabase
            """

    result = graph.query(query)

    questions.append(f"Has the author {result[0]['author']} published any paper indexed in the database {result[0]['database']}?")
    answers.append(f"{result[0]['hasPublishedInDatabase']}. The author {result[0]['author']} has following papers indexed at the {result[0]['database']} database: {result[0]['papers']}")

    # 20. Has the keyword <Keyword> been used in any paper published in the database <Database>?
    query = """
            MATCH (k:keyword), (d:database)
            WITH k, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (k)<-[:has_keyword]-(p:paper)-[:indexed_at]->(d)
            WITH 
            k.keyword AS keyword, 
            d.database AS database, 
            [x IN COLLECT(p) WHERE x IS NOT NULL] AS validPapers
            RETURN 
            keyword, 
            database, 
            [p IN validPapers | p.paper] AS papers,
            SIZE(validPapers) > 0 AS isKeywordUsedInDatabase
            """

    result = graph.query(query)

    questions.append(f"Has the keyword {result[0]['keyword']} been used in any paper published in the database {result[0]['database']}?")
    answers.append(f"{result[0]['isKeywordUsedInDatabase']}. The keyword {result[0]['keyword']} was used in the following papers at the {result[0]['database']} database: {result[0]['papers']}")

    testset = pd.DataFrame({"Question": questions, "Ground Truth": answers})

    testset["Tool"] = "cypher_search"

    return testset


def generate_testset(filename="testset", save_results=True):

    # similarity tool
    print("Generating similarity related questions...")
    sim_testset = get_questions_similarity_tool()

    # cypher tool
    print("Generating cypher related questions...")
    cy_testset = get_questions_cypher_tool()

    print("Done!")

    final_testset = pd.concat([sim_testset, cy_testset], ignore_index=True)

    if save_results:
        final_testset.to_json(f"Evaluate/{filename}.json", orient="table", force_ascii=False)

    print(f"{filename} created with success!")

    return final_testset


def extract_statements(answer):
    # Initialize the language model for answer generation
    llm = OllamaLLM(model="mistral")

    prompt = f"""Given an answer within the xml tags, create one or more statements from each sentence in the given answer.

    <answer> 
    {answer}
    </answer>
    
    Return only the statements as a python list format. No pre-amble.
    
    Output examples:
        ['The paper was published in 2024']
        ['The author li X authored the paper', 'There were 3 authors that contributed to the paper']
        ['The main trends in mllm for healthcare include international collaboration', 'Limitations that research topic include the lack of proper training sets', 'Agentic strategies have been constantly adopted to increase different models performance']
    
    Do not deviate from the specified format.
    """

    response = llm.invoke(prompt)

    response = response.strip().replace(r"\n", "")

    try:
        response = ast.literal_eval(response)

    except:
        try:
            matches = re.findall(r"\['(.*?)'\]", response)
            if matches:
                response = [s.strip() for s in matches[0].split("', '")]

        except:
            response = []

    print(f"statements: {response}")
    return response


def verify_statements(context, statements):
    verdicts = {}

    llm = OllamaLLM(model="llama3")

    for statement in statements:

        prompt = f"""Your task is to judge the faithfulness of a series of statements based on a given context. 
        For each statement you must return verdict as 1 if the statement can be directly inferred based on the context 
        or 0 if the statement can not be directly inferred based on the context.

        <context> 
        {context}
        </context>

        <statement> 
        {statement}
        </statement>
        
        Only output the number 0 or 1. Do not include explanations or any other text. Do not deviate from the specified format.
        """

        response = llm.invoke(prompt)

        try:
            response = int(response)

        except:
            response = None

        verdicts[statement] = response

    return verdicts


def compute_faithfulness_score(question, answer, context):
    print(f"question: {question}")
    print(f"answer: {answer}")

    statements = extract_statements(answer)

    if isinstance(statements, str):
        try:
            statements = ast.literal_eval(statements)
        except:
            return np.nan

    if len(statements) == 0:
        return np.nan

    print(f"statements: {statements}")

    verification_output = verify_statements(context, statements)

    print(f"verification output: {verification_output}")

    supported_count = sum(1 for v in verification_output.values() if v == 1)
    total = len(verification_output.values())

    faithfulness_score = supported_count / total if total > 0 else 0

    return faithfulness_score


def generate_questions(answer, n=3):
    llm = OllamaLLM(model="mistral")
    prompt = f"""Generate {n} different questions for the answer provided within xml tags.
    
    <answer> 
    {answer}
    </answer>
    
    Return only the questions as a python list format. No pre-amble.
    
    Examples:
        if 1 question:
            output: ['What are the main trends in mllm for healtcare?']
        if 2 questions:
            output: ['What are the current research limitations?', 'What is being done to deal with the existing research limitations?']
        if 3 questions:
            output: ['How many papers are in the database?', 'How many were published in 2024?', 'What was the most frequent keyword?']
        
        
    Do not deviate from the specified format.
    """

    response = llm.invoke(prompt)
    response = response.strip().replace(r"\n", "")

    try:
        response = ast.literal_eval(response)

    except:
        try:
            matches = re.findall(r"\['(.*?)'\]", response)
            if matches:
                response = [s.strip() for s in matches[0].split("', '")]

        except:
            response = []

    return response


def get_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    embedding = OllamaEmbeddings(model="llama3")
    return embedding.embed_documents(texts)


def is_noncommittal(answer):
    phrases = [
        "I don't know", "I'm not sure", "I cannot",
        "unsure", "unknown", "unaware", "it depends",
        "I could't find"
    ]
    return any(p.lower() in answer.lower() for p in phrases)


def answer_relevance(original_question, answer, n=3):
    if is_noncommittal(answer):
        return 0.0  # Fully penalize noncommittal answers

    generated_questions = generate_questions(answer, n=n)
    print(f"Generated questions: {generated_questions}")

    if not generated_questions:
        return np.nan

    q_embedding = get_embedding(original_question)[0]
    generated_embeddings = get_embedding(generated_questions)

    if not q_embedding or not generated_embeddings:
        return np.nan

    similarities = [
        cosine_similarity([q_embedding], [gen_emb])[0][0]
        for gen_emb in generated_embeddings
    ]

    return np.mean(similarities)


def judge_relevance(question, answer, context):
    """
    Use a language model to judge the relevance of a context to the query.
    Returns 1 for relevant and 0 for not relevant.
    """
    llm = OllamaLLM(model="llama3")

    prompt = f"""Given question, a answer and a context within xml tags, verify if the context was useful in arriving at the given answer.
             Give verdict as "1" if useful and "0" if not. Think carefully before providing the answer. 
             <question>
             {question}
             </question>
             
             <answer>
             {answer}
             </answer>
             
             <context> 
             {context}
             </context>
             Only output the number 0 or 1. Do not include explanations or any other text. Do not deviate from the specified format.
             """

    response = llm.invoke(prompt)

    try:
        response = int(response)
    except:
        response = None

    return response


def compute_precision_at_k(retrieved_contexts, query, answer, k=3):
    """
    Compute the Context Precision at K for a given query and retrieved contexts.

    :param retrieved_contexts: List of context chunks (strings).
    :param query: The query string.
    :param answer: The generated answer
    :param k: The number of top contexts to evaluate.
    :return: Context Precision at K
    """
    # Limit the retrieved contexts to top K
    selected_contexts = retrieved_contexts[:k]

    print(f"[compute_precision_at_k] Top-{k} retrieved contexts: {selected_contexts}")

    # Judge the relevance of each context using the LLM
    relevance_scores = [judge_relevance(query, answer, context) for context in selected_contexts]

    print(f"[compute_precision_at_k] Relevance scores: {relevance_scores}")

    # Total number of relevant items in the top K
    total_relevant = sum(relevance_scores)

    if total_relevant == 0:
        return 0.0  # No relevant items found

    # Calculate Context Precision at K using the rank-weighted formula
    cumulative_precision = 0.0
    relevant_found = 0

    for rank, relevance in enumerate(relevance_scores, start=1):
        if relevance == 1:
            relevant_found += 1
            precision_at_rank = relevant_found / rank
            cumulative_precision += precision_at_rank

    context_precision_at_k = cumulative_precision / total_relevant

    return context_precision_at_k


def context_recall_classifier(truth, context):

    sentences = re.split(r'(?<=[.!?;])\s+', truth.strip())

    classifications = []

    llm = OllamaLLM(model="llama3")

    for sentence in sentences:
        prompt = f""""Given a context and an sentence within the xml tags, analyze the sentence and classify if it can be attributed to the given context or not. 
                Give verdict as "1" if yes and "0" if no. Think carefully before providing the answer. 
                 
                <context> 
                {context}
                </context>
                <sentence>
                {sentence}
                </sentence>
                 
                Only output the number 0 or 1. Do not include explanations or any other text. Do not deviate from the specified format.
                """

        response = llm.invoke(prompt)

        try:
            response = int(response)

        except:
            response = np.nan

        classifications.append(response)

    return classifications


def context_recall(truth, context):

    classifications = context_recall_classifier(truth, context)

    valid = [c for c in classifications if c in [0, 1]]
    if not valid:
        return np.nan
    return sum(valid) / len(valid)


# Confidence intervals calculation
def calculate_confidence_interval(data, confidence=0.95):
    # Calculate the confidence interval for the mean using the t-distribution
    data = np.array(data)
    data = data[~np.isnan(data)]

    n = len(data)

    if n == 0:
        return np.nan, np.nan

    mean = np.mean(data)
    stderr = stats.sem(data)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, margin_of_error


def evaluate_agent_with_bootstrap(testset, agent, num_iterations=15, confidence_level=0.95, file_name="agentic_model"):
    print("Starting evaluation...")

    # Set up vectors for metrics
    s_recalls = []
    s_faithfulnesses = []
    s_precisions = []
    s_relevances = []
    c_recalls = []
    c_faithfulnesses = []
    c_precisions = []
    c_relevances = []
    sim_accuracy = []
    cy_accuracy = []

    # Bootstrap loop for multiple iterations
    for z in range(num_iterations):
        print(f"Iteration: {z+1}")

        sample = testset.groupby("Tool", group_keys=False).apply(
            lambda x: x.sample(5)
        ).reset_index(drop=True)

        print(f"Sample:")
        print(sample)

        # Loop through the test set
        for i, (question, truth, tool) in tqdm(enumerate(zip(sample["Question"], sample["Ground Truth"], sample["Tool"])),
                                                total=len(sample), desc=f"Evaluating questions (Iteration {z+1})"):

            # safely move on to next iteration if internal error happens
            try:

                if tool == "similarity_search":
                    print("Similarity task")
                    result = agent.run(user_msg=question)
                    final_answer = result["final_response"]

                    # Evaluate tool used by agent
                    if result["tool_call"] is not None:
                        tool_used = json.loads(result["tool_call"])
                        if tool_used["name"] == tool:
                            sim_accuracy.append(1)
                        else:
                            sim_accuracy.append(0)

                    # Vector search evaluation
                    #response = similarity_search(question)
                    #retrieved_context = json.loads(response)
                    #context = [retrieved_context["Answer"]]

                    context = result["context"]

                    print(f"Context: {context}")

                    # Metrics calculations for similarity search
                    faithfulness = compute_faithfulness_score(question, final_answer, context)
                    s_faithfulnesses.append(faithfulness)
                    relevance = answer_relevance(question, final_answer, n=2)
                    s_relevances.append(relevance)
                    precision = compute_precision_at_k(context, question, final_answer, k=10)
                    s_precisions.append(precision)
                    recall = context_recall(truth, context)
                    s_recalls.append(recall)

                    print(f"Sim Accuracy: {sim_accuracy}")
                    print(f"Sim F: {faithfulness}")
                    print(f"Sim AR: {relevance}")
                    print(f"Sim P: {precision}")
                    print(f"Sim CR: {recall}")

                # Cypher task evaluation
                else:
                    print("Cypher task")
                    result = agent.run(user_msg=question)
                    final_answer = result["final_response"]

                    # Evaluate tool used by agent
                    if result["tool_call"] is not None:
                        tool_used = json.loads(result["tool_call"])
                        if tool_used["name"] == tool:
                            cy_accuracy.append(1)
                        else:
                            cy_accuracy.append(0)

                    # Cypher search evaluation
                    #response = cypher_search(question)
                    #retrieved_context = json.loads(response)
                    #context = [retrieved_context["Answer"]]

                    context = result["context"]
                    print(f"Context: {context}")

                    # Metrics calculations for Cypher search
                    faithfulness = compute_faithfulness_score(question, final_answer, context)
                    c_faithfulnesses.append(faithfulness)
                    relevance = answer_relevance(question, final_answer, n=2)
                    c_relevances.append(relevance)
                    precision = compute_precision_at_k(context, question, final_answer, k=10)
                    c_precisions.append(precision)
                    recall = context_recall(truth, context)
                    c_recalls.append(recall)

                    print(f"Cy Accuracy: {cy_accuracy}")
                    print(f"Cy F: {faithfulness}")
                    print(f"Cy AR: {relevance}")
                    print(f"Cy P: {precision}")
                    print(f"Cy CR: {recall}")

                # After each iteration, calculate averages
                print(f"Iteration completed. Collecting metrics...")

            except Exception as e:
                print(f"Iteration failed due to: {e}")
                pass

    # Final aggregation of results after all iterations
    print("Calculating final aggregated results...")

    print(s_faithfulnesses)
    print(s_recalls)
    print(s_precisions)
    print(s_relevances)

    print(c_faithfulnesses)
    print(c_recalls)
    print(c_precisions)
    print(c_relevances)

    # Calculate confidence intervals for each metric
    sf_mean, sf_margin = calculate_confidence_interval(s_faithfulnesses, confidence_level)
    sar_mean, sar_margin = calculate_confidence_interval(s_relevances, confidence_level)
    sp_mean, sp_margin = calculate_confidence_interval(s_precisions, confidence_level)
    sr_mean, sr_margin = calculate_confidence_interval(s_recalls, confidence_level)

    cf_mean, cf_margin = calculate_confidence_interval(c_faithfulnesses, confidence_level)
    car_mean, car_margin = calculate_confidence_interval(c_relevances, confidence_level)
    cp_mean, cp_margin = calculate_confidence_interval(c_precisions, confidence_level)
    cr_mean, cr_margin = calculate_confidence_interval(c_recalls, confidence_level)

    of_mean, of_margin = calculate_confidence_interval(s_faithfulnesses + c_faithfulnesses, confidence_level)
    oar_mean, oar_margin = calculate_confidence_interval(s_relevances + c_relevances, confidence_level)
    op_mean, op_margin = calculate_confidence_interval(s_precisions + c_precisions, confidence_level)
    or_mean, or_margin = calculate_confidence_interval(s_recalls + c_recalls, confidence_level)

    # Calculate confidence intervals for accuracy metrics
    cy_final_mean, cy_final_margin = calculate_confidence_interval(cy_accuracy, confidence_level)
    sim_final_mean, sim_final_margin = calculate_confidence_interval(sim_accuracy, confidence_level)
    agent_final_mean, agent_final_margin = calculate_confidence_interval(sim_accuracy + cy_accuracy, confidence_level)

    # Results for accuracy metrics
    metrics = [
        "Overall Agent Accuracy", "Agent Accuracy on Similarity Tasks", "Agent Accuracy on Cypher Tasks",
        "Similarity Faithfulness", "Cypher Faithfulness", "Overall Faithfulness",
        "Similarity Answer Relevance", "Cypher Answer Relevance", "Overall Answer Relevance",
        "Similarity Precision", "Cypher Precision", "Overall Precision",
        "Similarity Context Recall", "Cypher Context Recall", "Overall Context Recall"
    ]

    results = [
        agent_final_mean, sim_final_mean, cy_final_mean, sf_mean, cf_mean, of_mean, sar_mean, car_mean,
        oar_mean, sp_mean, cp_mean, op_mean, sr_mean, cr_mean, or_mean,
    ]

    margins = [
        agent_final_margin, sim_final_margin, cy_final_margin, sf_margin, cf_margin, of_margin, sar_margin, car_margin,
        oar_margin, sp_margin, cp_margin, op_margin, sr_margin, cr_margin, or_margin
    ]

    final_results = pd.DataFrame({
        "Metric": metrics,
        "Result": results,
        "Confidence Interval": margins
    })

    final_results.to_excel(f"Evaluate/evaluation_results_with_ci_{file_name}.xlsx", index=False)

    print("Finished evaluating pipeline with Bootstrap and Confidence Intervals! Check results in Evaluate folder.")

    return


def evaluate_baselineRAG_with_bootstrap(testset, num_iterations=12, confidence_level=0.95, file_name="baseline_model"):

    print("Starting evaluation...")

    # Set up vectors for metrics
    s_recalls = []
    s_faithfulnesses = []
    s_precisions = []
    s_relevances = []
    c_recalls = []
    c_faithfulnesses = []
    c_precisions = []
    c_relevances = []

    # Bootstrap loop for multiple iterations
    for z in range(num_iterations):
        print(f"Iteration: {z + 1}")

        sample = testset.groupby("Tool", group_keys=False).apply(
            lambda x: x.sample(5)
        ).reset_index(drop=True)

        print(f"Sample:")
        print(sample)

        # Loop through the test set
        for i, (question, truth, tool) in tqdm(
                enumerate(zip(sample["Question"], sample["Ground Truth"], sample["Tool"])),
                total=len(sample), desc=f"Evaluating questions (Iteration {z+1})"):

            # safely move on to next iteration if internal error happens
            try:

                if tool == "similarity_search":
                    print("Similarity task")

                    context, final_answer = baseline_RAG(question)

                    print(f"Context: {context}")

                    # Metrics calculations for similarity search
                    faithfulness = compute_faithfulness_score(question, final_answer, context)
                    s_faithfulnesses.append(faithfulness)
                    relevance = answer_relevance(question, final_answer, n=2)
                    s_relevances.append(relevance)
                    precision = compute_precision_at_k(context, question, final_answer, k=10)
                    s_precisions.append(precision)
                    recall = context_recall(truth, context)
                    s_recalls.append(recall)

                    print(f"Sim F: {faithfulness}")
                    print(f"Sim AR: {relevance}")
                    print(f"Sim P: {precision}")
                    print(f"Sim CR: {recall}")

                # Cypher task evaluation
                else:
                    print("Cypher task")

                    context, final_answer = baseline_RAG(question)

                    print(f"Context: {context}")

                    # Metrics calculations for Cypher search
                    faithfulness = compute_faithfulness_score(question, final_answer, context)
                    c_faithfulnesses.append(faithfulness)
                    relevance = answer_relevance(question, final_answer, n=2)
                    c_relevances.append(relevance)
                    precision = compute_precision_at_k(context, question, final_answer, k=10)
                    c_precisions.append(precision)
                    recall = context_recall(truth, context)
                    c_recalls.append(recall)

                    print(f"Cy F: {faithfulness}")
                    print(f"Cy AR: {relevance}")
                    print(f"Cy P: {precision}")
                    print(f"Cy CR: {recall}")

                # After each iteration, calculate averages
                print(f"Iteration completed. Collecting metrics...")

            except Exception as e:
                print(f"Iteration failed due to: {e}")
                pass

    # Final aggregation of results after all iterations
    print("Calculating final aggregated results...")

    # Calculate confidence intervals for each metric
    sf_mean, sf_margin = calculate_confidence_interval(s_faithfulnesses, confidence_level)
    sar_mean, sar_margin = calculate_confidence_interval(s_relevances, confidence_level)
    sp_mean, sp_margin = calculate_confidence_interval(s_precisions, confidence_level)
    sr_mean, sr_margin = calculate_confidence_interval(s_recalls, confidence_level)

    cf_mean, cf_margin = calculate_confidence_interval(c_faithfulnesses, confidence_level)
    car_mean, car_margin = calculate_confidence_interval(c_relevances, confidence_level)
    cp_mean, cp_margin = calculate_confidence_interval(c_precisions, confidence_level)
    cr_mean, cr_margin = calculate_confidence_interval(c_recalls, confidence_level)

    of_mean, of_margin = calculate_confidence_interval(s_faithfulnesses + c_faithfulnesses, confidence_level)
    oar_mean, oar_margin = calculate_confidence_interval(s_relevances + c_relevances, confidence_level)
    op_mean, op_margin = calculate_confidence_interval(s_precisions + c_precisions, confidence_level)
    or_mean, or_margin = calculate_confidence_interval(s_recalls + c_recalls, confidence_level)

    print(s_faithfulnesses)
    print(c_faithfulnesses)

    print(sf_mean, sf_margin)
    print(cf_mean, cf_margin)
    print(of_mean, of_margin)

    # Results for accuracy metrics
    metrics = [
        "Similarity Faithfulness", "Cypher Faithfulness", "Overall Faithfulness",
        "Similarity Answer Relevance", "Cypher Answer Relevance", "Overall Answer Relevance",
        "Similarity Precision", "Cypher Precision", "Overall Precision",
        "Similarity Context Recall", "Cypher Context Recall", "Overall Context Recall"
    ]

    results = [
        sf_mean, cf_mean, of_mean, sar_mean, car_mean,
        oar_mean, sp_mean, cp_mean, op_mean, sr_mean, cr_mean, or_mean,
    ]

    margins = [
        sf_margin, cf_margin, of_margin, sar_margin,
        car_margin,
        oar_margin, sp_margin, cp_margin, op_margin, sr_margin, cr_margin, or_margin
    ]

    final_results = pd.DataFrame({
        "Metric": metrics,
        "Result": results,
        "Confidence Interval": margins
    })

    final_results.to_excel(f"Evaluate/evaluation_results_with_ci_{file_name}.xlsx", index=False)

    print("Finished evaluating pipeline with Bootstrap and Confidence Intervals! Check results in Evaluate folder.")

    return


def generate_dpo_data(data_points=6):

    dataset = generate_testset("dpo_data", save_results=False)

    questions = []
    ground_truth = []
    correct_response = []
    incorrect_response = []

    cypher = dataset.loc[dataset["Tool"] == "cypher_search", :]
    sim = dataset.loc[dataset["Tool"] == "similarity_search", :]

    if len(cypher) >= data_points and len(sim) >= data_points:
        sample_cy = cypher.sample(data_points)
        sample_sim = sim.sample(data_points)

    else:
        print("Data available does not have that many amount of samples")
        return

    questions += sample_cy["Question"].to_list()
    questions += sample_sim["Question"].to_list()
    ground_truth += sample_cy["Answer"].to_list()

    llm = OllamaLLM(model="mistral")

    for question, context in zip(sample_sim["Question"], sample_sim["Answer"]):
        prompt_ground_truth = f"""Based on the question and context given within the xml tags, generate a concise answer.
        Do not create new information. Base your answer only on the given context.
        
        <question>
        {question}
        </question>
        
        <context>
        {context}
        </context>
        
        Do not deviate from the specified format.
        
        Answer:
        """

        ground_truth.append(llm.invoke(prompt_ground_truth))

        print(f"Question: {question}")
        print(f"Truth: {ground_truth[-1]}")

    for gt, q in zip(ground_truth, questions):
        prompt_correct = f"""Based on the ground truth to the question provided within the xml tags, generate a response that is slightly different from the ground truth.
        Be concise and do not add extra information. The response must be really faithfull to the ground truth.
        
        <question>
        {q}
        </question>
        
        <ground truth>
        {gt}
        </ground truth>
        
        Do not deviate from the specified format.
        
        response:
        """

        correct_response.append(llm.invoke(prompt_correct))
        print(f"Question: {q}")
        print(f"Correct: {correct_response[-1]}")

        prompt_incorrect = f"""Based on the ground truth and question provided within the xml tags, generate a response that ignores the ground truth partially or completely.
                You may generate information that is not contained in the ground truth or even oppose it. Be concise
                
                </question>
                {q}
                <question>
                
                <ground truth>
                {gt}
                </ground truth>
                
                Do not deviate from the specified format.

                response:
                """

        incorrect_response.append(llm.invoke(prompt_incorrect))

        print(f"Incorrect: {incorrect_response[-1]}")

    dpo_data = pd.DataFrame({"question": questions, "ground_truth": ground_truth, "correct_response": correct_response, "wrong_response": incorrect_response})

    dpo_data.to_json(f"Evaluate/dpo_data.json", orient="table", force_ascii=False)

    print("dpo_data created with success!")

    return dpo_data
