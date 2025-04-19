import pickle
import pandas as pd
from tqdm import tqdm
import os
import re
import json
import ast
import numpy as np
import nltk

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jGraph
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.prompts import ChatPromptTemplate
from Functions.perform_rag import agent_rag
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from nltk.tokenize import sent_tokenize
from Functions.functions import cypher_search


nltk.download('punkt', quiet=True)


def get_questions_similarity_tool():

    with open("Database/text_splits.pkl", "rb") as f:
        documents = pickle.load(f)

    testset = pd.DataFrame({"Answer": documents})

    if len(testset) >= 20:
        testset = testset.sample(20).reset_index(drop=True)

    llm = OllamaLLM(model="llama3")

    questions = []
    for _, row in tqdm(testset.iterrows(), total=len(testset), desc="Generating Similarity Retrieval Questions"):
        prompt = f"Document: {row['Answer']} \n Generate a question that can only be answered from this document. Don't create generic questions. Don't mention specific figures, tables, sections or even the actual document in the question. Focus only in its the content and main ideas. The output should just be the question. No pre-amble. \n Output examples:  'What is the main purpose of the partnership between OpenAI and Guardian Media Group announced on February 14, 2025? \n What is the purpose of OpenAI's agreement with the U.S. National Laboratories as described in the document? \n What was the specific AI model used during the '1,000 Scientist AI Jam Session' event across the nine national labs?"
        questions.append(llm.invoke(prompt))

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
    MATCH (p:Paper)
    WITH p ORDER BY rand()
    LIMIT 1
    RETURN p.Paper AS title, p.Abstract AS abstract
    """

    result = graph.query(query)

    questions.append(f"What's the abstract of the paper {result[0]['title']}?")
    answers.append(result[0]["abstract"])

    # 2. What's the doi of the paper <paper.title>?
    query = """
        MATCH (p:Paper)
        WITH p ORDER BY rand()
        LIMIT 1
        RETURN p.Paper AS title, p.DOI AS doi
        """

    result = graph.query(query)

    questions.append(f"What's the DOI of the paper {result[0]['title']}?")
    answers.append(result[0]["doi"])

    # 3. How many keywords were used?
    query = """
            MATCH (k:Keyword)
            RETURN count(k) AS total_keywords
            """

    result = graph.query(query)

    questions.append("How many keywords were used?")
    answers.append(result[0]["total_keywords"])

    # 4. Which are the databases available?
    query = """
            MATCH (d:Database)
            RETURN DISTINCT d.Database AS database_name
            ORDER BY database_name
            """

    result = graph.query(query)

    questions.append("Which are the databases available?")
    answers.append([db['database_name'] for db in result])

    # 5. Which are the years available? (commented to keep the proportion between the types of questions)
    #query = """
    #            MATCH (y:Year)
    #            RETURN DISTINCT y.Year AS year
    #            ORDER BY year
    #            """

    #result = graph.query(query)

    #questions.append("Which are the years available?")
    #answers.append([y['year'] for y in result])

    # Relationship focused questions
    """Questions about an entity (the subject) with specific relation constraints (the
    predicate) but misses the target entity (the object). The task is to answer
    one specific aspect of the entity. Example: “Which college did the current
    president of United States graduate from?”
    """

    # 6. Which year was the paper titled <Paper> published in?
    query = """
            MATCH (p:Paper)-[:PUBLISHED_IN]->(y:Year)
            WITH p, y, rand() AS r
            ORDER BY r
            LIMIT 1
            RETURN p.Paper AS title, y.Year AS publication_year
            """

    result = graph.query(query)

    questions.append(f"Which year was the paper titled {result[0]['title']} published in?")
    answers.append(result[0]["publication_year"])

    # 7. Which database indexed the paper <Paper>?
    query = """
            MATCH (p:Paper)-[:PUBLISHER]->(d:Database)
            WITH p, d, rand() AS r
            ORDER BY r
            LIMIT 1
            RETURN p.Paper AS title, d.Database AS database
            """

    result = graph.query(query)

    questions.append(f"Which database indexed the paper {result[0]['title']}?")
    answers.append(result[0]["database"])

    # 8. Who are the authors of the research paper <Paper>?
    query = """
            MATCH (p:Paper)
            WITH p
            ORDER BY rand()
            LIMIT 1
            MATCH (p)-[:AUTHORED_BY]->(a:Author)
            RETURN p.Paper AS title, collect(DISTINCT a.Author) AS authors
            """

    result = graph.query(query)

    questions.append(f"Who are the authors of the research paper {result[0]['title']}?")
    answers.append(result[0]["authors"])

    # 9. What keywords are associated with the paper <Paper>?
    query = """
            MATCH (p:Paper)
            WITH p
            ORDER BY rand()
            LIMIT 1
            MATCH (p)-[:KEYWORDS]->(k:Keyword)
            RETURN p.Paper AS title, collect(DISTINCT k.Keyword) AS keywords
            """

    result = graph.query(query)

    questions.append(f"What keywords are associated with the paper {result[0]['title']}?")
    answers.append(result[0]["keywords"])

    # 10. Which papers were written by the author <Author>? (commented to keep the proportion between the types of questions)
    #query = """
    #        MATCH (a:Author)
    #        WITH a
    #        ORDER BY rand()
    #        LIMIT 1
    #        MATCH (a)<-[:AUTHORED_BY]-(p:Paper)
    #        RETURN a.Author AS author, collect(DISTINCT p.Paper) AS papers
    #        """

    #result = graph.query(query)

    #questions.append(f"Which papers were written by the author {result[0]['author']}?")
    #answers.append(result[0]["papers"])

    # Relation discovery questions
    """Questions about any relations (the predicate) between two entities (the
    subject and object). The task is to provide the relations between two entities.
    Example: “What is the relationship between Donald Trump and Joe Biden?”
    """

    # 11. What is the relationship between the author <Author> and the paper <Paper>?
    query = """
            MATCH (p:Paper)-[r]-(a:Author)
            WITH p, a, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.Paper AS paper, a.Author AS author, relationship
            """

    result = graph.query(query)

    questions.append(f"What is the relationship between the author {result[0]['author']} and the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # 12. How is the paper <Paper> linked to the year <Year>?
    query = """
            MATCH (p:Paper)-[r]-(y:Year)
            WITH p, y, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.Paper AS paper, y.Year AS year, relationship
            """

    result = graph.query(query)

    questions.append(f"How is the paper {result[0]['paper']} linked to the year {result[0]['year']}?")
    answers.append(result[0]["relationship"])

    # 13. What connection exists between the keyword <Keyword> and the paper <Paper>?
    query = """
            MATCH (p:Paper)-[r]-(k:Keyword)
            WITH p, k, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.Paper AS paper, k.Keyword AS keyword, relationship
            """

    result = graph.query(query)

    questions.append(f"What connection exists between the keyword {result[0]['keyword']} and the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # 14. In what way is the database <Database> related to the paper <Paper>?
    query = """
            MATCH (p:Paper)-[r]-(d:Database)
            WITH p, d, type(r) AS relationship
            ORDER BY rand()
            LIMIT 1
            RETURN p.Paper AS paper, d.Database AS database, relationship
            """

    result = graph.query(query)

    questions.append(f"In what way is the database {result[0]['database']} related to the paper {result[0]['paper']}?")
    answers.append(result[0]["relationship"])

    # 15. What is the connection between the author <Author> and the keyword <Keyword>? (indirect relationship)
    #query = """
    #        MATCH (a:Author), (k:Keyword)
    #        WITH a, k
    #        ORDER BY rand()
    #        LIMIT 1
    #        RETURN a.Author AS author, k.Keyword AS keyword
    #        """

    #result = graph.query(query)

    #questions.append(f"What is the connection between the author {result[0]['author']} and the keyword {result[0]['keyword']}?")
    #answers.append('No relationship')

    # Fact-check questions
    """Questions about specific relations (the predicate) between two entities (the
    subject and object). The task is to check the existence of a specific relationship between the two entities. Example: “Have Donald Trump and Joe
    Biden ever run for the same presidential term, and if so, when is that?”
    """

    # 16. Was the paper <Paper> authored by <Author>?
    query = """
            MATCH (p:Paper), (a:Author)
            WITH p, a
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:AUTHORED_BY]->(a)
            RETURN 
            p.Paper AS paper, 
            a.Author AS author, 
            CASE WHEN r IS NULL THEN false ELSE true END AS wasAuthored
            """

    result = graph.query(query)

    questions.append(f"Was the paper {result[0]['paper']} authored by {result[0]['author']}")
    answers.append(result[0]['wasAuthored'])

    # 17. Is the keyword <Keyword> used to describe the paper <Paper>?
    query = """
            MATCH (p:Paper), (k:Keyword)
            WITH p, k
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:KEYWORDS]->(k)
            RETURN 
            p.Paper AS paper, 
            k.Keyword AS keyword, 
            CASE WHEN r IS NULL THEN false ELSE true END AS keywordUsed
            """

    result = graph.query(query)

    questions.append(f"Is the keyword {result[0]['keyword']} used to describe the paper {result[0]['paper']}?")
    answers.append(result[0]['keywordUsed'])

    # 18. Was the paper <Paper> published in the year <Year>?
    query = """
            MATCH (p:Paper), (y:Year)
            WITH p, y
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:PUBLISHED_IN]->(y)
            RETURN 
            p.Paper AS paper, 
            y.Year AS year, 
            CASE WHEN r IS NULL THEN false ELSE true END AS publishedInYear
            """

    result = graph.query(query)

    questions.append(f"Was the paper {result[0]['paper']} published in the year {result[0]['year']}?")
    answers.append(result[0]['publishedInYear'])

    # 19. Is the paper <Paper> indexed in the database <Database>?
    query = """
            MATCH (p:Paper), (d:Database)
            WITH p, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p)-[r:PUBLISHER]->(d)
            RETURN 
            p.Paper AS paper, 
            d.Database AS database, 
            CASE WHEN r IS NULL THEN false ELSE true END AS isIndexed
            """

    result = graph.query(query)

    questions.append(f"Is the paper {result[0]['paper']} indexed in the database {result[0]['database']}?")
    answers.append(result[0]['isIndexed'])

    # 20. Did the author <Author> write any paper that contains the keyword <Keyword>? (indirect relationship)
    #query = """
    #        MATCH (a:Author), (k:Keyword)
    #        WITH a, k
    #        ORDER BY rand()
    #        LIMIT 1
    #        OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(p:Paper)-[:KEYWORDS]->(k)
    #        RETURN
    #        a.Author AS author,
    #        k.Keyword AS keyword,
    #        CASE WHEN p IS NULL THEN false ELSE true END AS wroteWithKeyword
    #        """

    #result = graph.query(query)

    #questions.append(f"Did the author {result[0]['author']} write any paper that contains the keyword {result[0]['keyword']}?")
    #answers.append(result[0]['wroteWithKeyword'])

    # Indirect relationship questions (added)
    """
    These questions involve connecting two entities that are not directly related in the schema but are 
    linked through an intermediate node, typically a Paper.
    """

    # 17. Did the author <Author> write any paper that contains the keyword <Keyword>?
    query = """
            MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper)-[:KEYWORDS]->(k:Keyword)
            WITH a, k, COLLECT(p) AS papers
            WHERE SIZE(papers) > 0
            WITH a, k, papers
            ORDER BY rand()
            LIMIT 1
            RETURN 
            a.Author AS author,
            k.Keyword AS keyword,
            [paper IN papers | paper.Paper] AS papers,
            true AS wroteWithKeyword
            """

    result = graph.query(query)

    questions.append(f"Did the author {result[0]['author']} write any paper that contains the keyword {result[0]['keyword']}?")
    answers.append(f"{result[0]['wroteWithKeyword']}: {result[0]['papers']}")

    # 18. Was the keyword <Keyword> associated with any paper published in the year <Year>?
    query = """
            MATCH (y:Year), (k:Keyword)
            WITH y, k
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (p:Paper)-[:PUBLISHED_IN]->(y)
            WITH y, k, COLLECT(p) AS papersInYear
            UNWIND papersInYear AS paper
            OPTIONAL MATCH (paper)-[:KEYWORDS]->(k)
            WITH y.Year AS year, k.Keyword AS keyword, paper
            WHERE paper IS NOT NULL
            WITH year, keyword, COLLECT(DISTINCT paper.Paper) AS papersWithKeyword
            RETURN 
            year,
            keyword,
            papersWithKeyword,
            CASE WHEN SIZE(papersWithKeyword) > 0 THEN true ELSE false END AS isAssociated
            """

    result = graph.query(query)

    questions.append(f"Was the keyword {result[0]['keyword']} associated with any paper published in the year {result[0]['year']}?")
    answers.append(f"{result[0]['isAssociated']}: {result[0]['papersWithKeyword']}")

    # 19. Has the author <Author> published any paper indexed in the database <Database>?
    query = """
            MATCH (a:Author), (d:Database)
            WITH a, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(p:Paper)-[:PUBLISHER]->(d)
            WHERE p IS NOT NULL
            WITH a.Author AS author, d.Database AS database, COLLECT(p.Paper) AS papers
            RETURN 
            author,
            database,
            papers,
            CASE WHEN SIZE(papers) > 0 THEN true ELSE false END AS hasPublishedInDatabase
            """

    result = graph.query(query)

    questions.append(f"Has the author {result[0]['author']} published any paper indexed in the database {result[0]['database']}?")
    answers.append(f"{result[0]['hasPublishedInDatabase']}: {result[0]['papers']}")

    # 20. Has the keyword <Keyword> been used in any paper published in the database <Database>?
    query = """
            MATCH (k:Keyword), (d:Database)
            WITH k, d
            ORDER BY rand()
            LIMIT 1
            OPTIONAL MATCH (k)<-[:KEYWORDS]-(p:Paper)-[:PUBLISHER]->(d)
            WITH 
            k.Keyword AS keyword, 
            d.Database AS database, 
            [x IN COLLECT(p) WHERE x IS NOT NULL] AS validPapers
            RETURN 
            keyword, 
            database, 
            [p IN validPapers | p.Paper] AS papers,
            SIZE(validPapers) > 0 AS isKeywordUsedInDatabase
            """

    result = graph.query(query)

    questions.append(f"Has the keyword {result[0]['keyword']} been used in any paper published in the database {result[0]['database']}?")
    answers.append(f"{result[0]['isKeywordUsedInDatabase']}: {result[0]['papers']}")

    testset = pd.DataFrame({"Question": questions, "Answer": answers})

    testset["Tool"] = "cypher_search"

    return testset


def generate_testset():

    # similarity tool
    sim_testset = get_questions_similarity_tool()

    # cypher tool
    print("Generating cypher related questions...")
    cy_testset = get_questions_cypher_tool()

    print("Done!")

    final_testset = pd.concat([sim_testset, cy_testset], ignore_index=True)

    final_testset.to_json("Evaluate/testset.json", orient="table", force_ascii=False)

    print("Testset created with success!")

    return final_testset


def extract_statements(question, answer):
    # Initialize the language model for answer generation
    llm = OllamaLLM(model="llama3")

    prompt = f"""Given a question and answer, create one or more statements from each sentence in the given answer.
    question: {question}
    answer: {answer}
    Return only the statements as a python list format. No pre-amble.
    """

    response = llm.invoke(prompt)

    try:
        response = ast.literal_eval(response)
    except:
        response = []

    return response


def verify_statements(context, statements):
    verdicts = {}

    llm = OllamaLLM(model="llama3")

    for statement in statements:

        prompt = f"""Consider the given context and the following statement, then determine whether it is supported by the information present in the context.
        Think carefully about each statement before arriving at the verdict (Yes/No).

        Context: {context}

        Statement: {statement}

        If the verdict is Yes, output: 1
        If the verdict is No, output: 0
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
    statements = extract_statements(question, answer)

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
    # Initialize the language model for answer generation
    llm = OllamaLLM(model="llama3")

    prompt = f"""Generate {n} different questions for the following answer:
        Answer: {answer}
        Return only the statements as a python list format. No pre-amble.
        """

    response = llm.invoke(prompt)

    try:
        response = ast.literal_eval(response)
    except:
        response = []

    return response


def get_embedding(texts):
    embedding = OllamaEmbeddings(model="llama3")

    vectors = embedding.embed_documents(texts)

    return vectors


def answer_relevance(original_question, answer, n=3):
    generated_questions = generate_questions(answer, n=n)

    print(f"Generated questions: {generated_questions}")

    if len(generated_questions) == 0:
        return np.nan

    q_embedding = get_embedding(original_question)[0]
    generated_embeddings = get_embedding(generated_questions)

    similarities = [
        cosine_similarity([q_embedding], [gen_emb])[0][0]
        for gen_emb in generated_embeddings
    ]

    return np.mean(similarities)


def judge_relevance(query, context):
    """
    Use a language model to judge the relevance of a context to the query.
    Returns 1 for relevant and 0 for not relevant.
    """
    llm = OllamaLLM(model="llama3")

    prompt = f"""Given the query: '{query}', is the following context relevant?
             Think carefully before providing the answer.         
             Context: {context}

             If the context is relevant, output: 1
             If the context is not relevant, output: 0
             Only output the number 0 or 1. Do not include explanations or any other text. Do not deviate from the specified format.
             """

    response = llm.invoke(prompt)

    try:
        response = int(response)
    except:
        response = None

    return response


def compute_precision_at_k(retrieved_contexts, query, k=3):
    """
    Compute the Context Precision at K for a given query and retrieved contexts.

    :param retrieved_contexts: List of context chunks (strings).
    :param query: The query string.
    :param k: The number of top contexts to evaluate.
    :return: Context Precision at K
    """
    # Limit the retrieved contexts to top K
    retrieved_contexts = retrieved_contexts[:k]

    print(f"Retrieved contexts: {retrieved_contexts}")

    # Judge the relevance of each context
    relevance_scores = [judge_relevance(query, context) for context in retrieved_contexts]

    print(f"relevance scores: {relevance_scores}")

    # Calculate Precision at each rank k
    true_positives = 0
    false_positives = 0
    total_relevant = sum(1 for v in relevance_scores if v == 1)

    context_precision_at_k = 0

    for rank, relevance in enumerate(relevance_scores, 1):
        if relevance == 1:
            true_positives += 1
        else:
            false_positives += 1

        # Precision at rank k
        precision_at_k = true_positives / (
                true_positives + false_positives) if true_positives + false_positives > 0 else 0

        # Binary relevance indicator (v_k)
        v_k = relevance  # 1 if relevant, 0 otherwise

        # Add the product of precision at rank k and relevance to the total score
        context_precision_at_k += precision_at_k * v_k

    # Return the final Context Precision at K (scaled by total relevant items in top K)
    return context_precision_at_k / total_relevant if total_relevant > 0 else 0


def context_recall_semantic(retrieved_context, ground_truth_answer, threshold=0.7):
    embedding_model = OllamaEmbeddings(model="llama3")

    # Tokenize the ground truth into sentences
    ground_truth_sentences = sent_tokenize(str(ground_truth_answer))

    # Get embeddings
    try:
        retrieved_context = [doc.page_content for doc in retrieved_context] # similarity
    except:
        retrieved_context = retrieved_context # cypher

    context_embeddings = embedding_model.embed_documents(retrieved_context)
    ground_truth_embeddings = embedding_model.embed_documents(ground_truth_sentences)

    matched_count = 0

    # Check each ground truth sentence against all context chunks
    for gt_emb in ground_truth_embeddings:
        similarities = cosine_similarity([gt_emb], context_embeddings)[0]

        print(f"Max similarity = {np.max(similarities):.4f}")
        if np.max(similarities) >= threshold:
            matched_count += 1

    # Avoid division by zero
    if len(ground_truth_sentences) == 0:
        return 0.0

    recall = matched_count / len(ground_truth_sentences)
    return recall


def evaluate_agent(testset, agent):

    print("Starting evaluation...")

    # create llm to check accuracy
    check_llm = OllamaLLM(model="llama3")

    # similarity parameters
    sim_accuracy = 0
    total_queries_sim = len(testset.loc[testset["Tool"] == "similarity_search", :])
    total_queries_cy = len(testset.loc[testset["Tool"] == "cypher_search", :])

    # agent parameters
    sim_right = 0
    cy_right = 0

    # Evaluate on cypher_query retrievals
    cy_accuracy = 0
    node_centered = 0
    relationship_centered = 0
    relationship_discovery = 0
    fact_check = 0
    indirect_relationship = 0

    # setup vector search tool
    with open("Database/text_splits.pkl", "rb") as f:
        text_splits = pickle.load(f)

    # Initialize a keyword-based retriever using BM25
    keyword_retriever = BM25Retriever.from_documents(text_splits)
    keyword_retriever.k = 5  # Retrieve top 5 most relevant documents

    # Initialize embedding model for vector similarity search
    embedding = OllamaEmbeddings(model="llama3")

    # Load the FAISS vector index from local storage
    vectorstore = FAISS.load_local("Database/faiss_index", embedding, allow_dangerous_deserialization=True)

    # Create a retriever using FAISS with top-5 search
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Combine both retrievers with equal weight
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    # Set API key for Cohere's reranker model
    os.environ["COHERE_API_KEY"] = "Ni2SuJm5hKdJict4OAblCsQ3l08tA3AYZwbQa2CL"

    # Apply Cohere's reranking model to compress and filter context
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    s_recalls = []
    s_faithfulnesses = []
    s_precisions = []
    s_relevances = []
    c_recalls = []
    c_faithfulnesses = []
    c_precisions = []
    c_relevances = []

    total_nodeCentered = total_relationshipCentered = total_relationshipDiscovery = total_factCheck = total_indirectRelationship = 4

    for i, (question, answer, tool) in tqdm(enumerate(zip(testset["Question"], testset["Answer"], testset["Tool"])), total=len(testset), desc="Evaluating questions"):

        if tool == "similarity_search":

            print("similarity task")

            print("evaluate agent decision and final response")

            result = agent.run(user_msg=question)

            print(result)

            final_answer = result["final_response"]

            if result["tool_call"] is not None:
                tool_used = json.loads(result["tool_call"])

                if tool_used["name"] == tool:
                    sim_right += 1
                    print("used right function")
                else:
                    print("didn't use right function")

            PROMPT = f"""You are given a question, a context, and a generated result based on the context.
                    Your task is to compare the generated result with the context and evaluate whether the result is correct.
                    If the generated result matches the context in meaning and fact, output: 1
                    If it does not match, output: 0
                    Only output the number 0 or 1. Do not include explanations or any other text.

                    Question:
                    {question}

                    Context:
                    {answer}

                    Generated Result:
                    {final_answer}
                    """

            feedback = check_llm.invoke(PROMPT)
            print(f"Feedback: {feedback}")
            try:
                feedback = int(feedback)
                print("Converted in int")

                sim_accuracy += feedback

            except:

                print("didn't convert in int")
                total_queries_sim -= 1

            print("evaluate vector search")

            # Retrieve the most relevant context for the input question
            context = compression_retriever.invoke(question)

            print(f"context: {context}")

            # compute faithfullness
            faithfulness = compute_faithfulness_score(question, final_answer, context)
            print(f"Faithfullness: {faithfulness}")
            s_faithfulnesses.append(faithfulness)

            # compute answer relevance
            relevance = answer_relevance(question, final_answer, n=3)
            print(f"Answer Relevance: {relevance}")
            s_relevances.append(relevance)

            # compute precision at k=3
            precision = compute_precision_at_k(context, question, k=3)
            print(f"Precision at k=3: {precision}")
            s_precisions.append(precision)

            # compute context recall
            recall = context_recall_semantic(context, answer)
            print(f"Context Recall: {recall}")
            s_recalls.append(recall)

        # cypher questions
        else:
            print("cypher task")

            print("evaluate agent decision and final response")

            result = agent.run(user_msg=question)

            final_answer = result["final_response"]

            if result["tool_call"] is not None:
                tool_used = json.loads(result["tool_call"])

                if tool_used["name"] == tool:
                    cy_right += 1
                    print("used right function")
                else:
                    print("didn't use the right function")

            PROMPT = f"""You are given a question, an expected answer, and a retrieved result from a graph database.
            Your task is to compare the retrieved result with the expected answer and evaluate whether the retrieval is correct.
            If the retrieved result matches the expected answer in meaning and fact, output: 1
            If it does not match, output: 0
            Only output the number 0 or 1. Do not include explanations or any other text.

            Question:
            {question}

            Expected Answer:
            {answer}

            Retrieved Result:
            {final_answer}
            """

            feedback = check_llm.invoke(PROMPT)
            print(f"Feedback: {feedback}")

            try:

                feedback = int(feedback)
                print("converted in in")

                cy_accuracy += feedback

                if i <= 23:
                    node_centered += feedback
                elif i <= 27:
                    relationship_centered += feedback
                elif i <= 31:
                    relationship_discovery += feedback
                elif i <= 35:
                    fact_check += feedback
                else:
                    indirect_relationship += feedback

            except:
                print("didn't convert in int")
                total_queries_cy -= 1

                if i <= 23:
                    total_nodeCentered -= 1
                elif i <= 27:
                    total_relationshipCentered -= 1
                elif i <= 31:
                    total_relationshipDiscovery -= 1
                elif i <= 35:
                    total_factCheck -= 1
                else:
                    total_indirectRelationship -= 1

            print("evaluate graph search")

            response = cypher_search(question)

            # Safely extract the Answer from the JSON response
            retrieved_context = json.loads(response)

            # Treat context as a single retrieved chunk
            context = [retrieved_context["Answer"]]

            print(f"context: {context}")

            # compute faithfullness
            faithfulness = compute_faithfulness_score(question, final_answer, context)
            print(f"Faithfullness: {faithfulness}")
            c_faithfulnesses.append(faithfulness)

            # compute answer relevance
            relevance = answer_relevance(question, final_answer, n=1)
            print(f"Answer Relevance: {relevance}")
            c_relevances.append(relevance)

            # compute precision at k=1
            precision = compute_precision_at_k(context, question, k=1)
            print(f"Precision at k=3: {precision}")
            c_precisions.append(precision)

            # compute context recall
            recall = context_recall_semantic(context, answer)
            print(f"Context Recall: {recall}")
            c_recalls.append(recall)

    print(f"Similarity faithfulness: {s_faithfulnesses}")
    print(f"Similarity answer relevance: {s_relevances}")
    print(f"Similarity precision at k=3: {s_precisions}")
    print(f"Similarity context recall: {s_recalls}")
    print(f"Cypher faithfulness: {c_faithfulnesses}")
    print(f"Cypher answer relevance: {c_relevances}")
    print(f"Cypher precision at k=1: {c_precisions}")
    print(f"Cypher context recall: {c_recalls}")

    # get means
    mean_sf = np.nanmean(s_faithfulnesses)
    print(f"Mean similarity faithfulness: {mean_sf}")

    mean_sar = np.nanmean(s_relevances)
    print(f"Mean similarity answer relevance: {mean_sar}")

    mean_sp = np.nanmean(s_precisions)
    print(f"Mean similarity precision at k=3: {mean_sp}")

    mean_sr = np.nanmean(s_recalls)
    print(f"Mean similarity context recall: {mean_sr}")

    mean_cf = np.nanmean(c_faithfulnesses)
    print(f"Mean cypher faithfulness: {mean_cf}")

    mean_car = np.nanmean(c_relevances)
    print(f"Mean cypher answer relevance: {mean_car}")

    mean_cp = np.nanmean(c_precisions)
    print(f"Mean cypher precision at k=3: {mean_cp}")

    mean_cr = np.nanmean(c_recalls)
    print(f"Mean cypher context recall: {mean_cr}")

    mean_of = np.nanmean(s_faithfulnesses + c_faithfulnesses)
    print(f"Mean overall faithfulness: {mean_of}")

    mean_oar = np.nanmean(s_relevances + c_relevances)
    print(f"Mean overall answer relevance: {mean_oar}")

    mean_op = np.nanmean(s_precisions + c_precisions)
    print(f"Mean overall precision at k=3: {mean_op}")

    mean_or = np.nanmean(s_recalls + c_recalls)
    print(f"Mean overall context recall: {mean_or}")

    # get medians
    median_sf = np.nanmedian(s_faithfulnesses)
    print(f"Median similarity faithfulness: {median_sf}")

    median_sar = np.nanmedian(s_relevances)
    print(f"Median similarity answer relevance: {median_sar}")

    median_sp = np.nanmedian(s_precisions)
    print(f"Median similarity precision at k=3: {median_sp}")

    median_sr = np.nanmedian(s_recalls)
    print(f"Median similarity context recall: {median_sr}")

    median_cf = np.nanmedian(c_faithfulnesses)
    print(f"Median cypher faithfulness: {median_cf}")

    median_car = np.nanmedian(c_relevances)
    print(f"Median cypher answer relevance: {median_car}")

    median_cp = np.nanmedian(c_precisions)
    print(f"Median cypher precision at k=3: {median_cp}")

    median_cr = np.nanmedian(c_recalls)
    print(f"Median cypher context recall: {median_cr}")

    median_of = np.nanmedian(s_faithfulnesses + c_faithfulnesses)
    print(f"Median overall faithfulness: {median_of}")

    median_oar = np.nanmedian(s_relevances + c_relevances)
    print(f"Median overall answer relevance: {median_oar}")

    median_op = np.nanmedian(s_precisions + c_precisions)
    print(f"Median overall precision at k=3: {median_op}")

    median_or = np.nanmedian(s_recalls + c_recalls)
    print(f"Median overall context recall: {median_or}")

    # overall accuracy similarity search
    accuracy_overall_sim = sim_accuracy / total_queries_sim

    print(f"Averall similarity accuracy: {accuracy_overall_sim}")

    # accuracy individual cypher search types
    accuracy_1 = node_centered / total_nodeCentered
    accuracy_2 = relationship_centered / total_relationshipCentered
    accuracy_3 = relationship_discovery / total_relationshipDiscovery
    accuracy_4 = fact_check / total_factCheck
    accuracy_5 = indirect_relationship / total_indirectRelationship

    print(f"Accuracy node centered: {accuracy_1}")
    print(f"Accuracy relationship centered: {accuracy_2}")
    print(f"Accuracy relationship discovery: {accuracy_3}")
    print(f"Accuracy fact check: {accuracy_4}")
    print(f"Accuracy indirect relationship: {accuracy_5}")

    # overall accuracy cypher search
    accuracy_overall_cy = cy_accuracy / total_queries_cy

    print(f"Overall cypher accuracy: {accuracy_overall_cy}")

    # overall accuracy
    accuracy_overall = (total_queries_sim*accuracy_overall_sim + total_queries_cy*accuracy_overall_cy) / (total_queries_sim + total_queries_cy)

    print(f"Overall accuracy: {accuracy_overall}")

    print(f"Cypher right: {cy_right}")
    print(f"Cypher total queries: {total_queries_cy}")
    print(f"Similarity right: {sim_right}")
    print(f"Similarity total queries: {total_queries_sim}")
    
    # accuracy agent
    cy_final = cy_right / total_queries_cy
    sim_final = sim_right / total_queries_sim

    print(f"Agent accuracy on cypher: {cy_final}")
    print(f"Agent accuracy on similarity {sim_final}")

    # overall accuracy agent
    agent_final = (sim_right + cy_right) / len(testset)

    print(f"Overall agent accuracy: {agent_final}")

    metrics = ["Overall Agent Accuracy",
               "Agent Accuracy on Similarity Tasks",
               "Agent Accuracy on Cypher Tasks",
               "Overall Accuracy",
               "Overall Similarity Accuracy",
               "Overall Cypher Accuracy",
               "Accuracy (Node Centered)",
               "Accuracy (Relationship Centered)",
               "Accuracy (Relationship Discovery)",
               "Accuracy (Fact Check)",
               "Accuracy (Indirect Relationship)",
               "Similarity Faithfulness (mean)",
               "Similarity Faithfulness (median)",
               "Cypher Faithfulness (mean)",
               "Cypher Faithfulness (median)",
               "Overall Faithfulness (mean)",
               "Overall Faithfulness (median)",
               "Similarity Answer Relevance (mean)",
               "Similarity Answer Relevance (median)",
               "Cypher Answer Relevance (mean)",
               "Cypher Answer Relevance (median)",
               "Overall Answer Relevance (mean)",
               "Overall Answer Relevance (median)",
               "Similarity Precision at k=3 (mean)",
               "Similarity Precision at k=3 (median)",
               "Cypher Precision at k=1 (mean)",
               "Cypher Precision at k=1 (median)",
               "Overall Precision at k (mean)",
               "Overall Precision at k (median)",
               "Similarity Context Recall (mean)",
               "Similarity Context Recall (median)",
               "Cypher Context Recall (mean)",
               "Cypher Context Recall (median)",
               "Overall Context Recall (mean)",
               "Overall Context Recall (median)",
               ]

    results = [
        agent_final,
        sim_final,
        cy_final,
        accuracy_overall,
        accuracy_overall_sim,
        accuracy_overall_cy,
        accuracy_1,
        accuracy_2,
        accuracy_3,
        accuracy_4,
        accuracy_5,
        mean_sf,
        median_sf,
        mean_cf,
        median_cf,
        mean_of,
        median_of,
        mean_sar,
        median_sar,
        mean_car,
        median_car,
        mean_oar,
        median_oar,
        mean_sp,
        median_sp,
        mean_cp,
        median_cp,
        mean_op,
        median_op,
        mean_sr,
        median_sr,
        mean_cr,
        median_cr,
        mean_or,
        median_or
    ]

    final_results = pd.DataFrame({"Metric": metrics, "Result": results})

    final_results.to_excel("Evaluate/evaluation_results.xlsx", index=False)

    print("Finished evaluating pipeline!")
    print("Please check results within the Evaluate folder")

    return
