from Functions.functions import cypher_search, similarity_search
from Functions.tool import tool
from Functions.tool_agent import ToolAgent


def agent_rag():
    """
    Creates and returns a Retrieval-Augmented Generation (RAG) agent that combines Cypher and similarity-based tools.

    Parameters:
    - None

    Functionality:
    - Wraps the Cypher-based graph query function and similarity-based PDF search function as tools.
    - Constructs an agent capable of choosing between these tools to answer user queries effectively.
      The agent can decide whether to query the Neo4j graph or retrieve context from PDFs based on the question type.

    Returns:
    - ToolAgent: An agent instance capable of performing tool-augmented question answering.
    """

    # Convert the cypher and similarity retrieval functions into usable tools for the agent
    cypher_tool = tool(cypher_search)
    similarity_tool = tool(similarity_search)

    # Create a ToolAgent with both tools registered
    agent = ToolAgent(tools=[cypher_tool, similarity_tool])

    # Return the configured agent
    return agent
