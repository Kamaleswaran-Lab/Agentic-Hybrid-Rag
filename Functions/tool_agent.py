import json

from colorama import Fore
from groq import Groq
from typing import Union
import os

from Functions.tool import Tool, validate_arguments
from Functions.completions import build_prompt_structure, ChatHistory, completions_create, update_chat_history
from Functions.extraction import extract_tag_content


os.environ["GROQ_API_KEY"] = "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"


TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

You must not create original answers. If the tool was unable to retrieve data, say you weren't able to answer the question.
Do not provide alternatives or speculative content.

Return only a JSON object with the function name and arguments inside <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>, "arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Do not add any text or explanation outside the XML tags. Do not rephrase or interpret the answer.
Do not output more than one <tool_call> unless explicitly required.

---

Here are the available tools:

<tools>
%s
</tools>

---

Here are some examples:

User: What papers mention optimization in the context of diabetes?

<tool_call>
{"name": "similarity_retrieval", "arguments": {"query": "optimization in the context of diabetes"}, "id": 1}
</tool_call>

User: Find authors who collaborated with João Silva and published in 2021.

<tool_call>
{"name": "cypher_retrieval", "arguments": {"query": "authors who collaborated with João Silva and published in 2021"}, "id": 2}
</tool_call>

User: List all methods used in papers related to deep learning and genomics.

<tool_call>
{"name": "similarity_retrieval", "arguments": {"query": "methods used in deep learning and genomics"}, "id": 3}
</tool_call>

User: Which institutions are connected to publications on tuberculosis in Brazil?

<tool_call>
{"name": "cypher_retrieval", "arguments": {"query": "institutions connected to publications on tuberculosis in Brazil"}, "id": 4}
</tool_call>

IMPORTANT: If a question does not match any function or if data is not found, do NOT generate an answer. Just say: "I was not able to answer the question."
Never mix tools. Always use only one function per question, and return results ONLY inside <tool_call> XML tags.
"""

class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (Groq): The Groq client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
    """

    def __init__(
        self,
        tools: Union[Tool, list[Tool]],
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self.client = Groq()
        self.model = model
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """
        Collects the function signatures of all available tools.

        Returns:
            str: A concatenated string of all tool function signatures in JSON format.
        """
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = result

        return observations

    def run(
        self,
        user_msg: str,
    ) -> dict:
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
        user_msg (str): The user's message that prompts the tool agent to act.

        Returns:
        dict: A dictionary containing:
        - "final_response" (str): The final answer generated by the model after executing the tool.
        - "agent_reasoning" (str): The full reasoning and response from the model that includes tool choice and intermediate steps.
        - "tool_call" (str): The original tool call returned by the model before executing any tool.
        """

        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history = ChatHistory([user_prompt])

        tool_call_response = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )

        agent_reasoning = str(tool_call_response)

        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(
                agent_chat_history, f'f"Observation: {observations}"', "user"
            )

        return {
            "final_response": completions_create(self.client, agent_chat_history, self.model),
            "agent_reasoning": agent_reasoning,
            "tool_call": tool_calls.content[0] if tool_calls.found else None
        }

        #return completions_create(self.client, agent_chat_history, self.model)
