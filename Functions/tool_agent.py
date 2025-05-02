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

    user: What is mentioned about optimization in the context of diabetes?
    <tool_call>
    {"name": "similarity_retrieval", "arguments": {"query": "optimization in the context of diabetes"}, "id": 1}
    </tool_call>
    Answer: Optimization in the context of diabetes focuses on finding the best strategies to control blood glucose levels, such as adjusting insulin doses, meal timings, and exercise plans.
    final_response: Optimization in the context of diabetes focuses on finding the best strategies to control blood glucose levels, such as adjusting insulin doses, meal timings, and exercise plans.
    
    user: Find authors who collaborated with João Silva and published in 2021.
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "authors who collaborated with João Silva and published in 2021"}, "id": 2}
    </tool_call>
    Answer: no authors who collaborated with João Silva and published in 2021 were found.
    final_response: no authors who collaborated with João Silva and published in 2021 were found.
    
    user: List all methods used in papers related to deep learning and genomics.
    <tool_call>
    {"name": "similarity_retrieval", "arguments": {"query": "methods used in deep learning and genomics"}, "id": 3}
    </tool_call>
    Answer: Deep learning methods used in genomics papers include convolutional neural networks (CNNs), recurrent neural networks (RNNs), transformers, autoencoders, graph neural networks (GNNs), and generative adversarial networks (GANs).
    final_response: Deep learning methods used in genomics papers include convolutional neural networks (CNNs), recurrent neural networks (RNNs), transformers, autoencoders, graph neural networks (GNNs), and generative adversarial networks (GANs).
    
    user: Which institutions are connected to publications on tuberculosis in Brazil?
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "institutions connected to publications on tuberculosis in Brazil"}, "id": 4}
    </tool_call>
    Answer: UnB and USP have published about tuberculosis in Brazil. 
    final_response: UnB and USP have published about tuberculosis in Brazil.
    
    user: What are the models being used for text-image tasks in healthcare?
    <tool_call>
    {"name": "similarity_retrieval", "arguments": {"query": "models being used for text-image tasks in healthcare"}, "id": 5}
    </tool_call>
    Answer: Models used for text-image tasks in healthcare include CLIP, BioViL, MedCLIP, GLoRIA, and LLaVA-Med, often combining vision transformers (ViT) and language models (like BioBERT or LLaMA) to link medical images and clinical text.
    final_response: Models used for text-image tasks in healthcare include CLIP, BioViL, MedCLIP, GLoRIA, and LLaVA-Med, often combining vision transformers (ViT) and language models (like BioBERT or LLaMA) to link medical images and clinical text.
    
    user: What is the abstract of the paper A Comprehensive Survey of Large Language Models and Multimodal Large Language Models in Medicine?
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "abstract of the paper A Comprehensive Survey of Large Language Models and Multimodal Large Language Models in Medicine"}, "id": 6}
    </tool_call>
    Answer: The paper surveys the development, applications, challenges, and future directions of large language models (LLMs) and multimodal large language models (MLLMs) in medicine.
    final_response: The paper surveys the development, applications, challenges, and future directions of large language models (LLMs) and multimodal large language models (MLLMs) in medicine.
    
    user: What connection exists between the keyword healthcare and the paper From Text to Multimodality: Exploring the Evolution and Impact of Large Language Models in Medical Practice?
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "relationship between the keyword healthcare and the paper From Text to Multimodality: Exploring the Evolution and Impact of Large Language Models in Medical Practice, if any"}, "id": 7}
    </tool_call>
    Answer: The paper mentioned has that keyword representing it.
    final_response: The paper mentioned has that keyword representing it.
    
    user: What the the current limitations in mllm research?
    <tool_call>
    {"name": "similarity_retrieval", "arguments": {"query": "current limitations in mllm research"}, "id": 8}
    </tool_call>
    Answer: Current limitations in MLLM (Multimodal Large Language Model) research include limited medical domain knowledge, small and biased datasets, poor interpretability, challenges in aligning multimodal inputs, and concerns over patient safety, privacy, and regulatory compliance.
    final_response: Current limitations in MLLM (Multimodal Large Language Model) research include limited medical domain knowledge, small and biased datasets, poor interpretability, challenges in aligning multimodal inputs, and concerns over patient safety, privacy, and regulatory compliance.
    
    user: How many keywords describe the sample?
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "total amount of keywords available"}, "id": 9}
    </tool_call>
    Answer: 5 keywords describe the sample.
    final_response: 5 keywords describe the sample.
    
    user: What is the meaning of meta-learning?
    <tool_call>
    {"name": "similarity_retrieval", "arguments": {"query": "meaning of meta-learning"}, "id": 10}
    </tool_call>
    Answer: Meta-learning, or "learning to learn," refers to training models that can quickly adapt to new tasks by leveraging experience from learning many previous tasks.
    final_response: Meta-learning, or "learning to learn," refers to training models that can quickly adapt to new tasks by leveraging experience from learning many previous tasks.
    
    user: how many databases are available?
    <tool_call>
    {"name": "cypher_retrieval", "arguments": {"query": "total amount of keywords available"}, "id": 9}
    </tool_call>
    Answer: ArXiv, IEEEXplore and Web of Science are available for your research.
    final_response: ArXiv, IEEEXplore and Web of Science are available for your research.

IMPORTANT: If a question does not match any function or if data is not found, do NOT generate an answer. Just say: "I was not able to answer the question." 
Always answer based only on the context provided and be concise. Do not create new information or add anything that is not related to the question.
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

            first_result = next(iter(observations.values()))

            first_result = json.loads(first_result)

            tool_observations = first_result.get("Answer")
            context = first_result.get("Context")

            update_chat_history(
                agent_chat_history, f"Observation: {context}", "user"
            )

        else:
            tool_observations = None
            context = None

        return {
            #"final_response": completions_create(self.client, agent_chat_history, self.model),
            "final_response": tool_observations,
            "agent_reasoning": agent_reasoning,
            "tool_call": tool_calls.content[0] if tool_calls.found else None,
            "context": context
        }

        #return completions_create(self.client, agent_chat_history, self.model)
