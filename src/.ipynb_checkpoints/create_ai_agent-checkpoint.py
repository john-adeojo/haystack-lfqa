from haystack.agents import Agent
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.agents import Tool


def create_agent(document_qa, API_KEY):
    react_prompt = PromptTemplate(
        prompt="""You are a helpful and knowledgeable agent. To achieve your goal of answering complex questions
    correctly, you have access to the following tools:
    
    {tool_names_with_descriptions}
    
    To answer questions, you'll need to go through multiple steps involving step-by-step thinking and
    selecting appropriate tools and their inputs; tools will respond with observations. When you are ready
    for a final answer, respond with the `Final Answer:`
    
    Use the following format:
    
    Question: the question to be answered
    Thought: Reason if you have the final answer. If yes, answer the question. If not, find out the missing information needed to answer it.
    Tool: pick one of {tool_names} 
    Tool Input: the input for the tool
    Observation: the tool will respond with the result
    ...
    
    Final Answer: the final answer to the question, make it short (200 words)
    Thought, Tool, Tool Input, and Observation steps can be repeated multiple times, but sometimes we can find an answer in the first pass
    ---
    
    Question: {query}
    Thought: Let's think step-by-step, I first need to
    """,
        output_parser=AnswerParser(),
    )
    prompt_node = PromptNode(model_name_or_path="gpt-4", default_prompt_template=react_prompt, api_key=API_KEY, stop_words=["Observation:"], model_kwargs={"temperature":0})
    agent = Agent(prompt_node=prompt_node)

    search_tool = Tool(
    name="document_qa",
    pipeline_or_node=document_qa,
    description="useful for when you need to answer any question",
    output_variable="answers",
)
    agent.add_tool(search_tool)

    return agent