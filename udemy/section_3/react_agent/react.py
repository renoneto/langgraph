from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent # Create an agent that uses ReAct prompting.
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# taking the react prompt from the hub > with Tool names and descriptions
react_prompt: PromptTemplate = hub.pull("hwchase17/react")


@tool
def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: the number tripled ->  multiplied by 3
    """
    return 3 * float(num)

# Create a wrapper for TavilySearchResults that returns string output
@tool
def tavily_search(query: str) -> str:
    """Search the internet for the query using Tavily API and return results as a string."""
    search = TavilySearchResults(max_results=1)
    results = search.invoke(query)
    return str(results[0]) if results else "No results found"

# Defining list of tools
tools = [tavily_search, triple]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Create the REACT agent using the prompt, tools, and LLM
# The REACT agent will use the provided prompt to generate its responses
react_agent_runnable = create_react_agent(llm, tools, react_prompt)
