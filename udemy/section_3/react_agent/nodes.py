from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

from react import react_agent_runnable, tools
from langchain_core.messages import AIMessage
from state import AgentState

load_dotenv()


def run_agent_reasoning_engine(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolNode(tools=tools)


def execute_tools(state: AgentState):
    agent_action = state["agent_outcome"]
    tool_args = {}
    if agent_action.tool == "triple":
        tool_args = {"num": agent_action.tool_input}
    else:
        tool_args = {"query": agent_action.tool_input}
    
    message_with_single_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": agent_action.tool,
                "args": tool_args,
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    raw_output = tool_executor.invoke({"messages": [message_with_single_tool_call]})
    output = str(raw_output) if isinstance(raw_output, list) else raw_output
    return {"intermediate_steps": [(agent_action, str(output))]}
