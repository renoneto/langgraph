from dotenv import load_dotenv

load_dotenv()

# AgentFinish: represents the final outcome of the agent's run
from langchain_core.agents import AgentFinish
# StateGraph: a graph structure to manage the flow of the agent's reasoning
from langgraph.graph import END, StateGraph

from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

AGENT_REASON = "agent_reason"
ACT = "act"

# Conditional logic to determine if the agent should continue or finish
# based on the outcome of its reasoning
# If the agent's outcome is an AgentFinish, we end the flow; otherwise, we continue with the next action
# This is a simple function that checks the type of the agent's outcome
def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT

# Create the state graph for the agent's reasoning process
flow = StateGraph(AgentState)
# Node for the agent's reasoning engine
flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
# Node for executing the tools based on the agent's reasoning
flow.set_entry_point(AGENT_REASON)
# Node for executing the tools, which is dependent on the agent's reasoning outcome
flow.add_node(ACT, execute_tools)
# Add edges to connect the nodes in the flow
# The flow starts with the agent's reasoning engine and then moves to execute the tools
# based on the agent's outcome
# The conditional edge checks if the agent's outcome is an AgentFinish to determine the next step
# If the agent's outcome is an AgentFinish, we end the flow; otherwise, we continue with the next action
# The flow will continue to execute the tools until the agent's outcome is an AgentFinish
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
)
# Connect the agent's reasoning node to the action node (ACT) to execute the tools
# based on the agent's reasoning outcome
# The action node (ACT) will execute the tools based on the agent's reasoning outcome
flow.add_edge(ACT, AGENT_REASON)

# Compile the flow to create the final application
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello ReAct with LangGraph")
    res = app.invoke(
        input={
            "input": "what is the weather in sf? List it and then Triple it ",
        }
    )
    print(res["agent_outcome"].return_values["output"])
