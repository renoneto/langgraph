from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()
# First node to receive the user's question
builder.add_node("draft", first_responder)
# Second node to execute the tools
builder.add_node("execute_tools", execute_tools)
# Third node to revise the answer
builder.add_node("revise", revisor)
# First edge to execute the tools
builder.add_edge("draft", "execute_tools")
# Second edge to revise the answer
builder.add_edge("execute_tools", "revise")

# Event loop to check if the number of tool visits is greater than the maximum iterations
def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# Add the event loop to the graph > simple one based on the number of tool visits
builder.add_conditional_edges("revise", event_loop)
# Set the entry point to the first node
builder.set_entry_point("draft")
# Compile the graph
graph = builder.compile()
# Draw the graph in mermaid format
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Invoke the graph with the user's question
res = graph.invoke(
    "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
)
# Print the final answer
print(res[-1].tool_calls[0]["args"]["answer"])
# Print the entire response
print(res)
