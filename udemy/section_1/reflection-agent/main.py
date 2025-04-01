from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage

# END: Indicates the end of the graph
# MessageGraph: A graph structure for managing message flows
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain

# Defining the nodes for the graph
REFLECT = "reflect"
GENERATE = "generate"

# Defining what happens in the generation node
# Input is the list of messages (state) and it returns the generated message
# Once the generate is done it will append the generated message to the state using the "messages" key
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

# This one returns a HumanMessage object with the generated content
# This was we can trick the LLM to think it is a human message and not a system one
# This is important for the reflection node to work properly
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

# Graph Initialization
builder = MessageGraph()
# Adding nodes to the graph
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
# Defining the starting node
builder.set_entry_point(GENERATE)

# We are defining a conditional edge that will determine if we should continue reflecting or end the process
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

# In this case we are adding a conditional edge that will check if the length of the state is greater than 6 but we could have another LLM here
# If the condition is met it will go to the END node, otherwise, it goes to REFLECT
builder.add_conditional_edges(GENERATE, should_continue)

# Creating a connection between the REFLECT node and the GENERATE node
# This means that once we reflect on the tweet we will generate a new one unconditionally
# This is important because we want to keep generating tweets until we reach the END node
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)