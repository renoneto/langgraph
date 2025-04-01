import json
from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

from chains import parser
from schemas import AnswerQuestion, Reflection

load_dotenv()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
tool_executor = ToolNode(tools=[tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    # Assuming the last message in the state is an AIMessage containing tool invocations
    tool_invocation: AIMessage = state[-1]

    # Json Parser
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    messages_list = []
    
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            # Create a list of messages for each tool invocation
            messages = [
                HumanMessage(content=query),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": parsed_call["id"],
                        "name": "tavily_search_results_json",
                        "args": {
                            "query": query
                        }
                    }]
                )
            ]
            messages_list.append(messages)
            ids.append(parsed_call["id"])

    # Execute all messages in batch
    outputs = tool_executor.batch(messages_list)

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, messages in zip(ids, outputs, messages_list):
        query = messages[0].content  # Get the query from the HumanMessage
        outputs_map[id_][query] = output[0].content

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=mapped_output, tool_call_id=id_)
        )

    return tool_messages


# For testing purposes, we can run this script directly to see the output
# if __name__ == "__main__":
#     print("Tool Executor Enter")

#     human_message = HumanMessage(
#         content="Write about AI-Powered SOC / autonomous soc  problem domain,"
#         " list startups that do that and raised capital."
#     )

#     answer = AnswerQuestion(
#         answer="",
#         reflection=Reflection(missing="", superfluous=""),
#         search_queries=[
#             "AI-powered SOC startups funding",
#             "AI SOC problem domain specifics",
#             "Technologies used by AI-powered SOC startups",
#         ],
#         id="call_KpYHichFFEmLitHFvFhKy1Ra",
#     )

#     raw_res = execute_tools(
#         state=[
#             human_message,
#             AIMessage(
#                 content="",
#                 tool_calls=[
#                     {
#                         "name": AnswerQuestion.__name__,
#                         "args": answer.dict(),
#                         "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
#                     }
#                 ],
#             ),
#         ]
#     )
#     print(raw_res)
