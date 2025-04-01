import operator
from typing import Annotated, TypedDict, Union

# AgentAction: represents the function we need to run and why
# AgentFinish: represents the final outcome of the agent's run
from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    # the user's input string
    input: str
    # The outcome of a given call to the agent. Needs 'None' as a valid type since this is what this will start as - the current state of the agent
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations (output) from the agent
    # Here we annotate this with 'operator.add' to indicate that operations on this list will be additive, not to overwrite
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
