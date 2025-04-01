import datetime

from dotenv import load_dotenv

load_dotenv()

# This will take in the response and extract the JSON or the Pydantic object
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage # to generate the human message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # to create the prompt template and store responses
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
# This will return the response as a dictionary
parser = JsonOutputToolsParser(return_id=True)
# This takes the answer from the LLM and create the AnswerQuestion Pydantic object
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# The agent template
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"), # all the information of what to research and the response will be stored here
        # ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(), # this will pass the 'time' to the prompt template
)

# This is using the template above and defining the first instruction
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# Then we pipe this with the tools we want to use (we are forcing the LLM to use the AnswerQuestion tool) > the pydantic object will ground the answer from the LLM
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
validator = PydanticToolsParser(tools=[AnswerQuestion])

# Revisor Agent - Instructions
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# Using the template we pass the revised instructions and bind the ReviseAnswer tool to the LLM
# this will force the LLM to use the ReviseAnswer tool to revise the answer
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


# if __name__ == "__main__":
#     human_message = HumanMessage(
#         content="Write about AI-Powered SOC / autonomous soc  problem domain,"
#         " list startups that do that and raised capital."
#     )
#     chain = (
#         first_responder_prompt_template
#         | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
#         | parser_pydantic # in the end it will parse the response to the Pydantic object (AnswerQuestion) so we can use it later
#     )

#     res = chain.invoke(input={"messages": [human_message]})
#     print(res)
