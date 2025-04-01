# This file holds the schema of the responses we are expecting
from typing import List

from pydantic import BaseModel, Field

# The schema to define the structure of the reflection on the answer
# this will ground the response to ensure it's returning the expected format
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

# The answer must have the following fields
class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.") # the description helps the LLM as well
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

# For the revisor Agent, it takes the same schema from the AnswerQuestion and adds the references
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
