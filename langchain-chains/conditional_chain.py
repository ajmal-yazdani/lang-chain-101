from typing import Literal

from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-08-01-preview")
parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (
        lambda x: isinstance(x, Feedback) and x.sentiment == "positive",
        prompt2 | model | parser,
    ),
    (
        lambda x: isinstance(x, Feedback) and x.sentiment == "negative",
        prompt3 | model | parser,
    ),
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a terrible phone"}))

chain.get_graph().print_ascii()
