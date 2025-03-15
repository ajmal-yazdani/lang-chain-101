from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)

model = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-08-01-preview")
parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({"topic": "elephants"})
print(result)

chain.get_graph().print_ascii()
