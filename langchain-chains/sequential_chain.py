from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

model = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-08-01-preview")
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in India"})

print(result)

chain.get_graph().print_ascii()
