import os
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureOpenAI(
    azure_deployment="gpt-35-turbo-instruct",
    api_version="2024-08-01-preview",
)

result = llm.invoke("What is the capital of India")
print(result)
