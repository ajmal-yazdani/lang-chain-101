from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

azure_openai_api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
if azure_openai_api_key:
    os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_api_key

## Langsmith Tracking and tracing
langchain_api_key: str | None = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"

langchain_project: str | None = os.getenv("LANGCHAIN_PROJECT")
if langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

llm = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-08-01-preview")
print(llm.invoke("What is the capital of India"))
