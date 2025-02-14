from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-08-01-preview")

result = model.invoke("What is the capital of India")
print(result.content)
