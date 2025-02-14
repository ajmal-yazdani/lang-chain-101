from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]

result = embedding.embed_documents(documents)
print(str(result))
