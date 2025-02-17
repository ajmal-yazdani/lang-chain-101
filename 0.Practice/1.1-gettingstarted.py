from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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

result = llm.invoke("What is Agentic AI")
print(result)
print(result.content)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI Engineer. Provide me answer based on the question",
        ),
        ("user", "{input}"),
    ]
)

chain = prompt | llm
response = chain.invoke({"input": "Can you tell me about Langsmith"})
print(response.content)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
response = chain.invoke({"input": "Can you tell me about Langsmith?"})
print(response)

output_parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

chain = prompt | llm | output_parser
response = chain.invoke({"query": "What is Agentic AI"})
print(response)

loader = WebBaseLoader("https://python.langchain.com/docs/tutorials/llm_chain/")

document = loader.load()
print(document)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(document)
print(documents)


embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(documents, embedding)

query = "This is a relatively simple LLM application"
result = vectorstore.similarity_search(query)
result[0].page_content
