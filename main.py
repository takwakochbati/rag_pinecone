from dotenv import load_dotenv
import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":

    print("retrieving...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    llm = ChatOllama(model="llama3.1")
    query = "what is Pinecone in machine learning?"
    #chain = PromptTemplate.from_template(template=query)|llm
    #result = chain.invoke(input={})
    #print(result.content)
    #Answer the query using Pinecone vectorStor
    vectorstore = PineconeVectorStore(index_name = os.environ["INDEX_NAME"], embedding=embeddings)

    #prompt to send to the LLM after retrieving the information
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke(input = {"input" :query})
    print(result)