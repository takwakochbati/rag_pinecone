from dotenv import load_dotenv
import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
    return"\n\n".join(doc.page_content for doc in docs)

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

    template = """ Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question : {question}
    
    Helpful answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = ({"context" : vectorstore.as_retriever()|format_docs , "question" : RunnablePassthrough()}|custom_rag_prompt|llm)
    result = rag_chain.invoke(query)
    print(result)