from venv import create

from dotenv import load_dotenv
import os

from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()

if __name__ == "__main__":
    print("hi")
    pdf_path = "/Users/Takwa/Documents/projects/udemy/rag-vectorDB/paper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents = documents)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    llm = ChatOllama(model="llama3.1")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(new_vectorstore.as_retriever(),combine_docs_chain)
    res = retrieval_chain.invoke({"input":"Give me the Gist of ReAct in 3 sentences"})
    print(res["answer"])
