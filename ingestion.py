import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub


from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
if __name__ == "__main__":
    print("ingestion ...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("splitting ....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f'created {len(texts)} chunks')

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])
    print("finish")

