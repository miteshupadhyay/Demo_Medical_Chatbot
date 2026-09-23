import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embedding_model_from_huggingface
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


os.environ["PINECONE_API_KEY"] =PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] =OPENAI_API_KEY


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

extracted_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embedding_model = download_embedding_model_from_huggingface()

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384, # Dimension of the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    index_name = index_name
)