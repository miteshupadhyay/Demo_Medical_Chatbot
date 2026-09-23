from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Extract text from PDF Files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# Filter the minimal Docs required
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """ 
    Given a list of Documents Object, return a new list of Document Objects
    containing only 'source' in metadata and the original page_content.
    """

    minimal_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs


# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    text_chunk= text_splitter.split_documents(minimal_docs)
    return text_chunk

# Download the embedding model from huggingface
def download_embedding_model_from_huggingface():
    """ 
    Download and return the Huggingface embedding Model.    
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name= model_name
    )
    return embeddings

embedding_model = download_embedding_model_from_huggingface()