import os
from typing import List
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ai.settings import CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_TYPES = ['.pdf', '.docx', '.txt']

def load_document(file_path: str) -> List[Document]:
    """
    Load a document into LangChain Document objects.
    Adds normalized metadata for consistent citations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    extension = os.path.splitext(file_path)[-1].lower()
    
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    docs: List[Document] = loader.load()
    for doc in docs:
        doc.metadata = {
            **(doc.metadata or {}),
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": extension,
        }
    return docs

def split_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Split a list of documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
    return splitter.split_documents(docs)