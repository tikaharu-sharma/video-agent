import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def meeting_id(source: str) -> str:
    """Stable per-meeting collection name derived from the input URL/path."""
    return "meeting_" + hashlib.sha256(source.encode()).hexdigest()[:16]


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )


def build_vector_store(transcript: str, collection_name: str):
    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index': i}) for i, chunk in enumerate(chunks)
    ]

    embedding_model = get_embedding_model()

    # Drop any previous run of this meeting so re-processing doesn't duplicate chunks
    load_vector_store(collection_name, embedding_model).delete_collection()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name
    )

    return vector_store


def load_vector_store(collection_name: str, embedding_model=None):
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=embedding_model or get_embedding_model()
    )


def get_retriever(vector_store: Chroma, k: int=4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
