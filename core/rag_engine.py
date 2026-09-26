from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

SYSTEM_PROMPT = (
    "You are an expert meeting assistant. Answer the user's questions based "
    "only on the meeting transcript context provided below. "
    "If the answer is not in the context, say 'I could not find this "
    "information in the meeting transcript.' Always be precise and concise. "
    "If quoting someone, mention it clearly.\n\n"
    "Context from meeting transcript: {context}"
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _assemble_chain(retriever):
    llm = get_llm("claude-sonnet-5")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    return (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def build_rag_chain(transcript: str, collection_name: str):
    vector_store = build_vector_store(transcript, collection_name)
    retriever = get_retriever(vector_store, k=4)
    return _assemble_chain(retriever)


def load_rag_chain(collection_name: str):
    vector_store = load_vector_store(collection_name)
    retriever = get_retriever(vector_store, k=4)
    return _assemble_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke(question)