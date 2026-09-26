from core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []
    for chunk in chunks:
        chunk_summaries.append(map_chain.invoke({"text": chunk}))

    # Short transcript: one chunk is already a full summary, skip the reduce step
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n".join(chunk_summaries)

    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting summarizer. Combine the following summaries into a single, coherent summary."),
        ("human", "{text}")
    ])
    combine_chain = combine_prompt | llm | StrOutputParser()

    return combine_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a concise and descriptive title for the following meeting transcript."),
        ("human", "{text}")
    ])
    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:5000]})