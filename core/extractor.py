from core.llm import get_llm
from core.summarize import split_transcript
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_chain(system_prompt: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    return prompt | llm | StrOutputParser()


def _map_reduce_extract(transcript: str, map_prompt: str, reduce_prompt: str) -> str:
    """Extract from each chunk, then merge/dedupe -- keeps a single full-transcript
    call for short meetings, same as before, while not paying for the whole
    transcript on every chunk once it's long enough to need chunking."""
    chunks = split_transcript(transcript)
    map_chain = build_chain(map_prompt)
    partials = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    if len(partials) == 1:
        return partials[0]

    combined = "\n\n".join(partials)
    return build_chain(reduce_prompt).invoke({"text": combined})


def extract_action_items(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting "
        "transcript, extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        "You are given several partial lists of action items extracted from "
        "different segments of the same meeting. Merge them into a single "
        "numbered list, removing duplicates (the segments overlap). Keep the "
        "same per-item format (task, owner, deadline). If none found say "
        "'No action items found.'"
    )


def extract_key_decisions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting "
        "transcript, extract all key decisions made. For each provide:\n"
        "- Decision description\n"
        "- Owner (who is responsible)\n"
        "- Date (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No key decisions found.'",
        "You are given several partial lists of key decisions extracted from "
        "different segments of the same meeting. Merge them into a single "
        "numbered list, removing duplicates (the segments overlap). Keep the "
        "same per-item format (decision, owner, date). If none found say "
        "'No key decisions found.'"
    )


def extract_questions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting "
        "transcript, extract all unresolved questions. Format as a numbered "
        "list. If none found say 'No open questions found.'",
        "You are given several partial lists of unresolved questions extracted "
        "from different segments of the same meeting. Merge them into a single "
        "numbered list, removing duplicates (the segments overlap). If none "
        "found say 'No open questions found.'"
    )
