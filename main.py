from dotenv import load_dotenv

# Must run before core imports: transcriber reads WHISPER_MODEL at import time
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.vector_store import meeting_id

def run_pipeline(source: str):
    print("Starting AI video assistant...")
    chunks = process_input(source)

    transcript = transcribe_all(chunks)
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    print("Generating title...")
    title = generate_title(transcript)

    print("Summarizing meeting...")
    summary = summarize(transcript)

    print("Extracting action items...")
    action_items = extract_action_items(transcript)

    print("Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    print("Extracting open questions...")
    questions = extract_questions(transcript)

    print("Building searchable knowledge base...")
    rag_chain = build_rag_chain(transcript, meeting_id(source))

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    source = input("Enter the video URL or local file path: ").strip()
    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"Summary: {result['summary']}")
    print(f"Action Items: {result['action_items']}")
    print(f"Key Decisions: {result['key_decisions']}")
    print(f"Questions: {result['questions']}")
    print("="*60)

    print("\n Chat with your meeting (type'exit' to quit)\n")
    rag_chain = result['rag_chain']

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            response = ask_question(rag_chain, question)
            print(f"\nAI: {response}\n")
        except Exception as e:
            print(f"\nSomething went wrong answering that: {e}\n")