---
title: AI Meeting Assistant
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.64.0"
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Summarize meetings, extract action items, and chat over the transcript.
---

# 🎙️ AI Meeting Assistant

Takes audio from a YouTube URL or an uploaded file, transcribes it locally with Whisper,
summarizes it and extracts action items / key decisions / open questions via LangChain LCEL
chains + Claude, and supports RAG-based chat over the transcript via ChromaDB + HuggingFace
embeddings.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `ffmpeg` as a system binary (`brew install ffmpeg` on macOS) and an `ANTHROPIC_API_KEY`
in a local `.env` file (see `.env` — never commit this file).

## Deploying to Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK and push this repo to it.
2. Under **Settings → Secrets**, add `ANTHROPIC_API_KEY` (never commit a real key —
   `.env` stays gitignored).
3. Under **Settings → Variables**, optionally set `WHISPER_MODEL=base` (or `tiny`) — the
   free CPU tier is slower than a local machine, and the code defaults to `small` if unset.
4. `packages.txt` (ffmpeg) and `requirements.txt` are picked up automatically by the
   Streamlit SDK build.

**Known limitations on the free tier:**
- `downloads/` and `vector_db/` are ephemeral — they're wiped whenever the Space restarts
  or rebuilds. Fine for a live demo; don't rely on it for persistence across restarts.
- `python_version` above is pinned to 3.11, not the 3.14 this project uses locally — Whisper,
  ChromaDB, and sentence-transformers are far more likely to have prebuilt wheels for 3.11 on
  HF's build image. If the build fails, that's the first thing to check.
