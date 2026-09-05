# 🎥 YouTube Multimodal RAG Chatbot

Chat with a YouTube video's transcript *and* its frames, in one conversation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-6E56CF?style=flat)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20Cloud-FF6F00?style=flat)
![CLIP](https://img.shields.io/badge/Vision-OpenAI%20CLIP-black?style=flat)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20(via%20LiteLLM)-4285F4?style=flat&logo=googlegemini&logoColor=white)
![Phoenix](https://img.shields.io/badge/Observability-Arize%20Phoenix-9C27B0?style=flat)
![yt--dlp](https://img.shields.io/badge/Ingestion-yt--dlp%20%2B%20OpenCV-critical?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

## 📌 Overview

This repository contains a CLI-based multimodal Retrieval-Augmented Generation (RAG) assistant that lets you have a conversation with a YouTube video using both channels the video actually communicates through. What's said in the transcript, and what's shown in the frames. It's built as a single script, end-to-end pipeline. Point it at a YouTube URL and it downloads the video, pulls the transcript, samples frames, embeds and indexes both modalities, then opens an interactive chat loop grounded in whichever modality (or both) actually answers the question.

The application runs as a single orchestrated script (`multimodal-youtube-reader.py`). It configures LlamaIndex's global `Settings` for the LLM, text embedding model, node parser, and callbacks, extracts and ingests data only when the target vector collections are empty, builds a `MultiModalVectorStoreIndex` on top of ChromaDB Cloud, and wires that index into a `MultiModalContextChatEngine` with cross-encoder reranking, persistent PostgreSQL-backed memory, and full request-level cost and token observability.

Every answer is retrieved, not memorized, from a combination of transcript chunks and sampled video frames. Every turn also prints exactly which source nodes (text or image) were used to ground the response.

### ✨ Key Features
* 📥 **Automated ingestion pipeline.** Downloads the target video with `yt-dlp`, pulls its transcript with `youtube-transcript-api`, and samples frames at a fixed time interval with OpenCV, all in one pass.
* 🧠 **True multimodal indexing.** Video frames are embedded with CLIP and transcript chunks are embedded with Google's Gemini embedding model. Both are stored as separate collections in ChromaDB Cloud and queried together through a single `MultiModalVectorStoreIndex`.
* ♻️ **Idempotent by design.** On startup the app checks whether the text and image collections already contain data. If they do, it skips re-downloading, re-extracting, and re-indexing entirely, and just loads the existing index.
* 🎯 **Retrieve-then-rerank retrieval.** Pulls the top 5 text nodes and top 3 image nodes per query, then narrows them down with a `cross-encoder/ms-marco-MiniLM-L-2-v2` reranker before they ever reach the LLM.
* 💬 **Persistent conversational memory.** Chat history is stored via `Memory` backed by AsyncPG and PostgreSQL, so context isn't lost between runs of the same session.
* 📊 **Built-in observability and cost tracking.** Every turn reports prompt, completion, and total token counts plus a LiteLLM-computed USD cost estimate, while Arize Phoenix (OpenTelemetry) traces the full request lifecycle for later inspection.
* 🛡️ **Layered, categorized error handling.** Download, frame extraction, transcript fetching, and top-level runtime failures are each caught and routed into clear, human-readable error categories such as network, auth, database, permission, or quota issues, instead of raw stack traces.

---

## 🎯 Context & Problem Statement

This project exists to solve a concrete, everyday research problem rather than to show off a tech stack for its own sake, so it's worth spelling out exactly what that problem is and why a plain transcript chatbot doesn't already solve it.

### 🎬 The Problem

Long-form video, whether it's a tutorial, a conference talk, a product walkthrough, or a lecture, has become one of the densest formats for sharing technical knowledge online, and also one of the hardest to search inside of. A blog post or a PDF can be full-text searched in seconds. A 40-minute video can't, unless you're willing to scrub through the timeline by hand and guess where the part you need actually is, hoping it was even said out loud in the first place. That gap between how much useful information a video actually holds and how hard it is to search inside it is the real problem this project is trying to close, and it breaks down into three more specific issues.

1. **A video conveys information in two very different ways at once, and only one of them is easy to search.** Part of the content is narrated out loud, and part of it only ever appears visually, in a slide, a chart, an on-screen label, or a piece of code shown but not read aloud. When you need one specific detail from a 40-minute video, your only real option today is to scrub through the timeline by hand and hope you land near the right moment.
2. **A transcript-only chatbot has a structural blind spot, not just a quality problem.** Even a well-built RAG system that only indexes the transcript literally has no data to retrieve from for anything that was shown but never spoken. It's not that the retrieval is imperfect, it's that the information was never captured in the first place, so the system either admits it doesn't know or, worse, guesses.
3. **Mixing two data types into one answer makes it hard to trust or verify.** Once you do combine spoken and visual context, a user has no way to tell which part of an answer came from what was said versus what was shown, which makes it harder to sanity-check the answer or jump back to the right point in the source video.

### 💡 The Solution

Rather than bolting a chatbot onto the transcript and calling it a day, this project treats a video as two parallel sources of truth that need to be captured, indexed, and retrieved together, the audio track and the visual track, instead of picking just one and hoping it's enough. Each of the three problems above maps to one deliberate piece of the pipeline, not a generic feature bolted on for its own sake.
* 🗣️ **Problem 1 → chunk and embed the full transcript.** A spoken explanation buried 40 minutes into a video becomes just as retrievable through semantic search as one from the first minute. This still assumes the video actually has captions to pull from, which won't always be true, so it's not a complete fix on its own (see System Limitations).
* 🖼️ **Problem 2 → sample and embed video frames as their own index.** Frames are pulled at a fixed interval and embedded with CLIP as first-class citizens alongside the text, so a question about something that was only ever shown on screen can be grounded in an actual retrieved frame instead of coming back empty or hallucinated. This only works as well as the sampling interval allows, so a very brief on-screen moment between two samples can still slip through (see System Limitations and Future Work).
* 🔎 **Problem 3 → print the exact source nodes behind every answer.** Each source is labeled as a text chunk or an image frame with its similarity score, so a user can immediately see which modality actually answered their question. It's still up to the user to manually match that printed node back to a moment in the video, which is one of the improvements listed under Future Work.

Taken together, these three pieces close the core gap this project set out to solve, but they don't make the pipeline finished. The sections below spell out exactly where the current version still falls short and what's planned to address it.

---

## 🧩 Tech Stack Highlights

| Layer | Tool | Role |
| :--- | :--- | :--- |
| **Video/transcript ingestion** | `yt-dlp`, `youtube-transcript-api`, OpenCV (`cv2`) | Downloads the video, fetches subtitles, samples frames at a fixed interval |
| **Orchestration** | LlamaIndex core (`Settings`, `Document`, `SentenceSplitter`) | Global LLM/embedding config and chunking |
| **Text embeddings** | `GoogleGenAIEmbedding` (`gemini-embedding-2`) | Embeds transcript chunks |
| **Image embeddings** | `ClipEmbedding` (OpenAI CLIP) | Embeds sampled video frames |
| **Vector storage** | ChromaDB Cloud (`ChromaVectorStore`) | One collection for text, one for images |
| **Indexing** | `MultiModalVectorStoreIndex` | Unified multimodal index over both collections |
| **Retrieval** | `MultiModalVectorStoreIndex.as_retriever` + `SentenceTransformerRerank` | Top-k text and image retrieval, then cross-encoder rerank |
| **LLM** | Gemini 2.5 Flash via `LiteLLM` (with a fallback model) | Answer generation |
| **Conversation memory** | `Memory` + AsyncPG (PostgreSQL) | Persistent chat history across turns |
| **Observability** | Arize Phoenix (OpenTelemetry) + `TokenCountingHandler` + `LlamaDebugHandler` | Tracing, token counts, per-turn cost estimation |

---

## 🖼️ Sample Extracted Frames

A few sample frames pulled straight from `extract_frames()` are kept in the [`youtube_image_example/`](youtube_image_example) folder, so you can see what the sampled visual input actually looks like before it gets embedded and indexed.

| ![frame_1.jpg](youtube_image_example/frame_1.jpg) | ![frame_159.jpg](youtube_image_example/frame_159.jpg) | ![frame_238.jpg](youtube_image_example/frame_238.jpg) |
| :---: | :---: | :---: |
| `frame_1.jpg` | `frame_159.jpg` | `frame_238.jpg` |

---

## 📁 Repository Structure

| Path | Description |
| :--- | :--- |
| `multimodal-youtube-reader.py` | The full pipeline. Ingestion, indexing, retrieval, chat engine, and CLI loop, all in one script |
| `requirements.txt` | All Python dependencies, including the ones not obviously implied by the imports (see notes inside the file) |
| `youtube_image_example/` | A handful of sample frames pulled from `extract_frames()`, kept as a reference for what the sampled visual input looks like |
| `LICENSE` | MIT License |
| `README.md` | You're reading it |

---

## 🏗️ Architecture & Data Flow

### 🔄 End-to-End System Flowchart

```mermaid
flowchart TD
    A["YouTube URL(s)"] --> B["download_youtube_video() via yt-dlp"]
    A --> C["load_youtube_transcripts() via youtube-transcript-api"]
    B --> D["extract_frames() via OpenCV, fixed interval"]
    D --> E["SimpleDirectoryReader loads Image Documents"]
    C --> F["Text Documents"]

    subgraph Indexing ["Runs only if both collections are empty"]
        E --> G["ClipEmbedding produces Image Vectors"]
        F --> H["GoogleGenAIEmbedding produces Text Vectors"]
        G --> I["ChromaDB Cloud Image Collection"]
        H --> J["ChromaDB Cloud Text Collection"]
        I --> K["MultiModalVectorStoreIndex"]
        J --> K
    end

    K --> L["Retriever, top 5 text + top 3 image"]
    L --> M["SentenceTransformerRerank narrows to top 6"]
    M --> N["MultiModalContextChatEngine"]
    O["AsyncPG / PostgreSQL Persistent Memory"] --> N
    P["Gemini 2.5 Flash via LiteLLM"] --> N
    N --> Q["Interactive CLI Chat Loop"]
    Q --> R["Answer + Cited Source Nodes + Token/Cost Report"]
    N -.-> S["Arize Phoenix OpenTelemetry Tracing"]
```

> **Note.** On every run, the app first checks whether the ChromaDB text and image collections are already populated. If they are, it skips download, frame extraction, and indexing entirely, and loads the existing index directly. The full ingestion path above only runs once per fresh video.

---

## 💻 Installation & Reproduction Steps

### 📋 Prerequisites
* **Python 3.10 or newer** is required.
* **FFmpeg** must be installed as a system binary, not via pip, and available on your `PATH`. `yt-dlp` needs it to merge the separately-downloaded video and audio streams. Without it, downloads will silently fall back to a lower, unmerged quality or fail outright.
  * macOS, run `brew install ffmpeg`
  * Ubuntu/Debian, run `sudo apt install ffmpeg`
  * Windows, install a build from the [official FFmpeg site](https://ffmpeg.org/download.html) and add it to your `PATH`.
* **Accounts and keys** you'll need include a Google AI Studio API key for Gemini, a [Chroma Cloud](https://www.trychroma.com/) API key, an [Arize Phoenix](https://phoenix.arize.com/) API key, and a reachable PostgreSQL database for conversation memory.

### 🛠️ CLI Installation & Execution

#### 1. Clone the Repository
```bash
git clone https://github.com/viochris/youtube-multimodal-rag-chatbot.git
cd youtube-multimodal-rag-chatbot
```

#### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> The LlamaIndex ecosystem ships as many small, fast-moving sub-packages. If you hit a resolver conflict, try installing into a clean virtual environment and letting `pip` resolve every package from `requirements.txt` in a single command rather than one at a time, since that gives it the full picture needed to pick mutually compatible versions.

#### 4. Configure Environment Variables
Create a `.env` file in the project root.
```env
GOOGLE_API_KEY=your_google_ai_studio_key
CHROMA_API_KEY=your_chroma_cloud_key
PHOENIX_API_KEY=your_arize_phoenix_key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space/v1/traces
ASYNCPG_DATABASE_URL=postgresql://user:password@host:port/dbname
```

#### 5. Set the Target Video
Currently the target video(s) are set directly in the script.
```python
youtube_urls = ["https://www.youtube.com/watch?v=your_video_id"]
```
Edit this list before running to point the pipeline at a different video (see System Limitations below).

#### 6. Run the Application
```bash
python multimodal-youtube-reader.py
```
Type your questions at the `🗣️ [USER] You:` prompt. Type `exit`, `quit`, or `x` to end the session.

---

## 📝 Conclusion

Putting the whole pipeline together, this project set out to solve three concrete problems: long-form video splitting its content across narration and visuals with only one side easy to search, a transcript-only chatbot's structural blind spot for anything that was only ever shown on screen, and the difficulty of trusting or verifying an answer that mixes two very different kinds of source material. The pipeline built here addresses all three directly. It chunks and embeds the full transcript so spoken content stays searchable regardless of where it falls in the video, it samples and embeds video frames as a first-class citizen of the same index so visually-only information has somewhere to be retrieved from, and it prints the exact source nodes behind every answer, labeled by modality and similarity score, so a user can see for themselves what an answer was actually grounded in.

The design choices behind this also hold up reasonably well on their own terms. Treating the ingestion path as idempotent, so re-running the app on the same video skips straight to loading the existing index, keeps repeated use cheap. Reranking retrieved nodes with a cross-encoder before they reach the LLM is a deliberate quality step rather than a shortcut, and pairing that with persistent PostgreSQL-backed memory and full token and cost tracking means the chat behaves like something meant to be used repeatedly, not just demoed once.

That said, this is a solution to the three problems above specifically, not a finished product. It still has real, honest gaps. The video to ingest is hardcoded in the script rather than passed in as an argument, ingestion only re-runs when the ChromaDB collections are completely empty so swapping videos requires manual cleanup, frame sampling is fixed-interval rather than scene-aware so a brief but important visual moment can still be missed, and the whole system depends on four external services being reachable with no offline fallback. The sections immediately below go through each of these gaps in detail and lay out concrete next steps for closing them.

---

## ⚠️ System Limitations

### 🏗️ Architectural Limitations
* **Hardcoded video target.** The video URL(s) to ingest are set by editing the `youtube_urls` list directly in the script, not passed in as an argument. This means switching to a different video means opening the code, not just running a command, and it's easy to forget which video is currently configured. This is addressed by the "CLI arguments for video input" item under Future Work.
* **One-time ingestion gate.** The extraction and indexing path only runs when both the text and image ChromaDB collections are empty, as a way to avoid re-processing the same video on every run. The downside is that swapping in a new video without first manually clearing both collections doesn't trigger re-ingestion at all, so the app will silently keep answering from the old video's data. This is what the "collection versioning" item under Future Work is meant to fix.
* **Synchronous, in-process ingestion.** Downloading the video, extracting frames, and building the index all happen one after another in the same process before the chat loop even starts. There's no progress bar beyond console prints and no way to run a long ingestion in the background, so first-run setup time scales directly with video length and how small `interval_seconds` is set, and a very long video can mean a long wait before you can ask your first question.
* **CLI-only interface.** All interaction happens through a terminal input loop with plain-text prompts and responses. There's no web page, no visual chat history, and no way to click into a cited frame image directly from the terminal. See the "lightweight web UI" item under Future Work.

### 🔬 Model & Domain Limitations
* **Dependence on four external, metered services.** The pipeline needs a live connection to Gemini, ChromaDB Cloud, Arize Phoenix, and a PostgreSQL database just to start up, since each of those environment variables is checked and required before anything else runs. There's currently no offline or local-only mode, so the app simply won't run if any one of those services is unreachable or its API key is invalid.
* **Fixed-interval frame sampling.** Frames are captured every `interval_seconds` of video time regardless of what's actually happening on screen at that moment. A slow, static presentation slide and a fast-cutting montage both get sampled at the same rate, which means a visually important but brief moment, like a chart that's only on screen for two seconds, can fall entirely between two sampled frames and never make it into the index at all.
* **Compute-heavy local inference.** Both the CLIP image embedding step and the cross-encoder reranking step run locally rather than through a hosted API, so their speed depends entirely on the machine running the script. Without a GPU, indexing a video with a large number of sampled frames, or reranking on every single chat turn, can noticeably slow the whole pipeline down.
* **Dormant graph-store imports.** The script imports `Neo4jPropertyGraphStore`, `PropertyGraphIndex`, and `SimpleLLMPathExtractor` at the top, but none of them are actually constructed or used anywhere in the current pipeline. They read as scaffolding left in place for a future entity-and-relationship-graph extension rather than functionality that exists today, and the "activate the graph layer" item under Future Work is exactly about turning that scaffolding into a real feature.
* **Hard dependency on captions existing.** Transcript extraction is not optional. If a video has captions disabled, or simply doesn't have any (auto-generated or otherwise), the ingestion step raises an error and stops before any indexing happens, even if the video's visual content alone might have been enough to answer a question.

---

## 🚀 Future Work
* **CLI arguments for video input.** Right now the target video is a hardcoded list inside the script. Accepting one or more YouTube URLs as a command-line argument or a config value would let anyone point the pipeline at a new video without touching the code at all.
* **Collection versioning.** Because ingestion only runs when the ChromaDB collections are empty, switching videos today means manually deleting the old collections first. Namespacing collections per video, or auto-detecting and clearing stale data, would make swapping videos a one-command operation instead of a manual cleanup step.
* **Scene-aware frame sampling.** Fixed-interval sampling treats every second of video the same regardless of what's happening on screen. Replacing it with keyframe or scene-change detection would sample more frames during visually dense moments and fewer during static ones, catching brief but important visuals that the current fixed interval can miss entirely.
* **Activate the graph layer.** `Neo4jPropertyGraphStore`, `PropertyGraphIndex`, and `SimpleLLMPathExtractor` are already imported but sit unused. Wiring them into the retrieval path would add a layer for entity and relationship questions, like "how are these two people or concepts connected," that plain vector similarity search on transcript chunks and frames isn't well suited to answer on its own.
* **Lightweight web UI.** The current interface is a terminal chat loop with plain-text output. A simple Streamlit or Gradio front end would let a user see cited frame images directly instead of just their file paths, and would make the chat history and source citations easier to read at a glance than scrolling terminal output.
* **Batch and playlist ingestion.** The pipeline currently expects a single video (or a short manually-listed set of URLs) per run. Extending it to accept an entire playlist and ingest every video in it in one pass would make it usable for larger source material, like a full course or a multi-part series, without running the script once per video.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
**Author:** [Silvio Christian Joe](https://github.com/viochris)

*"Answering from what a video actually says, and shows."*
