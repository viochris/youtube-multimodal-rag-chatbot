# ==============================================================================
# 0. REQUIRED LIBRARIES & DEPENDENCIES
# ==============================================================================

# Standard library imports for system operations, logging, and asynchronous execution
import os
import sys
import subprocess
import logging
import traceback
import asyncio
from urllib.parse import urlparse, parse_qs

# Third-party utility libraries for environment management, tokenization, and API interaction
import cv2
import chromadb
import nest_asyncio
import litellm
import tiktoken
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# LlamaIndex core components for document modeling, vector storage, and application context
from llama_index.core import (
    Settings,
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    PropertyGraphIndex,
)
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

# LlamaIndex core modules for nodes, message blocks, and processing
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank

# LlamaIndex core modules for memory, retrieval, and chat engines
from llama_index.core.memory import Memory
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.chat_engine import ContextChatEngine, MultiModalContextChatEngine

# LlamaIndex callbacks and observability schemas
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler
from llama_index.core.callbacks.schema import EventPayload

# LlamaIndex integrations for vector databases, graph stores, generation models, and embeddings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.llms.litellm import LiteLLM

# Phoenix OpenTelemetry instrumentation for distributed tracing and performance observability
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION
# ==============================================================================
# Apply nested asyncio to permit the execution of asynchronous event loops within 
# environments that already manage their own loops (e.g., Jupyter, interactive terminals).
nest_asyncio.apply()

# Suppress verbose underlying library logs (such as HTTP requests and raw API debugs) 
# to maintain a clean, readable Command Line Interface (CLI) for the end user.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(level=logging.WARNING)
logging.getLogger("google_genai").setLevel(level=logging.WARNING)
logging.getLogger("litellm").setLevel(level=logging.WARNING)

# Securely load environment variables from the local .env file.
load_dotenv()

# Validate the presence of critical API keys before proceeding to prevent late runtime failures.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("🚨 [SYSTEM ERROR] GOOGLE_API_KEY is missing. Please check your .env file.")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
if not CHROMA_API_KEY:
    raise ValueError("🚨 [SYSTEM ERROR] CHROMA_API_KEY is missing. Please check your .env file.")

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
if not PHOENIX_API_KEY:
    raise ValueError("🚨 [SYSTEM ERROR] PHOENIX_API_KEY is missing. Please check your .env file.")

# Configure Phoenix OpenTelemetry routing parameters to ingest telemetry traces into the cloud.
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/viochristian12/v1/traces")


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
# Define the target video URLs for data ingestion.
youtube_urls = ["https://www.youtube.com/watch?v=i3OYlaoj-BM"]

def download_youtube_video(youtube_url: str, output_path: str = "video_clip.webm") -> str:
    """
    Downloads a YouTube video using yt-dlp via subprocess.
    Requires yt-dlp to be installed and available in the system PATH.
    """
    try:
        # Validate that a URL was provided before attempting to invoke the CLI tool.
        if not youtube_url:
            raise ValueError("🚨 [SYSTEM ERROR] A valid YouTube URL must be provided for download.")

        # Execute yt-dlp to fetch the best available webm format.
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
                "-o", output_path,
                youtube_url
            ],
            check=True,
            capture_output=True,
            text=True
        )

        return output_path

    except FileNotFoundError:
        # Catch errors where yt-dlp is not installed or not found in the system PATH.
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        raise FileNotFoundError("🚨 [SYSTEM ERROR] 'yt-dlp' executable not found. Ensure it is installed via 'pip install yt-dlp'.")

    except Exception as e:
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        
        # Extract exception attributes to facilitate precise conditional error matching.
        error_type = type(e).__name__
        error_msg = str(e).lower()
        
        print("📌 [ERROR SUMMARY]:")
        # Route specific subprocess or network errors using conditional matching.
        if error_type == "CalledProcessError":
            # Surface the actual stderr output from yt-dlp, since str(e) alone hides the real reason for failure.
            stderr_output = getattr(e, "stderr", "") or "(no stderr captured)"
            raise RuntimeError(f"🚨 [PROCESS ERROR] yt-dlp failed during video download. Details: {stderr_output}")
        elif "timeout" in error_msg or "network" in error_msg:
            raise RuntimeError(f"🚨 [NETWORK ERROR] Connection timed out while downloading the video. Details: {e}")
        elif "permission" in error_msg:
            raise PermissionError(f"🚨 [ACCESS ERROR] System denied write access for output path. Details: {e}")
        else:
            raise RuntimeError(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during download. Details: {e}")

def extract_frames(video_path: str, output_folder: str, interval_seconds: int = 7) -> int:
    """
    Extracts frames from a downloaded video file at specified second intervals using OpenCV.
    """
    try:
        # Validate the existence of the source video file before processing.
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"🚨 [SYSTEM ERROR] The specified video file was not found: {video_path}")

        # Ensure the destination folder exists; create it if necessary.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Initialize the OpenCV VideoCapture object.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"🚨 [SYSTEM ERROR] OpenCV failed to open the video file: {video_path}")

        # Retrieve the Frames Per Second (FPS) metadata from the video.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("⚠️ [SYSTEM WARNING] FPS detected as 0. This might be a corrupted file or an unsupported codec.")
            raise ValueError("🚨 [SYSTEM ERROR] Video FPS is 0. Cannot calculate frame intervals.")

        # Calculate the mathematical interval of frames to skip based on the target seconds.
        frame_interval = max(1, int(fps * interval_seconds))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimated_saves = total_frames // frame_interval
        
        print(f"⚙️ [SYSTEM] FPS: {fps:.1f} | Interval: every {interval_seconds}s | Estimated frames to save: ~{estimated_saves}")
        
        frame_count = 0
        saved_count = 0
        
        # Iterate through the video sequence and extract specific frames based on the calculated interval.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = f"{output_folder}/frame_{saved_count}.jpg"
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                print(f"🖼️ [SYSTEM] Saved frame {saved_count}/{estimated_saves}", end="\r")
            
            frame_count += 1
        
        # Release hardware/memory resources attached to the VideoCapture object.
        cap.release()
        print(f"\n✅ [SYSTEM] Process completed. Total {saved_count} frames saved to: {output_folder}")
        return saved_count

    except Exception as e:
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        
        # Extract exception attributes to facilitate precise conditional error matching.
        error_type = type(e).__name__
        error_msg = str(e).lower()
        
        print("📌 [ERROR SUMMARY]:")
        # Route specific OpenCV or OS-level access errors using conditional matching.
        if error_type == "FileNotFoundError" or "not found" in error_msg:
            raise FileNotFoundError(f"🚨 [FILE ERROR] The specified video file was not found: {video_path}")
        elif error_type == "IOError" or "isopened" in error_msg:
            raise IOError(f"🚨 [IO ERROR] OpenCV failed to open or read the video file. Details: {e}")
        elif "permission" in error_msg:
            raise PermissionError(f"🚨 [ACCESS ERROR] System denied write access to the output folder. Details: {e}")
        else:
            raise RuntimeError(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during frame extraction. Details: {e}")

def load_youtube_transcripts(youtube_urls: list[str]) -> list[Document]:
    """
    Parses YouTube URLs, extracts their subtitles utilizing the YouTubeTranscriptApi, 
    and formats the resulting text into LlamaIndex Document objects with relevant metadata.
    """
    print("📥 [SYSTEM] Fetching YouTube transcripts...")
    try:
        ytt_api = YouTubeTranscriptApi()
        documents = []

        for url in youtube_urls:
            # Parse the video ID dynamically to support both standard and shortened YouTube URL formats.
            if "youtube" in url:
                video_id = url.split("/")[-1].split("v=")[-1].split("&")[0]
            elif "youtu.be" in url:
                video_id = url.split("/")[-1].split("?")[0]
            else:
                parsed_url = urlparse(url=url)

                if "v" in parse_qs(parsed_url.query):
                    print(f"🔍 [SYSTEM] Extracting video ID from URL query parameters: {url}")
                    video_id = parse_qs(parsed_url.query)["v"][0]
                elif parsed_url.netloc in ["youtu.be"]:
                    print(f"🔍 [SYSTEM] Extracting video ID from shortened URL: {url}")
                    video_id = parsed_url.path.lstrip("/")
                else:
                    raise ValueError(f"🚨 [SYSTEM ERROR] Invalid or unsupported YouTube URL format: {url}")

            # Retrieve the transcript chunks and compile them into a single cohesive text block.
            fetched_transcripts = ytt_api.fetch(video_id=video_id)
            # Note: newer versions of youtube_transcript_api return FetchedTranscriptSnippet
            # dataclass objects instead of dicts, so attribute access (.text) is used instead of ["text"].
            full_text = " ".join([snippet.text for snippet in fetched_transcripts])

            # Encapsulate the raw text and contextual metadata into a LlamaIndex Document instance.
            documents.append(Document(text=full_text.strip(), metadata={"video_id": video_id, "url": url}))

        return documents

    except TranscriptsDisabled:
        # Catch errors where the video creator has explicitly disabled subtitles/transcripts.
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        raise RuntimeError("🚨 [SYSTEM ERROR] Transcripts are disabled for this YouTube video.")
        
    except NoTranscriptFound:
        # Catch errors where the video does not have any accessible transcripts (e.g., no spoken audio).
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        raise RuntimeError("🚨 [SYSTEM ERROR] No transcripts were found for this YouTube video.")
        
    except VideoUnavailable:
        # Catch errors where the video is private, deleted, or geoblocked.
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        raise RuntimeError("🚨 [SYSTEM ERROR] The requested YouTube video is currently unavailable.")

    except Exception as e:
        print("\n🔍 [TRACEBACK LOG]:")
        traceback.print_exc()
        
        # Extract exception attributes to facilitate precise conditional error matching.
        error_type = type(e).__name__
        error_msg = str(e).lower()
        
        print("📌 [ERROR SUMMARY]:")
        # Route specific parsing, formatting, or network errors using conditional matching.
        if error_type == "ValueError" and "format" in error_msg:
            raise ValueError(f"🚨 [FORMAT ERROR] The provided YouTube URL format is invalid. Details: {e}")
        elif "connection" in error_msg or "timeout" in error_msg:
            raise RuntimeError(f"🚨 [NETWORK ERROR] Failed to connect to YouTube API. Details: {e}")
        elif "quota" in error_msg or "rate limit" in error_msg:
            raise RuntimeError(f"🚨 [API ERROR] YouTube API rate limit exceeded. Details: {e}")
        else:
            raise RuntimeError(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure during transcript extraction. Details: {e}")

try:
    # ==============================================================================
    # 3. SYSTEM SETTINGS & OBSERVABILITY
    # ==============================================================================
    print("⚙️ [SYSTEM] Configuring LlamaIndex Settings (LLM, Embeddings, & Node Parser)...")

    # Initialize Token Counter utilizing the OpenAI cl100k_base encoding standard.
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.get_encoding("cl100k_base").encode,
        verbose=False
    )

    # Initialize the debug handler to trace LLM inputs and outputs for cost calculations.
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)

    # Establish the global Language Model utilizing LiteLLM as a unified proxy router.
    Settings.llm = LiteLLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.3,
        additional_kwargs={
            "fallbacks": ["gemini/gemini-3.6-flash"],
            "drop_params": True,
            "num_retries": 3
        }
    )
    
    # Establish the global Semantic Embedding Model for text.
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-2", 
        api_key=GOOGLE_API_KEY, 
        embed_batch_size=10
    )
    
    # Configure the document parsing strategy to segment large texts.
    Settings.node_parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    
    # Attach the observability callbacks to the global Settings manager.
    Settings.callback_manager = CallbackManager([token_counter, debug_handler])

    # Initialize the dedicated embedding model for image vectorization.
    print("⚙️ [SYSTEM] Initializing CLIP Embedding model for image processing...")
    image_embed_model = ClipEmbedding()

    print("📊 [SYSTEM] Initializing Phoenix Observability & Tracing...")
    # Register the OpenTelemetry provider to route telemetry data to Arize Phoenix Cloud.
    tracer_provider = register(
        project_name="youtube_multimodal_project",
        auto_instrument=True
    )

    # Instrument LlamaIndex to dispatch tracing data automatically during query execution.
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

    print("⚙️ [SYSTEM] Initializing Cross-Encoder Reranker...")
    # Configure the cross-encoder model to refine and re-score retrieved nodes.
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=6
    )

    # ==============================================================================
    # 4. DATABASE INITIALIZATION & DATA INGESTION
    # ==============================================================================
    print("⏳ [SYSTEM] Connecting to ChromaDB Cloud Client...")
    
    # Initialize the ChromaDB vector storage client.
    chroma_client = chromadb.CloudClient(api_key=CHROMA_API_KEY)
    
    # Retrieve or create designated collections for text and images.
    chroma_text_collection = chroma_client.get_or_create_collection("youtube_text_collection")
    chroma_image_collection = chroma_client.get_or_create_collection("youtube_image_collection")

    # Instantiate ChromaVectorStore adapters.
    print("🔌 [SYSTEM] Attaching ChromaVectorStore adapters for Multimodal data...")
    text_vector_store = ChromaVectorStore(chroma_collection=chroma_text_collection)
    image_vector_store = ChromaVectorStore(chroma_collection=chroma_image_collection)

    # Establish storage context utilizing the text vector store as the primary default.
    text_storage_context = StorageContext.from_defaults(vector_store=text_vector_store)

    # Execute data extraction and indexing exclusively if both collections are empty.
    if (chroma_text_collection.count() == 0) and (chroma_image_collection.count() == 0):
        print("⏳ [SYSTEM] Both databases are empty. Initiating extraction phase...")
        
        # Extract transcripts from the provided YouTube URLs.
        text_documents = load_youtube_transcripts(youtube_urls=youtube_urls)

        # Check if the video file is already downloaded to avoid redundant network requests.
        if not os.path.exists("video_clip.webm"):
            print("⏳ [SYSTEM] Video file not found locally. Initiating download...")
            downloaded_path = download_youtube_video(youtube_url=youtube_urls[0])
        else:
            print("📂 [SYSTEM] Video file 'video_clip.webm' already exists. Skipping download.")
            downloaded_path = "video_clip.webm"

        # Define the output directory for storing extracted video frames.
        youtube_dir_path = "youtube_image"
        
        # Check if the image directory exists to avoid redundant frame extraction operations.
        if not os.path.exists(youtube_dir_path) or len(os.listdir(youtube_dir_path)) == 0:
            print("⏳ [SYSTEM] Image directory not found. Extracting frames from video...")
            total_frames = extract_frames(video_path=downloaded_path, output_folder=youtube_dir_path)
            print(f"📊 [SYSTEM] Total frames extracted: {total_frames}")
        else:
            print("📂 [SYSTEM] Image directory already exists. Skipping frame extraction.")
            # Calculate the total number of existing frames by measuring the directory list length.
            total_frames = len(os.listdir(youtube_dir_path))
            print(f"📊 [SYSTEM] Total existing frames found: {total_frames}")

        # Load the extracted image frames as LlamaIndex multimodal documents.
        print("📂 [SYSTEM] Loading image documents from directory...")
        image_documents = SimpleDirectoryReader(youtube_dir_path).load_data()

        # Validate that data was successfully extracted.
        if not text_documents:
            raise ValueError("🚨 [DATA ERROR] Failed to extract text transcripts from the provided YouTube URLs.")

        if not image_documents:
            raise ValueError("🚨 [DATA ERROR] Failed to extract image frames from the downloaded video.")

        # Combine multimodal documents.
        all_documents = text_documents + image_documents

        print("⚙️ [SYSTEM] Indexing multimodal documents to ChromaDB...")
        # Generate the Multimodal Vector Store Index and persist into ChromaDB.
        index = MultiModalVectorStoreIndex.from_documents(
            documents=all_documents,
            storage_context=text_storage_context,
            image_vector_store=image_vector_store,
            image_embed_model=image_embed_model
        )
    
    else:
        print("📂 [SYSTEM] Databases contain existing data. Bypassing extraction phase.")
        
        if chroma_text_collection.count() != 0:
            print(f"📊 [SYSTEM] Found {chroma_text_collection.count()} items in text collection.")
        else:
            print("⚠️ [SYSTEM] Text collection is currently empty.")

        if chroma_image_collection.count() != 0:
            print(f"📊 [SYSTEM] Found {chroma_image_collection.count()} items in image collection.")
        else:
            print("⚠️ [SYSTEM] Image collection is currently empty.")
            
        print("⚙️ [SYSTEM] Loading existing Multimodal Index from ChromaDB...")
        # Load the existing index directly from the initialized vector stores.
        index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=text_vector_store,
            image_vector_store=image_vector_store,
            image_embed_model=image_embed_model
        )

    print("✅ [SYSTEM] Multimodal Index successfully loaded!")

    # ==============================================================================
    # 5. RETRIEVER & MULTIMODAL CHAT ENGINE SETUP
    # ==============================================================================
    print("🧠 [SYSTEM] Initializing Multimodal Chat Engine and PostgreSQL memory...")
    
    # Instantiate the dual retriever for semantic text and image similarity.
    retriever = index.as_retriever(
        similarity_top_k=5,
        image_similarity_top_k=3
    )

    # Establish a persistent conversational memory backend using AsyncPG.
    memory = Memory.from_defaults(
        session_id="youtube_multimodal_session",
        token_limit=40000,
        async_database_uri=os.getenv("ASYNCPG_DATABASE_URL"),
        table_name="youtube_multimodal_memory"
    )

    # Define the operational boundaries and strict behavioral constraints for the AI Agent.
    ai_system_prompt = (
        "You are an intelligent multimodal assistant. "
        "Your primary objective is to answer user queries strictly based on the provided context retrieved from both the text transcripts and the video frames. "
        "Provide clear, comprehensive, and well-structured explanations. "
        "If the answer cannot be deduced from the context, explicitly state that you do not have the information."
    )

    # Construct the comprehensive Multimodal Context Chat Engine.
    chat_engine = MultiModalContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=ai_system_prompt,
        llm=Settings.llm,
        node_postprocessors=[reranker]
    )
    print("✅ [SYSTEM] Multimodal Chat Engine successfully integrated and ready for interaction.")

    # ==============================================================================
    # 6. INTERACTIVE ASYNCHRONOUS CHAT LOOP
    # ==============================================================================
    async def main():
        print("\n" + "="*70)
        print("💬 [SYSTEM] Interactive Multimodal Session Started. Type 'exit' to stop.")
        print("="*70)

        # Clear existing buffers prior to starting the session.
        token_counter.reset_counts()
        debug_handler.flush_event_logs()

        while True:
            # Capture continuous user input from the terminal CLI.
            user_input = input("\n🗣️ [USER] You: ").strip()

            # Process session termination commands gracefully.
            if user_input.lower() in ["exit", "close", "out", "break", "quit", "x"]:
                print("🛑 [SYSTEM] Terminating session. Goodbye!")
                break

            # Bypass engine execution if the input is entirely empty.
            if not user_input:
                print("⚠️ [SYSTEM] Input cannot be empty. Please ask a question.")
                continue

            # Execute the asynchronous query utilizing the multimodal context retrieval.
            print("🧠 [AI] Querying text transcripts and image frames...")
            ai_response = await chat_engine.achat(user_input)

            # Initialize variables to compute inference costs based on LiteLLM's event payload.
            total_cost = 0.0
            model_used = "Unknown"

            # Retrieve tracked inputs and outputs from the LlamaDebugHandler.
            llm_events = debug_handler.get_llm_inputs_outputs()
            
            # Iterate through the captured LLM events safely to accumulate completion costs.
            for start_event, end_event in llm_events:
                response_obj = end_event.payload.get(EventPayload.RESPONSE)
                if response_obj is not None and getattr(response_obj, "raw", None):
                    response_raw = response_obj.raw
                    total_cost += litellm.completion_cost(completion_response=response_raw)
                    model_used = response_raw.get("model", "Unknown")

            # Validate the existence of an AI response to prevent downstream rendering errors.
            if not ai_response:
                print("⚠️ [SYSTEM] AI failed to generate a response or no relevant context found.")
                continue

            # Render the final synthesized AI response to the terminal output.
            print("-" * 70)
            print("🤖 [AI RESPONSE]:")
            print(ai_response)
            print("-" * 70)

            # Extract and display the underlying source nodes evaluated during the RAG generation phase.
            source_nodes = ai_response.source_nodes
            print(f"\n🔍 [SOURCE NODES] Retrieved {len(source_nodes)} relevant nodes from multimodal databases:")
            
            # Iterate through the retrieved source nodes and dynamically identify their data type.
            for i, node in enumerate(source_nodes, 1):
                node_type = type(node.node).__name__
                
                # Print standard metadata applicable to all node types.
                print(f"--- Node {i} | Type: {node_type} | Score: {node.score:.4f} ---")
                
                # Route content extraction based on the specific node modality.
                if node_type == "ImageNode":
                    print(f"🖼️ Path: {node.node.image_path}")
                else:
                    print(f"📄 Content: {node.get_content()}")
                    
                # Append a trailing newline for visual separation between nodes.
                print()

            # ==============================================================================
            # 7. TOKEN USAGE & COST INSPECTION
            # ==============================================================================
            print("\n📊 [LLAMAINDEX CALLBACK TOKEN OVERVIEW]:")
            print(f"Model Used        : {model_used}")
            print("------------------+")
            print(f"Prompt Tokens     : {token_counter.prompt_llm_token_count}")
            print(f"Completion Tokens : {token_counter.completion_llm_token_count}")
            print("------------------+")
            print(f"Total LLM Tokens  : {token_counter.total_llm_token_count}")
            print("=" * 70)

            print(f"\n💰 [COST ANALYSIS]:")
            print(f"Total Estimated Cost: ${total_cost:.6f} USD")
            print("=" * 70)

            # Reset token counters and flush debug logs to prevent event overlap in the next loop iteration.
            token_counter.reset_counts()
            debug_handler.flush_event_logs()

    # Execute the asynchronous event loop.
    if __name__ == "__main__":
        asyncio.run(main())

# ==============================================================================
# 8. EXCEPTION HANDLING & ERROR ROUTING
# ==============================================================================
except Exception as e:
    # Extract exception attributes to facilitate precise conditional error matching.
    error_type = type(e).__name__
    error_msg = str(e).lower()
    error_raw = str(e)

    print("\n" + "="*70)
    print("💥 [CRITICAL FAILURE] Agent execution aborted!")
    print("="*70)
    
    # Print the comprehensive traceback stack for deep developer debugging.
    print("🔍 [TRACEBACK LOG]:")
    traceback.print_exc()
    print("-" * 70)

    # Output a structured, human-readable root cause summary for CLI diagnostics.
    print("📌 [ERROR SUMMARY]:")
    
    # Catch Google API quota limitations or rate limits (HTTP 429).
    if error_type == "ResourceExhausted" or "quota" in error_msg or "429" in error_msg:
        print(f"🚨 [API ERROR] {error_type}: Google API quota exceeded or rate limited. Details: {error_raw}")
        
    # Catch data deficiency errors such as missing context or empty initialization data.
    elif error_type == "ValueError" and "empty" in error_msg:
        print(f"🚨 [DATA ERROR] {error_type}: A data source is empty or missing valid inputs. Details: {error_raw}")
        
    # Catch authentication rejections from LiteLLM or upstream providers.
    elif error_type == "InvalidArgument" or "api_key" in error_msg:
        print(f"🚨 [AUTH ERROR] {error_type}: Invalid API Key configuration. Details: {error_raw}")
        
    # Catch ChromaDB cloud connection or initialization failures.
    elif error_type in ["OperationalError", "DatabaseError", "InvalidCollectionException"] or "chroma" in error_msg:
        print(f"🚨 [DATABASE ERROR] {error_type}: ChromaDB Cloud connection failed. Details: {error_raw}")
        
    # Catch AsyncPG (PostgreSQL) persistent memory connection failures.
    elif "asyncpg" in error_msg or "postgres" in error_msg:
        print(f"🚨 [DATABASE ERROR] {error_type}: Failed to connect or write to the PostgreSQL memory database. Details: {error_raw}")
        
    # Catch OS-level file system or network permission blocks.
    elif error_type == "PermissionError":
        print(f"🚨 [ACCESS ERROR] {error_type}: System denied access. Details: {error_raw}")
        
    # Default fallback routing for unhandled or unexpected exceptions.
    else:
        print(f"🚨 [UNKNOWN ERROR] {error_type}: Unexpected failure. Details: {error_raw}")
        
    print("="*70 + "\n")
    
    # Terminate the application with a non-zero exit code to signal execution failure to the OS.
    sys.exit(1)