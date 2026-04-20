"""
Video-based Chat Ingestion Pipeline (Sentence-wise Timestamping)

Flow:
1. Transcribe video with sentence-level timestamps
2. Generate embeddings for each sentence
3. Store in Vector DB with exact timestamps
4. Supports video-based chat interfaces (jump-to-video UX)

Dependencies:
  pip install openai-whisper sentence-transformers chromadb ffmpeg-python
  (ffmpeg must be installed on the system)
"""

import os
import shutil

FFMPEG_BIN = r"C:\Users\subrata_ghosh\Downloads"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]

print("ffmpeg :", shutil.which("ffmpeg"))
print("ffprobe:", shutil.which("ffprobe"))

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
import whisper
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict


# -----------------------------
# CONFIG
# -----------------------------
WHISPER_MODEL = "base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_DIR = "./video_vector_db"
COLLECTION_NAME = "video_chat_collection"


# -----------------------------
# STEP 1: TRANSCRIBE VIDEO
# (Sentence-wise timestamps)
# -----------------------------
def transcribe_video(video_path: str) -> List[Dict]:
    """
    Returns sentence-level transcription with timestamps.
    Whisper treats each segment as a sentence/phrase.
    """
    print("🔊 Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL)

    print(f"🎬 Transcribing video: {video_path}")
    result = model.transcribe(video_path, verbose=False)

    sentences = []
    for i, seg in enumerate(result["segments"]):
        sentences.append({
            "sentence_id": i,
            "text": seg["text"].strip(),
            "start_time": round(seg["start"], 2),
            "end_time": round(seg["end"], 2),
        })

    return sentences


# -----------------------------
# STEP 2: CREATE EMBEDDINGS
# -----------------------------
def embed_sentences(sentences: List[str]) -> List[List[float]]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(sentences).tolist()


# -----------------------------
# STEP 3: STORE IN VECTOR DB
# -----------------------------
def store_sentences_in_vector_db(
    video_id: str,
    sentences: List[Dict]
):
    print("🧠 Initializing Vector DB...")
    client = chromadb.Client(
        Settings(persist_directory=VECTOR_DB_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    texts = [s["text"] for s in sentences]
    embeddings = embed_sentences(texts)

    metadatas = [
        {
            "video_id": video_id,
            "sentence_id": s["sentence_id"],
            "start_time": s["start_time"],
            "end_time": s["end_time"]
        }
        for s in sentences
    ]

    ids = [
        f"{video_id}_sentence_{s['sentence_id']}"
        for s in sentences
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Stored {len(sentences)} sentences for video_id='{video_id}'")


# -----------------------------
# STEP 4: FULL INGESTION PIPELINE
# -----------------------------
def ingest_video_for_chat(video_path: str, video_id: str):
    sentences = transcribe_video(video_path)
    store_sentences_in_vector_db(video_id, sentences)


# -----------------------------
# STEP 5: QUERY RETRIEVAL
# -----------------------------
def query_video_chat(
    query: str,
    video_id: str,
    top_k: int = 3
):
    """
    Query the vector DB and retrieve best matching video sentences
    with timestamps.
    """

    print(f"🔍 Querying video '{video_id}' for: '{query}'")

    # Load embedding model
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Embed the query
    query_embedding = embedder.encode(query).tolist()

    # Load vector DB
    client = chromadb.Client(
        Settings(persist_directory=VECTOR_DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Query the DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"video_id": video_id},
        include=["documents", "metadatas", "distances"]
    )

    responses = []

    for i in range(len(results["documents"][0])):
        responses.append({
            "text": results["documents"][0][i],
            "score": round(results["distances"][0][i], 4),
            "start_time": results["metadatas"][0][i]["start_time"],
            "end_time": results["metadatas"][0][i]["end_time"],
        })

    return responses

# -----------------------------
# STEP 6: JUMP-TO-VIDEO HELPERS
# -----------------------------
def seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def build_jump_link(video_url: str, start_time: float) -> str:
    """
    Builds a jump-to-time URL for web players (YouTube-style).
    """
    return f"{video_url}?t={int(start_time)}"
# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    VIDEO_PATH = "C:\\Work\\GenAI\\SG\\POC_RAG\\src_bakup\\tmp.wav"   # <-- your video file
    VIDEO_ID = "video_chat_001"       # <-- unique identifier

    ingest_video_for_chat(
        video_path=VIDEO_PATH,
        video_id=VIDEO_ID
    )

    print("🎉 Video is ready for chat-based retrieval!")

    # -----------------------------
    # QUERY DEMO
    # -----------------------------

    QUERY = "What is being discussed about hop craft buns?"
    VIDEO_ID = "video_chat_001"

    results = query_video_chat(
        query=QUERY,
        video_id=VIDEO_ID,
        top_k=3
    )

    print("\n📌 Top Matches:\n")
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] {r['text']}")
        print(f"    ⏱ {seconds_to_hhmmss(r['start_time'])} → {seconds_to_hhmmss(r['end_time'])}")
        print(f"    🎯 Similarity score: {r['score']}")
        print("-" * 60)
