import streamlit as st
import os
import uuid

from video_chatbot import (
    ingest_video_for_chat,
    query_video_chat,
    seconds_to_hhmmss
)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Video Chatbot", layout="wide")

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_ingested" not in st.session_state:
    st.session_state.video_ingested = False

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🎬 Video Chatbot (Ingest Once, Query Anytime)")

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
st.header("📤 Upload Video / Audio")

uploaded_file = st.file_uploader(
    "Upload MP4 / WAV / MP3",
    type=["mp4", "wav", "mp3"]
)

if uploaded_file and not st.session_state.video_ingested:
    unique_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.video_id = f"video_{unique_id}"
    st.session_state.video_path = file_path

    st.video(file_path)

    if st.button("Ingest Video "):
        with st.spinner("Ingesting video..."):
            ingest_video_for_chat(
                video_path=st.session_state.video_path,
                video_id=st.session_state.video_id,
            )

        st.session_state.video_ingested = True
        st.success("✅ Video ingested successfully!")

# -----------------------------
# QUERY SECTION (MULTI-QUERY)
# -----------------------------
if st.session_state.video_ingested:

    st.header("💬 Ask Question")

    query = st.text_input(
        "Ask anything about the video",
        placeholder="e.g. Where is system architecture explained?"
    )

    if query:
        with st.spinner("Searching relevant moments..."):
            results = query_video_chat(
                query=query,
                video_id=st.session_state.video_id,
                top_k=3
            )

        if not results:
            st.warning("No relevant moment found.")
        else:
            st.subheader("📌 Relevant Moments")

            for idx, r in enumerate(results, 1):
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown(f"**Result {idx}**")
                    st.write(r["text"])
                    st.caption(
                        f"⏱ {seconds_to_hhmmss(r['start_time'])} → "
                        f"{seconds_to_hhmmss(r['end_time'])}"
                    )

                with col2:
                    st.video(
                        st.session_state.video_path,
                        start_time=int(r["start_time"])
                    )

                st.divider()

    # -----------------------------
    # RESET OPTION
    # -----------------------------
    if st.button(" Upload & Ingest New Video"):
        st.session_state.video_id = None
        st.session_state.video_path = None
        st.session_state.video_ingested = False
        st.rerun()