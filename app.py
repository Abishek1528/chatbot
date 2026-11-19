import os

import streamlit as st
from dotenv import load_dotenv

from rag_backend import PDFRAGAssistant

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS layer to match the dark dashboard shown in the reference image.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.08); }
      .app-subtitle { color: #9CA3AF; font-size: .88rem; margin-top: -.55rem; }
      .welcome-box { background: #202C40; border: 1px solid #2C3B55; padding: 13px 16px;
                     border-radius: 8px; margin: 18px 0; }
      .status-box { background: #103A2D; color: #9AE6C2; padding: 10px 12px;
                    border-radius: 7px; font-size: .9rem; }
      div[data-testid="stChatMessage"] { border-radius: 10px; }
      .small-muted { color: #9CA3AF; font-size: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("## 📑 PDF RAG Assistant")
    st.caption("Your intelligent document assistant")
    st.divider()

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF that contains selectable text.",
    )

    if uploaded_file is not None:
        st.caption(f"📄 {uploaded_file.name} · {uploaded_file.size / 1024:.1f} KB")

    process_clicked = st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if process_clicked and uploaded_file is not None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY is missing. Add it to your .env file and restart Streamlit.")
        else:
            try:
                with st.spinner("Reading PDF, creating embeddings and indexing it..."):
                    assistant = PDFRAGAssistant(google_api_key=api_key)
                    chunk_count = assistant.process_pdf(
                        uploaded_file.getvalue(), uploaded_file.name
                    )
                st.session_state.assistant = assistant
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []
                st.session_state.last_evidence = []
                st.success(f"PDF processed! Created {chunk_count} text chunks.")
            except Exception as exc:
                st.session_state.assistant = None
                st.error(f"Could not process the PDF: {exc}")

    st.divider()
    st.markdown("### Document Status")
    if st.session_state.assistant is not None:
        st.markdown(
            f'<div class="status-box">● Document ready</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"📄 {st.session_state.processed_file}")
        if st.button("Clear document", use_container_width=True):
            st.session_state.assistant = None
            st.session_state.processed_file = None
            st.session_state.messages = []
            st.session_state.last_evidence = []
            st.rerun()
    else:
        st.info("No document processed yet.")

# ------------------------------ Main chat ------------------------------
st.markdown("## 🌐 Ask Your PDF")
st.markdown(
    '<p class="app-subtitle">Upload a document and get intelligent answers based on its content.</p>',
    unsafe_allow_html=True,
)

if st.session_state.assistant is None:
    st.markdown(
        '<div class="welcome-box">👋 Welcome! Upload a PDF from the sidebar and click '
        '<b>Process PDF</b> to start asking questions.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="welcome-box">📘 Ready to answer questions from '
        f'<b>{st.session_state.processed_file}</b>.</div>',
        unsafe_allow_html=True,
    )

# Show existing conversation after a rerun.
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("evidence"):
            with st.expander("🔎 View Retrieved Evidence"):
                for item in message["evidence"]:
                    st.markdown(f"**Page {item['page']}** · `{item['source']}`")
                    st.write(item["text"])
                    if index < len(st.session_state.messages) - 1:
                        st.divider()

question = st.chat_input(
    "Ask something about your PDF...",
    disabled=st.session_state.assistant is None,
)

if question:
    previous_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer..."):
            try:
                result = st.session_state.assistant.ask_question(
                    question,
                    chat_history=previous_history,
                )
                answer = result["answer"]
                evidence = result["evidence"]
                st.markdown(answer)
                with st.expander("🔎 View Retrieved Evidence"):
                    if evidence:
                        for item in evidence:
                            st.markdown(f"**Page {item['page']}** · `{item['source']}`")
                            st.write(item["text"])
                            st.divider()
                    else:
                        st.write("No matching passages were retrieved.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "evidence": evidence}
                )
            except Exception as exc:
                error_message = f"Something went wrong: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "evidence": []}
                )
