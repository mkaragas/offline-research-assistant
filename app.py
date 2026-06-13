#!/usr/bin/env python3
# Offline Research Assistant — Streamlit web interface
# Copyright (C) 2026 mkaragas
#
# Generated with Claude (Anthropic) to the author's specifications.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
app.py - Streamlit web UI for offline ZIM + PDF research.

Run it with:
    streamlit run app.py

Set your library location in the sidebar (e.g. E:\\ on Windows) and start asking.
It scans for both .zim and .pdf files. Everything runs locally against Ollama.
"""

import streamlit as st

from zim_rag import (
    Researcher,
    discover_sources,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
)

st.set_page_config(page_title="Offline Research", page_icon="📚", layout="centered")
st.title("📚 Offline ZIM + PDF Research")
st.caption("Ask questions about your offline Wikipedia / ZIM library and your PDF "
           "collection using a local Ollama model. Nothing leaves your machine.")

# --------------------------------------------------------------------------- #
# Sidebar configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    lib_path = st.text_input("Library location", value="E:\\",
                             help="A .zim/.pdf file, a folder, or a drive root. "
                                  "Scans for both ZIM and PDF.")
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL,
                               help="Name from `ollama list`, e.g. gemma3 or llama3.1.")
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL,
                                help="e.g. nomic-embed-text (run `ollama pull nomic-embed-text`).")
    host = st.text_input("Ollama host (optional)", value="",
                         help="Leave blank for the default localhost:11434.")
    col1, col2 = st.columns(2)
    use_zim = col1.checkbox("ZIM", value=True)
    use_pdf = col2.checkbox("PDF", value=True)
    top_k = st.slider("Context passages (top-k)", 1, 12, 5)
    max_articles = st.slider("Max candidate passages", 3, 30, 12)
    rebuild = st.checkbox("Rebuild PDF index", value=False,
                          help="Re-read & re-embed all PDFs (after adding new files).")
    load = st.button("Load / reload library", type="primary")


@st.cache_resource(show_spinner=False)
def get_researcher(path, chat, embed, host, use_zim, use_pdf, rebuild, _nonce):
    zims, pdfs = discover_sources(path)
    if not use_zim:
        zims = []
    if not use_pdf:
        pdfs = []
    if not zims and not pdfs:
        raise FileNotFoundError(f"No .zim or .pdf files found at: {path}")

    status = st.empty()

    def progress(done, total, msg):
        status.info(f"Indexing PDFs [{done}/{total}] — {msg}")

    researcher = Researcher(
        zim_paths=zims,
        pdf_paths=pdfs,
        chat_model=chat,
        embed_model=embed,
        ollama_host=host or None,
        rebuild_pdf_index=rebuild,
        pdf_progress=progress if pdfs else None,
    )
    status.empty()
    return researcher, zims, pdfs


if "nonce" not in st.session_state:
    st.session_state.nonce = 0
if load:
    st.session_state.nonce += 1
    get_researcher.clear()

researcher = None
try:
    with st.spinner("Opening library..."):
        researcher, zims, pdfs = get_researcher(
            lib_path, chat_model, embed_model, host, use_zim, use_pdf,
            rebuild, st.session_state.nonce,
        )
    st.sidebar.success(f"Loaded {len(zims)} ZIM + {len(pdfs)} PDF file(s).")
    with st.sidebar.expander("Files"):
        for z in zims:
            st.write(f"📘 {z}")
        for d in pdfs:
            st.write(f"📄 {d}")
    if researcher.pdf_lib and researcher.pdf_lib.skipped:
        st.sidebar.warning(
            "Some PDFs had no extractable text (likely scanned images) and were "
            "skipped — they'd need OCR to be searchable:\n\n"
            + "\n".join(f"• {p}" for p in researcher.pdf_lib.skipped)
        )
except Exception as e:
    st.sidebar.error(str(e))

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for title, zim_name in m["sources"]:
                    st.write(f"• **{title}**  _( {zim_name} )_")

question = st.chat_input("Ask a question about your offline library (ZIM + PDF)...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if researcher is None:
        with st.chat_message("assistant"):
            st.error("Load a valid library in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            collected = []
            answer_obj = None
            with st.spinner("Searching your library..."):
                try:
                    for kind, payload in researcher.answer(
                        question, top_k=top_k, max_articles=max_articles, stream=True
                    ):
                        if kind == "token":
                            collected.append(payload)
                            placeholder.markdown("".join(collected) + "▌")
                        elif kind == "done":
                            answer_obj = payload
                    placeholder.markdown("".join(collected))
                except Exception as e:
                    placeholder.error(f"Error: {e}")

            sources = answer_obj.sources() if answer_obj else []
            if sources:
                with st.expander("Sources"):
                    for title, zim_name in sources:
                        st.write(f"• **{title}**  _( {zim_name} )_")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "".join(collected),
                "sources": sources,
            })
