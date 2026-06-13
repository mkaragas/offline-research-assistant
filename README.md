# Offline Research Assistant 📚

Ask questions about your **offline** knowledge library and get grounded, cited answers
from a **local** AI model. Two kinds of sources are supported side by side:

- **ZIM files** — offline Wikipedia, Wiktionary, Stack Overflow, Project Gutenberg, and
  anything else from the [Kiwix](https://library.kiwix.org) library.
- **PDF files** — your own books, papers, manuals, and booklets.

Everything runs on your machine through [Ollama](https://ollama.com). No internet
connection is needed once your models and content are downloaded, and nothing you ask
ever leaves your computer.

> **Why "offline"?** This is a Retrieval-Augmented Generation (RAG) pipeline that reads
> your local files, finds the passages most relevant to your question, and hands them to
> a local language model to write a cited answer. It's built for air-gapped, private, or
> low-connectivity use.

---

## Features

- 🔌 **Fully local & private** — uses your own Ollama install for both embeddings and chat.
- 📖 **ZIM + PDF in one answer** — a single question can draw on Wikipedia *and* your PDFs.
- ⚡ **Scales to full Wikipedia** — ZIM files are queried through their own built-in
  full-text index, so there's no need to embed millions of articles up front.
- 🧠 **Smart PDF indexing** — PDFs are embedded once and cached; only new or changed files
  are re-processed when you add to your library.
- 🔁 **Interruptible indexing** — long indexing runs save progress and resume where they
  left off.
- 🗂️ **Cited sources** — answers reference the article title (ZIM) or file name + page
  number (PDF) behind each claim.
- 💻 **CLI and Web UI** — a command-line tool and a Streamlit app.

---

## How it works

```
                       your question
                            |
        +-------------------+-------------------+
        |                                       |
   ZIM sources                             PDF sources
   query each ZIM's built-in full-text     semantic search over a prebuilt
   index (full question + each keyword,    embedding index of your PDFs
   then union the hits) -> candidate       (built once, cached on disk)
   articles -> extract & chunk text
        |                                       |
        +-------------------+-------------------+
                            |
                    unified ranking
        embed the question, score every candidate chunk by
        cosine similarity, keep the best top-k across BOTH types
                            |
                            v
                       generation
        feed the top passages to your local chat model with a
        "use only this context and cite it" instruction
                            |
                            v
              grounded answer + sources
```

The two source types are handled differently on purpose. **ZIM** files ship with a
built-in [Xapian](https://xapian.org) full-text index, so they need no preprocessing and
scale to enormous dumps. **PDFs** have no built-in index, so the app builds and caches one.

> **Note:** ZIM is *not* a ZIP archive — it's the openZIM/Kiwix format, read here with the
> official `libzim` library, not Python's `zipfile`.

---

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com)**, installed and running
- A **chat model** and an **embedding model** pulled into Ollama:

  ```bash
  ollama pull gemma3            # or llama3.1, mistral, qwen3 — your choice of chat model
  ollama pull nomic-embed-text  # embedding model used for ranking and PDF search
  ```

  Check what you have with `ollama list`.

- At least one `.zim` and/or `.pdf` file to search.

---

## Installation

```bash
git clone https://github.com/<your-username>/offline-research-assistant.git
cd offline-research-assistant
pip install -r requirements.txt
```

`requirements.txt`:

```text
libzim>=3.4.0        # read ZIM files (Wikipedia, etc.)
pypdf>=4.0.0         # extract text from PDF files
numpy>=1.24.0        # fast cosine search over the PDF embedding index
ollama>=0.3.0        # local LLM + embeddings client
streamlit>=1.30.0    # optional: the web UI in app.py
```

---

## Usage

### Command line

```bash
# Ask against EVERYTHING (.zim and .pdf) found in a folder or drive:
python ask.py --path /path/to/library "Why did the Western Roman Empire collapse?"

# Windows example (a whole drive):
python ask.py --path E:\ --model gemma3 "Explain quantum entanglement"

# Combine specific locations (repeat --path):
python ask.py --path "E:\wikipedia.zim" --path "E:\papers" "Summarize the key findings"

# PDFs only, forcing a fresh index after adding new files:
python ask.py --path E:\papers --no-zim --rebuild-index "What does chapter 3 argue?"

# Interactive mode — just leave the question off:
python ask.py --path /path/to/library
```

### Web UI

```bash
streamlit run app.py
```

Set your library location (e.g. `E:\`), tick **ZIM** / **PDF** as desired, and click
**Load / reload library**. Answers stream in and list their sources.

### Command-line options

| Flag | Description | Default |
| --- | --- | --- |
| `--path` (alias `--zim`) | A file, folder, or drive root. Repeatable. Scans for both `.zim` and `.pdf`. | *(required)* |
| `--model` | Ollama chat model (see `ollama list`) | `gemma3` |
| `--embed-model` | Ollama embedding model | `nomic-embed-text` |
| `--host` | Ollama host URL | `localhost:11434` |
| `--top-k` | Number of passages fed to the model | `5` |
| `--max-articles` | Candidate passages pulled per question | `12` |
| `--no-zim` / `--no-pdf` | Restrict to one source type | both on |
| `--rebuild-index` | Re-read and re-embed all PDFs | off |
| `--pdf-index` | Where to store the PDF index | `.pdf_index.pkl` |
| `--no-cache` | Disable the embedding cache | off |

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `ZIM_CHAT_MODEL` | Default chat model | `gemma3` |
| `ZIM_EMBED_MODEL` | Default embedding model | `nomic-embed-text` |
| `OLLAMA_HOST` | Ollama host URL | unset (localhost) |
| `ZIM_EMBED_BATCH` | Chunks per embedding request | `64` |

---

## Project structure

| File | Purpose |
| --- | --- |
| `zim_rag.py` | Core: retrieval, chunking, embedding, ranking, generation (`Researcher` class) |
| `pdf_index.py` | PDF discovery, text extraction, and the persistent embedding index |
| `ask.py` | Command-line interface |
| `app.py` | Streamlit web interface |
| `requirements.txt` | Python dependencies |

The app writes two cache files next to where you run it: `.pdf_index.pkl` (the PDF
embedding index) and `.zim_embed_cache.json` (cached question/article embeddings). Both
are safe to delete; they'll be rebuilt on demand. Consider adding them to `.gitignore`.

---

## Troubleshooting

**`model "nomic-embed-text" not found, try pulling it first (status code: 404)`**
The embedding model isn't downloaded yet. Pull it once:

```bash
ollama pull nomic-embed-text
```

The embedding model is separate from your chat model — the app needs both. To use an
embedding model you already have instead, pass `--embed-model <name>` (or set it in the
Web UI).

**`model "gemma3" not found` (or a similar 404 for your chat model)**
Run `ollama list` and use the exact name shown for `--model`. Local names often include a
size tag, e.g. `gemma3:4b` or `llama3.1:8b`.

**Connection refused / can't reach Ollama**
Make sure Ollama is running (`ollama serve`, or the desktop app). If it's on another
machine or port, pass `--host http://host:11434` or set `OLLAMA_HOST`.

**`No .zim or .pdf files found at: ...`**
Check the path. On Windows, quote paths with spaces and use the drive root form `E:\`.
Use `--no-zim` or `--no-pdf` if you only want one type.

**Some PDFs are listed as "skipped"**
Those PDFs are scanned page images with no text layer, so nothing can be extracted. Run
them through OCR first (e.g. [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)) and re-run.

**The first PDF run is slow**
That's the one-time embedding pass. Later runs reuse `.pdf_index.pkl` and only embed new
or changed files. It's interruptible — stop and resume freely.

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the
[`LICENSE`](LICENSE) file for the full text. In short: you're free to use, study, share,
and modify it, but distributed versions (and works built on it) must remain open under the
same license.

GPL-3.0 is also the appropriate choice here because the `libzim` dependency that reads ZIM
files is itself GPL-licensed, so a distributed combined work must be GPL.

Copyright (C) 2026 mkaragas. This is not legal advice.

---

## Acknowledgements

- [openZIM / Kiwix](https://www.kiwix.org) for the ZIM format and offline content library
- [Ollama](https://ollama.com) for local model serving
