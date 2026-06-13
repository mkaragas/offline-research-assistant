# Offline Research Assistant — core retrieval, ranking, and answer generation
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
zim_rag.py - Offline Retrieval-Augmented Generation over ZIM files (Wikipedia, etc.)

How it works (no giant pre-indexing step required):
  1. RETRIEVE candidates: query each ZIM's *built-in* full-text index (Xapian) with
     both the full question and individual keywords, then union the results. This
     scales to a full Wikipedia dump because we never embed the whole corpus.
  2. EXTRACT + CHUNK: pull the candidate articles, strip HTML to clean text, split
     into overlapping chunks.
  3. RANK: embed the chunks + the question with a local Ollama embedding model and
     keep the most semantically similar chunks (cosine similarity).
  4. GENERATE: feed those chunks as context to a local Ollama chat model (e.g. Gemma)
     and stream a grounded answer with source attribution.

Everything runs offline once the Ollama models are pulled. No data leaves your machine.
"""

from __future__ import annotations

import os
import re
import json
import glob
import math
import time
import hashlib
import html as _htmllib
from html.parser import HTMLParser
from dataclasses import dataclass, field

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
    from libzim.suggestion import SuggestionSearcher
except ImportError:  # pragma: no cover
    Archive = None


# --------------------------------------------------------------------------- #
# Defaults — override via the CLI / Streamlit UI or environment variables.
# --------------------------------------------------------------------------- #
DEFAULT_CHAT_MODEL = os.environ.get("ZIM_CHAT_MODEL", "gemma3")
DEFAULT_EMBED_MODEL = os.environ.get("ZIM_EMBED_MODEL", "nomic-embed-text")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST")  # e.g. http://localhost:11434

# How many chunks to send to the embedding model per call. Keeps big books from
# being embedded in one oversized request.
EMBED_BATCH = int(os.environ.get("ZIM_EMBED_BATCH", "64"))

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "was",
    "were", "what", "which", "who", "whom", "when", "where", "why", "how", "did",
    "do", "does", "for", "with", "as", "at", "by", "that", "this", "it", "its",
    "be", "been", "from", "into", "about", "can", "could", "would", "should",
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "he", "she",
}


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #
class _HTMLToText(HTMLParser):
    _SKIP = {"script", "style", "head", "nav", "footer"}
    _BREAK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in self._BREAK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def clean_html(raw: str) -> str:
    parser = _HTMLToText()
    try:
        parser.feed(raw)
    except Exception:
        # Fall back to a crude tag strip if the parser chokes on malformed markup.
        raw = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", _htmllib.unescape(raw)).strip()
    txt = _htmllib.unescape(parser.text())
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt.strip()


def extract_keywords(question: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9]+", question.lower())
    seen, out = set(), []
    for t in toks:
        if t in _STOPWORDS or len(t) < 3 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def chunk_text(text: str, chunk_words: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, chunk_words - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_words]))
        i += step
    return chunks


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    text: str
    title: str          # article title (ZIM) or document title (PDF)
    path: str           # entry path inside the ZIM, or the PDF file path
    source_name: str    # basename of the .zim or .pdf file
    source_kind: str = "zim"   # "zim" or "pdf"
    loc: str = ""              # page label for PDFs, e.g. "p.3"; empty for ZIM
    score: float = 0.0

    @property
    def label(self) -> str:
        """Human-readable citation label."""
        if self.source_kind == "pdf" and self.loc:
            return f"{self.title} ({self.loc})"
        return self.title


@dataclass
class Answer:
    text: str
    chunks: list[Chunk] = field(default_factory=list)

    def sources(self) -> list[tuple[str, str]]:
        """Deduplicated (citation label, source file) pairs used as context."""
        seen, out = set(), []
        for c in self.chunks:
            key = (c.label, c.source_name)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out


# --------------------------------------------------------------------------- #
# ZIM discovery
# --------------------------------------------------------------------------- #
def discover_zims(path: str) -> list[str]:
    """Accept a single .zim file or a directory and return all .zim paths.

    On Windows you can pass a whole drive, e.g.  discover_zims(r"E:\\")
    """
    if os.path.isfile(path) and path.lower().endswith(".zim"):
        return [path]
    if os.path.isdir(path):
        found = sorted(
            glob.glob(os.path.join(path, "**", "*.zim"), recursive=True)
        )
        return found
    return []


# --------------------------------------------------------------------------- #
# Embedding cache (so repeated questions don't re-embed the same articles)
# --------------------------------------------------------------------------- #
class _EmbedCache:
    def __init__(self, cache_path: str | None):
        self.path = cache_path
        self.mem: dict[str, list[float]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.mem = json.load(f)
            except Exception:
                self.mem = {}

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, k: str):
        return self.mem.get(k)

    def put(self, k: str, vec: list[float]):
        self.mem[k] = vec

    def flush(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.mem, f)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# The researcher
# --------------------------------------------------------------------------- #
class Researcher:
    def __init__(
        self,
        zim_paths: list[str] | None = None,
        pdf_paths: list[str] | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        ollama_host: str | None = DEFAULT_OLLAMA_HOST,
        cache_path: str | None = ".zim_embed_cache.json",
        pdf_index_path: str = ".pdf_index.pkl",
        rebuild_pdf_index: bool = False,
        pdf_progress=None,
    ):
        if ollama is None:
            raise RuntimeError(
                "The ollama python client is not installed. Run:  pip install ollama"
            )
        zim_paths = zim_paths or []
        pdf_paths = pdf_paths or []
        if not zim_paths and not pdf_paths:
            raise ValueError("No ZIM or PDF files were provided/found.")
        if zim_paths and Archive is None:
            raise RuntimeError("libzim is not installed. Run:  pip install libzim")

        self.chat_model = chat_model
        self.embed_model = embed_model
        self.client = ollama.Client(host=ollama_host) if ollama_host else ollama
        self.cache = _EmbedCache(cache_path)

        # ZIM archives (query-time search via their built-in full-text index)
        self.archives: list[tuple[str, "Archive"]] = []
        for p in zim_paths:
            try:
                self.archives.append((os.path.basename(p), Archive(p)))
            except Exception as e:
                print(f"[warn] could not open {p}: {e}")

        # PDF library (persistent embedding index, built once / incrementally)
        self.pdf_lib = None
        self.pdf_stats = None
        if pdf_paths:
            from pdf_index import PdfLibrary
            if rebuild_pdf_index and os.path.exists(pdf_index_path):
                try:
                    os.remove(pdf_index_path)
                except OSError:
                    pass
            self.pdf_lib = PdfLibrary(
                pdf_paths=pdf_paths,
                embed_texts=self._embed_raw,
                embed_model=embed_model,
                index_path=pdf_index_path,
            )
            self.pdf_stats = self.pdf_lib.build(progress=pdf_progress)

        if not self.archives and (self.pdf_lib is None or self.pdf_lib.num_chunks == 0):
            raise RuntimeError("No readable sources: ZIM files failed to open and no "
                               "PDF text could be indexed.")

    # -- retrieval ---------------------------------------------------------- #
    def _search_one(self, archive: "Archive", query_str: str, limit: int) -> list[str]:
        if not query_str.strip():
            return []
        try:
            if archive.has_fulltext_index:
                searcher = Searcher(archive)
                res = searcher.search(Query().set_query(query_str))
                n = res.getEstimatedMatches()
                if not n:
                    return []
                return list(res.getResults(0, min(n, limit)))
            # No full-text index: fall back to title suggestions.
            sug = SuggestionSearcher(archive).suggest(query_str)
            n = sug.getEstimatedMatches()
            return list(sug.getResults(0, min(n, limit))) if n else []
        except Exception:
            return []

    def candidate_articles(self, question: str, max_articles: int = 12) -> list[Chunk]:
        """Union of full-question + per-keyword searches across all archives.

        Returns lightweight Chunk stubs (text filled later) — actually returns
        one (path, title, zim) per candidate article.
        """
        queries = [question] + extract_keywords(question)
        results: list[Chunk] = []
        seen: set[tuple[str, str]] = set()
        for zim_name, archive in self.archives:
            for q in queries:
                for path in self._search_one(archive, q, limit=max_articles):
                    key = (zim_name, path)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        entry = archive.get_entry_by_path(path)
                        if entry.is_redirect:
                            entry = entry.get_redirect_entry()
                        title = entry.title or path
                    except Exception:
                        title = path
                    results.append(Chunk(text="", title=title, path=path, source_name=zim_name, source_kind="zim"))
                    if len(results) >= max_articles:
                        return results
        return results

    def _entry_text(self, zim_name: str, path: str) -> str:
        for name, archive in self.archives:
            if name != zim_name:
                continue
            try:
                entry = archive.get_entry_by_path(path)
                if entry.is_redirect:
                    entry = entry.get_redirect_entry()
                item = entry.get_item()
                mimetype = item.mimetype or ""
                if "html" not in mimetype and "text" not in mimetype:
                    return ""
                raw = bytes(item.content).decode("utf-8", "replace")
                return clean_html(raw)
            except Exception:
                return ""
        return ""

    def gather_chunks(self, articles: list[Chunk]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for art in articles:
            body = self._entry_text(art.source_name, art.path)
            if not body:
                continue
            for piece in chunk_text(body):
                chunks.append(
                    Chunk(text=piece, title=art.title, path=art.path, source_name=art.source_name, source_kind="zim")
                )
        return chunks

    # -- embedding + ranking ------------------------------------------------ #
    def _embed_raw(self, texts: list[str], progress=None) -> list[list[float]]:
        """Embed texts in fixed-size batches (no cache). Used for PDF indexing,
        where the durable store is the PDF index itself, so the JSON cache would
        only bloat. `progress` is an optional callable(done, total)."""
        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            resp = self.client.embed(model=self.embed_model, input=batch)
            vectors = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
            out.extend(list(v) for v in vectors)
            if progress:
                progress(min(start + len(batch), total), total)
        return out

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Cached embedding for questions and ZIM chunks (small, recurring sets)."""
        out: list[list[float]] = [None] * len(texts)  # type: ignore
        todo_idx, todo_txt = [], []
        for i, t in enumerate(texts):
            k = self.cache.key(self.embed_model, t)
            cached = self.cache.get(k)
            if cached is not None:
                out[i] = cached
            else:
                todo_idx.append(i)
                todo_txt.append(t)
        if todo_txt:
            vectors = self._embed_raw(todo_txt)
            for j, vec in zip(todo_idx, vectors):
                out[j] = vec
                self.cache.put(self.cache.key(self.embed_model, texts[j]), vec)
            self.cache.flush()
        return out  # type: ignore

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-9)

    def rank_chunks(self, question: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []
        q_vec = self._embed([question])[0]
        c_vecs = self._embed([c.text for c in chunks])
        for c, v in zip(chunks, c_vecs):
            c.score = self._cosine(q_vec, v)
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

    def retrieve(self, question: str, top_k: int = 5, max_articles: int = 12) -> list[Chunk]:
        """Gather candidates from ZIM (built-in search) and PDF (vector index),
        score them all against the question, and return the best `top_k`."""
        q_vec = self._embed([question])[0]
        scored: list[Chunk] = []

        # ZIM candidates: keyword-union search -> articles -> chunks -> score.
        if self.archives:
            zim_chunks = self.gather_chunks(
                self.candidate_articles(question, max_articles=max_articles)
            )
            if zim_chunks:
                vecs = self._embed([c.text for c in zim_chunks])
                for c, v in zip(zim_chunks, vecs):
                    c.score = self._cosine(q_vec, v)
                scored.extend(zim_chunks)

        # PDF candidates: semantic search over the prebuilt index (already scored).
        if self.pdf_lib is not None:
            scored.extend(self.pdf_lib.search(q_vec, top_n=max(top_k * 2, max_articles)))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    # -- generation --------------------------------------------------------- #
    @staticmethod
    def _build_prompt(question: str, chunks: list[Chunk]) -> list[dict]:
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            origin = "PDF" if c.source_kind == "pdf" else "article"
            context_blocks.append(f"[{i}] (from {origin} \"{c.label}\")\n{c.text}")
        context = "\n\n".join(context_blocks) if context_blocks else "(no relevant passages found)"
        system = (
            "You are an offline research assistant. Answer the user's question using ONLY "
            "the numbered context passages provided. Cite the passages you rely on with "
            "their bracket numbers, e.g. [1], [2]. If the context does not contain the "
            "answer, say so plainly rather than guessing. Be concise and factual."
        )
        user = f"Context passages:\n\n{context}\n\n---\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def answer(self, question: str, top_k: int = 5, max_articles: int = 12, stream: bool = True):
        """Yields ("token", str) chunks while streaming, then ("done", Answer).
        For a simple blocking call use answer_blocking()."""
        top = self.retrieve(question, top_k=top_k, max_articles=max_articles)
        messages = self._build_prompt(question, top)

        if stream:
            collected = []
            for part in self.client.chat(model=self.chat_model, messages=messages, stream=True):
                token = part["message"]["content"]
                collected.append(token)
                yield ("token", token)
            yield ("done", Answer(text="".join(collected), chunks=top))
        else:
            resp = self.client.chat(model=self.chat_model, messages=messages)
            yield ("done", Answer(text=resp["message"]["content"], chunks=top))

    def answer_blocking(self, question: str, top_k: int = 5, max_articles: int = 12) -> Answer:
        result = None
        for kind, payload in self.answer(question, top_k, max_articles, stream=False):
            if kind == "done":
                result = payload
        return result  # type: ignore


# Backward-compatible alias (the class was ZimResearcher in the ZIM-only version).
ZimResearcher = Researcher


def discover_sources(paths) -> tuple[list[str], list[str]]:
    """Scan one or more files/folders/drive roots for both .zim and .pdf files.

    Returns (zim_paths, pdf_paths), each de-duplicated and sorted.
    """
    from pdf_index import discover_pdfs
    if isinstance(paths, str):
        paths = [paths]
    zims, pdfs = [], []
    for p in paths:
        zims.extend(discover_zims(p))
        pdfs.extend(discover_pdfs(p))
    return sorted(set(zims)), sorted(set(pdfs))
