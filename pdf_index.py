# Offline Research Assistant — PDF discovery, text extraction, and the embedding index
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
pdf_index.py - Build and search a persistent semantic index over a PDF collection.

Unlike ZIM files (which ship with their own full-text search index), PDFs have no
built-in index, so we build one: extract text page-by-page, chunk it, embed each
chunk with the local Ollama embedding model, and store the vectors on disk. PDF
collections are small enough (vs. a full Wikipedia dump) that embedding everything
once is cheap, and the index is cached so unchanged files are never re-embedded.
"""

from __future__ import annotations

import os
import glob
import pickle

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

from zim_rag import Chunk, chunk_text


def discover_pdfs(path: str) -> list[str]:
    """Accept a single .pdf file, a folder, or a drive root and return all PDFs."""
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.pdf"), recursive=True))
    return []


def _signature(path: str) -> list:
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime)]


def _extract_pages(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages that contain extractable text."""
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Run:  pip install pypdf")
    out = []
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"[warn] could not open PDF {path}: {e}")
        return out
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            out.append((i, text))
    return out


class PdfLibrary:
    """Persistent, incrementally-updated semantic index over a set of PDFs."""

    def __init__(
        self,
        pdf_paths: list[str],
        embed_texts,                 # callable: list[str] -> list[list[float]]
        embed_model: str,
        index_path: str = ".pdf_index.pkl",
        chunk_words: int = 220,
        overlap: int = 40,
    ):
        if np is None:
            raise RuntimeError("numpy is not installed. Run:  pip install numpy")
        self.pdf_paths = pdf_paths
        self.embed_texts = embed_texts
        self.embed_model = embed_model
        self.index_path = index_path
        self.chunk_words = chunk_words
        self.overlap = overlap

        self._files: dict = {}          # abspath -> {"sig", "chunks":[{text,page}], "vecs":[[..]]}
        self._matrix = None             # np.ndarray (N, d), L2-normalized
        self._flat: list[Chunk] = []    # aligned with matrix rows
        self.skipped: list[str] = []    # PDFs with no extractable text (likely scanned)

    # -- persistence -------------------------------------------------------- #
    def _load_existing(self) -> dict:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                if data.get("model") == self.embed_model:
                    return data.get("files", {})
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump({"model": self.embed_model, "files": self._files}, f)
        except Exception as e:
            print(f"[warn] could not save PDF index: {e}")

    # -- build -------------------------------------------------------------- #
    def build(self, progress=None, save_every: int = 5) -> dict:
        """(Re)build the index incrementally. Returns a small stats dict.

        `progress` is an optional callable(done, total, message).
        The index is saved every `save_every` newly-indexed files (and at the end),
        so a long run that gets interrupted keeps the work it already did.
        """
        existing = self._load_existing()
        discovered = {os.path.abspath(p): p for p in self.pdf_paths}
        # Start from what's already on disk so partial progress survives restarts.
        self._files = {}
        self.skipped = []
        stats = {"reused": 0, "indexed": 0, "skipped": 0, "chunks": 0}

        total = len(discovered)
        since_save = 0
        for n, (abspath, path) in enumerate(sorted(discovered.items()), start=1):
            sig = _signature(abspath)
            prior = existing.get(abspath)
            if prior and prior.get("sig") == sig and prior.get("vecs"):
                self._files[abspath] = prior
                stats["reused"] += 1
                if progress:
                    progress(n, total, f"cached: {os.path.basename(path)}")
                continue

            if progress:
                progress(n, total, f"reading: {os.path.basename(path)}")
            pages = _extract_pages(abspath)
            records = []
            for page_no, text in pages:
                for piece in chunk_text(text, self.chunk_words, self.overlap):
                    records.append({"text": piece, "page": page_no})

            if not records:
                # No extractable text -> almost certainly a scanned/image-only PDF.
                self.skipped.append(path)
                stats["skipped"] += 1
                continue

            name = os.path.basename(path)
            n_chunks = len(records)

            def batch_progress(done, tot, _n=n, _name=name, _nc=n_chunks):
                if progress:
                    progress(_n, total, f"embedding {_name}: {done}/{_nc} chunks")

            vecs = self.embed_texts([r["text"] for r in records], progress=batch_progress)
            self._files[abspath] = {"sig": sig, "chunks": records, "vecs": vecs}
            stats["indexed"] += 1

            since_save += 1
            if since_save >= save_every:
                self._save()
                since_save = 0

        self._save()
        self._rebuild_matrix()
        stats["chunks"] = len(self._flat)
        return stats

    def _rebuild_matrix(self):
        rows, flat = [], []
        for abspath, rec in self._files.items():
            stem = os.path.splitext(os.path.basename(abspath))[0]
            src = os.path.basename(abspath)
            for meta, vec in zip(rec["chunks"], rec["vecs"]):
                rows.append(vec)
                flat.append(Chunk(
                    text=meta["text"], title=stem, path=abspath,
                    source_name=src, source_kind="pdf", loc=f"p.{meta['page']}",
                ))
        if rows:
            mat = np.asarray(rows, dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = mat / norms
        else:
            self._matrix = None
        self._flat = flat

    # -- search ------------------------------------------------------------- #
    def search(self, query_vec, top_n: int = 8) -> list[Chunk]:
        if self._matrix is None or not self._flat:
            return []
        q = np.asarray(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self._matrix @ q
        k = min(top_n, len(self._flat))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        out = []
        for i in top_idx:
            c = self._flat[int(i)]
            out.append(Chunk(text=c.text, title=c.title, path=c.path,
                             source_name=c.source_name, source_kind="pdf",
                             loc=c.loc, score=float(sims[int(i)])))
        return out

    @property
    def num_chunks(self) -> int:
        return len(self._flat)
