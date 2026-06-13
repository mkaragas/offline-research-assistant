#!/usr/bin/env python3
# Offline Research Assistant — command-line interface
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
ask.py - Command-line interface for offline ZIM + PDF research.

Examples (Windows, your files live on drive E):

    # Ask against everything (.zim AND .pdf) found on drive E:
    python ask.py --path E:\\ --model gemma3 "Why did the Western Roman Empire fall?"

    # Mix specific locations (repeat --path):
    python ask.py --path "E:\\wikipedia.zim" --path "E:\\papers" "Explain entanglement"

    # PDFs only, and force a fresh index after adding new files:
    python ask.py --path E:\\papers --no-zim --rebuild-index "summarize the key findings"

    # Interactive mode (omit the question):
    python ask.py --path E:\\

The first run embeds your PDFs (cached afterwards). ZIM files use their own
built-in index, so they need no preprocessing.
"""

import sys
import time
import argparse

from zim_rag import (
    Researcher,
    discover_sources,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Research offline ZIM and PDF files with a local Ollama model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", nargs="*", help="Your question (omit for interactive mode).")
    # --path is the general flag; --zim is kept as a backward-compatible alias.
    p.add_argument("--path", "--zim", dest="paths", action="append", required=True,
                   metavar="PATH",
                   help="A .zim/.pdf file, a folder, or a drive root (e.g. E:\\). "
                        "Repeat to add more locations. Scans for BOTH .zim and .pdf.")
    p.add_argument("--model", default=DEFAULT_CHAT_MODEL,
                   help="Ollama chat model name (see `ollama list`).")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help="Ollama embedding model name.")
    p.add_argument("--host", default=None,
                   help="Ollama host URL, e.g. http://localhost:11434.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of context passages fed to the model.")
    p.add_argument("--max-articles", type=int, default=12,
                   help="Max candidate articles/passages pulled per question.")
    p.add_argument("--no-zim", action="store_true", help="Ignore ZIM files.")
    p.add_argument("--no-pdf", action="store_true", help="Ignore PDF files.")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force re-reading and re-embedding all PDFs.")
    p.add_argument("--pdf-index", default=".pdf_index.pkl",
                   help="Where to store the PDF embedding index.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the on-disk embedding cache.")
    return p


def make_researcher(args) -> Researcher:
    zims, pdfs = discover_sources(args.paths)
    if args.no_zim:
        zims = []
    if args.no_pdf:
        pdfs = []
    if not zims and not pdfs:
        sys.exit(f"No .zim or .pdf files found at: {', '.join(args.paths)}")

    print(f"Found {len(zims)} ZIM file(s) and {len(pdfs)} PDF file(s).")
    for z in zims:
        print(f"  [zim] {z}")
    for d in pdfs:
        print(f"  [pdf] {d}")

    def progress(done, total, msg):
        print(f"  indexing PDFs [{done}/{total}] {msg}", flush=True)

    if pdfs:
        print("\nPreparing PDF index (first run embeds them; later runs reuse the cache)...")

    researcher = Researcher(
        zim_paths=zims,
        pdf_paths=pdfs,
        chat_model=args.model,
        embed_model=args.embed_model,
        ollama_host=args.host,
        cache_path=None if args.no_cache else ".zim_embed_cache.json",
        pdf_index_path=args.pdf_index,
        rebuild_pdf_index=args.rebuild_index,
        pdf_progress=progress if pdfs else None,
    )

    if researcher.pdf_stats:
        s = researcher.pdf_stats
        print(f"PDF index ready: {s['indexed']} newly indexed, {s['reused']} cached, "
              f"{s['chunks']} searchable passages.")
    if researcher.pdf_lib and researcher.pdf_lib.skipped:
        print("\n[note] These PDFs had no extractable text (likely scanned images) "
              "and were skipped. OCR would be needed to include them:")
        for p in researcher.pdf_lib.skipped:
            print(f"  - {p}")
    return researcher


def ask_once(researcher: Researcher, question: str, top_k: int, max_articles: int):
    print("\nSearching the knowledge base...", flush=True)
    t0 = time.time()
    answer = None
    printed_header = False
    for kind, payload in researcher.answer(question, top_k=top_k,
                                           max_articles=max_articles, stream=True):
        if kind == "token":
            if not printed_header:
                print("\n" + "=" * 60 + "\nANSWER\n" + "=" * 60)
                printed_header = True
            print(payload, end="", flush=True)
        elif kind == "done":
            answer = payload
    if not printed_header:
        print("\n(no answer text returned by the model)")
    print()
    if answer and answer.sources():
        print("\n" + "-" * 60 + "\nSources used")
        for title, zim_name in answer.sources():
            print(f"  - {title}   ({zim_name})")
    print(f"\n[done in {time.time() - t0:.1f}s]")


def main():
    args = build_parser().parse_args()
    try:
        researcher = make_researcher(args)
    except Exception as e:
        sys.exit(f"Startup failed: {e}")

    if args.question:
        ask_once(researcher, " ".join(args.question), args.top_k, args.max_articles)
        return

    print("\nInteractive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if q.lower() in {"exit", "quit", "bye", ""}:
            print("bye")
            break
        try:
            ask_once(researcher, q, args.top_k, args.max_articles)
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    main()
