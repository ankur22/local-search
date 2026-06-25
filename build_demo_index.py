#!/usr/bin/env python3
"""Build the small, self-contained demo index for the k6 tool-agent demo.

Indexes ``demo_docs/*.md`` (including a deliberately poisoned doc) into a
separate Chroma DB so the real `my_corpus` index is never touched.

    python build_demo_index.py
    DEMO_DB=./demo_chroma DEMO_COLLECTION=demo_corpus python build_demo_index.py
"""

from __future__ import annotations

import glob
import os

import requests
import chromadb
from chromadb.config import Settings

DEMO_DB = os.environ.get("DEMO_DB", "./demo_chroma")
DEMO_COLLECTION = os.environ.get("DEMO_COLLECTION", "demo_corpus")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DOCS_GLOB = os.environ.get("DEMO_DOCS", "demo_docs/*.md")


def embed(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def main() -> None:
    client = chromadb.PersistentClient(path=DEMO_DB, settings=Settings(allow_reset=True))
    try:
        client.delete_collection(DEMO_COLLECTION)
    except Exception:
        pass
    col = client.create_collection(DEMO_COLLECTION)

    paths = sorted(glob.glob(DOCS_GLOB))
    if not paths:
        raise SystemExit(f"no docs matched {DOCS_GLOB!r}")

    ids, embs, docs, metas = [], [], [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        src = os.path.abspath(path)
        ids.append(src)
        docs.append(text)
        embs.append(embed(text))
        metas.append({
            "doc_id": src, "source": src, "ext": ".md",
            "kind": "note", "chunk_index": 0, "total_chunks": 1,
        })
        print(f"  + {os.path.basename(path)}")

    col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    print(f"indexed {len(ids)} docs into {DEMO_DB}::{DEMO_COLLECTION}")


if __name__ == "__main__":
    main()
