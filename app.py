#!/usr/bin/env python3
import atexit
import os
import re
import subprocess
from contextlib import nullcontext
from typing import List, Any, Dict, Optional

import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import chromadb
from chromadb.config import Settings

try:
    from sigil_sdk import (
        Client as SigilClient,
        ClientConfig,
        EmbeddingResult,
        EmbeddingStart,
        GenerationExportConfig,
        GenerationStart,
        ModelRef,
        TokenUsage,
        assistant_text_message,
        user_text_message,
    )
    _SIGIL_AVAILABLE = True
except ImportError:
    _SIGIL_AVAILABLE = False

# ---- Config ----

DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "my_corpus"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"  # must match indexer
DEFAULT_CHAT_MODEL = "llama3.1"            # must exist in `ollama list`
DEFAULT_TOP_K = 8

# Retrieval / ranking config
BASE_TOP_K = 12            # minimum number of chunks to retrieve from Chroma
MAX_CONTEXT_SEGMENTS = 8   # how many merged segments to feed to the LLM
NEIGHBOR_JOIN_GAP = 1      # join chunks from same doc if indices differ by <= this
RERANK_WITH_LLM = False    # optional extra rerank using the LLM (slower)

# Keyword search limits
KW_PER_KEYWORD_LIMIT = 40  # max chunks per keyword
KW_TOTAL_LIMIT = 200       # global cap on keyword-matched chunks

DB_DIR = os.environ.get("RAG_DB_DIR", DEFAULT_DB_DIR)
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", DEFAULT_COLLECTION)
OLLAMA_URL = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL)

# ---- Sigil instrumentation (opt-in via env var) ----

SIGIL_ENDPOINT = os.environ.get("SIGIL_GENERATION_EXPORT_ENDPOINT", "")

_sigil_client = None
if _SIGIL_AVAILABLE and SIGIL_ENDPOINT:
    OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if OTEL_ENDPOINT:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": "local-search"})
        tp = TracerProvider(resource=resource)
        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces")))
        trace.set_tracer_provider(tp)

        mp = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{OTEL_ENDPOINT}/v1/metrics"),
                export_interval_millis=5000,
            )],
        )
        metrics.set_meter_provider(mp)

    _sigil_client = SigilClient(
        ClientConfig(
            generation_export=GenerationExportConfig(
                protocol="http",
                endpoint=SIGIL_ENDPOINT,
            ),
        )
    )


def _shutdown_sigil():
    if _sigil_client is not None:
        _sigil_client.shutdown()


atexit.register(_shutdown_sigil)

# ---- Stopwords & keyword helpers ----

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "you",
    "your", "about", "can", "could", "would", "should", "please",
    "what", "when", "where", "which", "who", "how", "why", "is",
    "are", "was", "were", "to", "of", "in", "on", "at", "a", "an",
    "it", "as", "by", "or", "if", "be", "we", "they", "i",
}

def extract_keywords(q: str) -> List[str]:
    # Keep letters/digits/*/. (for getBy*, v1.3.0, etc.)
    tokens = re.findall(r"[A-Za-z0-9_\.\*]+", q)
    kws: List[str] = []
    for t in tokens:
        t_clean = t.strip().lower()
        if not t_clean:
            continue
        if len(t_clean) <= 2:
            continue
        if t_clean in STOPWORDS:
            continue
        kws.append(t_clean)
    # dedupe, preserve order
    deduped = list(dict.fromkeys(kws))
    return deduped

# ---- Ollama helpers ----

def ollama_embed(text: str) -> List[float]:
    ctx = (
        _sigil_client.start_embedding(
            EmbeddingStart(
                agent_name="local-search",
                agent_version="0.1.0",
                model=ModelRef(provider="ollama", name=EMBED_MODEL),
            )
        )
        if _sigil_client is not None
        else nullcontext()
    )

    with ctx as rec:
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            r.raise_for_status()
            js = r.json()

            if "embedding" in js:
                embedding = js["embedding"]
            elif "data" in js and js["data"] and "embedding" in js["data"][0]:
                embedding = js["data"][0]["embedding"]
            else:
                raise RuntimeError(
                    f"Unexpected embeddings response from Ollama: {js}"
                )

            if rec is not None:
                rec.set_result(
                    EmbeddingResult(input_count=1, response_model=EMBED_MODEL)
                )

            return embedding
        except Exception as exc:
            if rec is not None:
                rec.set_call_error(exc)
            raise

def ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    """
    Uses /api/generate. If your Ollama prefers /api/chat, you can switch implementation.
    """
    ctx = (
        _sigil_client.start_generation(
            GenerationStart(
                agent_name="local-search",
                agent_version="0.1.0",
                model=ModelRef(provider="ollama", name=CHAT_MODEL),
                temperature=temperature,
            )
        )
        if _sigil_client is not None
        else nullcontext()
    )

    with ctx as rec:
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=180,
            )
            r.raise_for_status()
            js = r.json()
            response_text = js.get("response", "")

            if rec is not None:
                rec.set_result(
                    input=[user_text_message(prompt)],
                    output=[assistant_text_message(response_text)],
                    usage=TokenUsage(
                        input_tokens=js.get("prompt_eval_count", 0),
                        output_tokens=js.get("eval_count", 0),
                    ),
                    stop_reason=js.get("done_reason", ""),
                    response_model=js.get("model", ""),
                )

            return response_text
        except Exception as exc:
            if rec is not None:
                rec.set_call_error(exc)
            raise

# ---- Chroma client ----

client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
collection = client.get_collection(COLLECTION_NAME)

# ---- Segment merging / reranking ----

def merge_neighbor_chunks(
    docs: List[str],
    metas: List[Dict[str, Any]],
    dists: List[float],
    max_segments: int = MAX_CONTEXT_SEGMENTS,
    neighbor_gap: int = NEIGHBOR_JOIN_GAP,
) -> List[Dict[str, Any]]:
    """
    Merge adjacent chunks from the same doc into larger segments and
    keep them ranked by the best (lowest) distance.
    Returns a list of dicts: [{id, text, best_dist, chunks, metas, primary_meta}, ...].
    """
    items: List[Dict[str, Any]] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        items.append({
            "doc": doc,
            "meta": meta,
            "dist": dist,
            "idx": i,
            "doc_id": meta.get("doc_id"),
            "chunk_index": meta.get("chunk_index", 0),
        })

    # Sort by doc_id then chunk_index so we can merge neighbors
    items.sort(key=lambda x: (x["doc_id"], x["chunk_index"]))

    segments: List[Dict[str, Any]] = []
    current_segment: Optional[Dict[str, Any]] = None

    for it in items:
        if current_segment is None:
            current_segment = {
                "doc_id": it["doc_id"],
                "chunks": [it],
                "best_dist": it["dist"],
            }
            continue

        same_doc = it["doc_id"] == current_segment["doc_id"]
        prev_chunk_index = current_segment["chunks"][-1]["chunk_index"]
        if same_doc and abs(it["chunk_index"] - prev_chunk_index) <= neighbor_gap:
            current_segment["chunks"].append(it)
            if it["dist"] < current_segment["best_dist"]:
                current_segment["best_dist"] = it["dist"]
        else:
            segments.append(current_segment)
            current_segment = {
                "doc_id": it["doc_id"],
                "chunks": [it],
                "best_dist": it["dist"],
            }

    if current_segment is not None:
        segments.append(current_segment)

    # Sort segments by best_dist ascending
    segments.sort(key=lambda s: s["best_dist"])
    segments = segments[:max_segments]

    merged: List[Dict[str, Any]] = []
    for seg_id, seg in enumerate(segments, start=1):
        text = "\n\n".join(chunk["doc"] for chunk in seg["chunks"])
        primary = seg["chunks"][0]
        meta_list = []
        for ch in seg["chunks"]:
            m = dict(ch["meta"])
            m["distance"] = ch["dist"]
            meta_list.append(m)

        merged.append({
            "id": seg_id,
            "text": text,
            "best_dist": seg["best_dist"],
            "chunks": seg["chunks"],
            "metas": meta_list,
            "primary_meta": primary["meta"],
        })

    return merged

def llm_rerank(question: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple reranker: ask the LLM which segments are most relevant.
    This is optional and slower because it adds an extra LLM call.
    """
    if not segments:
        return segments

    descr_lines: List[str] = []
    for seg in segments:
        preview = seg["text"][:300].replace("\n", " ")
        descr_lines.append(f"{seg['id']}. {preview}")

    prompt = (
        "You are a ranking assistant. You are given a question and several context passages.\n"
        "Rank the passages from most relevant to least relevant by their ID.\n"
        "Return ONLY a comma-separated list of IDs in order (e.g., '2,1,3').\n\n"
        f"Question: {question}\n\n"
        "Passages:\n" + "\n".join(descr_lines) + "\n\n"
        "Order:"
    )

    try:
        resp = ollama_generate(prompt, temperature=0.0)
        ids: List[int] = []
        for part in resp.strip().split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
    except Exception:
        return segments

    if not ids:
        return segments

    seg_by_id = {seg["id"]: seg for seg in segments}
    ordered = [seg_by_id[i] for i in ids if i in seg_by_id]
    remaining = [s for s in segments if s["id"] not in ids]
    return ordered + remaining

# ---- Keyword search over Chroma ----

def keyword_search(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Keyword-first search: find chunks whose document text contains any of the keywords.
    Uses Chroma's where_document $contains filter via .query().
    Returns a list of {doc, meta, kw_hits}.
    """
    matches: List[Dict[str, Any]] = []
    seen: set = set()

    for kw in keywords:
        try:
            res = collection.query(
                query_texts=[kw],                 # simple query to satisfy API
                n_results=KW_PER_KEYWORD_LIMIT,   # per-keyword cap
                where_document={"$contains": kw},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            doc_id = meta.get("doc_id")
            chunk_index = meta.get("chunk_index", 0)
            key = (doc_id, chunk_index)
            if key in seen:
                continue
            seen.add(key)

            text_lower = doc.lower()
            kw_hits = sum(text_lower.count(k) for k in keywords)
            if kw_hits <= 0:
                continue

            matches.append({
                "doc": doc,
                "meta": meta,
                "kw_hits": kw_hits,
            })

            if len(matches) >= KW_TOTAL_LIMIT:
                break

        if len(matches) >= KW_TOTAL_LIMIT:
            break

    # Sort by total keyword hits (desc)
    matches.sort(key=lambda x: x["kw_hits"], reverse=True)
    return matches


# ---- Prompt & sources helpers ----

def build_prompt(question: str, segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("You are a precise assistant.")
    lines.append(
        "Use the provided context as your primary source of truth. "
        "The user's query may sometimes be just a list of keywords; "
        "in that case, treat it as a request to find and explain where those "
        "keywords appear in the context, summarising the relevant information."
    )
    lines.append(
        "If the answer is clearly not supported by the context, say you don't know."
    )
    lines.append("")
    lines.append("Context:")

    for seg in segments:
        pm = seg["primary_meta"]
        source = pm.get("source")
        ext = pm.get("ext")
        first_idx = pm.get("chunk_index", 0)
        total = pm.get("total_chunks", 0)
        best_dist = seg.get("best_dist", 0.0)

        seg_header = (
            f"[{seg['id']}] ({source} • {ext} • "
            f"chunks starting at {first_idx+1}/{total} • best_distance={best_dist:.4f})"
        )
        lines.append(seg_header)
        lines.append(seg["text"])
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("Answer (cite sources like [1], [2]):")

    return "\n".join(lines)

def build_sources_payload(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for seg in segments:
        pm = seg["primary_meta"]
        source = pm.get("source")
        ext = pm.get("ext")
        first_idx = pm.get("chunk_index", 0)
        total = pm.get("total_chunks", 0)
        best_dist = seg.get("best_dist", 0.0)

        preview = seg["text"][:400]
        payload.append({
            "id": seg["id"],
            "source": source,
            "ext": ext,
            "chunk_index": first_idx,
            "total_chunks": total,
            "distance": best_dist,
            "doc_id": pm.get("doc_id"),
            "preview": preview,
        })
    return payload

# ---- Flask app ----

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.post("/api/query")
def api_query():
    data = request.get_json(force=True)
    question = (data or {}).get("query", "").strip()
    user_k = int((data or {}).get("k", DEFAULT_TOP_K))
    temperature = float((data or {}).get("temperature", 0.2))

    if not question:
        return jsonify({"error": "query is required"}), 400

    keywords = extract_keywords(question)

    segments: List[Dict[str, Any]] = []

    # ---- 1) Keyword-first search ----
    if keywords:
        kw_matches = keyword_search(keywords)
        if kw_matches:
            docs = [m["doc"] for m in kw_matches]
            metas = [m["meta"] for m in kw_matches]
            # distances are dummy here; we rank by keyword hits later
            dists = [1.0 for _ in kw_matches]

            segments = merge_neighbor_chunks(docs, metas, dists)

            # propagate keyword hits into segments
            kw_hits_by_key: Dict[Any, int] = {}
            for m in kw_matches:
                meta = m["meta"]
                key = (meta.get("doc_id"), meta.get("chunk_index", 0))
                kw_hits_by_key[key] = m["kw_hits"]

            for seg in segments:
                total_hits = 0
                for ch in seg["chunks"]:
                    meta = ch["meta"]
                    key = (meta.get("doc_id"), meta.get("chunk_index", 0))
                    total_hits += kw_hits_by_key.get(key, 0)
                seg["keyword_hits"] = total_hits

            # sort segments: most keyword hits first, then best_dist
            segments.sort(key=lambda s: (-s.get("keyword_hits", 0), s["best_dist"]))
            segments = segments[:MAX_CONTEXT_SEGMENTS]

    # ---- 2) Fallback to semantic vector search if no keyword segments ----
    if not segments:
        try:
            qvec = ollama_embed(question)
        except Exception as e:
            return jsonify({"error": f"embedding failed: {e}"}), 500

        top_k = max(user_k, BASE_TOP_K)
        try:
            results = collection.query(
                query_embeddings=[qvec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return jsonify({"error": f"chroma query failed: {e}"}), 500

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            return jsonify({
                "answer": "I couldn't find anything in your index for that.",
                "sources": [],
            })

        segments = merge_neighbor_chunks(docs, metas, dists)

        # Even for vector search, we can still apply simple keyword boosting
        if keywords:
            for seg in segments:
                text_lower = seg["text"].lower()
                hits = sum(text_lower.count(k) for k in keywords)
                seg["keyword_hits"] = hits
            segments.sort(key=lambda s: (-s.get("keyword_hits", 0), s["best_dist"]))

        segments = segments[:MAX_CONTEXT_SEGMENTS]

    # ---- Optional LLM-based reranking ----
    if RERANK_WITH_LLM:
        segments = llm_rerank(question, segments)

    # ---- Build answer ----
    prompt = build_prompt(question, segments)

    try:
        answer = ollama_generate(prompt, temperature=temperature)
    except Exception as e:
        return jsonify({"error": f"generation failed: {e}"}), 500

    sources_payload = build_sources_payload(segments)

    if _sigil_client is not None:
        _sigil_client.flush()

    return jsonify({
        "answer": answer,
        "sources": sources_payload,
    })

@app.post("/api/open-file")
def api_open_file():
    data = request.get_json(force=True)
    path = (data or {}).get("path", "").strip()

    if not path or not os.path.exists(path):
        return jsonify({"error": "Invalid or missing file path"}), 400

    try:
        subprocess.Popen(["open", "-a", "Cursor", path])
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
