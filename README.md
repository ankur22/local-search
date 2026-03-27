# 🧠 Local Search + Hybrid RAG System  
**Using Ollama + ChromaDB + Flask + Browser UI**

This project lets you index local documents (PDF, Markdown, text, code, Office files) and run a **hybrid keyword + vector search** over them, with answers summarised by a local LLM via Ollama.

Everything runs **100% locally** — no cloud APIs, no external data transfer.

---

## 📦 1. Installation

### 1.1 Enter the project directory

```bash
cd /path/to/local-search
```

### 1.2 Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Install and start Ollama

Download:  
https://ollama.com/download

Start the server:

```bash
ollama serve
```

### 1.5 Pull required models

```bash
ollama pull mxbai-embed-large   # embeddings
ollama pull llama3.1            # chat / reasoning
```

You can change these in `rag_indexer.py` / `app.py` via `DEFAULT_EMBED_MODEL` and `DEFAULT_CHAT_MODEL`.

---

## 📂 2. Indexing Your Documents (Multi-Path)

The indexer supports scanning **multiple root folders**, extracting text, chunking, and embedding into ChromaDB with useful metadata (`kind`, `ext`, `version`, etc.).

### 2.1 Example indexing command

```bash
source .venv/bin/activate

python rag_indexer.py   --root ~/obsidian/work   --root ~/Documents/manuals   --root ~/go/src/github.com/grafana/k6   --db ./chroma_db   --collection my_corpus   --embed-model mxbai-embed-large   --use-tokens --chunk-tokens 256 --overlap-tokens 64
```

### 2.2 What this does

- Recursively scans each `--root` folder  
- Skips large files and common build/venv dirs  
- Extracts text from:
  - `.pdf`, `.md`, `.txt`, `.docx`, `.pptx`, `.xlsx`, `.html/.htm`, many code file types  
- Cleans and chunks text (token-aware if `--use-tokens` is set)  
- Embeds each chunk via Ollama (`/api/embeddings`)  
- Stores:
  - **document text**
  - **embeddings**
  - **metadata**:
    - `doc_id`: absolute path  
    - `source`: absolute path  
    - `ext`: file extension  
    - `kind`: `release_notes | code | note | pdf | other`  
    - `version`: version-like string if found (e.g. `v1.3.0`)  
    - `chunk_index`, `total_chunks`

It also keeps a small `index_state.json` in `./chroma_db` so future runs only reindex **changed files**, unless `--refresh-all` is used.

### 2.3 Full rebuild

To wipe and rebuild the index:

```bash
rm -rf ./chroma_db

python rag_indexer.py   --root ~/obsidian/work   --root ~/Documents/manuals   --root ~/go/src/github.com/grafana/k6   --db ./chroma_db   --collection my_corpus   --embed-model mxbai-embed-large   --use-tokens --chunk-tokens 256 --overlap-tokens 64   --refresh-all
```

---

## 🔍 3. How Search Works (Hybrid Keyword + Vector)

The backend implements a **hybrid retrieval strategy**:

### 3.1 Keyword-first behaviour

1. Your query is parsed into **keywords** (e.g.:
   - `prometheus`, `ql`, `health`, `check`
   - `getby`
   - `v1.3.0`
2. For each keyword, Chroma is queried with `where_document={"$contains": kw}` to find chunks whose **text** contains that keyword.
3. All matching chunks are:
   - deduplicated
   - scored by how many keyword hits they contain
   - limited to a safe maximum (configurable)

4. Neighboring chunks from the same file are **merged** into larger segments so context isn’t cut mid-sentence/section.
5. Segments are sorted:
   - first by keyword hit count (desc),
   - then by a simple distance score (if available).

If keyword matches exist, these segments are used as context for the LLM.

### 3.2 Semantic fallback

If **no keyword-based matches** are found, the system falls back to **semantic vector search**:

1. The full query is embedded via Ollama.
2. Chroma `query()` retrieves the top-k most similar chunks.
3. Neighboring chunks are merged into segments.
4. Segments are mildly re-ordered using keyword presence (if any keywords were extracted).
5. These segments become the context for the LLM.

### 3.3 LLM prompt

The backend builds a prompt like:

- A short instruction ("use context as primary source; say 'I don't know' if unsupported"),  
- A **Context** section listing segments `[1]`, `[2]`, … with file paths and chunk ranges,  
- Your **Question**,  
- A request to answer and cite sources like `[1]`, `[2]`.

---

## 🚀 4. Running the Local RAG Server (Flask backend)

From the project root:

```bash
source .venv/bin/activate
python app.py
```

Flask will start on:

```text
http://localhost:8000
```

Open that URL in your browser.

### 4.1 Running with Sigil observability

To send LLM generation and embedding telemetry to a [Grafana Sigil](https://github.com/grafana/sigil) instance, set the following environment variables:

```bash
# Required: enables Sigil generation export
export SIGIL_GENERATION_EXPORT_ENDPOINT=http://localhost:8080/api/v1/generations:export

# Optional: enables OTEL traces and metrics (for dashboard panels)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

python app.py
```

When `SIGIL_GENERATION_EXPORT_ENDPOINT` is set and `sigil-sdk` is installed, both `app.py` and `rag_indexer.py` will record:

- **Generation spans** for every LLM call (`ollama_generate`) with input/output messages, token usage, latency, model, and stop reason
- **Embedding spans** for every embedding call (`ollama_embed`) with model and input count
- **OTEL metrics** (`gen_ai.client.operation.duration`, `gen_ai.client.token.usage`) when `OTEL_EXPORTER_OTLP_ENDPOINT` is set

When neither variable is set, the app behaves identically to before — zero overhead.

See section 9 below for full Sigil stack setup.

---

## 💬 5. Using the Browser UI

The front-end (simple `index.html`) provides:

- A text input to type queries (keywords or natural language)  
- A chat-like view of:
  - **Your questions**
  - **LLM answers**
  - **Sources** with file paths and distances  

For your current workflow, the intended pattern is:

- Use **keyword-style queries** such as:
  - `getBy`
  - `v1.3.0`
  - `prometheus ql health check`
  - `zstd compressor Close semantics`
- Let the backend:
  - find all occurrences,
  - merge relevant context,
  - feed that to the LLM for summarisation.

### 5.1 Opening source files in Cursor

Each answer includes a **Sources** section listing the segments used.

Clicking a source entry:

- sends `POST /api/open-file { path: "/absolute/path/to/file" }`
- macOS opens that file in **Cursor** (`open -a "Cursor"`)

You can change the editor command in `app.py` if you prefer VS Code or another editor.

---

## 🔐 6. Privacy

- Ollama runs fully **locally** on `http://localhost:11434`  
- ChromaDB stores vectors/text under `./chroma_db`  
- Flask serves `http://localhost:8000` only on your machine  
- The browser talks to your own backend only  
- No external network calls are made unless you add them

Your notes, code, and documents never leave your machine.

---

## 🧪 7. Troubleshooting & Tuning

### 7.1 Clear the index

```bash
rm -rf ./chroma_db
```

Then re-run `rag_indexer.py`.

### 7.2 Ensure Ollama is running

```bash
ollama ps
```

You should see your models present (or at least `ollama serve` running).

### 7.3 Check installed models

```bash
ollama list
```

Ensure `mxbai-embed-large` and `llama3.1` are present (or adjust the defaults in the scripts).

### 7.4 If embeddings fail

Try:

- Smaller chunk sizes:

```bash
--chunk-tokens 200 --overlap-tokens 40
```

- Or temporarily index only a subset of your roots.

### 7.5 If results feel off

You can tweak in `app.py`:

- `KW_PER_KEYWORD_LIMIT` / `KW_TOTAL_LIMIT` for how many matches are considered  
- `MAX_CONTEXT_SEGMENTS` for how many merged segments go into context  
- `BASE_TOP_K` for semantic fallback retrieval depth  
- Turn on `RERANK_WITH_LLM = True` to let the LLM re-order segments for trickier questions (slower, but sometimes nicer).

---

## 🧩 8. Typical Commands You’ll Reuse

### Recreate environment

```bash
cd /path/to/local-search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Rebuild index

```bash
rm -rf ./chroma_db

python rag_indexer.py   --root ~/obsidian/work   --root ~/Documents/manuals   --root ~/go/src/github.com/grafana/k6   --db ./chroma_db   --collection my_corpus   --embed-model mxbai-embed-large   --use-tokens --chunk-tokens 256 --overlap-tokens 64   --refresh-all
```

### Run server

```bash
source .venv/bin/activate
python app.py
```

Then visit `http://localhost:8000`.

---

## 📊 9. Sigil Observability Setup (Optional)

[Grafana Sigil](https://github.com/grafana/sigil) provides an AI observability dashboard for tracking LLM generations, token usage, latency, errors, and conversations.

### 9.1 Start the Sigil stack

Clone and run the Sigil development stack:

```bash
git clone https://github.com/grafana/sigil.git
cd sigil
docker compose --profile core up -d
```

The first run builds from source and takes several minutes. Once ready:

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | http://localhost:3000 | Dashboards + Sigil plugin (login: `admin`/`admin`) |
| Sigil | http://localhost:8080 | Generation ingest API |
| Alloy | http://localhost:4318 | OTEL collector (traces + metrics) |
| Prometheus | http://localhost:9090 | Metrics storage |
| Tempo | http://localhost:3200 | Trace storage |

### 9.2 Run local-search with Sigil

```bash
source .venv/bin/activate

SIGIL_GENERATION_EXPORT_ENDPOINT=http://localhost:8080/api/v1/generations:export \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
python app.py
```

### 9.3 Run the indexer with Sigil

```bash
SIGIL_GENERATION_EXPORT_ENDPOINT=http://localhost:8080/api/v1/generations:export \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
python rag_indexer.py --root ~/obsidian/work --db ./chroma_db --collection my_corpus \
  --embed-model mxbai-embed-large --use-tokens --chunk-tokens 256 --overlap-tokens 64
```

### 9.4 What you'll see in Grafana

Open http://localhost:3000 and navigate to **AI Observability**:

- **Overview**: total requests, latency P95, error rate, token usage, cost
- **Performance**: generation duration breakdown by agent/model
- **Usage**: input/output token counts over time
- **Conversations**: full input/output message explorer

### 9.5 Stop the Sigil stack

```bash
cd /path/to/sigil
docker compose --profile core down
```

### 9.6 Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `SIGIL_GENERATION_EXPORT_ENDPOINT` | Yes (for Sigil) | HTTP endpoint for generation export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTEL collector for traces + metrics (enables dashboard panels) |

When neither variable is set, the app runs without any instrumentation overhead.

---

## 🧪 10. Running Tests

```bash
source .venv/bin/activate
pip install pytest
python -m pytest test_app_sigil.py test_indexer_sigil.py -v
```

---

This README describes the current **hybrid keyword + vector** behaviour.  
If you later decide to add an MCP layer or more advanced routing, you can extend from here without changing these core building blocks.
