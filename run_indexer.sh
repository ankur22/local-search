#!/bin/bash
# Wrapper script to run the indexer with venv activation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Run the indexer with the exact command
exec python rag_indexer.py \
  --root /Users/ankuragarwal/obsidian/work \
  --root "/Users/ankuragarwal/go/src/github.com/grafana/k6/release notes" \
  --db ./chroma_db \
  --collection my_corpus \
  --embed-model mxbai-embed-large \
  --use-tokens --chunk-tokens 256 --overlap-tokens 64 \
  --refresh-all

