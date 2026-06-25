# Setup Instructions for Background Services

This guide will help you set up the RAG app and indexer to run automatically.

## Prerequisites

- macOS with LaunchAgent support
- Python 3 installed
- Virtual environment (`.venv` or `venv`) set up in the project directory
- Ollama installed and configured

## Installation Steps

### 1. Make scripts executable

```bash
chmod +x start_app.sh
chmod +x run_indexer.sh
chmod +x check_ollama.py
chmod +x rag_indexer.py
chmod +x app.py
```

**Note**: The scripts automatically detect and activate your virtual environment (`.venv` or `venv`). Make sure your venv is set up in the project directory.

### 2. Create logs directory

```bash
mkdir -p logs
```

### 3. Update paths if needed

**Important**: If your repository is located elsewhere, update the paths in:
- `com.localrag.app.plist` - Update the path `/Users/ankuragarwal/go/src/github.com/ankur22/local-search` if needed
- `com.localrag.indexer.plist` - Update the path if needed
- `run_indexer.sh` - Update the `--root` directories to match your setup (currently configured for your Obsidian work directory and k6 release notes)

### 4. Install LaunchAgent for Flask App

```bash
# Copy the plist to LaunchAgents directory
cp com.localrag.app.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.localrag.app.plist

# Start it immediately (optional)
launchctl start com.localrag.app
```

### 5. Install LaunchAgent for Daily Indexer

```bash
# Copy the plist to LaunchAgents directory
cp com.localrag.indexer.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.localrag.indexer.plist
```

## Managing Services

### Check status

```bash
# Check if Flask app is running
launchctl list | grep com.localrag.app

# Check if indexer is scheduled
launchctl list | grep com.localrag.indexer
```

### View logs

```bash
# Flask app logs
tail -f logs/app.log
tail -f logs/app.error.log

# Indexer logs
tail -f logs/indexer.log
tail -f logs/indexer.error.log
```

### Stop services

```bash
# Stop Flask app
launchctl unload ~/Library/LaunchAgents/com.localrag.app.plist

# Stop indexer schedule
launchctl unload ~/Library/LaunchAgents/com.localrag.indexer.plist
```

### Restart services

```bash
# Restart Flask app
launchctl unload ~/Library/LaunchAgents/com.localrag.app.plist
launchctl load ~/Library/LaunchAgents/com.localrag.app.plist

# Restart indexer schedule
launchctl unload ~/Library/LaunchAgents/com.localrag.indexer.plist
launchctl load ~/Library/LaunchAgents/com.localrag.indexer.plist
```

## Customization

### Chat model provider (Ollama / OpenAI / Anthropic)

The chat/generation model is called over the OpenAI-compatible
`/v1/chat/completions` API, so Ollama, OpenAI, and Anthropic share one code path.
Pick the provider + model with environment variables (embeddings always use
Ollama via `RAG_EMBED_MODEL`):

| Variable | Meaning |
|----------|---------|
| `RAG_CHAT_PROVIDER` | `ollama` (default), `openai`, or `anthropic` — which API + auth to use |
| `RAG_CHAT_MODEL` | the model name at that provider (e.g. `llama3.1`, `gpt-4o`, `claude-haiku-4-5`) |
| `OPENAI_API_KEY` | required when `RAG_CHAT_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | required when `RAG_CHAT_PROVIDER=anthropic` |
| `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` / `OLLAMA_OPENAI_BASE_URL` | optional endpoint overrides |

```bash
# Local (default)
RAG_CHAT_PROVIDER=ollama    RAG_CHAT_MODEL=llama3.1          python app.py
# OpenAI
RAG_CHAT_PROVIDER=openai    RAG_CHAT_MODEL=gpt-4o            OPENAI_API_KEY=sk-...      python app.py
# Anthropic
RAG_CHAT_PROVIDER=anthropic RAG_CHAT_MODEL=claude-haiku-4-5  ANTHROPIC_API_KEY=sk-ant-... python app.py
```

The model must exist for the chosen provider and that provider's key must be set.

**Reasoning models (e.g. GPT-5):** they only accept the default temperature, so set
`RAG_CHAT_TEMPERATURE=none` to omit it, and optionally `RAG_CHAT_REASONING_EFFORT=high`
(`minimal|low|medium|high`). Caveat: `gpt-5.5` does not allow `tools` +
`reasoning_effort` together on `/v1/chat/completions`, so for the tool-calling agent
(`/api/agent`) use `RAG_CHAT_TEMPERATURE=none` *without* `reasoning_effort`.

See `.env.example` for the full set of variables.

### Change Flask app port

Edit `app.py` or set the `PORT` environment variable in `com.localrag.app.plist`:

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>PORT</key>
    <string>8000</string>
</dict>
```

### Change indexer schedule

Edit `com.localrag.indexer.plist` to change the `StartCalendarInterval`:

```xml
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>17</integer>  <!-- 5 PM -->
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

### Change indexer command

Edit `run_indexer.sh` to modify the indexer command, including root directories and other arguments:

```bash
exec python rag_indexer.py \
  --root /path/to/your/documents \
  --root /another/path \
  --db ./chroma_db \
  --collection my_corpus \
  --embed-model mxbai-embed-large \
  --use-tokens --chunk-tokens 256 --overlap-tokens 64 \
  --refresh-all
```

## Troubleshooting

1. **Flask app won't start**: Check if Ollama is running (`ollama list`)
2. **Indexer not running**: Check the logs in `logs/indexer.error.log`
3. **Permission errors**: Make sure all scripts are executable (`chmod +x`)
4. **Path issues**: Verify all paths in plist files are correct and absolute

