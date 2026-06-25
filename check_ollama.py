#!/usr/bin/env python3
"""Check if Ollama is ready and responding."""
import sys
import time
import requests

OLLAMA_URL = "http://localhost:11434"
MAX_RETRIES = 30
RETRY_DELAY = 2

def check_ollama():
    """Check if Ollama is responding."""
    try:
        # Try to ping Ollama
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False

if __name__ == "__main__":
    for i in range(MAX_RETRIES):
        if check_ollama():
            sys.exit(0)
        if i < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    print("Ollama is not ready after waiting", file=sys.stderr)
    sys.exit(1)

