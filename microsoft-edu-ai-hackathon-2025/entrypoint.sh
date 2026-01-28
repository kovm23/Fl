#!/bin/bash
set -e

# Wait for Ollama to be up
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama..."
  sleep 2
done

# Check if Qwen2.5-VL is pulled
if ! curl -s http://ollama:11434/api/tags | grep -q 'qwen2.5-vl'; then
  echo "Pulling Qwen2.5-VL model..."
  ollama pull qwen2.5-vl || curl -X POST http://ollama:11434/api/pull -d '{"name":"qwen2.5-vl"}'
fi

# Start Flask app
exec python3 /app/backend/app.py
