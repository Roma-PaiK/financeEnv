#!/bin/bash
set -e

# Start the environment server in the background
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    sleep 1
done

# Run the inference agent if LLM proxy credentials are available
# (injected by the hackathon evaluator; skipped during plain env-only usage)
if [ -n "$API_BASE_URL" ] && [ -n "$API_KEY" ]; then
    echo "LLM proxy detected — running inference agent..."
    python inference.py
fi

# Keep the server alive
wait $SERVER_PID
