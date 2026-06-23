#!/bin/bash
set -e

echo "Downloading model weights (if needed)..."
python scripts/download_models.py

echo "Starting Gridlock AI API..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Starting Gridlock AI Dashboard..."
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 &
DASHBOARD_PID=$!

echo "Gridlock AI is ready!"
echo "  API:      http://localhost:8000"
echo "  Docs:     http://localhost:8000/docs"
echo "  Dashboard: http://localhost:8501"

cleanup() {
    kill $API_PID $DASHBOARD_PID 2>/dev/null || true
}
trap cleanup EXIT

wait
