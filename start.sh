#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "Starting Enterprise Contract & Vendor Risk Automation Engine ($0 Free Tier)"
echo "Port: 7860 (Hugging Face Spaces / Render Free Tier Compatible)"
echo "================================================================================"

# Initialize FAISS cache if not already present
python3 -c "import asyncio; from app.policy_rag import policy_retriever; asyncio.run(policy_retriever.initialize())" || true

# Start FastAPI server in the background on port 8000
echo "Starting FastAPI backend server on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
echo "Waiting for FastAPI backend to initialize..."
sleep 2

# Start Streamlit frontend in foreground on port 7860
echo "Starting Streamlit frontend dashboard on port 7860..."
exec streamlit run ui/dashboard.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
