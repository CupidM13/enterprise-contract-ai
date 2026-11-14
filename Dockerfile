# Multi-stage production Dockerfile ($0 Free-Tier Edition)
# Fully compatible with Hugging Face Spaces (Port 7860, user 1000) and Render Free Tier

# ==============================================================================
# Stage 1: Build Dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runner

# Create non-root user for security & Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install runtime dependencies for FAISS CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder --chown=user:user /root/.local /home/user/.local

# Copy application files
COPY --chown=user:user . /app

# Ensure start script is executable
RUN chmod +x /app/start.sh

# Create local data cache directory with correct permissions
RUN mkdir -p /app/data/faiss_index && chown -R user:user /app

USER user

# Expose Streamlit default port for Hugging Face Spaces & Render
EXPOSE 7860
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/start.sh"]
