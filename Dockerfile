FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy project files
COPY --chown=user . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    scikit-learn>=1.7.1 \
    pandas>=2.2.3 \
    numpy>=1.24.0 \
    umap-learn \
    scikit-activeml>=0.6.2 \
    librosa>=0.10.0 \
    matplotlib>=3.7.0 \
    pyyaml \
    soundfile

# Build React frontend (empty VITE_API_URL = same-origin requests)
WORKDIR $HOME/app/app
ENV VITE_API_URL=""
RUN npm install && npm run build

# Back to root
WORKDIR $HOME/app

# Railway sets PORT env var; default to 7860 for local dev
ENV PORT=7860
EXPOSE $PORT

# Start the application
CMD ["python", "app.py"]
