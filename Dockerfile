# Use NVIDIA CUDA runtime as base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory in container
WORKDIR /app

# Install system dependencies required for audio processing and libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    curl \
    cuda-compat-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY req*.txt ./

# Install PyTorch with CUDA support first, then other dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=2000 \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 && \
    pip install --no-cache-dir --default-timeout=2000 -r requirements.txt

# Install production WSGI server
RUN pip install --no-cache-dir gunicorn

# Copy the entire application
COPY . .

# Copy and set up entrypoint
RUN chmod +x /app/entrypoint.sh

# Expose ports for HTTP and HTTPS
EXPOSE 8080 8443

# Set environment variables
ENV FLASK_APP=main.py \
    FLASK_ENV=production \
    NVIDIA_VISIBLE_DEVICES=all \
    PORT=8443 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Health check - generous timeouts since the app loads large ML models
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -k -f https://localhost:8443/health || exit 1

# Run via entrypoint which generates SSL cert then starts Gunicorn
CMD ["/app/entrypoint.sh"]
