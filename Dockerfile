# Dockerfile para Whisper Stream Server
#
# Suporta múltiplos backends via build arg:
#   - faster-whisper (universal, word timestamps)
#   - crisper-whisper (verbatim, fillers, ±50ms precision)
#   - mlx (Apple Silicon)
#   - whisperx (diarization, CUDA only)
#
# Build example:
#   docker build --build-arg BACKEND=crisper-whisper -t whisper-stream:crisper .
#
# Run example:
#   docker run -p 9090:9090 -e WHISPER_MODEL=base whisper-stream:crisper

# Argumento para selecionar backend em build time
ARG BACKEND=faster-whisper
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

# Base image (CUDA para GPU, ou slim para CPU)
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS base-cuda
FROM python:${PYTHON_VERSION}-slim AS base-cpu

# Escolher base correta (usar CUDA se disponível)
FROM base-cuda AS base

# Metadados
LABEL maintainer="Whisper Stream"
LABEL description="Whisper Stream Server - Speech to Text with multiple backends"
LABEL backend="${BACKEND}"

# Configuração de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios
WORKDIR /app
RUN mkdir -p /app/server /app/models /app/cache /app/logs

# Copiar requirements base
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Instalar backend específico baseado em ARG
ARG BACKEND
RUN echo "Installing backend: ${BACKEND}" && \
    if [ "$BACKEND" = "faster-whisper" ]; then \
        pip install --no-cache-dir faster-whisper; \
    elif [ "$BACKEND" = "crisper-whisper" ]; then \
        pip install --no-cache-dir git+https://github.com/nyrahealth/transformers.git@crisper_whisper; \
    elif [ "$BACKEND" = "whisperx" ]; then \
        pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git; \
    elif [ "$BACKEND" = "mlx" ]; then \
        pip install --no-cache-dir mlx-whisper; \
    else \
        echo "Unknown backend: ${BACKEND}"; \
        exit 1; \
    fi

# Copiar código do servidor
COPY server/ /app/server/

# Copiar configurações de exemplo
COPY server-config.example.yaml /app/server-config.yaml

# Variáveis de ambiente para configuração runtime
ENV WHISPER_BACKEND=${BACKEND}
ENV WHISPER_MODEL=base
ENV WHISPER_LANGUAGE=pt
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=9090
ENV MODELS_DIR=/app/models
ENV CACHE_DIR=/app/cache
ENV LOG_LEVEL=info

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Expor porta
EXPOSE 9090

# Comando para iniciar servidor
CMD ["sh", "-c", "python -m server.main \
    --backend ${WHISPER_BACKEND} \
    --model ${WHISPER_MODEL} \
    --language ${WHISPER_LANGUAGE} \
    --host ${SERVER_HOST} \
    --port ${SERVER_PORT}"]
