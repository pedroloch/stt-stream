#!/bin/bash
#
# Build Docker images para Whisper Stream Server
#
# Uso:
#   ./scripts/build-docker.sh faster-whisper
#   ./scripts/build-docker.sh crisper-whisper
#   ./scripts/build-docker.sh all
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "============================================================"
echo "🐳  WHISPER STREAM - DOCKER BUILD"
echo "============================================================"
echo -e "${NC}"

# Função para build de um backend específico
build_backend() {
    local backend=$1
    local cuda_version=${2:-12.1.0}

    echo -e "${YELLOW}Building backend: ${backend}${NC}"
    echo ""

    docker build \
        --build-arg BACKEND="${backend}" \
        --build-arg CUDA_VERSION="${cuda_version}" \
        -t whisper-stream:"${backend}" \
        -t whisper-stream:latest \
        -f Dockerfile \
        .

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Build successful: whisper-stream:${backend}${NC}"
        echo ""
    else
        echo -e "${RED}❌ Build failed: ${backend}${NC}"
        exit 1
    fi
}

# Parse argumentos
BACKEND=${1:-""}

if [ -z "$BACKEND" ]; then
    echo -e "${RED}Erro: Backend não especificado${NC}"
    echo ""
    echo "Uso:"
    echo "  $0 faster-whisper    # Universal backend"
    echo "  $0 crisper-whisper   # Verbatim transcription"
    echo "  $0 whisperx          # Diarization (CUDA only)"
    echo "  $0 mlx               # Apple Silicon"
    echo "  $0 all               # Build all backends"
    echo ""
    exit 1
fi

if [ "$BACKEND" = "all" ]; then
    echo -e "${BLUE}Building all backends...${NC}"
    echo ""

    build_backend "faster-whisper"
    build_backend "crisper-whisper"
    build_backend "whisperx"
    build_backend "mlx"

    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}✅ All backends built successfully!${NC}"
    echo -e "${GREEN}============================================================${NC}"

else
    build_backend "$BACKEND"

    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}✅ Build complete!${NC}"
    echo -e "${GREEN}============================================================${NC}"
fi

echo ""
echo "Imagens disponíveis:"
docker images | grep whisper-stream

echo ""
echo -e "${BLUE}Para rodar:${NC}"
echo "  docker run -p 9090:9090 whisper-stream:${BACKEND}"
echo ""
echo -e "${BLUE}Para testar:${NC}"
echo "  curl http://localhost:9090/health"
echo ""
