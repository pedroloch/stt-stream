#!/bin/bash
#
# Deploy Whisper Stream Server no RunPod
#
# Pré-requisitos:
#   1. Build da imagem Docker
#   2. Push para Docker Hub (ou outro registry)
#   3. Criar pod no RunPod com GPU
#
# Uso:
#   ./scripts/deploy-runpod.sh faster-whisper
#   ./scripts/deploy-runpod.sh crisper-whisper
#

set -e

BACKEND=${1:-"faster-whisper"}
DOCKER_USERNAME=${DOCKER_USERNAME:-"your-username"}
IMAGE_TAG="${DOCKER_USERNAME}/whisper-stream:${BACKEND}"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "🚀  DEPLOY TO RUNPOD"
echo "============================================================"
echo -e "${NC}"
echo ""

# Verificar se imagem existe localmente
if ! docker images | grep -q "whisper-stream.*${BACKEND}"; then
    echo -e "${YELLOW}⚠️  Imagem não encontrada localmente${NC}"
    echo "   Rodando build..."
    ./scripts/build-docker.sh "${BACKEND}"
fi

# Tag da imagem para registry
echo -e "${YELLOW}📦 Tagging image...${NC}"
docker tag "whisper-stream:${BACKEND}" "${IMAGE_TAG}"

# Push para Docker Hub
echo -e "${YELLOW}☁️  Pushing to Docker Hub...${NC}"
echo "   Image: ${IMAGE_TAG}"
echo ""

# Check if logged in
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  Não logado no Docker Hub${NC}"
    echo "   Rode: docker login"
    exit 1
fi

docker push "${IMAGE_TAG}"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Image pushed successfully!${NC}"
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}📋 NEXT STEPS - RUNPOD SETUP${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "1. Acesse: https://runpod.io/console/pods"
    echo ""
    echo "2. Clique em 'Deploy'"
    echo ""
    echo "3. Configuração:"
    echo "   • GPU: RTX A4000 / A5000 / A6000 (recomendado)"
    echo "   • Container Image: ${IMAGE_TAG}"
    echo "   • Container Disk: 20 GB"
    echo "   • Volume: 50 GB (para models cache)"
    echo ""
    echo "4. Environment Variables:"
    echo "   WHISPER_BACKEND=${BACKEND}"
    echo "   WHISPER_MODEL=base"
    echo "   WHISPER_LANGUAGE=pt"
    echo "   SERVER_PORT=9090"
    echo ""
    echo "5. Exposed Ports:"
    echo "   HTTP: 9090"
    echo ""
    echo "6. Depois de criado, pegue o Public IP:"
    echo "   curl http://<POD_IP>:9090/health"
    echo ""
    echo -e "${GREEN}7. Conectar cliente Bun:${NC}"
    echo "   Editar src/index.ts:"
    echo "   const WS_URL = 'ws://<POD_IP>:9090/ws';"
    echo ""
    echo -e "${BLUE}============================================================${NC}"
else
    echo -e "${RED}❌ Push failed${NC}"
    exit 1
fi
