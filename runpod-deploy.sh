#!/bin/bash
# Script de Deploy para RunPod - Whisper Stream
#
# Este script facilita o deploy do Whisper Stream no RunPod
# com modelo large-v3 para melhor qualidade de transcrição.
#
# Uso:
#   ./runpod-deploy.sh build    # Build da imagem Docker
#   ./runpod-deploy.sh push     # Push para Docker Hub
#   ./runpod-deploy.sh all      # Build + Push
#   ./runpod-deploy.sh test     # Testa localmente (requer GPU)

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variáveis configuráveis
DOCKER_USERNAME="${DOCKER_USERNAME:-seu-usuario}"  # Altere ou defina via env
IMAGE_NAME="whisper-stream-runpod"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Banner
print_banner() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ${GREEN}🚀 Whisper Stream - RunPod Deploy Script${BLUE}      ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Info
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar se Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker não está instalado!"
        echo "Instale Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker encontrado: $(docker --version)"
}

# Build da imagem
build_image() {
    print_info "Fazendo build da imagem Docker..."
    echo ""

    docker build \
        -f Dockerfile.runpod \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${FULL_IMAGE_NAME}" \
        --progress=plain \
        .

    echo ""
    print_success "Imagem construída: ${FULL_IMAGE_NAME}"

    # Mostrar tamanho da imagem
    SIZE=$(docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "{{.Size}}")
    print_info "Tamanho da imagem: ${SIZE}"
}

# Push para Docker Hub
push_image() {
    print_info "Fazendo push para Docker Hub..."

    # Verificar se está logado
    if ! docker info 2>/dev/null | grep -q "Username"; then
        print_warning "Não está logado no Docker Hub"
        print_info "Fazendo login..."
        docker login
    fi

    echo ""
    docker push "${FULL_IMAGE_NAME}"

    echo ""
    print_success "Imagem enviada para Docker Hub: ${FULL_IMAGE_NAME}"
    print_info "Use esta imagem no RunPod!"
}

# Testar localmente (requer GPU NVIDIA)
test_local() {
    print_info "Testando imagem localmente..."
    print_warning "IMPORTANTE: Requer GPU NVIDIA com CUDA instalado!"
    echo ""

    # Verificar se nvidia-docker está disponível
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "GPU NVIDIA não detectada ou nvidia-docker não configurado"
        print_info "Para testar, você precisa de:"
        echo "  1. GPU NVIDIA"
        echo "  2. Drivers NVIDIA"
        echo "  3. nvidia-docker2"
        exit 1
    fi

    print_success "GPU detectada!"
    echo ""

    # Rodar container
    print_info "Iniciando container de teste..."
    print_info "Pressione Ctrl+C para parar"
    echo ""

    docker run --rm -it \
        --gpus all \
        -p 9090:9090 \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        "${IMAGE_NAME}:${IMAGE_TAG}"
}

# Instruções pós-deploy
show_instructions() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 Deploy preparado com sucesso!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}📋 Próximos passos:${NC}"
    echo ""
    echo "1. Acesse RunPod: https://www.runpod.io/"
    echo "2. Crie um novo pod:"
    echo "   - GPU: RTX 4090 ou A100 (24GB+ VRAM)"
    echo "   - Template: PyTorch (CUDA 12.1+)"
    echo "   - Docker Image: ${FULL_IMAGE_NAME}"
    echo "   - Volume: 20GB persistente montado em /app/models"
    echo "   - Porta: 9090 (TCP)"
    echo ""
    echo "3. Aguarde o pod iniciar (~2-5 minutos)"
    echo "   - Primeiro uso: modelo large-v3 será baixado (~3GB)"
    echo ""
    echo "4. Teste o servidor:"
    echo "   curl http://SEU-POD-IP:9090/health"
    echo ""
    echo "5. Configure o cliente local:"
    echo "   Edite config.yaml:"
    echo "   server:"
    echo "     url: \"ws://SEU-POD-IP:9090/ws\""
    echo ""
    echo "6. Rode o cliente:"
    echo "   bun start"
    echo ""
    print_info "Documentação completa: docs/RUNPOD.md"
    echo ""
    echo -e "${BLUE}💰 Custo estimado (RTX 4090):${NC}"
    echo "   ~$0.60-0.80/hora = ~$432-576/mês (24/7)"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
}

# Main
main() {
    print_banner

    # Verificar argumentos
    if [ $# -eq 0 ]; then
        print_error "Uso: $0 {build|push|all|test}"
        echo ""
        echo "Comandos:"
        echo "  build  - Build da imagem Docker"
        echo "  push   - Push para Docker Hub"
        echo "  all    - Build + Push"
        echo "  test   - Testa localmente (requer GPU)"
        exit 1
    fi

    check_docker

    echo ""
    print_info "Imagem: ${FULL_IMAGE_NAME}"
    echo ""

    case "$1" in
        build)
            build_image
            ;;
        push)
            push_image
            show_instructions
            ;;
        all)
            build_image
            echo ""
            push_image
            show_instructions
            ;;
        test)
            test_local
            ;;
        *)
            print_error "Comando inválido: $1"
            echo "Use: build, push, all ou test"
            exit 1
            ;;
    esac
}

main "$@"
