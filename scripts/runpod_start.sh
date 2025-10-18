#!/bin/bash
# RunPod Start Script - Whisper Stream Server
#
# Configura LD_LIBRARY_PATH para cuDNN e inicia o servidor
#
# Uso:
#   bash scripts/runpod_start.sh
#   bash scripts/runpod_start.sh --model base --verbose
#   bash scripts/runpod_start.sh --backend whisperx

set -e  # Exit on error

echo "=========================================="
echo "🚀 Iniciando Whisper Stream Server"
echo "=========================================="

# Detectar diretório do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Configurar LD_LIBRARY_PATH para cuDNN
echo ""
echo "⚙️  Configurando ambiente..."

# Tentar carregar do bashrc primeiro (se já configurado)
if [ -f ~/.bashrc ]; then
    source ~/.bashrc 2>/dev/null || true
fi

# Verificar se cuDNN está disponível e configurar path
CUDNN_PATH=$(poetry run python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null || echo "")

if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
    echo "✅ cuDNN path configurado: $CUDNN_PATH/lib"
else
    echo "⚠️  Aviso: cuDNN não encontrado (servidor pode falhar com CUDA)"
    echo "   Instale com: poetry run pip install nvidia-cudnn-cu12"
fi

# Verificar se ctranslate2 detecta GPU (se cuDNN configurado)
if [ -n "$CUDNN_PATH" ]; then
    GPU_COUNT=$(poetry run python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "✅ CUDA GPU detectada: $GPU_COUNT device(s)"
    else
        echo "⚠️  Aviso: CUDA GPU não detectada pelo ctranslate2"
    fi
fi

# Iniciar servidor
echo ""
echo "🎤 Iniciando servidor..."
echo ""

# Passar todos argumentos para o servidor
poetry run python -m server.main --config server-config.yaml "$@"
