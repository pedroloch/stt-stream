#!/bin/bash
# RunPod Development Setup - Whisper Stream com Diarization
#
# Script para setup rápido de ambiente de desenvolvimento/teste no RunPod com GPU
#
# Uso:
#   bash runpod_setup.sh [sortformer|whisperx]
#
# Exemplos:
#   bash runpod_setup.sh              # Default: sortformer (BATCH API)
#   bash runpod_setup.sh sortformer   # BATCH API + Sortformer SOTA
#   bash runpod_setup.sh whisperx     # Streaming + WhisperX v3.3.2
#
# Via URL (one-liner):
#   bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh)
#   bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh) whisperx

set -e  # Exit on error

# Parse argumentos
SETUP_TYPE="${1:-sortformer}"  # Default: sortformer

if [[ "$SETUP_TYPE" != "sortformer" && "$SETUP_TYPE" != "whisperx" ]]; then
    echo "❌ Uso: bash runpod_setup.sh [sortformer|whisperx]"
    echo ""
    echo "Opções:"
    echo "  sortformer  - BATCH API + Sortformer (SOTA 2025, 4 speakers) [DEFAULT]"
    echo "  whisperx    - Streaming + WhisperX v3.3.2 (timestamps ±50ms)"
    exit 1
fi

echo "=========================================="
echo "🚀 Whisper Stream - RunPod GPU Setup"
echo "   Setup: $SETUP_TYPE"
echo "=========================================="

# 1. Verificar CUDA
echo ""
echo "📊 Verificando GPU/CUDA..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ CUDA version: {torch.version.cuda}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 1.5. Instalar ffmpeg (para NeMo/pydub)
echo ""
echo "📦 Instalando ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
    echo "✅ ffmpeg instalado"
else
    echo "✅ ffmpeg já instalado"
fi

# 2. Clone repositório
echo ""
echo "📦 Clonando repositório..."
cd /workspace
if [ -d "stt-stream" ]; then
    echo "⚠️  Repositório já existe, atualizando..."
    cd stt-stream
    git pull
else
    git clone https://github.com/pedroloch/stt-stream.git
    cd stt-stream
fi

# 3. Instalar Poetry
echo ""
echo "📦 Instalando Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/root/.local/bin:$PATH"
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    echo "✅ Poetry instalado"
else
    echo "✅ Poetry já instalado"
fi

# 4. Instalar dependências
echo ""
echo "📦 Instalando dependências (pode demorar 5-10 min)..."

if [[ "$SETUP_TYPE" == "sortformer" ]]; then
    echo "   Instalando: Base + CUDA + Sortformer + Pyannote + VAD"
    poetry install --extras "cuda vad diarization-all"
else
    echo "   Instalando: Base + CUDA + VAD (WhisperX será instalado depois)"
    poetry install --extras "cuda vad"
fi

# 4.5. Instalar cuDNN para ctranslate2 (faster-whisper)
echo ""
echo "📦 Instalando cuDNN (necessário para faster-whisper GPU)..."
poetry run pip install nvidia-cudnn-cu12
echo "✅ cuDNN instalado"

# 4.6. Configurar LD_LIBRARY_PATH permanentemente
echo ""
echo "⚙️  Configurando LD_LIBRARY_PATH para cuDNN..."
CUDNN_PATH=$(poetry run python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    # Adicionar ao bashrc se ainda não existe
    if ! grep -q "nvidia.cudnn" ~/.bashrc; then
        echo '' >> ~/.bashrc
        echo '# cuDNN path for faster-whisper' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=$(poetry run python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo "✅ LD_LIBRARY_PATH configurado em ~/.bashrc"
    else
        echo "✅ LD_LIBRARY_PATH já configurado"
    fi
    # Exportar para sessão atual
    export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
else
    echo "⚠️  Aviso: Não foi possível detectar path do cuDNN"
fi

# 5. Configurar server-config.yaml
echo ""
echo "⚙️  Configurando server..."
if [ ! -f "server-config.yaml" ]; then
    cp server-config.example.yaml server-config.yaml

    # Configurar para GPU
    sed -i 's/host: "127.0.0.1"/host: "0.0.0.0"/' server-config.yaml
    sed -i 's/backend: "auto"/backend: "faster-whisper"/' server-config.yaml
    sed -i 's/device: "auto"/device: "cuda"/' server-config.yaml
    sed -i 's/compute_type: "auto"/compute_type: "float16"/' server-config.yaml

    # Usar modelo distil-large-v3 (6x mais rápido)
    sed -i 's/model: "base"/model: "distil-large-v3"/' server-config.yaml

    # Habilitar diarization Sortformer
    sed -i 's/backend: "none"  # "none"/backend: "sortformer"  # "sortformer"/' server-config.yaml

    echo "✅ server-config.yaml criado e configurado para GPU"
else
    echo "⚠️  server-config.yaml já existe, não sobrescrevendo"
    echo "   Se quiser reconfigurar, delete o arquivo e execute novamente"
fi

# 6. Testar imports
echo ""
echo "🧪 Testando imports..."
poetry run python -c "
import sys

# Base dependencies
try:
    import torch
    import torchaudio
    import faster_whisper
    print('✅ Base (faster-whisper + torch): OK')
except ImportError as e:
    print(f'❌ Base dependencies: {e}')
    sys.exit(1)

# WhisperX (opcional)
try:
    import whisperx
    print('✅ WhisperX: OK')
except ImportError:
    print('⚠️  WhisperX: Não instalado (opcional)')

# NeMo Sortformer (opcional)
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    print('✅ NeMo Sortformer: OK')
except ImportError:
    print('⚠️  NeMo Sortformer: Não instalado (opcional)')

# Pyannote (opcional)
try:
    import pyannote.audio
    print('✅ Pyannote Audio: OK')
except ImportError:
    print('⚠️  Pyannote Audio: Não instalado (opcional)')

# cuDNN
try:
    import nvidia.cudnn
    print('✅ cuDNN: OK')
except ImportError:
    print('⚠️  cuDNN: Não instalado (pode causar problemas com CUDA)')

print('')
print('✅ Imports base OK! Backends opcionais podem não estar instalados.')
"

# 6.5. Instalar WhisperX se necessário (baseado em argumento)
if [[ "$SETUP_TYPE" == "whisperx" ]]; then
    echo ""
    echo "=========================================="
    echo "📦 Instalando WhisperX v3.3.2"
    echo "=========================================="
    echo ""
    echo "WhisperX e Sortformer são MUTUAMENTE EXCLUSIVOS (conflito numpy)"
    echo "  - Desinstalando NeMo (se instalado)..."
    echo "  - Instalando WhisperX v3.3.2 (última versão com numpy 1.x)"
    echo ""

    poetry run pip uninstall nemo-toolkit -y 2>/dev/null || true
    poetry run pip install git+https://github.com/m-bain/whisperx.git@v3.3.2

    echo "✅ WhisperX v3.3.2 instalado"

    # Atualizar config para não usar sortformer
    if [ -f "server-config.yaml" ]; then
        sed -i 's/backend: "sortformer"/backend: "none"/' server-config.yaml
        echo "   server-config.yaml atualizado (diarization via WhisperX)"
    fi
else
    echo ""
    echo "✅ Sortformer instalado (SOTA 2025, 4 speakers)"
fi

# 7. Informações
echo ""
echo "=========================================="
echo "✅ Setup completo!"
echo "=========================================="
echo ""
echo "📂 Diretório: /workspace/stt-stream"
echo ""
echo "🚀 Para iniciar o servidor:"
echo "   cd /workspace/stt-stream"
echo "   bash scripts/runpod_start.sh"
echo ""
echo "📝 Ou manualmente (Poetry 2.0):"
echo "   cd /workspace/stt-stream"
echo "   source ~/.bashrc  # Carregar LD_LIBRARY_PATH"
echo "   poetry run python -m server.main --config server-config.yaml"
echo ""
echo "🔧 Ou com argumentos personalizados:"
echo "   poetry run python -m server.main --config server-config.yaml --model base --verbose"
echo ""
echo "📝 Em background (tmux recomendado):"
echo "   tmux new -s whisper"
echo "   cd /workspace/stt-stream"
echo "   bash scripts/runpod_start.sh"
echo "   # Pressione Ctrl+B, depois D para detach"
echo ""
echo "📊 Monitorar GPU (em outra janela):"
echo "   watch -n 1 nvidia-smi"
echo "   # ou"
echo "   nvtop  # (se instalado: apt install nvtop)"
echo ""
echo "🔗 URL do servidor (após iniciar):"
echo "   Verifique no dashboard RunPod a porta 9090 mapeada"
echo "   Exemplo: https://<pod-id>-9090.proxy.runpod.net"
echo ""
echo "🧪 Testar health:"
echo "   curl https://<pod-id>-9090.proxy.runpod.net/health"
echo ""
echo "⚙️  Configuração instalada:"
echo "   Setup: $SETUP_TYPE"
echo "   Backend: faster-whisper"
echo "   Modelo: distil-large-v3 (6x mais rápido!)"
if [[ "$SETUP_TYPE" == "sortformer" ]]; then
    echo "   Diarization: Sortformer (4 speakers) + Pyannote"
else
    echo "   Diarization: Pyannote (via WhisperX)"
fi
echo "   Device: CUDA (GPU)"
echo ""
echo "🔄 Para alternar entre setups:"
echo "   bash scripts/runpod_setup.sh sortformer"
echo "   bash scripts/runpod_setup.sh whisperx"
echo ""
echo "=========================================="
