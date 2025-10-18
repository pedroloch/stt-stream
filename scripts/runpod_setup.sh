#!/bin/bash
# RunPod Development Setup - Whisper Stream com Diarization
#
# Script para setup rápido de ambiente de desenvolvimento/teste no RunPod com GPU
#
# Uso:
#   bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh)
#
# Ou:
#   wget -O - https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh | bash

set -e  # Exit on error

echo "=========================================="
echo "🚀 Whisper Stream - RunPod GPU Setup"
echo "   (Ambiente de Desenvolvimento)"
echo "=========================================="

# 1. Verificar CUDA
echo ""
echo "📊 Verificando GPU/CUDA..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ CUDA version: {torch.version.cuda}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

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
echo "   Instalando: Base + CUDA + Sortformer + Pyannote + VAD"
poetry install --extras "all"

# 5. Configurar server-config.yaml
echo ""
echo "⚙️  Configurando server..."
if [ ! -f "server-config.yaml" ]; then
    cp server-config.example.yaml server-config.yaml

    # Configurar para GPU
    sed -i 's/host: "127.0.0.1"/host: "0.0.0.0"/' server-config.yaml
    sed -i 's/backend: "mlx"/backend: "cuda"/' server-config.yaml
    sed -i 's/device: "auto"/device: "cuda:0"/' server-config.yaml
    sed -i 's/compute_type: "auto"/compute_type: "float16"/' server-config.yaml

    # Habilitar diarization Sortformer
    sed -i 's/backend: "none"/backend: "sortformer"/' server-config.yaml

    echo "✅ server-config.yaml criado e configurado para GPU"
else
    echo "⚠️  server-config.yaml já existe, não sobrescrevendo"
    echo "   Se quiser reconfigurar, delete o arquivo e execute novamente"
fi

# 6. Testar imports
echo ""
echo "🧪 Testando imports..."
poetry run python -c "
import torch
import torchaudio
import faster_whisper
from nemo.collections.asr.models import SortformerEncLabelModel
print('✅ Todos os imports OK!')
"

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
echo "   poetry shell"
echo "   python -m server.main --config server-config.yaml"
echo ""
echo "📝 Ou em background (tmux recomendado):"
echo "   tmux new -s whisper"
echo "   cd /workspace/stt-stream && poetry shell"
echo "   python -m server.main --config server-config.yaml"
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
echo "=========================================="
