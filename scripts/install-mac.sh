#!/bin/bash
# Script de instalação para Mac (Apple Silicon com MLX ou Intel)
# Instala tudo que funciona no Mac

set -e  # Exit on error

echo "🍎 Instalando dependências para Mac..."
echo ""

# Detectar Apple Silicon
ARCH=$(uname -m)

# 1. Dependências base
echo "📦 Instalando dependências base..."
poetry install

# 2. MLX (Apple Silicon only)
if [ "$ARCH" = "arm64" ]; then
    echo "📦 Instalando MLX Whisper (Apple Silicon)..."
    poetry install --extras "mlx"
else
    echo "⚠️  MLX não disponível (apenas Apple Silicon)"
fi

# 3. VAD (Silero)
echo "📦 Instalando VAD..."
poetry install --extras "vad"

# 4. Diarization: Pyannote (funciona em Mac)
echo "📦 Instalando Pyannote para diarization..."
poetry install --extras "diarization-pyannote"

# 5. Verificar instalações
echo ""
echo "✅ Verificando instalações..."
echo ""

echo "🔍 Verificando Python e Poetry..."
poetry run python --version

echo ""
echo "🔍 Verificando PyTorch..."
poetry run python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'MPS disponível: {torch.backends.mps.is_available()}')" || echo "⚠️  PyTorch opcional não instalado"

echo ""
echo "🔍 Verificando faster-whisper..."
poetry run python -c "import faster_whisper; print(f'faster-whisper {faster_whisper.__version__}')"

if [ "$ARCH" = "arm64" ]; then
    echo ""
    echo "🔍 Verificando MLX Whisper..."
    poetry run python -c "import mlx_whisper; print('MLX Whisper OK')" || echo "⚠️  MLX não instalado"
fi

echo ""
echo "🔍 Verificando Pyannote..."
poetry run python -c "from pyannote.audio import Pipeline; print('Pyannote OK')" || echo "⚠️  Pyannote não instalado"

echo ""
echo "✅ Instalação completa!"
echo ""
echo "📝 Backends disponíveis:"
if [ "$ARCH" = "arm64" ]; then
    echo "  - faster-whisper (CPU)"
    echo "  - MLX Whisper (Apple Silicon GPU)"
else
    echo "  - faster-whisper (CPU)"
fi
echo "  - Diarization: Pyannote (CPU/MPS)"
echo ""
echo "💡 Para iniciar o servidor:"
echo "   poetry run python -m server.main"
echo ""
echo "💡 Variáveis de ambiente necessárias (.env):"
echo "   HUGGING_FACE_HUB_TOKEN=hf_..."
echo ""
