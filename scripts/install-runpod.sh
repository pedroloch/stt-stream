#!/bin/bash
# Script de instalação para RunPod (CUDA GPU)
# Instala tudo que funciona no RunPod

set -e  # Exit on error

echo "🚀 Instalando dependências para RunPod (CUDA GPU)..."
echo ""

# 1. Dependências base + CUDA + faster-whisper
echo "📦 Instalando dependências base..."
poetry install --extras "cuda vad"

# 2. Diarization: Pyannote E Sortformer (compatíveis)
echo "📦 Instalando backends de diarization (Pyannote + Sortformer)..."
poetry install --extras "diarization-all"

# 3. Verificar instalações
echo ""
echo "✅ Verificando instalações..."
echo ""

echo "🔍 Verificando Python e Poetry..."
poetry run python --version

echo ""
echo "🔍 Verificando PyTorch e CUDA..."
poetry run python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "🔍 Verificando faster-whisper..."
poetry run python -c "import faster_whisper; print(f'faster-whisper {faster_whisper.__version__}')"

echo ""
echo "🔍 Verificando NeMo (Sortformer)..."
poetry run python -c "import nemo; print(f'NeMo {nemo.__version__}')" || echo "❌ NeMo não instalado"

echo ""
echo "🔍 Verificando Pyannote..."
poetry run python -c "from pyannote.audio import Pipeline; print('Pyannote OK')" || echo "❌ Pyannote não instalado"

echo ""
echo "✅ Instalação completa!"
echo ""
echo "📝 Backends disponíveis:"
echo "  - faster-whisper (CUDA)"
echo "  - Diarization: Sortformer (NeMo) + Pyannote"
echo ""
echo "💡 Para iniciar o servidor:"
echo "   poetry run python -m server.main"
echo ""
echo "💡 Variáveis de ambiente necessárias (.env):"
echo "   HUGGING_FACE_HUB_TOKEN=hf_..."
echo ""
