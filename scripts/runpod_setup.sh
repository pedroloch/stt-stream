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

    # Adicionar ao PATH permanentemente
    if ! grep -q '/root/.local/bin' ~/.bashrc; then
        echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    fi

    # Carregar PATH para sessão atual
    export PATH="/root/.local/bin:$PATH"

    # Verificar se Poetry agora está disponível
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry instalado: $(poetry --version)"
    else
        echo "⚠️  Poetry instalado mas não encontrado no PATH"
        echo "   Execute: source ~/.bashrc"
        echo "   Ou use caminho completo: /root/.local/bin/poetry"
    fi
else
    echo "✅ Poetry já instalado: $(poetry --version)"
fi

# Garantir que poetry está disponível (fallback para caminho completo)
if ! command -v poetry &> /dev/null; then
    if [ -f "/root/.local/bin/poetry" ]; then
        echo "⚙️  Configurando PATH do Poetry..."
        export PATH="/root/.local/bin:$PATH"
        alias poetry='/root/.local/bin/poetry'
    else
        echo "❌ Poetry não encontrado em /root/.local/bin/poetry"
        echo "   Por favor, instale manualmente ou verifique instalação"
        exit 1
    fi
fi

# 3.5. Limpeza de conflitos (garantir instalação limpa)
echo ""
echo "🧹 Limpando dependências conflitantes..."

# Remover versões incorretas do NeMo se existirem
poetry run pip uninstall nemo-toolkit -y 2>/dev/null || true

# Remover cuDNN antigo que pode estar no sistema
poetry run pip uninstall nvidia-cudnn-cu12 nvidia-cudnn-cu11 -y 2>/dev/null || true

echo "✅ Limpeza concluída"

# 4. Instalar dependências
echo ""
echo "📦 Instalando dependências (pode demorar 5-10 min)..."

if [[ "$SETUP_TYPE" == "sortformer" ]]; then
    echo "   Instalando: Base + CUDA + VAD"
    poetry install --extras "cuda vad"

    # Instalar NeMo 2.5.x EXPLICITAMENTE (não deixar poetry escolher)
    echo ""
    echo "📦 Instalando NeMo 2.5.x (versão testada com modelo Sortformer)..."
    poetry run pip install "nemo-toolkit[asr]>=2.5.0,<2.6.0"

    # Instalar Pyannote
    echo "📦 Instalando Pyannote..."
    poetry run pip install "pyannote-audio>=3.3.0"

else
    echo "   Instalando: Base + CUDA + VAD (WhisperX será instalado depois)"
    poetry install --extras "cuda vad"
fi

# 4.5. Instalar cuDNN para ctranslate2 (faster-whisper) e PyTorch
echo ""
echo "📦 Instalando cuDNN 9.x (necessário para PyTorch 2.6 + faster-whisper)..."

# IMPORTANTE: Desinstalar qualquer cuDNN anterior (pode estar no sistema)
poetry run pip uninstall nvidia-cudnn-cu12 nvidia-cudnn-cu11 -y 2>/dev/null || true

# Instalar versão correta (9.x para PyTorch 2.6)
poetry run pip install "nvidia-cudnn-cu12>=9.1.0,<10.0.0" --force-reinstall

echo "✅ cuDNN 9.x instalado"

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

# 6. Testar imports COM DIAGNÓSTICO DETALHADO
echo ""
echo "🧪 Verificando instalação e diagnosticando..."
poetry run python -c "
import sys

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('📊 DIAGNÓSTICO DE DEPENDÊNCIAS')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print()

# 1. PyTorch e CUDA
print('1️⃣  PyTorch e CUDA:')
try:
    import torch
    print(f'   ✅ PyTorch: {torch.__version__}')
    print(f'   ✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   ✅ CUDA version: {torch.version.cuda}')
        print(f'   ✅ GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('   ⚠️  CUDA não disponível (CPU only)')
except ImportError as e:
    print(f'   ❌ PyTorch não disponível: {e}')
    sys.exit(1)
print()

# 2. cuDNN
print('2️⃣  cuDNN (requerido para faster-whisper + CUDA):')
try:
    import nvidia.cudnn
    cudnn_version = nvidia.cudnn.__version__
    print(f'   ✅ cuDNN: {cudnn_version}')

    # Verificar se versão é compatível (9.x para PyTorch 2.6)
    major_version = int(cudnn_version.split('.')[0])
    if major_version < 9:
        print(f'   ⚠️  cuDNN {cudnn_version} pode ser incompatível com PyTorch {torch.__version__}')
        print('   💡 PyTorch 2.6+ requer cuDNN 9.x')
    else:
        print(f'   ✅ Versão compatível com PyTorch {torch.__version__}')
except ImportError:
    print('   ❌ cuDNN não instalado')
    print('   💡 Instale com: poetry run pip install nvidia-cudnn-cu12')
except Exception as e:
    print(f'   ⚠️  Erro ao verificar cuDNN: {e}')
print()

# 3. faster-whisper
print('3️⃣  faster-whisper (backend base):')
try:
    import faster_whisper
    print(f'   ✅ faster-whisper: {faster_whisper.__version__}')
except ImportError as e:
    print(f'   ❌ faster-whisper: {e}')
    sys.exit(1)
print()

# 4. NeMo (para Sortformer)
print('4️⃣  NeMo Toolkit (Sortformer diarization):')
try:
    import nemo
    print(f'   ✅ NeMo: {nemo.__version__}')

    # Verificar se versão é compatível (2.5.x requerido para Sortformer)
    version_parts = nemo.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0

    if major == 2 and minor >= 5:
        print('   ✅ Versão compatível com modelo Sortformer (2.5+)')

        # Tentar importar modelo
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
            print('   ✅ SortformerEncLabelModel importado com sucesso')
        except ImportError as e:
            print(f'   ⚠️  Não foi possível importar SortformerEncLabelModel: {e}')
    else:
        print(f'   ⚠️  NeMo {nemo.__version__} pode não ter suporte ao modelo Sortformer')
        print('   💡 Recomendado: NeMo >= 2.5.0 (tem parâmetros spkcache_len, etc)')

except ImportError:
    print('   ⚠️  NeMo não instalado (diarization Sortformer não disponível)')
    print('   💡 Instale com: poetry run pip install \"nemo-toolkit[asr]>=2.5.0\"')
except Exception as e:
    print(f'   ⚠️  Erro ao verificar NeMo: {e}')
print()

# 5. Pyannote
print('5️⃣  Pyannote Audio (diarization alternativo):')
try:
    import pyannote.audio
    print(f'   ✅ Pyannote Audio instalado')

    # Verificar se pode importar Pipeline
    try:
        from pyannote.audio import Pipeline
        print('   ✅ Pipeline disponível')
    except ImportError as e:
        print(f'   ⚠️  Pipeline não disponível: {e}')

except ImportError:
    print('   ⚠️  Pyannote não instalado')
    print('   💡 Instale com: poetry run pip install pyannote-audio')
except Exception as e:
    print(f'   ⚠️  Erro ao verificar Pyannote: {e}')
print()

# 6. WhisperX (conflita com NeMo)
print('6️⃣  WhisperX (streaming + diarization integrado):')
try:
    import whisperx
    print(f'   ✅ WhisperX instalado')
    print('   ⚠️  WhisperX e NeMo são MUTUAMENTE EXCLUSIVOS (conflito numpy)')
except ImportError:
    print('   ⚠️  WhisperX não instalado')
print()

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('✅ DIAGNÓSTICO COMPLETO')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐛 TROUBLESHOOTING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "❌ Erro: 'CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH'"
echo "   Causa: PyTorch 2.6 compilado com cuDNN 9.x, mas sistema tem 8.x"
echo "   Solução:"
echo "     poetry run pip uninstall nvidia-cudnn-cu12 -y"
echo "     poetry run pip install 'nvidia-cudnn-cu12>=9.1.0' --force-reinstall"
echo "     source ~/.bashrc"
echo ""
echo "❌ Erro: 'spkcache_len' parameter not accepted"
echo "   Causa: NeMo 2.2.x não tem parâmetros do modelo Sortformer"
echo "   Solução:"
echo "     poetry run pip uninstall nemo-toolkit -y"
echo "     poetry run pip install 'nemo-toolkit[asr]>=2.5.0,<2.6.0'"
echo ""
echo "❌ Erro: 'WhisperX e NeMo conflitam' (numpy)"
echo "   Causa: Ambos instalados ao mesmo tempo"
echo "   Solução:"
echo "     bash scripts/runpod_setup.sh sortformer  # Remove WhisperX"
echo "     bash scripts/runpod_setup.sh whisperx    # Remove NeMo"
echo ""
echo "❌ Erro: 'Out of memory' (CUDA)"
echo "   Causa: GPU sem VRAM suficiente"
echo "   Solução:"
echo "     1. Usar modelo menor (base em vez de large-v3)"
echo "     2. Desabilitar diarization temporariamente"
echo "     3. Trocar para GPU maior (A5000/A6000)"
echo ""
echo "❌ Servidor não inicia / erros de import"
echo "   Solução: Refazer setup completo"
echo "     cd /workspace/stt-stream"
echo "     poetry env remove --all"
echo "     bash scripts/runpod_setup.sh sortformer"
echo ""
echo "📚 Documentação completa:"
echo "   - Batch API: docs/BATCH_API.md"
echo "   - Sortformer: docs/INSTALL.md"
echo "   - WhisperX: docs/WHISPERX_RUNPOD.md"
echo ""
echo "=========================================="
