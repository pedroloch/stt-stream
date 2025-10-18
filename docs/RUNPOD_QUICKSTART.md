# RunPod GPU Quickstart - Whisper Stream

Guia rápido para deploy e teste no RunPod com GPU.

## 1. Setup RunPod Pod

### 1.1 Criar Pod

1. Acesse https://runpod.io
2. **Templates**: Escolha `PyTorch` ou `RunPod PyTorch`
   - Já vem com CUDA, PyTorch, torchaudio instalados
3. **GPU**: Recomendado:
   - **RTX 4090** (24GB VRAM, ~$0.44/h) - Melhor custo-benefício
   - **RTX A6000** (48GB VRAM, ~$0.79/h) - Se precisar mais memória
   - **RTX 3090** (24GB VRAM, ~$0.34/h) - Mais barato
4. **Container Disk**: 50GB mínimo (modelos Whisper + Sortformer ocupam espaço)
5. **Volume (persistente)**: Opcional, mas recomendado (15GB+)
6. **Expose Ports**:
   - HTTP: `9090` (porta do servidor)
   - TCP: `22` (SSH, geralmente já exposto)

### 1.2 Conectar via SSH ou Web Terminal

```bash
# SSH (se configurado)
ssh root@<pod-ip> -p <ssh-port>

# Ou use o Web Terminal no dashboard do RunPod
```

## 2. Setup do Projeto

### 2.1 Clonar Repositório

```bash
cd /workspace  # Diretório padrão RunPod

git clone https://github.com/<seu-usuario>/whisper-stream.git
cd whisper-stream
```

### 2.2 Instalar Poetry

```bash
# Verificar PyTorch + CUDA (já instalado no template)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Instalar Poetry (se não estiver instalado)
curl -sSL https://install.python-poetry.org | python3 -

# Adicionar Poetry ao PATH (temporário)
export PATH="/root/.local/bin:$PATH"

# Ou adicionar permanentemente ao bashrc
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2.3 Instalar Dependências com Poetry

```bash
# Instalar dependências base + CUDA + Diarization
poetry install --extras "cuda diarization-sortformer"

# Ou para instalar TUDO (Sortformer + Pyannote):
poetry install --extras "all"

# Ou apenas Pyannote (mais leve):
poetry install --extras "cuda diarization-pyannote"

# Ativar ambiente Poetry
poetry shell
```

**Opções de extras disponíveis:**
- `cuda`: PyTorch + torchaudio (GPU)
- `vad`: VAD (Silero) - incluído em cuda
- `diarization-sortformer`: NeMo Sortformer (4 speakers)
- `diarization-pyannote`: Pyannote Audio (auto-detect)
- `diarization-all`: Ambos backends
- `all`: Tudo (recomendado para testes completos)

**Verificar instalação:**
```bash
# Dentro do poetry shell
python -c "import torch; import torchaudio; print('✅ PyTorch + torchaudio OK')"
python -c "import faster_whisper; print('✅ faster-whisper OK')"
python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('✅ NeMo OK')"
```

## 3. Configuração

### 3.1 Copiar e Editar Config

```bash
cp server-config.example.yaml server-config.yaml
nano server-config.yaml  # ou vim
```

### 3.2 Configuração Recomendada para GPU

```yaml
# server-config.yaml

server:
  host: "0.0.0.0"  # ⚠️ IMPORTANTE: 0.0.0.0 para aceitar conexões externas
  port: 9090
  max_clients: 5

whisper:
  # Modelo: use large-v3-turbo para melhor qualidade
  model: "large-v3-turbo"

  language: "pt"  # ou "auto" para auto-detect

  # Backend CUDA
  backend: "cuda"
  device: "cuda:0"  # ou "auto"
  compute_type: "float16"  # float16 é rápido e preciso em GPU

  # VAD (economiza GPU)
  use_vad: true
  vad_threshold: 0.25

  # Streaming settings
  min_chunk_size: 1.0
  buffer_trimming: "segment"
  buffer_trimming_sec: 10.0
  max_buffer_size: 20.0

  # Pause detection
  pause_detection_enabled: true
  pause_threshold_sec: 2.5
  auto_punctuate_on_pause: true

# ⭐ DIARIZATION (teste aqui!)
diarization:
  # Escolha: "sortformer" (SOTA) ou "pyannote" (simples)
  backend: "sortformer"

  # Sortformer: fixo em 4 speakers
  # Pyannote: pode especificar ou auto-detect
  num_speakers: null  # null = auto (apenas pyannote)

  device: "cuda"  # ⚠️ IMPORTANTE: usar GPU!

  # Apenas pyannote (se usar)
  min_speakers: 1
  max_speakers: 4
  auth_token: null  # HuggingFace token se necessário

performance:
  threads: "auto"
  batch_size: 1
  prefer_gpu: true

logging:
  level: "info"
  save_to_file: true
  log_dir: "./logs"
  log_audio_stats: false  # true para debug

debug:
  verbose: false
  show_hardware_info: true
```

## 4. Iniciar Servidor

### 4.1 Primeira Execução (Download de Modelos)

Na primeira vez, os modelos serão baixados (~2-5GB):
- Whisper large-v3-turbo: ~1.5GB
- Sortformer 4spk: ~500MB
- Silero VAD: ~50MB

```bash
# Ativar ambiente Poetry (se não estiver ativo)
poetry shell

# Executar servidor
poetry run python -m server.main --config server-config.yaml

# OU usando o script configurado (se preferir)
poetry run whisper-server --config server-config.yaml

# OU com log verbose (para debug)
poetry run python -m server.main --config server-config.yaml --verbose

# OU em background (tmux/screen)
tmux new -s whisper
poetry shell
python -m server.main --config server-config.yaml
# Ctrl+B, D para detach
```

### 4.2 Verificar Logs

```bash
# Se salvando em arquivo
tail -f logs/whisper-stream-*.log

# Se usando tmux
tmux attach -t whisper
```

**Você deve ver:**
```
🔧 Hardware detectado: NVIDIA GeForce RTX 4090 (CUDA 12.1)
✅ Backend Whisper inicializado: faster-whisper (device=cuda:0)
✅ Diarization habilitado (backend=sortformer, speakers=4)
🚀 Servidor WebSocket iniciado: ws://0.0.0.0:9090
```

## 5. Testar Conexão

### 5.1 Encontrar URL Pública

No dashboard do RunPod:
- Vá em **TCP Port Mappings** ou **HTTP Service**
- Encontre a URL pública mapeada para porta `9090`
- Formato: `https://<pod-id>-9090.proxy.runpod.net`

### 5.2 Testar Health Check

```bash
# No seu Mac local (ou qualquer máquina)
curl https://<pod-id>-9090.proxy.runpod.net/health

# Resposta esperada:
# {"status":"healthy","uptime":123.45}
```

### 5.3 Testar WebSocket

**No seu Mac:**

```bash
# Teste com wscat (se tiver instalado)
npm install -g wscat
wscat -c "wss://<pod-id>-9090.proxy.runpod.net"

# Ou use o cliente Python (se tiver no projeto)
```

**Teste com cliente Python:**

```python
# test_runpod.py
import asyncio
import websockets
import json
import numpy as np

async def test_connection():
    uri = "wss://<pod-id>-9090.proxy.runpod.net"

    async with websockets.connect(uri) as ws:
        # Receber welcome
        welcome = await ws.recv()
        print("📨 Welcome:", json.loads(welcome))

        # Enviar áudio de teste (silêncio)
        audio = np.zeros(16000, dtype=np.int16)  # 1s de silêncio
        await ws.send(audio.tobytes())

        # Receber resposta
        response = await ws.recv()
        print("📨 Response:", json.loads(response))

asyncio.run(test_connection())
```

## 6. Teste de Performance

### 6.1 Monitorar GPU

```bash
# Em outra janela tmux/SSH
watch -n 1 nvidia-smi

# Ou
nvtop  # Se instalado (apt install nvtop)
```

### 6.2 Benchmarks Esperados

**Whisper large-v3-turbo (RTX 4090):**
- Latência: ~50-150ms por chunk (1s de áudio)
- Throughput: ~10-20x realtime
- VRAM: ~2-4GB

**Sortformer Diarization:**
- Latência adicional: ~20-50ms por chunk
- VRAM: ~500MB-1GB

**Total esperado:**
- End-to-end: ~100-200ms por 1s de áudio
- VRAM total: ~3-5GB
- Sobra bastante espaço na RTX 4090 (24GB)!

## 7. Troubleshooting

### Erro: "CUDA out of memory"

```yaml
# Reduzir tamanho do modelo
whisper:
  model: "base"  # ou "small"

# Ou reduzir batch size
performance:
  batch_size: 1
```

### Erro: "Connection refused"

```bash
# Verificar se servidor está rodando
ps aux | grep python

# Verificar porta
netstat -tuln | grep 9090

# Verificar firewall (RunPod geralmente está OK)
```

### Erro: "NeMo/Pyannote não encontrado"

```bash
# Reinstalar com verbose
pip install nemo_toolkit[asr] --verbose
pip install pyannote.audio --verbose

# Verificar
python -c "import nemo; print(nemo.__version__)"
```

### Servidor travando/lento

```yaml
# Desabilitar diarization temporariamente
diarization:
  backend: "none"

# Reduzir buffer
whisper:
  buffer_trimming_sec: 5.0
  max_buffer_size: 10.0
```

## 8. Próximos Passos

### 8.1 Testes Recomendados

1. **Teste básico**: Transcrição sem diarization
   ```yaml
   diarization:
     backend: "none"
   ```

2. **Teste Sortformer**: 4 speakers simultâneos
   ```yaml
   diarization:
     backend: "sortformer"
   ```

3. **Teste Pyannote**: Auto-detect speakers
   ```yaml
   diarization:
     backend: "pyannote"
     num_speakers: null
   ```

4. **Benchmark**: Medir latência end-to-end

### 8.2 Métricas a Coletar

- Latência média (ms)
- Throughput (realtime factor)
- VRAM usage
- Accuracy de diarization (quantos speakers identifica corretamente)
- CPU/GPU utilization

### 8.3 Configuração de Produção

```yaml
server:
  max_clients: 10  # Aumentar se GPU aguenta

whisper:
  model: "large-v3-turbo"  # Melhor qualidade
  use_vad: true  # Economizar GPU

diarization:
  backend: "sortformer"  # SOTA accuracy
  device: "cuda"

logging:
  level: "info"  # Menos verbose
  log_audio_stats: false
```

## 9. Custos Estimados

**RTX 4090 (~$0.44/h):**
- 1 hora de testes: ~$0.44
- 10 horas (dia todo testando): ~$4.40
- 100 horas (4 dias full): ~$44

**Dicas para economizar:**
- Use **Spot Instances** (50% mais barato, mas pode ser interrompido)
- **PAUSE** o pod quando não estiver usando
- Configure **Auto-Stop** (idle timeout)

## 10. Script de Deploy Completo

### 10.1 Script de Inicialização SSH (`scripts/runpod_setup.sh`)

Salve este script como `scripts/runpod_setup.sh` e execute no RunPod:

```bash
#!/bin/bash
# RunPod Setup Script - Whisper Stream com Diarization
#
# Uso: bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh)
# Ou: wget -O - https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh | bash

set -e  # Exit on error

echo "=========================================="
echo "🚀 Whisper Stream - RunPod GPU Setup"
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
echo "   Instalando: Base + CUDA + Sortformer + Pyannote"
poetry install --extras "all"

# 5. Configurar server-config.yaml
echo ""
echo "⚙️  Configurando server..."
if [ ! -f "server-config.yaml" ]; then
    cp server-config.example.yaml server-config.yaml

    # Configurar para GPU
    sed -i 's/host: "127.0.0.1"/host: "0.0.0.0"/' server-config.yaml
    sed -i 's/backend: "auto"/backend: "cuda"/' server-config.yaml
    sed -i 's/device: "auto"/device: "cuda:0"/' server-config.yaml
    sed -i 's/compute_type: "auto"/compute_type: "float16"/' server-config.yaml

    # Habilitar diarization Sortformer
    sed -i 's/backend: "none"  # "none"/backend: "sortformer"  # "sortformer"/' server-config.yaml

    echo "✅ server-config.yaml criado e configurado"
else
    echo "⚠️  server-config.yaml já existe, não sobrescrevendo"
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
echo "🚀 Para iniciar o servidor:"
echo "   cd /workspace/stt-stream"
echo "   poetry shell"
echo "   python -m server.main --config server-config.yaml"
echo ""
echo "📝 Ou em background (tmux):"
echo "   tmux new -s whisper"
echo "   cd /workspace/stt-stream && poetry shell"
echo "   python -m server.main --config server-config.yaml"
echo "   # Ctrl+B, D para detach"
echo ""
echo "📊 Monitorar GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "=========================================="
```

### 10.2 Script de Start (`scripts/runpod_start.sh`)

Script simplificado para iniciar o servidor após setup:

```bash
#!/bin/bash
# RunPod Start Script - Inicia servidor Whisper Stream
#
# Uso: bash scripts/runpod_start.sh

cd /workspace/stt-stream

# Ativar Poetry e iniciar
poetry shell
poetry run python -m server.main --config server-config.yaml
```

### 10.3 Uso Rápido

**Primeira vez (setup completo):**
```bash
# No terminal SSH do RunPod
bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh)
```

**Depois (apenas start):**
```bash
cd /workspace/stt-stream
poetry run python -m server.main --config server-config.yaml

# Ou em background
tmux new -s whisper
cd /workspace/stt-stream
poetry shell
python -m server.main --config server-config.yaml
# Ctrl+B, D para detach
```

---

**Boa sorte! 🚀**

Qualquer dúvida durante o deploy, me avise!
