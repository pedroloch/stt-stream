# Guia de Instalação

Instruções claras para instalar dependências em diferentes ambientes.

## TL;DR - Quick Install

### 🍎 Mac (Desenvolvimento Local)
```bash
chmod +x scripts/install-mac.sh
./scripts/install-mac.sh
```

**Inclui:**
- ✅ faster-whisper (CPU)
- ✅ MLX Whisper (se Apple Silicon M1/M2/M3)
- ✅ Pyannote diarization (CPU/MPS)

---

### 🚀 RunPod (CUDA GPU) - BATCH API ⭐ RECOMENDADO

**Setup completo**: clone + deps + config automaticamente

```bash
# Via URL (one-liner)
bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh)

# Ou se já clonou o repo:
cd /workspace/stt-stream
bash scripts/runpod_setup.sh  # Default: sortformer
```

**Inclui:**
- ✅ faster-whisper (CUDA)
- ✅ **Sortformer** diarization (SOTA 2025, 4 speakers)
- ✅ Pyannote diarization (backup)
- ✅ Configuração automática do server-config.yaml

---

### 🚀 RunPod (CUDA GPU) - Streaming com WhisperX (Alternativa)

**Para timestamps ultra-precisos (±50ms)**

```bash
# Via URL
bash <(curl -s https://raw.githubusercontent.com/pedroloch/stt-stream/main/scripts/runpod_setup.sh) whisperx

# Ou se já clonou:
cd /workspace/stt-stream
bash scripts/runpod_setup.sh whisperx
```

**Inclui:**
- ✅ faster-whisper (CUDA)
- ✅ **WhisperX v3.3.2** (timestamps ±50ms, 4x melhor)
- ✅ Pyannote diarization (integrado no WhisperX)
- ⚠️ **REMOVE Sortformer** (conflito numpy)

---

## ⚠️ RunPod: Sortformer vs WhisperX - Você DEVE escolher

**Eles são MUTUAMENTE EXCLUSIVOS** (conflito numpy)

| Feature | Sortformer ⭐ | WhisperX |
|---------|--------------|----------|
| **Uso recomendado** | **BATCH API (atual)** | Streaming (futuro) |
| **Speakers simultâneos** | 4 fixos (SOTA 2025) | Auto-detect |
| **Precisão timestamps** | ±200ms | ±50ms (4x melhor) |
| **Performance** | Rápido | Mais lento |
| **Conflito** | ⚠️ Com WhisperX | ⚠️ Com Sortformer |

**Default:** Sortformer (melhor para seu uso BATCH API)

**Para alternar:**
```bash
cd /workspace/stt-stream
bash scripts/runpod_setup.sh sortformer  # Volta para Sortformer
bash scripts/runpod_setup.sh whisperx    # Vai para WhisperX
```

---

## Entendendo as Dependências

### Backends de Transcrição
- **faster-whisper**: Funciona em CPU, CUDA, CoreML (sempre instalado)
- **MLX Whisper**: Apenas Apple Silicon (opcional)
- **WhisperX**: GPU only, **CONFLITA com NeMo** (numpy) - não recomendado

### Backends de Diarization
- **Pyannote**: CPU/MPS/CUDA, simples, funciona em qualquer lugar
- **Sortformer (NeMo)**: CUDA only, SOTA 2025, requer NeMo toolkit

### ⚠️ IMPORTANTE: Conflitos
- **WhisperX 3.3.2** requer numpy<2.0
- **NeMo 2.2.x** requer numpy<2.0
- **WhisperX 3.3.3+** requer numpy>=2.0 (incompatível com NeMo)

**Solução**: Não use WhisperX. Use faster-whisper ou MLX.

---

## Instalação Manual

### RunPod (CUDA GPU)

**Cenário: GPU NVIDIA com CUDA**

```bash
# 1. Dependências base + CUDA
poetry install --extras "cuda vad"

# 2. Diarization: Pyannote + Sortformer
poetry install --extras "diarization-all"

# 3. Verificar
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
poetry run python -c "import nemo; print('NeMo OK')"
poetry run python -c "from pyannote.audio import Pipeline; print('Pyannote OK')"
```

**Backends disponíveis:**
- ✅ faster-whisper (CUDA)
- ✅ Sortformer diarization (CUDA, 4 speakers)
- ✅ Pyannote diarization (CUDA/CPU)

### Mac (Apple Silicon)

**Cenário: MacBook M1/M2/M3**

```bash
# 1. Dependências base
poetry install

# 2. MLX (Apple Silicon GPU)
poetry install --extras "mlx"

# 3. VAD
poetry install --extras "vad"

# 4. Diarization: Pyannote
poetry install --extras "diarization-pyannote"

# 5. Verificar
poetry run python -c "import mlx_whisper; print('MLX OK')"
poetry run python -c "from pyannote.audio import Pipeline; print('Pyannote OK')"
```

**Backends disponíveis:**
- ✅ faster-whisper (CPU)
- ✅ MLX Whisper (Apple Silicon GPU)
- ✅ Pyannote diarization (MPS/CPU)

### Mac (Intel)

**Cenário: MacBook Intel (sem GPU acelerada)**

```bash
# 1. Dependências base
poetry install

# 2. VAD
poetry install --extras "vad"

# 3. Diarization: Pyannote
poetry install --extras "diarization-pyannote"
```

**Backends disponíveis:**
- ✅ faster-whisper (CPU)
- ✅ Pyannote diarization (CPU)

---

## Variáveis de Ambiente

Crie arquivo `.env` na raiz do projeto:

```bash
# HuggingFace token (necessário para Pyannote)
HUGGING_FACE_HUB_TOKEN=hf_...

# Opcional: Log level
LOG_LEVEL=INFO
```

### Como obter HuggingFace token:

1. Criar conta em https://huggingface.co
2. Ir em Settings → Access Tokens
3. Create new token (read)
4. Aceitar termos dos modelos:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

---

## Verificar Instalação

```bash
# Verificar imports
poetry run python -c "
import faster_whisper
print(f'✅ faster-whisper {faster_whisper.__version__}')

try:
    import mlx_whisper
    print('✅ MLX Whisper OK')
except ImportError:
    print('⚠️  MLX Whisper não instalado')

try:
    import nemo
    print(f'✅ NeMo {nemo.__version__}')
except ImportError:
    print('⚠️  NeMo não instalado')

try:
    from pyannote.audio import Pipeline
    print('✅ Pyannote OK')
except ImportError:
    print('⚠️  Pyannote não instalado')

import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'   CUDA: {torch.cuda.is_available()}')
print(f'   MPS: {torch.backends.mps.is_available()}')
"
```

---

## Troubleshooting

### Erro: "NeMo e WhisperX conflitam"

**Causa**: numpy incompatível

**Solução**: Desinstale WhisperX
```bash
poetry run pip uninstall whisperx -y
poetry install --extras "cuda diarization-sortformer"
```

### Erro: "Pyannote pipeline gated"

**Causa**: Não aceitou termos ou token inválido

**Solução**:
1. Aceite termos dos modelos (links acima)
2. Verifique `.env` com token correto
3. Reinicie servidor

### Erro: "CUDA out of memory"

**Causa**: GPU sem memória suficiente

**Solução**:
1. Use modelo menor: `--model-size base` em vez de `large-v3`
2. Desabilite diarization
3. Reduza batch size (código)

### Erro: "MLX não funciona"

**Causa**: Não é Apple Silicon ou instalação falhou

**Solução**:
```bash
# Verificar arquitetura
uname -m  # Deve ser "arm64"

# Reinstalar MLX
poetry install --extras "mlx" --no-cache
```

---

## Referência Rápida - Poetry Extras

```bash
# Tudo (RunPod)
poetry install --extras "cuda vad diarization-all"

# Mac Apple Silicon
poetry install --extras "mlx vad diarization-pyannote"

# CPU only (qualquer OS)
poetry install --extras "vad diarization-pyannote"

# Apenas Sortformer (CUDA required)
poetry install --extras "cuda diarization-sortformer"

# Apenas Pyannote
poetry install --extras "vad diarization-pyannote"
```

---

## Próximos Passos

Após instalação:

1. **Configurar servidor**: Edite `server-config.example.yaml`
2. **Iniciar servidor**: `poetry run python -m server.main`
3. **Testar**: `bun sandbox/test-batch-api.ts --diarization`

Ver [README.md](../README.md) para uso completo.
