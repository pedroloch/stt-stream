# Dependências do Whisper Stream

Guia completo das dependências do projeto, organizadas por plataforma e caso de uso.

## Tabela de Conteúdos

- [Dependências Base](#dependências-base)
- [Por Plataforma](#por-plataforma)
- [Por Caso de Uso](#por-caso-de-uso)
- [Instalação](#instalação)
- [Verificação](#verificação)

---

## Dependências Base

Instaladas por padrão em todas as plataformas:

| Dependência | Versão | Propósito |
|-------------|--------|-----------|
| **Python** | 3.11+ | Runtime |
| **aiohttp** | ^3.11.0 | WebSocket server |
| **aiohttp-cors** | ^0.7.0 | CORS support |
| **numpy** | 1.x | Audio processing (< 2.0.0 para NeMo) |
| **pyyaml** | ^6.0.2 | Config files |
| **faster-whisper** | ^1.2.0 | Backend principal (CPU/GPU) |

---

## Por Plataforma

### macOS Apple Silicon (M1/M2/M3)

**Recomendado:** Usar MLX backend (otimizado para Metal).

```bash
poetry install --extras "mlx"
```

**Dependências adicionais:**
- `mlx-whisper` ^0.4.0

**Backends disponíveis:**
- ✅ MLX (recomendado)
- ✅ faster-whisper (CPU)
- ❌ WhisperX (apenas Linux CUDA)

---

### Linux com CUDA GPU (RunPod, Cloud GPU)

**Recomendado:** Instalar tudo para máxima funcionalidade.

```bash
poetry install --extras "all"
```

**Dependências GPU:**
- `torch` ^2.6.0
- `torchaudio` ^2.6.0
- `nvidia-cudnn-cu12` ^9.1.0 (para faster-whisper GPU)

**Backends opcionais:**
- `whisperx` (git, diarization integrada)
- `nemo-toolkit[asr]` ^2.2.0 (Sortformer diarization)
- `pyannote-audio` ^3.3.0 (Pyannote diarization)

**Backends disponíveis:**
- ✅ faster-whisper (GPU, recomendado)
- ✅ WhisperX (GPU, diarization integrada)
- ✅ Sortformer (4 speakers, SOTA)
- ✅ Pyannote (auto-detect speakers)

---

### Linux CPU / Windows CPU

**Recomendado:** Base apenas (faster-whisper CPU).

```bash
poetry install
```

**Backends disponíveis:**
- ✅ faster-whisper (CPU)
- ❌ WhisperX (apenas CUDA)
- ❌ MLX (apenas Apple Silicon)

---

## Por Caso de Uso

### 1. Desenvolvimento Local (Básico)

**Mac:**
```bash
poetry install --extras "mlx"
```

**Linux/Windows:**
```bash
poetry install
```

---

### 2. RunPod GPU - Transcrição Básica

Apenas faster-whisper com GPU:

```bash
poetry install --extras "cuda"
```

**Inclui:**
- faster-whisper (GPU)
- torch + torchaudio
- nvidia-cudnn-cu12

---

### 3. RunPod GPU - Com Diarization (Sortformer)

Transcrição + identificação de 4 speakers:

```bash
poetry install --extras "cuda diarization-sortformer"
```

**Inclui:**
- faster-whisper (GPU)
- torch + torchaudio
- nvidia-cudnn-cu12
- nemo-toolkit[asr] (Sortformer)

---

### 4. RunPod GPU - WhisperX (Backend Alternativo)

⚠️ **ATENÇÃO:** WhisperX e NeMo são **MUTUAMENTE EXCLUSIVOS**!

**Conflito de versão numpy:**
- WhisperX 3.3.3+ requer numpy >= 2.0
- NeMo 2.2.x requer numpy < 2.0
- **Solução:** Use WhisperX v3.3.2 (última versão com numpy 1.x)

**Instalação manual (após `poetry install --extras "cuda"`):**

```bash
poetry install --extras "cuda"

# Desinstalar NeMo se estiver instalado
poetry run pip uninstall nemo-toolkit -y

# Instalar WhisperX v3.3.2 (compatível com numpy 1.x)
poetry run pip install git+https://github.com/m-bain/whisperx.git@v3.3.2
```

**Inclui:**
- faster-whisper (fallback)
- torch + torchaudio
- nvidia-cudnn-cu12
- whisperx v3.3.2 (manual, diarization pyannote integrada)

**Para voltar ao Sortformer:**
```bash
poetry run pip uninstall whisperx -y
poetry install --extras "diarization-sortformer"
```

---

### 5. RunPod GPU - Completo (Sortformer + Pyannote)

Todos backends exceto WhisperX (que é mutuamente exclusivo):

```bash
poetry install --extras "all"
```

**Inclui:**
- faster-whisper (GPU)
- torch + torchaudio
- nvidia-cudnn-cu12
- nemo-toolkit[asr] (Sortformer)
- pyannote-audio

**NOTA:** `--extras "all"` NÃO inclui WhisperX devido ao conflito numpy. Para usar WhisperX, siga as instruções da seção 4 acima.

---

## Instalação

### Via Poetry

```bash
# Base apenas
poetry install

# CUDA básico
poetry install --extras "cuda"

# Tudo
poetry install --extras "all"
```

### Via Makefile

```bash
# Base
make install

# CUDA
make install-cuda

# Tudo
make install-all
```

### RunPod (Automatizado)

```bash
bash scripts/runpod_setup.sh
```

Instala automaticamente:
- Base dependencies
- CUDA (torch, torchaudio, cudnn)
- WhisperX
- NeMo Sortformer
- Pyannote Audio
- FFmpeg (para NeMo)

---

## Verificação

### Via Script

```bash
poetry run python scripts/check_deps.py
```

**Saída esperada:**
```
======================================================================
📦 WHISPER STREAM - DEPENDENCY CHECK
======================================================================

⭐ REQUIRED (Base):
----------------------------------------------------------------------
  aiohttp                        ✅ v3.11.0
  numpy                          ✅ v1.26.4
  pyyaml                         ✅
  faster-whisper                 ✅ v1.2.0

🔧 BACKENDS (Optional):
----------------------------------------------------------------------
  whisperx                       ✅ (git)
  mlx-whisper                    ⚠️  opcional (Mac Apple Silicon apenas)

🎮 CUDA/GPU (Optional):
----------------------------------------------------------------------
  torch                          ✅ v2.6.0 (CUDA 12.4)
  torchaudio                     ✅ v2.6.0
  nvidia-cudnn-cu12              ✅ @ /path/to/cudnn

👥 DIARIZATION (Optional):
----------------------------------------------------------------------
  nemo-toolkit                   ✅ v2.2.0 (Sortformer)
  pyannote-audio                 ✅ (Pyannote diarization)

======================================================================

✅ Todas dependências obrigatórias instaladas!
⭐ Extras instalados: CUDA/GPU, WhisperX, Sortformer, Pyannote
```

### Via Makefile

```bash
make check-deps
```

---

## Troubleshooting

### Problema: `ModuleNotFoundError: No module named 'whisperx'`

**Causa:** WhisperX e NeMo são mutuamente exclusivos (conflito numpy).

**Solução:** Instale WhisperX v3.3.2 manualmente (última versão com numpy 1.x):

```bash
# Desinstalar NeMo se estiver instalado
poetry run pip uninstall nemo-toolkit -y

# Instalar WhisperX v3.3.2
poetry install --extras "cuda"
poetry run pip install git+https://github.com/m-bain/whisperx.git@v3.3.2
```

**Para voltar ao Sortformer:**
```bash
poetry run pip uninstall whisperx -y
poetry install --extras "diarization-sortformer"
```

---

### Problema: `Unable to load libcudnn_ops.so`

**Solução:** cuDNN não instalado ou não no PATH.

```bash
# Instalar
poetry run pip install nvidia-cudnn-cu12

# Configurar LD_LIBRARY_PATH (RunPod)
export LD_LIBRARY_PATH=$(poetry run python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH

# Ou usar script
bash scripts/runpod_start.sh
```

---

### Problema: `nemo.collections.asr not found`

**Solução:** NeMo toolkit não instalado.

```bash
poetry install --extras "diarization-sortformer"
```

---

### Problema: `numpy version conflict`

**Solução:** NeMo requer numpy < 2.0.0.

```bash
# Já configurado no pyproject.toml
numpy = "^1.24.0,<2.0.0"
```

---

## Matriz de Compatibilidade

| Backend | macOS Intel | macOS M1/M2/M3 | Linux CPU | Linux CUDA | Windows |
|---------|-------------|----------------|-----------|------------|---------|
| **faster-whisper** | ✅ CPU | ✅ CPU | ✅ CPU | ✅ GPU | ✅ CPU |
| **MLX** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **WhisperX** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Sortformer** | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ✅ GPU | ⚠️ CPU |
| **Pyannote** | ⚠️ CPU | ⚠️ CPU | ⚠️ CPU | ✅ GPU | ⚠️ CPU |

**Legenda:**
- ✅ Suportado e recomendado
- ⚠️ Funciona mas lento
- ❌ Não suportado

---

## Referências

- [Poetry Dependency Management](https://python-poetry.org/docs/managing-dependencies/)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

---

**Última atualização:** 2025-10-18
