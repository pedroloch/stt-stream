# 🎤 WhisperX no RunPod - Guia Completo

Deploy do Whisper Stream com **WhisperX** (diarization + word timestamps precisos) no RunPod com GPU CUDA.

---

## 🌟 O que é WhisperX?

WhisperX é uma extensão do Whisper que adiciona:

1. **Word timestamps precisos** - Alinhamento com wav2vec2 (±50ms)
2. **Speaker diarization** - Identifica quem está falando (SPEAKER_00, SPEAKER_01, etc)
3. **Suporte multilingual** - Português, Inglês, 90+ idiomas

**Perfeito para:**
- Reuniões com múltiplos participantes
- Entrevistas
- Podcasts
- Call centers
- Atas de reunião automatizadas

---

## 📋 Pré-requisitos

### 1. HuggingFace Token (OBRIGATÓRIO para diarization)

WhisperX usa `pyannote/speaker-diarization` que requer aceitar termos de uso:

**Passo 1:** Criar conta no HuggingFace
- Acesse: https://huggingface.co/join
- Confirme email

**Passo 2:** Aceitar termos dos modelos
Acesse e clique em "Agree and access repository":
1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0

**Passo 3:** Criar Access Token
- Acesse: https://huggingface.co/settings/tokens
- Clique em "New token"
- Nome: `whisperx-runpod`
- Type: **Read**
- Copie o token (ex: `hf_AbCdEfGhIjKlMnOpQrStUvWxYz`)

⚠️ **IMPORTANTE:** Guarde esse token! Você vai precisar dele no deploy.

---

## 🚀 Build e Deploy

### 1. Build da Imagem Docker

```bash
# Build WhisperX
./scripts/build-docker.sh whisperx
```

**Tempo estimado:** 5-10 minutos

### 2. Push para Docker Hub

```bash
export DOCKER_USERNAME=seu-username
docker tag whisper-stream:whisperx ${DOCKER_USERNAME}/whisper-stream:whisperx
docker push ${DOCKER_USERNAME}/whisper-stream:whisperx
```

**Tempo estimado:** 10-30 minutos (dependendo da conexão)

### 3. Deploy no RunPod

**Acesse:** https://runpod.io/console/pods

**Clique em:** Deploy

#### GPU Recomendadas:

| GPU | VRAM | Preço/hora | Modelo Recomendado |
|-----|------|-----------|-------------------|
| RTX A4000 | 16GB | $0.44 | `base`, `small` |
| **RTX A5000** ⭐ | 24GB | $0.64 | `medium`, `large-v2` |
| RTX A6000 | 48GB | $0.79 | `large-v3` |

WhisperX carrega **3 modelos** (Whisper + Alignment + Diarization), então precisa mais VRAM que Faster-Whisper.

#### Container Configuration:

| Campo | Valor |
|-------|-------|
| Container Image | `seu-username/whisper-stream:whisperx` |
| Container Disk | 25 GB |
| Volume Disk | 80 GB (modelos de diarization são grandes!) |
| Expose HTTP Ports | 9090 |

#### Environment Variables:

```bash
WHISPER_BACKEND=whisperx
WHISPER_MODEL=base                    # ou small, medium, large-v2
WHISPER_LANGUAGE=pt
SERVER_HOST=0.0.0.0
SERVER_PORT=9090
LOG_LEVEL=info

# ⭐ OBRIGATÓRIO para diarization
HUGGING_FACE_HUB_TOKEN=hf_seu_token_aqui
```

⚠️ **Substitua `hf_seu_token_aqui` pelo seu token real!**

---

## 🧪 Testar

### 1. Health Check

```bash
# Substituir pelo IP do seu pod
curl http://123.45.67.89:9090/health
```

**Output esperado:**
```json
{
  "status": "healthy",
  "processor_ready": true
}
```

### 2. Info do Backend

```bash
curl http://123.45.67.89:9090/info
```

**Output esperado:**
```json
{
  "server": {...},
  "processor": {
    "backend_type": "whisperx",
    "model": "base",
    "language": "pt",
    "capabilities": [
      "transcription",
      "word_timestamps",
      "speaker_diarization",  // ⭐ Diarization ativo!
      "vad",
      "streaming"
    ]
  }
}
```

### 3. Conectar Cliente Bun

Editar `src/index.ts`:

```typescript
// Substituir pelo IP do RunPod
const WS_URL = 'ws://123.45.67.89:9090/ws';
```

Rodar cliente:
```bash
DEBUG=1 bun start
```

### 4. Testar com Múltiplos Speakers

Fale alternando entre 2+ pessoas. Você deve ver:

```json
{
  "type": "transcription",
  "text": "Olá, como vai?",
  "is_final": true,
  "speaker": "SPEAKER_00",        // ⭐ Speaker principal
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "Olá, como vai?",
      "speaker_id": "SPEAKER_00",   // ⭐ Speaker do segment
      "words": [
        {
          "word": "Olá",
          "start": 0.0,
          "end": 0.5,
          "probability": 0.92,
          "speaker_id": "SPEAKER_00"  // ⭐ Speaker por palavra!
        },
        ...
      ]
    }
  ]
}
```

---

## ⚙️ Configurações Avançadas

### Limitar Número de Speakers

Se você sabe quantas pessoas vão falar, adicione:

```yaml
# server-config.yaml
whisper:
  backend: whisperx
  min_speakers: 2
  max_speakers: 4
```

Ou via ENV vars:
```bash
MIN_SPEAKERS=2
MAX_SPEAKERS=4
```

### Modelos Disponíveis

| Modelo | VRAM | Qualidade | Velocidade |
|--------|------|-----------|------------|
| `tiny` | ~3GB | ⭐⭐ | 🚀🚀🚀🚀🚀 |
| `base` | ~4GB | ⭐⭐⭐ | 🚀🚀🚀🚀 |
| `small` | ~6GB | ⭐⭐⭐⭐ | 🚀🚀🚀 |
| `medium` | ~10GB | ⭐⭐⭐⭐ | 🚀🚀 |
| `large-v2` | ~15GB | ⭐⭐⭐⭐⭐ | 🚀 |
| `large-v3` | ~15GB | ⭐⭐⭐⭐⭐ | 🚀 |

**Recomendado:** `base` ou `small` para produção (bom equilíbrio)

### Batch Size

Para GPU mais potente, aumente o batch_size:

```yaml
whisper:
  batch_size: 32  # Padrão: 16
```

---

## 🐛 Troubleshooting

### "HF token não fornecido - diarization desabilitado"

**Causa:** Token do HuggingFace não foi configurado

**Solução:**
1. Verificar se `HUGGING_FACE_HUB_TOKEN` está nas env vars do pod
2. Verificar se o token é válido
3. Verificar se você aceitou os termos dos modelos pyannote

### "Not enough VRAM"

**Causa:** GPU sem memória suficiente

**Soluções:**
- Trocar para modelo menor (`large-v3` → `base`)
- Ou trocar para GPU maior (RTX A5000/A6000)
- Reduzir batch_size

### Diarization muito lento

**Causa:** Diarization é computacionalmente pesado

**Soluções:**
1. Usar GPU mais potente (A5000 ou A6000)
2. Limitar número de speakers (`max_speakers=3`)
3. Para transcrição sem diarization, use `faster-whisper` (3x mais rápido)

### Speakers incorretos

**Causa:** Áudio com muito ruído ou overlap

**Soluções:**
1. Melhorar qualidade do áudio
2. Evitar overlap (pessoas falando ao mesmo tempo)
3. Usar microfones individuais se possível

---

## 💰 Custos Estimados

### GPU Recomendada (RTX A5000):

- **Preço:** $0.64/hora
- **Modelo:** base ou small
- **Uso 8h/dia:** ~$5/dia = $150/mês
- **Uso 24/7:** ~$460/mês

### Comparação com Faster-Whisper:

| Feature | Faster-Whisper | WhisperX |
|---------|---------------|----------|
| Word Timestamps | ✅ ±200ms | ✅ ±50ms (4x melhor!) |
| Diarization | ❌ | ✅ |
| Velocidade | 🚀🚀🚀🚀 | 🚀🚀 (mais lento) |
| VRAM | ~2GB (base) | ~4GB (base) |
| GPU Mínima | RTX 3090 | RTX A5000 |

**Quando usar WhisperX:**
- ✅ Reuniões/conferências
- ✅ Entrevistas
- ✅ Podcasts
- ✅ Timestamps ultra-precisos

**Quando usar Faster-Whisper:**
- ✅ Transcrição individual
- ✅ Legendagem
- ✅ Velocidade é prioridade
- ✅ Custo menor

---

## 📚 Recursos Adicionais

- **WhisperX GitHub:** https://github.com/m-bain/whisperx
- **Pyannote Audio:** https://github.com/pyannote/pyannote-audio
- **RunPod Docs:** https://docs.runpod.io/
- **HuggingFace Tokens:** https://huggingface.co/settings/tokens

---

## 🎯 Quick Start - Resumo

```bash
# 1. Build
./scripts/build-docker.sh whisperx

# 2. Push
export DOCKER_USERNAME=seu-username
docker tag whisper-stream:whisperx ${DOCKER_USERNAME}/whisper-stream:whisperx
docker push ${DOCKER_USERNAME}/whisper-stream:whisperx

# 3. Deploy no RunPod
# - GPU: RTX A5000
# - Image: seu-username/whisper-stream:whisperx
# - ENV: WHISPER_BACKEND=whisperx, WHISPER_MODEL=base, HUGGING_FACE_HUB_TOKEN=hf_...

# 4. Testar
curl http://<POD_IP>:9090/health

# 5. Conectar cliente
# Editar src/index.ts → WS_URL = 'ws://<POD_IP>:9090/ws'
DEBUG=1 bun start
```

✅ **Pronto! Speaker diarization funcionando!** 🎤
