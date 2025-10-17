# 🚀 Deploy no RunPod com GPU

Guia completo para deploy do Whisper Stream Server no RunPod com diferentes backends.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Build Local](#build-local)
4. [Push para Registry](#push-para-registry)
5. [Deploy no RunPod](#deploy-no-runpod)
6. [Configuração de Backends](#configuração-de-backends)
7. [Conectar Cliente Bun](#conectar-cliente-bun)
8. [Troubleshooting](#troubleshooting)
9. [Custos Estimados](#custos-estimados)

---

## 🎯 Visão Geral

O Whisper Stream Server é **agnóstico ao modelo** - você escolhe qual backend usar no **build time** da imagem Docker.

**Backends disponíveis:**
- `faster-whisper`: Universal, word timestamps, VAD ✅ **Recomendado**
- `crisper-whisper`: Verbatim transcription, ±50ms precision, fillers 🌟
- `whisperx`: Diarization (quem está falando), CUDA only
- `mlx`: Apple Silicon (não funciona no RunPod)

**Filosofia:** Servidor agnóstico, modelo escolhido no Docker build.

---

## 🔧 Pré-requisitos

### 1. Docker Hub Account

```bash
# Criar conta em: https://hub.docker.com/
# Depois, fazer login:
docker login
```

### 2. RunPod Account

- Acesse: https://runpod.io/
- Crie conta e adicione créditos
- Mínimo recomendado: $10 USD

### 3. Repositório Local

```bash
cd /path/to/whisper-stream
```

---

## 🏗️ Build Local

### Opção 1: Build manual

```bash
# Backend: Faster-Whisper (recomendado)
docker build \
  --build-arg BACKEND=faster-whisper \
  -t whisper-stream:faster-whisper \
  .

# Backend: CrisperWhisper (verbatim)
docker build \
  --build-arg BACKEND=crisper-whisper \
  -t whisper-stream:crisper-whisper \
  .
```

### Opção 2: Usar script

```bash
# Build um backend
./scripts/build-docker.sh faster-whisper

# Build todos os backends
./scripts/build-docker.sh all
```

### Verificar imagens

```bash
docker images | grep whisper-stream
```

**Output esperado:**
```
whisper-stream   faster-whisper   abc123   2 hours ago   5.2GB
whisper-stream   crisper-whisper  def456   1 hour ago    6.1GB
```

---

## ☁️ Push para Registry

### 1. Tag da imagem

```bash
# Substituir YOUR_USERNAME pelo seu username do Docker Hub
export DOCKER_USERNAME=your-username

docker tag whisper-stream:faster-whisper \
  ${DOCKER_USERNAME}/whisper-stream:faster-whisper
```

### 2. Push

```bash
docker push ${DOCKER_USERNAME}/whisper-stream:faster-whisper
```

**⏱️ Tempo estimado:** 10-30 minutos (dependendo da conexão)

### 3. Verificar no Docker Hub

Acesse: `https://hub.docker.com/r/${YOUR_USERNAME}/whisper-stream/tags`

---

## 🎮 Deploy no RunPod

### Passo 1: Acessar Console

1. Acesse: https://runpod.io/console/pods
2. Clique em **"Deploy"**

### Passo 2: Escolher GPU

**Recomendado para produção:**
- **RTX A4000** (16GB VRAM) - $0.44/hr
- **RTX A5000** (24GB VRAM) - $0.64/hr
- **RTX A6000** (48GB VRAM) - $0.79/hr

**Para testes:**
- **RTX 3090** (24GB VRAM) - $0.39/hr
- **RTX 4090** (24GB VRAM) - $0.69/hr

### Passo 3: Configurar Pod

**Container Configuration:**

| Campo | Valor |
|-------|-------|
| Container Image | `your-username/whisper-stream:faster-whisper` |
| Container Disk | 20 GB |
| Volume Disk | 50 GB (para cache de modelos) |
| Expose HTTP Ports | 9090 |

**Environment Variables:**

```bash
WHISPER_BACKEND=faster-whisper
WHISPER_MODEL=base
WHISPER_LANGUAGE=pt
SERVER_HOST=0.0.0.0
SERVER_PORT=9090
LOG_LEVEL=info
```

### Passo 4: Deploy

1. Clique em **"Deploy"**
2. Aguarde ~2-5 minutos (download da imagem)
3. Status deve ficar **"Running"** 🟢

### Passo 5: Testar

#### Health Check

```bash
# Pegar Public IP do pod (ex: 123.45.67.89)
curl http://123.45.67.89:9090/health
```

**Output esperado:**
```json
{
  "status": "healthy",
  "backend": "faster-whisper",
  "model": "base",
  "initialized": true
}
```

#### WebSocket Info

```bash
curl http://123.45.67.89:9090/info
```

---

## ⚙️ Configuração de Backends

### Faster-Whisper (Universal) ✅

**Características:**
- Universal (CUDA, CPU, Apple Silicon)
- Word timestamps
- VAD (Voice Activity Detection)
- Multilingual (Português, Inglês, 90+ idiomas)

**Configuração:**
```yaml
WHISPER_BACKEND=faster-whisper
WHISPER_MODEL=base
WHISPER_LANGUAGE=pt
```

**Modelos disponíveis:**
- `tiny`: Mais rápido, menor qualidade
- `base`: ⭐ **Recomendado para início**
- `small`: Boa qualidade
- `medium`: Alta qualidade
- `large-v3`: Melhor qualidade (lento)
- `distil-large-v3`: 6x mais rápido que large-v3 🚀

### CrisperWhisper (Verbatim) 🌟

**Características:**
- Verbatim transcription (captura fillers: "um", "uh", "er")
- ±50ms timestamp precision (4x melhor que Whisper)
- Word-level timestamps
- **Idiomas:** English, German (⚠️ Não tem PT)

**Configuração:**
```yaml
WHISPER_BACKEND=crisper-whisper
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=en
```

**Quando usar:**
- Transcrição médica (importante capturar hesitações)
- Análise de fala (disfluências, stutters)
- Legendagem de alta precisão

### WhisperX (Diarization)

**Características:**
- Diarization (identifica quem está falando)
- Word-level timestamps com speaker ID
- **Requer:** CUDA (não funciona em CPU)

**Configuração:**
```yaml
WHISPER_BACKEND=whisperx
WHISPER_MODEL=large-v3
WHISPER_LANGUAGE=pt
```

**Quando usar:**
- Reuniões com múltiplos participantes
- Entrevistas
- Podcasts

---

## 🔌 Conectar Cliente Bun

### 1. Pegar IP do RunPod

No console do RunPod, copie o **Public IP** do pod.

### 2. Atualizar Cliente

Editar `src/index.ts`:

```typescript
// Antes (local)
const WS_URL = 'ws://localhost:9090/ws';

// Depois (RunPod)
const WS_URL = 'ws://123.45.67.89:9090/ws';  // Substituir pelo IP real
```

### 3. Rodar Cliente

```bash
bun start
```

**Output esperado:**
```
✅ Conectado ao servidor Whisper
   Backend: faster-whisper
   Model: base
   Capabilities: transcription, word_timestamps, vad, streaming
```

### 4. Testar Transcrição

Fale no microfone. Você deve ver:

```
📝 Transcrição:
Olá mundo [0.00s - 1.50s]

Words:
  • Olá (0.00s - 0.50s)
  • mundo (0.60s - 1.50s)
```

---

## 🐛 Troubleshooting

### Problema: Pod não inicia

**Sintoma:** Pod fica em "Pending" ou "Error"

**Solução:**
1. Verificar logs: Console RunPod → Pod → "View Logs"
2. Erros comuns:
   - **"Image not found"**: Verificar se push foi concluído
   - **"Out of memory"**: Trocar para GPU maior ou modelo menor
   - **"CUDA error"**: Verificar se GPU é compatível (deve ser NVIDIA)

### Problema: Health check falha

**Sintoma:** `curl http://IP:9090/health` retorna erro

**Solução:**
1. Verificar se porta 9090 está exposta no RunPod
2. Aguardar 1-2 minutos (modelo pode estar carregando)
3. Ver logs do container:
   ```bash
   # No console RunPod, abrir terminal do pod
   tail -f /app/logs/server.log
   ```

### Problema: "Backend não inicializado"

**Sintoma:** Health check retorna `"initialized": false`

**Causas:**
1. Modelo ainda está baixando (primeira vez demora ~5-10 min)
2. GPU sem memória suficiente
3. Backend incompatível com plataforma

**Solução:**
1. Aguardar alguns minutos
2. Verificar logs do servidor
3. Trocar para modelo menor (`tiny` ou `base`)

### Problema: Cliente não conecta

**Sintoma:** Cliente Bun fica tentando conectar indefinidamente

**Solução:**
1. Verificar se IP está correto
2. Verificar se porta 9090 está acessível:
   ```bash
   telnet 123.45.67.89 9090
   ```
3. Firewall do RunPod pode estar bloqueando - verificar configuração

### Problema: Transcrição muito lenta

**Sintoma:** Demora >5 segundos para transcrever 1 segundo de áudio

**Solução:**
1. Trocar para modelo menor (`base` → `tiny`)
2. Ou trocar para `distil-large-v3` (6x mais rápido)
3. Ou escolher GPU mais potente

---

## 💰 Custos Estimados

### GPU no RunPod

| GPU | VRAM | Preço/hora | Preço/mês (24/7) | Recomendado |
|-----|------|-----------|------------------|-------------|
| RTX 3090 | 24GB | $0.39 | ~$280 | Testes |
| RTX A4000 | 16GB | $0.44 | ~$320 | ✅ Produção |
| RTX A5000 | 24GB | $0.64 | ~$460 | Alta demanda |
| RTX A6000 | 48GB | $0.79 | ~$570 | Múltiplos modelos |

### Modelos - Uso de VRAM

| Modelo | VRAM | Velocidade | Qualidade |
|--------|------|-----------|-----------|
| tiny | ~1GB | 🚀🚀🚀🚀🚀 | ⭐⭐ |
| base | ~1.5GB | 🚀🚀🚀🚀 | ⭐⭐⭐ |
| small | ~2.5GB | 🚀🚀🚀 | ⭐⭐⭐⭐ |
| medium | ~5GB | 🚀🚀 | ⭐⭐⭐⭐ |
| large-v3 | ~10GB | 🚀 | ⭐⭐⭐⭐⭐ |
| distil-large-v3 | ~6GB | 🚀🚀🚀🚀 | ⭐⭐⭐⭐⭐ |

### Recomendação Custo-Benefício

**Configuração ideal para produção:**
- GPU: **RTX A4000** ($0.44/hr)
- Modelo: **distil-large-v3** (~6GB VRAM)
- Resultado: Alta qualidade + velocidade razoável
- Custo: ~$320/mês (24/7)

**Para testes/desenvolvimento:**
- GPU: **RTX 3090** ($0.39/hr)
- Modelo: **base** (~1.5GB VRAM)
- Resultado: Qualidade OK + rápido
- Custo: $3-5/dia (~8h uso)

---

## 🎯 Quick Start - Resumo

```bash
# 1. Build
./scripts/build-docker.sh faster-whisper

# 2. Tag & Push
export DOCKER_USERNAME=your-username
docker tag whisper-stream:faster-whisper ${DOCKER_USERNAME}/whisper-stream:faster-whisper
docker push ${DOCKER_USERNAME}/whisper-stream:faster-whisper

# 3. Deploy no RunPod
# - Acessar https://runpod.io/console/pods
# - Deploy com imagem: your-username/whisper-stream:faster-whisper
# - GPU: RTX A4000
# - Ports: 9090
# - ENV: WHISPER_BACKEND=faster-whisper, WHISPER_MODEL=base, WHISPER_LANGUAGE=pt

# 4. Testar
curl http://<POD_IP>:9090/health

# 5. Conectar cliente
# Editar src/index.ts → WS_URL = 'ws://<POD_IP>:9090/ws'
bun start
```

---

## 📚 Próximos Passos

Depois de deploy funcional:

1. **Adicionar SSL/TLS:** Usar Cloudflare Tunnel ou Nginx reverse proxy
2. **Monitoramento:** Logs centralizados (Grafana + Loki)
3. **Auto-scaling:** RunPod Serverless (on-demand)
4. **Rate limiting:** Implementar limites de requisições
5. **Autenticação:** API keys ou JWT

---

## 🔗 Links Úteis

- **RunPod Console:** https://runpod.io/console/pods
- **Docker Hub:** https://hub.docker.com/
- **Faster-Whisper Docs:** https://github.com/guillaumekln/faster-whisper
- **CrisperWhisper:** https://huggingface.co/nyrahealth/CrisperWhisper
- **Whisper Stream GitHub:** https://github.com/your-repo/whisper-stream

---

✅ **Pronto para produção!** 🚀
