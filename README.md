# 🎤 Whisper Stream - Real-Time Transcription

Sistema de transcrição em tempo real usando OpenAI Whisper com arquitetura cliente-servidor separada.

## 🏗️ Arquitetura

### Servidor Python (Principal)
**O core do projeto** - Servidor WebSocket independente que processa áudio com Whisper.

- ✅ Detecção automática de hardware (Apple Silicon/MLX, NVIDIA/CUDA, CPU)
- ✅ Backends otimizados para cada plataforma
- ✅ WebSocket server com suporte a múltiplos clientes
- ✅ Configuração flexível via YAML
- ✅ **Pode ser deployado separadamente com GPU**

### Cliente Bun (Exemplo)
**Cliente simples** para demonstrar uso do servidor.

- Captura áudio do microfone (via sox)
- Conecta ao servidor via WebSocket
- Exibe transcrições no terminal com cores
- **Agnóstico ao hardware** - não precisa de GPU

> O cliente Bun é apenas um **exemplo/sandbox**. Você pode criar seu próprio cliente em qualquer linguagem que suporte WebSocket.

---

## 🚀 Quick Start

### 1. Servidor Python

O servidor é o componente principal. Instale e rode em uma máquina com GPU (opcional mas recomendado):

```bash
# Instalar dependências com Poetry
poetry install

# Para Apple Silicon (MLX - mais rápido):
poetry install -E mlx

# Para NVIDIA GPU (CUDA):
poetry install -E cuda

# Copiar e configurar
cp server-config.example.yaml server-config.yaml
# Edite server-config.yaml se necessário

# Iniciar servidor
poetry run python -m server.main --config server-config.yaml

# Ou usar o Makefile:
make install  # instalar
make dev      # rodar servidor
```

O servidor estará disponível em `ws://localhost:9090/ws`

### 2. Cliente Bun (Exemplo)

Cliente simples para testar o servidor:

```bash
# Instalar sox (para captura de áudio)
# macOS:
brew install sox

# Linux:
sudo apt-get install sox

# Instalar dependências
bun install

# Copiar config
cp config.example.yaml config.yaml
# Ajuste a URL do servidor em config.yaml se necessário

# Iniciar cliente
bun start
```

**Pronto!** Fale no microfone e veja as transcrições aparecerem no terminal.

---

## ⚙️ Configuração

### Servidor (server-config.yaml)

```yaml
server:
  host: "0.0.0.0"      # 0.0.0.0 para aceitar conexões externas
  port: 9090

whisper:
  model: "base"        # tiny, base, small, medium, large
  language: "pt"       # pt, en, es, etc
  backend: "auto"      # auto detecta: mlx, cuda, ou cpu
  use_vad: true        # Voice Activity Detection
```

### Cliente (config.yaml)

```yaml
server:
  url: "ws://localhost:9090/ws"  # URL do servidor (pode ser remoto!)

display:
  show_partial: true
  colors:
    final: "green"
    partial: "cyan"
```

---

## 🔌 Criando seu Próprio Cliente

O cliente Bun é apenas um exemplo. Você pode criar clientes em qualquer linguagem:

### Protocolo WebSocket

**Enviar áudio:**
- Formato: Binary (raw PCM)
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Bits: 16-bit signed integer

**Receber transcrições:**
```json
{
  "type": "transcription",
  "text": "texto transcrito",
  "is_final": true,
  "language": "pt",
  "confidence": 0.95,
  "timestamp": "2025-01-17T10:30:00.000Z"
}
```

### Exemplo em Python

```python
import asyncio
import websockets
import pyaudio

async def stream_audio():
    async with websockets.connect('ws://localhost:9090/ws') as ws:
        # Receber confirmação
        msg = await ws.recv()
        print(msg)

        # Capturar e enviar áudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1,
                       rate=16000, input=True, frames_per_buffer=16000)

        while True:
            data = stream.read(16000)  # 1 segundo
            await ws.send(data)

            # Receber transcrição
            response = await ws.recv()
            print(response)

asyncio.run(stream_audio())
```

---

## 📋 Deployment

### Servidor com GPU (Produção)

Deploy o servidor Python em uma máquina com GPU:

1. **Railway / Render:**
   ```bash
   # Configure como Python app
   # Start command: python -m server.main --config server-config.yaml
   ```

2. **Docker:**
   ```dockerfile
   FROM python:3.10
   # ... (adicionar suporte CUDA se necessário)
   CMD ["python", "-m", "server.main"]
   ```

3. **VPS com GPU:**
   ```bash
   # Instalar CUDA drivers (NVIDIA)
   # Instalar dependências
   poetry install -E cuda

   # Rodar servidor
   poetry run python -m server.main --host 0.0.0.0
   ```

### Cliente (Local ou Proxy)

O cliente pode rodar localmente e conectar a um servidor remoto:

```yaml
# config.yaml
server:
  url: "ws://seu-servidor.com:9090/ws"
```

---

## 📊 Status do Projeto

**Versão**: 1.0 (pré-produção)
**Data**: 2025-10-18

### ✅ Implementado e Funcionando
- ✅ Servidor Python completo com detecção de hardware
- ✅ Backends: faster-whisper, MLX, WhisperX, CrisperWhisper, CPU/CUDA
- ✅ Backend Registry com capabilities system
- ✅ WebSocket server com múltiplos clientes
- ✅ Cliente Bun de exemplo
- ✅ Refatorações recentes (SRP, performance, type safety)

### ⏳ Em Desenvolvimento (Ver ROADMAP.md)
- ⏳ **Streaming inteligente** (LocalAgreement) - FASE 0
- ⏳ **Validação WhisperX** (diarization) - FASE 1
- ⏳ **Testes automatizados** (coverage target: 80%)
- ⏳ **Novos backends** (Voxtral, Parakeet, Granite, SeamlessM4T) - FASE 2
- ⏳ **Features premium** (Redact/PII, Translation) - FASE 3

### 📋 Documentação
- 📖 **[DEVELOPER_NOTES.md](docs/DEVELOPER_NOTES.md)** - ⭐ **COMECE AQUI** - Notas críticas
- 📖 **[ROADMAP.md](docs/ROADMAP.md)** - Roadmap completo (16 semanas, 5 fases)
- 📖 **[IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** - Código pronto (FASE 0)
- 📖 **[MODELS_COMPARISON.md](docs/MODELS_COMPARISON.md)** - Comparação 11 modelos
- 📖 [STATUS.md](docs/STATUS.md) - Status atual detalhado
- 📖 [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Arquitetura detalhada
- 📖 [API.md](docs/API.md) - Protocolo WebSocket
- 📖 [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deploy em produção

---

## 🛠️ Desenvolvimento

### Estrutura

```
whisper-stream/
├── server/              # 🐍 Servidor Python (PRINCIPAL)
│   ├── backends/        # Backends MLX/CUDA/CPU
│   ├── utils/           # Hardware detection, logging
│   ├── main.py          # Entry point
│   └── config.py        # Configuração
├── src/                 # 🥟 Cliente Bun (EXEMPLO)
│   └── index.ts         # Cliente simples
└── config files
```

### Rodar Servidor em Dev

```bash
# Com auto-reload (nodemon ou similar)
poetry run python -m server.main --config server-config.yaml
```

### Testar Endpoints

```bash
# Health check
curl http://localhost:9090/health

# Info
curl http://localhost:9090/info

# WebSocket (use wscat)
npm install -g wscat
wscat -c ws://localhost:9090/ws
```

---

## 🤝 Contribuindo

Este é um projeto sandbox/exemplo. Sinta-se livre para:
- Criar clientes em outras linguagens
- Melhorar os backends
- Adicionar features ao servidor
- Criar UI web

---

## 📄 Licença

MIT License - veja [LICENSE](LICENSE)

---

## 💡 Notas

- **Servidor Python** é o componente principal - foque nele para produção
- **Cliente Bun** é apenas exemplo - substitua pelo seu próprio
- Hardware detection acontece **apenas no servidor**
- Servidor pode rodar em máquina separada com GPU
- Cliente é agnóstico - só precisa de WebSocket

---

**Desenvolvido com** [Claude Code](https://claude.com/claude-code)
