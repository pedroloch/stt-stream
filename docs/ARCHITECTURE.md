# 🏗️ Arquitetura do Whisper Stream

## Visão Geral

Whisper Stream é um sistema de transcrição em tempo real com **arquitetura cliente-servidor completamente separada**.

### Princípio de Design

**Separação de responsabilidades:**
- ☁️ **Servidor Python**: Processing pesado (Whisper, GPU) - pode estar em servidor remoto
- 💻 **Cliente**: Apenas captura de áudio e UI - roda localmente

## Componentes

### 1. Servidor Python (Core)

**Responsabilidade**: Processar áudio e retornar transcrições

```
┌─────────────────────────────────────────┐
│         Servidor Python                 │
│  (Pode rodar em máquina separada)      │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   WebSocket Server               │  │
│  │   - aiohttp                      │  │
│  │   - Múltiplos clientes           │  │
│  │   - /health, /info, /ws          │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │   Whisper Processor              │  │
│  │   - Gerencia backends            │  │
│  │   - Carrega modelos              │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │   Backend Factory                │  │
│  │   - Detecta hardware             │  │
│  │   - Escolhe backend              │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│        ┌────────┴────────┐             │
│        │                 │             │
│  ┌─────▼─────┐   ┌──────▼──────┐     │
│  │ MLX       │   │ CUDA        │     │
│  │ Backend   │   │ Backend     │     │
│  │ (M1/M2)   │   │ (NVIDIA)    │     │
│  └───────────┘   └─────────────┘     │
│        │                 │             │
│  ┌─────▼─────────────────▼─────┐     │
│  │    CPU Backend              │     │
│  │    (Fallback)               │     │
│  └─────────────────────────────┘     │
│                                         │
└─────────────────────────────────────────┘
```

#### Detecção de Hardware

O servidor detecta automaticamente o hardware disponível:

1. **Apple Silicon** (M1/M2/M3):
   - Verifica: `platform.machine() == "arm64"` e `sys.platform == "darwin"`
   - Backend: **MLX** (otimizado para Apple Silicon)
   - Lib: `mlx-whisper`

2. **NVIDIA GPU**:
   - Verifica: `nvidia-smi` disponível
   - Backend: **CUDA** (muito rápido)
   - Lib: `faster-whisper` com CUDA

3. **CPU** (fallback):
   - Sempre disponível
   - Backend: `faster-whisper` com CPU
   - Mais lento, mas funciona em qualquer máquina

#### Backends

Todos implementam a mesma interface (`WhisperBackend`):

```python
class WhisperBackend(ABC):
    async def initialize() -> None
    async def transcribe_chunk(audio, context) -> dict
    async def transcribe_stream(audio_stream) -> AsyncIterator[dict]
    async def cleanup() -> None
    def get_backend_info() -> dict
```

**Fluxo de transcrição:**

```
Audio bytes → numpy array (float32) → Backend.transcribe_chunk() →
{
  "text": "...",
  "is_final": true,
  "language": "pt",
  "confidence": 0.95
}
```

### 2. Cliente (Exemplo)

**Responsabilidade**: Capturar áudio e exibir transcrições

Cliente Bun é apenas um **exemplo minimalista**. Você pode criar clientes em qualquer linguagem.

```
┌─────────────────────────────────────────┐
│         Cliente Bun                     │
│  (Roda localmente)                      │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   index.ts (270 linhas)          │  │
│  │                                  │  │
│  │  1. Carrega config.yaml          │  │
│  │  2. Health check servidor        │  │
│  │  3. Conecta WebSocket            │  │
│  │  4. Inicia captura áudio (sox)   │  │
│  │  5. Envia chunks via WS          │  │
│  │  6. Recebe transcrições          │  │
│  │  7. Exibe no terminal            │  │
│  └──────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

**Por que sox?**
- Simples e multiplataforma
- Funciona via CLI (fácil de usar com spawn)
- Suporta formato raw PCM necessário

## Protocolo de Comunicação

### WebSocket Protocol

**URL**: `ws://servidor:9090/ws`

#### 1. Conexão

Cliente conecta → Servidor envia:
```json
{
  "type": "connected",
  "message": "Conectado ao servidor Whisper Stream",
  "server_info": {
    "model": "base",
    "language": "pt",
    "backend_type": "mlx",
    "hardware": {...}
  }
}
```

#### 2. Envio de Áudio

**Formato**: Binary message (WebSocket Binary Frame)

**Especificação do áudio**:
- Format: Raw PCM
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit signed integer (little-endian)
- Chunk size: ~1 segundo (32000 bytes = 16000 samples * 2 bytes)

**Exemplo em Bun/Node**:
```typescript
// Capturar com sox
const sox = spawn("sox", [
  "-d",           // Device padrão
  "-t", "raw",    // Raw output
  "-r", "16000",  // 16kHz
  "-c", "1",      // Mono
  "-b", "16",     // 16-bit
  "-e", "signed-integer",
  "-"             // stdout
]);

sox.stdout.on("data", (chunk) => {
  ws.send(chunk); // Enviar binário direto
});
```

#### 3. Recebimento de Transcrições

**Formato**: JSON message (WebSocket Text Frame)

```json
{
  "type": "transcription",
  "text": "olá mundo",
  "is_final": true,
  "language": "pt",
  "confidence": 0.95,
  "timestamp": "2025-01-17T10:30:00.000Z"
}
```

**Campos**:
- `type`: Sempre "transcription"
- `text`: Texto transcrito
- `is_final`: `true` = transcrição final, `false` = parcial
- `language`: Idioma detectado
- `confidence`: Confiança (0-1)
- `timestamp`: ISO timestamp

#### 4. Comandos (Cliente → Servidor)

**Ping/Pong**:
```json
// Cliente envia:
{"type": "ping"}

// Servidor responde:
{"type": "pong"}
```

**Obter info**:
```json
// Cliente envia:
{"type": "get_info"}

// Servidor responde:
{
  "type": "info",
  "data": {...}
}
```

## Configuração

### Servidor (server-config.yaml)

```yaml
server:
  host: "0.0.0.0"        # Aceita conexões externas
  port: 9090
  max_clients: 5

whisper:
  model: "base"          # Modelo a usar
  language: "pt"         # Idioma
  backend: "auto"        # auto | mlx | cuda | cpu
  use_vad: true          # Voice Activity Detection

hardware:
  auto_detect: true      # Detectar automaticamente
  force_type: null       # Forçar tipo (para testes)
```

### Cliente (config.yaml)

```yaml
server:
  url: "ws://localhost:9090/ws"

audio:
  sample_rate: 16000
  channels: 1

display:
  show_partial: true
  colors:
    final: "green"
    partial: "cyan"
```

## Fluxo de Dados Completo

```
┌──────────┐                              ┌──────────┐
│ Cliente  │                              │ Servidor │
└────┬─────┘                              └────┬─────┘
     │                                         │
     │ 1. Connect WS                           │
     ├────────────────────────────────────────►│
     │                                         │
     │ 2. {"type": "connected", ...}           │
     │◄────────────────────────────────────────┤
     │                                         │
     │ 3. Capturar áudio (sox)                 │
     │    ┌─────────────┐                      │
     ├───►│ Microfone   │                      │
     │◄───┤             │                      │
     │    └─────────────┘                      │
     │                                         │
     │ 4. Enviar chunk (binary)                │
     ├────────────────────────────────────────►│
     │                                         │ 5. Processar
     │                                         │    ┌──────────┐
     │                                         ├───►│ Whisper  │
     │                                         │◄───┤ Backend  │
     │                                         │    └──────────┘
     │                                         │
     │ 6. {"type": "transcription", ...}       │
     │◄────────────────────────────────────────┤
     │                                         │
     │ 7. Exibir no terminal                   │
     │    ✅ "olá mundo"                       │
     │                                         │
```

## Extensibilidade

### Criar Novo Backend

1. Herdar de `WhisperBackend`
2. Implementar métodos abstratos
3. Adicionar ao `BackendFactory`

```python
# server/backends/meu_backend.py
from .base import WhisperBackend

class MeuBackend(WhisperBackend):
    async def initialize(self):
        # Carregar seu modelo
        pass

    async def transcribe_chunk(self, audio, context):
        # Processar áudio
        return {
            "text": "...",
            "is_final": True,
            "language": "pt",
            "confidence": 0.95
        }
    # ...
```

### Criar Novo Cliente

Qualquer linguagem com WebSocket:

**Python**:
```python
import asyncio
import websockets

async def client():
    async with websockets.connect('ws://localhost:9090/ws') as ws:
        # Enviar áudio
        await ws.send(audio_bytes)
        # Receber transcrição
        response = await ws.recv()
```

**JavaScript/Browser**:
```javascript
const ws = new WebSocket('ws://localhost:9090/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);
};
```

**Go**:
```go
import "github.com/gorilla/websocket"

conn, _, _ := websocket.DefaultDialer.Dial("ws://localhost:9090/ws", nil)
conn.WriteMessage(websocket.BinaryMessage, audioBytes)
```

## Deployment

### Servidor em Produção

**Opção 1: VPS com GPU**
```bash
# Instalar CUDA
# Instalar Python deps
uv pip install -e ".[cuda]"
# Rodar
uv run python -m server.main --host 0.0.0.0
```

**Opção 2: Cloud GPU (vast.ai, runpod.io)**
- Deploy container com servidor
- Expor porta 9090
- Configurar firewall

### Cliente

Cliente roda localmente e conecta ao servidor remoto:
```yaml
# config.yaml
server:
  url: "ws://seu-servidor.com:9090/ws"
```

## Notas de Desenvolvimento

### Para Próximo Desenvolvedor

**Servidor Python é o core**:
- Foque em melhorar backends
- Adicione suporte a mais modelos
- Otimize performance

**Cliente é exemplo**:
- Substitua pelo seu próprio
- Use qualquer stack que preferir
- WebSocket é o único requirement

**Hardware detection só no servidor**:
- Cliente nunca sabe se há GPU
- Toda lógica de hardware fica isolada
- Facilita deployment separado

### Testes

```bash
# Servidor
pytest server/

# Cliente
bun test

# Integration
# 1. Inicie servidor
# 2. Use wscat para testar WS
wscat -c ws://localhost:9090/ws
```

---

**Última atualização**: 2025-01-17
