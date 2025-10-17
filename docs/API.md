# 📡 API Documentation - Whisper Stream

Documentação completa da API WebSocket do Whisper Stream.

## Endpoints HTTP

### GET /health

Health check do servidor.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "processor_ready": true
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "initializing",
  "processor_ready": false
}
```

---

### GET /info

Informações sobre o servidor e backend.

**Response** (200 OK):
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 9090,
    "max_clients": 5
  },
  "processor": {
    "initialized": true,
    "model": "base",
    "language": "pt",
    "backend_type": "cuda",
    "hardware": {
      "type": "cuda",
      "device_name": "NVIDIA RTX 3090",
      "memory": "24 GB"
    },
    "backend": {
      "name": "CUDA Backend (faster-whisper)",
      "version": "1.0.0",
      "device": "cuda:0",
      "compute_type": "float16",
      "initialized": true,
      "gpu_name": "NVIDIA GeForce RTX 3090",
      "gpu_memory_total": "24.00 GB"
    }
  }
}
```

---

### GET /stats

Estatísticas do servidor.

**Response** (200 OK):
```json
{
  "connections": {
    "active_connections": 3,
    "max_clients": 5
  }
}
```

---

## WebSocket API

### URL

```
ws://servidor:9090/ws
```

### Mensagens

#### 1. Conexão Estabelecida

Quando cliente conecta com sucesso, servidor envia:

**Direção**: Servidor → Cliente

```json
{
  "type": "connected",
  "message": "Conectado ao servidor Whisper Stream",
  "server_info": {
    "initialized": true,
    "model": "base",
    "language": "pt",
    "backend_type": "auto",
    "hardware": {
      "type": "apple_silicon",
      "device_name": "Apple M2",
      "memory": "16.0 GB"
    }
  }
}
```

---

#### 2. Envio de Áudio

**Direção**: Cliente → Servidor

**Tipo**: Binary Frame (WebSocket Binary Message)

**Formato do áudio**:
- **Encoding**: Raw PCM
- **Sample Rate**: 16000 Hz
- **Channels**: 1 (mono)
- **Sample Format**: 16-bit signed integer (little-endian)
- **Chunk Size Recomendado**: 16000 samples = 1 segundo = 32000 bytes

**Exemplo em diferentes linguagens**:

**JavaScript/Bun**:
```javascript
const ws = new WebSocket('ws://localhost:9090/ws');

// Enviar buffer de áudio
const audioBuffer = new Int16Array(16000); // 1 segundo
ws.send(audioBuffer.buffer);
```

**Python**:
```python
import struct

# Criar chunk de áudio (exemplo: silêncio)
audio_samples = [0] * 16000  # 1 segundo de silêncio
# Converter para bytes (little-endian int16)
audio_bytes = struct.pack('<' + 'h' * len(audio_samples), *audio_samples)

# Enviar
await websocket.send(audio_bytes)
```

**Go**:
```go
import "encoding/binary"

// Criar buffer
audioData := make([]int16, 16000)
buf := new(bytes.Buffer)
binary.Write(buf, binary.LittleEndian, audioData)

// Enviar
conn.WriteMessage(websocket.BinaryMessage, buf.Bytes())
```

---

#### 3. Recebimento de Transcrições

**Direção**: Servidor → Cliente

**Tipo**: Text Frame (WebSocket Text Message - JSON)

```json
{
  "type": "transcription",
  "text": "olá mundo",
  "is_final": true,
  "language": "pt",
  "confidence": 0.95,
  "timestamp": "2025-01-17T10:30:15.123Z"
}
```

**Campos**:
- `type` (string): Sempre "transcription"
- `text` (string): Texto transcrito
- `is_final` (boolean):
  - `true`: Transcrição final, não vai mudar
  - `false`: Transcrição parcial, ainda processando
- `language` (string): Código do idioma detectado (ISO 639-1)
- `confidence` (number): Confiança da transcrição (0.0 - 1.0)
- `timestamp` (string): Timestamp ISO 8601 de quando foi gerada

**Exemplo de uso**:
```javascript
ws.on('message', (data) => {
  const msg = JSON.parse(data);

  if (msg.type === 'transcription') {
    if (msg.is_final) {
      // Transcrição final - salvar/exibir permanentemente
      console.log(`[${msg.timestamp}] ${msg.text} (${(msg.confidence * 100).toFixed(0)}%)`);
    } else {
      // Transcrição parcial - atualizar UI temporariamente
      updateTempUI(msg.text);
    }
  }
});
```

---

#### 4. Ping/Pong (Keep-Alive)

**Direção**: Cliente → Servidor

```json
{
  "type": "ping"
}
```

**Direção**: Servidor → Cliente

```json
{
  "type": "pong"
}
```

**Uso**: Manter conexão viva, especialmente em ambientes com timeouts agressivos.

**Exemplo**:
```javascript
// Enviar ping a cada 30 segundos
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'ping' }));
  }
}, 30000);
```

---

#### 5. Obter Informações

**Direção**: Cliente → Servidor

```json
{
  "type": "get_info"
}
```

**Direção**: Servidor → Cliente

```json
{
  "type": "info",
  "data": {
    "initialized": true,
    "model": "base",
    "language": "pt",
    "backend_type": "mlx",
    "hardware": { ... },
    "backend": { ... }
  }
}
```

---

#### 6. Reset de Contexto (Futuro)

**Direção**: Cliente → Servidor

```json
{
  "type": "reset_context"
}
```

**Direção**: Servidor → Cliente

```json
{
  "type": "ack",
  "message": "Context reset"
}
```

**Uso**: Resetar o contexto de transcrição (útil quando mudar de tópico).

---

#### 7. Erros

**Direção**: Servidor → Cliente

```json
{
  "type": "error",
  "message": "Descrição do erro"
}
```

**Exemplos de erros**:
- "Limite de clientes conectados excedido"
- "Erro ao processar áudio: ..."
- "JSON inválido"
- "Tipo de mensagem desconhecido: ..."

---

## Fluxo de Comunicação Típico

```
Cliente                                 Servidor
  │                                        │
  │  1. Conectar WebSocket                │
  ├───────────────────────────────────────►│
  │                                        │
  │  2. {"type":"connected",...}           │
  │◄───────────────────────────────────────┤
  │                                        │
  │  3. [Audio Binary Data - 1s chunk]    │
  ├───────────────────────────────────────►│
  │                                        │ Processing...
  │                                        │
  │  4. {"type":"transcription",...}       │
  │◄───────────────────────────────────────┤
  │                                        │
  │  5. [Audio Binary Data - 1s chunk]    │
  ├───────────────────────────────────────►│
  │                                        │
  │  6. {"type":"transcription",...}       │
  │◄───────────────────────────────────────┤
  │                                        │
  │  ... continua ...                      │
  │                                        │
  │  7. {"type":"ping"}                    │
  ├───────────────────────────────────────►│
  │                                        │
  │  8. {"type":"pong"}                    │
  │◄───────────────────────────────────────┤
  │                                        │
```

---

## Códigos de Exemplo

### Cliente Mínimo (JavaScript)

```javascript
const WebSocket = require('ws');

// Conectar
const ws = new WebSocket('ws://localhost:9090/ws');

ws.on('open', () => {
  console.log('Conectado!');

  // Enviar áudio de teste (1s de silêncio)
  const silence = new Int16Array(16000).fill(0);
  ws.send(Buffer.from(silence.buffer));
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  console.log('Recebido:', msg);

  if (msg.type === 'transcription') {
    console.log(`Transcrição: "${msg.text}"`);
  }
});

ws.on('error', (error) => {
  console.error('Erro:', error);
});

ws.on('close', () => {
  console.log('Desconectado');
});
```

---

### Cliente com Captura de Áudio (Python)

```python
import asyncio
import websockets
import pyaudio
import json

async def stream_audio():
    # Conectar
    async with websockets.connect('ws://localhost:9090/ws') as ws:
        # Receber mensagem de conexão
        conn_msg = await ws.recv()
        print(json.loads(conn_msg))

        # Setup PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=16000  # 1 segundo
        )

        print("Gravando... (Ctrl+C para parar)")

        # Task para receber transcrições
        async def receive():
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data['type'] == 'transcription':
                    print(f"[{'FINAL' if data['is_final'] else 'PARCIAL'}] {data['text']}")

        # Task para enviar áudio
        async def send():
            loop = asyncio.get_event_loop()
            while True:
                # Ler áudio em thread separada
                audio_data = await loop.run_in_executor(None, stream.read, 16000)
                # Enviar
                await ws.send(audio_data)
                await asyncio.sleep(0.01)  # Small delay

        # Rodar ambas tasks em paralelo
        await asyncio.gather(receive(), send())

# Rodar
asyncio.run(stream_audio())
```

---

### Cliente com Arquivo de Áudio (Python)

```python
import asyncio
import websockets
import numpy as np
import soundfile as sf
import json

async def transcribe_file(file_path: str):
    # Carregar arquivo
    audio, sr = sf.read(file_path, dtype='int16')

    # Resample para 16kHz se necessário
    if sr != 16000:
        from scipy import signal
        audio = signal.resample_poly(audio, 16000, sr)
        audio = audio.astype(np.int16)

    # Garantir mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Conectar
    async with websockets.connect('ws://localhost:9090/ws') as ws:
        # Receber mensagem de conexão
        await ws.recv()

        # Dividir em chunks de 1 segundo
        chunk_size = 16000
        transcription = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]

            # Pad se necessário
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Enviar
            await ws.send(chunk.tobytes())

            # Receber transcrição
            response = await ws.recv()
            msg = json.loads(response)

            if msg['type'] == 'transcription' and msg['text']:
                print(f"Chunk {i//chunk_size + 1}: {msg['text']}")
                transcription.append(msg['text'])

        print("\nTranscrição completa:")
        print(" ".join(transcription))

# Uso
asyncio.run(transcribe_file("audio.wav"))
```

---

## Rate Limiting

O servidor tem proteção contra sobrecarga:

- **max_clients**: Limite de conexões simultâneas (padrão: 5)
- **idle_timeout**: Desconecta clientes inativos após N segundos (padrão: 300)

Se exceder `max_clients`:
```json
{
  "type": "error",
  "message": "Limite de clientes conectados excedido"
}
```

A conexão será fechada imediatamente.

---

## Best Practices

### 1. Tamanho de Chunks

**Recomendado**: 1 segundo (16000 samples = 32000 bytes)

- Chunks muito pequenos (<0.5s): Overhead de network, transcrições ruins
- Chunks muito grandes (>3s): Latência alta, demora para ver resultado

### 2. Tratamento de Erros

Sempre tratar erros e reconectar:
```javascript
ws.on('close', () => {
  console.log('Desconectado, reconectando...');
  setTimeout(reconnect, 2000);
});

ws.on('error', (err) => {
  console.error('Erro:', err);
});
```

### 3. Buffer Management

Evitar enviar áudio mais rápido que consegue processar:
```javascript
let processing = false;

async function sendChunk(audio) {
  if (!processing) {
    processing = true;
    ws.send(audio);
    // Aguardar resposta antes de enviar próximo
    await waitForResponse();
    processing = false;
  }
}
```

### 4. Contexto

Transcrições usam contexto anterior. Para melhor qualidade:
- Envie chunks contínuos
- Não interrompa no meio de uma frase
- Use `reset_context` ao mudar de tópico

---

## Limitações

1. **Audio Format**: Apenas raw PCM 16kHz mono int16
   - Não suporta: MP3, AAC, OGG, etc (converter antes)

2. **Latência**: Depende do modelo e hardware
   - Tiny: ~1s
   - Base: ~2s
   - Small: ~3s
   - Large: ~5s+

3. **Idiomas**: Depende do modelo Whisper
   - Configurado no servidor via `language` parameter

4. **Concurrent Clients**: Limitado por `max_clients` (padrão: 5)

---

**Última atualização**: 2025-01-17
