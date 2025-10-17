# 🚀 Como Rodar o Servidor com Faster-Whisper

## Opção 1: Usar configuração via linha de comando

```bash
python -m server.main --backend faster-whisper --model base --language pt --port 9090
```

## Opção 2: Criar arquivo de config

Crie `server-config-distil.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 9090

whisper:
  backend: "faster-whisper"  # Usar o novo backend universal
  model: "base"  # Ou "distil-large-v3" para 6x mais rápido
  language: "pt"
  use_vad: true

logging:
  level: "info"
```

Depois rodar:

```bash
python -m server.main --config server-config-distil.yaml
```

## O que você verá:

```
============================================================
🎤 WHISPER STREAM SERVER
============================================================
Inicializando Whisper Processor...
Plataforma: macos_arm64
Backends disponíveis:
  - mlx: streaming, transcription
  - faster-whisper: streaming, word_timestamps, vad, transcription
Auto-selecionado: faster-whisper (universal + word timestamps)
Criando backend: faster-whisper
Carregando modelo Whisper...
✅ Whisper Processor inicializado
============================================================
✅ Servidor rodando em ws://0.0.0.0:9090
Health check: http://0.0.0.0:9090/health
WebSocket: ws://0.0.0.0:9090/ws
============================================================
```

## Testar com cliente Bun:

```bash
# Terminal 2
bun start
```

O cliente vai receber mensagens com word timestamps! ⭐

```json
{
  "type": "transcription",
  "text": "Olá mundo",
  "is_final": true,
  "segments": [
    {
      "words": [
        {"word": "Olá", "start": 0.0, "end": 0.5},
        {"word": "mundo", "start": 0.6, "end": 1.5}
      ]
    }
  ]
}
```
