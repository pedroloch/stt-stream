# 🎙️ Teste da API Batch

Script Bun para testar o endpoint `/v1/transcribe`

## 🚀 Quick Start

**1. Iniciar servidor (terminal 1):**
```bash
cd /Users/pedro/Development/Sandbox/whisper-stream
poetry install  # Se ainda não instalou
poetry run python -m server.main --config server-config.yaml
```

**2. Testar API (terminal 2):**
```bash
cd /Users/pedro/Development/Sandbox/whisper-stream/sandbox
bun test-batch-api.ts
```

---

## 📋 Uso

### Teste básico (audio-test.mp3 - 37s)
```bash
bun test-batch-api.ts
```

### Especificar arquivo
```bash
bun test-batch-api.ts --file audios/outro-audio.mp3
```

### Diferentes backends
```bash
# faster-whisper (padrão)
bun test-batch-api.ts --model faster-whisper

# MLX (se estiver no Mac M1)
bun test-batch-api.ts --model mlx

# Auto-select
bun test-batch-api.ts --model auto
```

### Diferentes idiomas
```bash
# Português (padrão)
bun test-batch-api.ts --language pt

# Inglês
bun test-batch-api.ts --language en

# Auto-detect
bun test-batch-api.ts --language null
```

### Com diarization (speaker identification)
```bash
bun test-batch-api.ts --diarization
```

### Modelo maior (melhor qualidade)
```bash
bun test-batch-api.ts --model-size medium
bun test-batch-api.ts --model-size large
bun test-batch-api.ts --model-size distil-large-v3  # Recomendado (6x rápido)
```

### Task: traduzir para inglês
```bash
bun test-batch-api.ts --task translate --language pt
```

---

## 📊 Output

O script mostra:
- ✅ Texto transcrito completo
- 📊 Metadata (idioma, duração, segmentos, backend)
- ⚡ Performance (RTF, tempos de processamento)
- 📄 Primeiros 3 segmentos com timestamps
- 💾 Resultado completo salvo em `output-<timestamp>.json`

**Exemplo de output:**
```
🎙️  Whisper Stream - Teste API Batch

📡 Servidor: http://localhost:9090
📁 Arquivo: audios/audio-test.mp3 (0.04 MB)
📤 Enviando request...

Parâmetros:
  model: faster-whisper
  language: pt
  task: transcribe
  response_format: verbose_json

✅ Transcrição completa!

============================================================
📝 TEXTO TRANSCRITO
============================================================
Olá, tudo bem? Este é um teste de transcrição de áudio.
============================================================

📊 METADATA:
  Idioma detectado: pt
  Duração do áudio: 37.0s
  Segmentos: 5
  Backend: faster-whisper (base)
  Device: cuda

⚡ PERFORMANCE:
  Tempo de processamento: 2.34s
  Real-Time Factor (RTF): 0.063x
  Tempo de carregamento: 0.50s
  Tempo de transcrição: 1.80s

📄 SEGMENTOS (primeiros 3):
  [0.0s - 2.5s] Olá, tudo bem?
    Words: Olá(0.98) ,(0.99) tudo(0.97) bem(0.96) ?(0.95)
  [2.5s - 5.0s] Este é um teste de transcrição.
    Words: Este(0.99) é(0.98) um(0.97) teste(0.96) de(0.95)...

✅ Request completo em 2.40s

💾 Resultado completo salvo em: output-1729273890123.json
```

---

## ⚙️ Opções Completas

```bash
bun test-batch-api.ts [opções]

Opções:
  -f, --file <path>        Arquivo de áudio (default: audios/audio-test.mp3)
  -m, --model <backend>    Backend (faster-whisper, mlx, whisperx, auto)
  --model-size <size>      Tamanho (tiny, base, small, medium, large, distil-large-v3)
  -l, --language <lang>    Idioma (pt, en, es, fr, etc)
  -t, --task <task>        Tarefa (transcribe, translate)
  --format <format>        Formato (json, verbose_json, text, srt, vtt)
  --no-metrics             Não retornar métricas
  --diarization            Habilitar speaker identification
  -h, --help               Mostrar ajuda
```

---

## 🐛 Troubleshooting

### Erro: "Arquivo não encontrado"
```bash
# Verificar se arquivo existe
ls -la audios/audio-test.mp3

# Usar caminho absoluto
bun test-batch-api.ts --file /full/path/to/audio.mp3
```

### Erro: "Connection refused"
```bash
# Verificar se servidor está rodando
curl http://localhost:9090/health

# Iniciar servidor
cd .. && poetry run python -m server.main
```

### Erro: "ffmpeg not found"
```bash
# Instalar ffmpeg
brew install ffmpeg  # Mac
# ou
apt-get install ffmpeg  # Linux
```

---

## 📝 Notas

- **Formatos suportados**: MP3, WAV, M4A, FLAC, OGG, etc (via ffmpeg)
- **Tamanho máximo**: 100MB
- **Duração máxima**: 1 hora
- **Diarization**: Ainda não implementado no MVP (retorna warning)
- **Translation**: Requer backend com suporte (Granite, SeamlessM4T)

---

## 🎯 Próximos Testes

1. ✅ Testar com audio-test.mp3 (37s)
2. ⏳ Testar com arquivo maior (5+ min)
3. ⏳ Testar com múltiplos idiomas
4. ⏳ Testar formatters (SRT, VTT)
5. ⏳ Testar com diarization (quando implementado)
6. ⏳ Benchmark: diferentes backends e tamanhos de modelo

---

**Happy testing!** 🎉
