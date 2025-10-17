# 🚀 Whisper Stream v2.0 - Próximos Passos

Este documento explica como continuar a partir da implementação atual.

---

## ✅ O Que Está Funcionando AGORA

### No Mac (sua máquina atual)
```bash
# Testar backends disponíveis
python -c "
from server.backends import BackendRegistry
print('Backends no Mac:', list(BackendRegistry.list_available().keys()))
"
```

**Resultado:** `['mlx', 'faster-whisper']`

### Usar Faster-Whisper com Word Timestamps
```python
from server.backends import BackendRegistry
import numpy as np

# Criar backend
backend = BackendRegistry.create("faster-whisper", {
    "model": "base",  # ou "distil-large-v3" para 6x mais rápido
    "language": "pt",
    "enable_word_timestamps": True,
})

# Inicializar
await backend.initialize()

# Transcrever (exemplo com áudio dummy)
audio = np.random.randn(16000).astype(np.float32)  # 1 segundo
result = await backend.transcribe_chunk(audio)

# Ver resultado
print(result.text)
if result.segments:
    for seg in result.segments:
        print(f"Segment: {seg.text}")
        if seg.words:
            for word in seg.words:
                print(f"  {word.word}: {word.start:.2f}s - {word.end:.2f}s")
```

---

## 🔄 Integrar com Servidor Existente

### 1. Atualizar `server/main.py` para usar Registry

**Antes:**
```python
from .backends.factory import BackendFactory
factory = BackendFactory(logger)
backend = factory.create_backend(backend_type="auto", ...)
```

**Depois:**
```python
from .backends import BackendRegistry

# Listar backends disponíveis no startup
available = BackendRegistry.list_available()
logger.info("Backends disponíveis:")
for name, info in available.items():
    caps = [c.value for c in info.capabilities]
    logger.info(f"  {name}: {caps}")

# Criar backend escolhido
backend_name = config["backend"]["name"]  # "faster-whisper", "mlx", etc
backend = BackendRegistry.create(backend_name, config["backend"])
await backend.initialize()
```

### 2. Atualizar `server/websocket_handler.py` para enviar TranscriptionResult

**Modificar:**
```python
# Quando receber resultado da transcrição
result = await backend.transcribe_chunk(audio)

# Converter para WebSocket JSON
ws_dict = result.to_websocket_dict()

# Enviar ao cliente
await websocket.send_json(ws_dict)
```

**Formato enviado:**
```json
{
  "type": "transcription",
  "text": "Olá mundo",
  "is_final": true,
  "confidence": 0.95,
  "language": "pt",
  "timestamp": "2025-10-17T12:00:00",
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "Olá mundo",
      "words": [
        {"word": "Olá", "start": 0.0, "end": 0.5, "probability": 0.99},
        {"word": "mundo", "start": 0.6, "end": 1.5, "probability": 0.98}
      ]
    }
  ]
}
```

---

## 🎨 Atualizar Cliente Bun (src/index.ts)

### Adicionar tipos
```typescript
interface Word {
  word: string;
  start: number;
  end: number;
  probability: number;
}

interface Segment {
  start: number;
  end: number;
  text: string;
  words?: Word[];
}

interface TranscriptionMessage {
  type: "transcription";
  text: string;
  is_final: boolean;
  confidence: number;
  language: string;
  timestamp: string;
  segments?: Segment[];
  speaker?: string;  // Para WhisperX
  translations?: Record<string, string>;  // Para SeamlessM4T
}
```

### Renderizar word-by-word
```typescript
function renderTranscription(msg: TranscriptionMessage) {
  // Se tem words, highlight palavra por palavra
  if (msg.segments?.[0]?.words) {
    for (const word of msg.segments[0].words) {
      const color = word.probability > 0.9 ? chalk.green : chalk.yellow;
      process.stdout.write(color(word.word) + " ");
    }
    console.log();
  } else {
    // Fallback: texto normal
    console.log(chalk.green(msg.text));
  }
}
```

---

## ⚙️ Configuração Recomendada

### Para desenvolvimento (Mac)
```yaml
# server-config.yaml
backend:
  name: "faster-whisper"
  model: "base"  # Começar pequeno
  language: "pt"
  enable_word_timestamps: true
  enable_vad: true

server:
  host: "0.0.0.0"
  port: 9090
```

### Para produção (performance)
```yaml
# server-config-prod.yaml
backend:
  name: "faster-whisper"
  model: "distil-large-v3"  # ⚡ 6x mais rápido!
  language: "pt"
  compute_type: "int8_float16"  # Otimizado para distil
  enable_word_timestamps: true
  enable_vad: true
  vad_threshold: 0.75

  # Anti-hallucination
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  compression_ratio_threshold: 2.4
```

---

## 🎯 Deploy em GPU (WhisperX + Diarization)

### Preparar servidor GPU Linux
```bash
# 1. Deploy em RunPod, Vast.ai, ou servidor próprio com GPU NVIDIA
# 2. Instalar dependências
pip install faster-whisper whisperx pyannote.audio torch

# 3. Obter HuggingFace token (gratuito)
# https://huggingface.co/settings/tokens
# Aceitar termos: https://huggingface.co/pyannote/speaker-diarization

# 4. Configurar
export HUGGINGFACE_TOKEN="hf_..."
```

### Config para diarization
```yaml
# server-config-gpu.yaml
backend:
  name: "whisperx"  # ← Agora disponível!
  model: "large-v3"
  language: "pt"
  enable_diarization: true
  huggingface_token: "${HUGGINGFACE_TOKEN}"
  enable_vad: true
```

### Testar
```python
from server.backends import BackendRegistry

# Em servidor GPU, whisperx estará disponível
available = BackendRegistry.list_available()
# Resultado: ['whisperx', 'faster-whisper', 'seamless-m4t']

# Criar backend com diarization
backend = BackendRegistry.create("whisperx", {
    "model": "large-v3",
    "enable_diarization": True,
})
await backend.initialize()

# Transcrever
result = await backend.transcribe_chunk(audio)
print(result.speaker)  # "SPEAKER_00", "SPEAKER_01", etc
```

---

## 🧪 Testes

### Rodar testes existentes
```bash
# Todos os testes unitários (75)
python -m pytest tests/unit/ -v

# Testes específicos
python -m pytest tests/unit/test_backend_registry.py -v
python -m pytest tests/unit/test_platform.py -v
python -m pytest tests/unit/test_models.py -v
```

### Adicionar novos testes
```python
# tests/integration/test_faster_whisper_full.py
import pytest
from server.backends import BackendRegistry

@pytest.mark.asyncio
async def test_faster_whisper_word_timestamps():
    backend = BackendRegistry.create("faster-whisper", {})
    await backend.initialize()

    # ... testar com áudio real
```

---

## 📚 Documentação Adicional

### Arquivos importantes
- `spec.md` - Especificação completa v2.0
- `PLAN.md` - Roadmap original
- `IMPLEMENTATION_SUMMARY.md` - O que foi feito
- `NEXT_STEPS.md` - Este arquivo

### Arquivos de código
- `server/backends/registry.py` - Backend Registry
- `server/backends/faster_whisper_backend.py` - Backend universal ⭐
- `server/backends/mlx_backend.py` - Backend Mac
- `server/backends/whisperx_backend.py` - Diarization (GPU)
- `server/backends/seamless_backend.py` - Translation (GPU)
- `server/models/result.py` - TranscriptionResult
- `server/models/capability.py` - Capabilities
- `server/utils/platform.py` - Platform detection

---

## 🐛 Troubleshooting

### Backend não aparece como disponível
```python
# Verificar plataforma
from server.utils.platform import detect_platform
print(detect_platform())

# Ver todos backends e plataformas suportadas
from server.backends import BackendRegistry
for name, info in BackendRegistry.list_all().items():
    print(f"{name}: {[p.value for p in info.supported_platforms]}")
```

### Erro ao criar backend
```python
# Erro vai mostrar backends disponíveis automaticamente
try:
    backend = BackendRegistry.create("whisperx", {})
except Exception as e:
    print(e)  # Mostra sugestões
```

---

## ✅ Checklist de Integração

- [ ] Atualizar `server/main.py` para usar BackendRegistry
- [ ] Atualizar `server/websocket_handler.py` para TranscriptionResult
- [ ] Atualizar cliente Bun com novos tipos
- [ ] Implementar renderização word-by-word no cliente
- [ ] Testar com áudio real usando faster-whisper
- [ ] Criar config de exemplo com faster-whisper
- [ ] Testar em servidor GPU (opcional)
- [ ] Implementar diarization UI (se usar WhisperX)
- [ ] Implementar translation UI (se usar SeamlessM4T)
- [ ] Atualizar README.md com nova arquitetura

---

## 🎉 Resumo

**O que você tem agora:**
- ✅ Foundation robusta com 75 testes
- ✅ 4 backends (2 funcionando no Mac)
- ✅ Word-level timestamps funcionando!
- ✅ Arquitetura capabilities-based
- ✅ Fail-fast validation
- ✅ Pronto para GPU (WhisperX, SeamlessM4T)

**Próximo passo mais importante:**
1. Integrar com servidor existente (main.py + websocket_handler.py)
2. Testar com áudio real
3. Deploy!

---

**Desenvolvido com TDD** 🧪
**Testes passando: 75/75** ✅
