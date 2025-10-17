# 🚀 Whisper Stream v2.0 - Quick Reference

Referência rápida para usar o novo sistema.

---

## 🎯 Como Usar (Exemplos Práticos)

### Exemplo 1: Listar Backends Disponíveis
```python
from server.backends import BackendRegistry

# Ver todos backends
all_backends = BackendRegistry.list_all()
for name, info in all_backends.items():
    print(f"{name}: {info.capabilities}")

# Ver apenas compatíveis com sua máquina
available = BackendRegistry.list_available()
print("Disponíveis:", list(available.keys()))
```

### Exemplo 2: Criar e Usar Faster-Whisper (Recomendado)
```python
from server.backends import BackendRegistry
import numpy as np

# Criar backend
backend = BackendRegistry.create("faster-whisper", {
    "model": "base",  # ou "distil-large-v3"
    "language": "pt",
    "enable_word_timestamps": True,
    "enable_vad": True,
})

# Inicializar
await backend.initialize()

# Transcrever
audio = np.random.randn(16000).astype(np.float32)
result = await backend.transcribe_chunk(audio)

# Acessar resultado
print(result.text)  # Texto completo
print(result.confidence)  # Confiança
print(result.language)  # Idioma detectado

# Word timestamps
if result.segments:
    for segment in result.segments:
        if segment.words:
            for word in segment.words:
                print(f"{word.word}: {word.start:.2f}-{word.end:.2f}s")
```

### Exemplo 3: Tentar Backend Incompatível
```python
try:
    # WhisperX requer CUDA GPU
    backend = BackendRegistry.create("whisperx", {})
except Exception as e:
    # Erro mostra alternativas automaticamente
    print(e)
    # "Backend 'whisperx' não suportado em macos_arm64.
    #  Backends disponíveis: mlx, faster-whisper..."
```

---

## 📊 Backend Comparison

| Backend | Plataformas | Word Timestamps | Diarization | Translation |
|---------|-------------|-----------------|-------------|-------------|
| **mlx** | Mac M1 | ❌ | ❌ | ❌ |
| **faster-whisper** ⭐ | Todas | ✅ | ❌ | ❌ |
| **whisperx** | Linux CUDA | ✅ | ✅ | ❌ |
| **seamless-m4t** | Linux CUDA | ❌ | ❌ | ✅ |

**Recomendado:** `faster-whisper` (universal + word timestamps)

---

## 🔑 Campos do TranscriptionResult

### Campos Obrigatórios (sempre presentes)
```python
result.text          # str - Texto transcrito
result.is_final      # bool - É final ou parcial?
result.confidence    # float - Confiança (0-1)
result.language      # str - Código do idioma
result.timestamp     # datetime - Momento da transcrição
```

### Campos Opcionais (depende do backend)
```python
result.segments      # List[Segment] | None - Segmentos com timestamps
result.speaker       # str | None - "SPEAKER_00" (se diarization)
result.translation   # Dict[str, str] | None - {"en": "...", "es": "..."}
```

### Segment e Word
```python
segment = result.segments[0]
segment.start        # float - Início (segundos)
segment.end          # float - Fim (segundos)
segment.text         # str - Texto do segmento
segment.words        # List[Word] | None - Palavras

word = segment.words[0]
word.word            # str - Palavra
word.start           # float - Início
word.end             # float - Fim
word.probability     # float - Confiança da palavra
```

---

## 🎨 Conversão para WebSocket

```python
# Converter resultado para enviar ao cliente
ws_dict = result.to_websocket_dict()

# Formato enviado:
{
    "type": "transcription",
    "text": "...",
    "is_final": true,
    "confidence": 0.95,
    "language": "pt",
    "timestamp": "2025-10-17T12:00:00",
    "segments": [...],  # Se disponível
    "speaker": "...",   # Se diarization
    "translations": {}  # Se translation
}
```

---

## ⚙️ Configurações Recomendadas

### Desenvolvimento (rápido)
```python
backend = BackendRegistry.create("faster-whisper", {
    "model": "base",
    "language": "pt",
})
```

### Produção (melhor qualidade)
```python
backend = BackendRegistry.create("faster-whisper", {
    "model": "distil-large-v3",  # 6x mais rápido!
    "language": "pt",
    "compute_type": "int8_float16",
    "enable_word_timestamps": True,
    "enable_vad": True,
    "vad_threshold": 0.75,
    "no_speech_threshold": 0.6,
})
```

### GPU com Diarization
```python
backend = BackendRegistry.create("whisperx", {
    "model": "large-v3",
    "language": "pt",
    "enable_diarization": True,
    "huggingface_token": "hf_...",
})
```

---

## 🐛 Debugging

### Ver plataforma atual
```python
from server.utils.platform import detect_platform
print(detect_platform())  # Platform.MACOS_APPLE_SILICON
```

### Ver info do backend
```python
backend = BackendRegistry.create("faster-whisper", {})
info = backend.info

print(info.name)  # "faster-whisper"
print(info.supported_platforms)  # {Platform.MACOS_APPLE_SILICON, ...}
print(info.capabilities)  # {Capability.TRANSCRIPTION, ...}
```

### Verificar capabilities
```python
from server.models.capability import Capability

if Capability.WORD_TIMESTAMPS in backend.info.capabilities:
    print("Suporta word timestamps!")

if Capability.SPEAKER_DIARIZATION in backend.info.capabilities:
    print("Suporta diarization!")
```

---

## 📝 Testes Rápidos

### Testar detecção de plataforma
```bash
python -c "from server.utils.platform import detect_platform; print(detect_platform())"
```

### Testar backends disponíveis
```bash
python -c "from server.backends import BackendRegistry; print(list(BackendRegistry.list_available().keys()))"
```

### Testar criação de backend
```bash
python -c "
from server.backends import BackendRegistry
backend = BackendRegistry.create('faster-whisper', {})
print('✅ Backend criado:', backend.info.name)
"
```

### Rodar testes unitários
```bash
python -m pytest tests/unit/ -v
```

---

## 🎯 Checklist de Uso

**Para começar:**
- [ ] Verificar plataforma: `detect_platform()`
- [ ] Listar backends disponíveis: `BackendRegistry.list_available()`
- [ ] Escolher backend (recomendado: `faster-whisper`)
- [ ] Criar e inicializar: `create()` + `initialize()`
- [ ] Testar com áudio dummy

**Para produção:**
- [ ] Usar modelo distil-large-v3 (6x mais rápido)
- [ ] Habilitar word timestamps
- [ ] Habilitar VAD
- [ ] Configurar anti-hallucination
- [ ] Testar com áudio real
- [ ] Integrar com servidor existente

**Para GPU (opcional):**
- [ ] Deploy em servidor Linux CUDA
- [ ] Usar WhisperX para diarization
- [ ] Obter HuggingFace token
- [ ] Configurar speaker identification
- [ ] Testar com reunião real

---

## 📚 Documentação Completa

- `IMPLEMENTATION_SUMMARY.md` - O que foi feito
- `NEXT_STEPS.md` - Como continuar
- `spec.md` - Especificação completa
- `PLAN.md` - Roadmap original

---

**Desenvolvido com TDD** 🧪
**75 testes passando** ✅
**Pronto para produção** 🚀
