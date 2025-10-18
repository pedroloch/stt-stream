# 🎯 Streaming Inteligente - LocalAgreement Policy

**Status**: ✅ Implementado (FASE 0)

Este documento explica como funciona o algoritmo de streaming inteligente implementado no Whisper Stream.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [LocalAgreement-n Policy](#localagreement-n-policy)
3. [StreamingBuffer](#streamingbuffer)
4. [Context Window Otimizado](#context-window-otimizado)
5. [Buffer Trimming](#buffer-trimming)
6. [VAD Chunking](#vad-chunking)
7. [Exemplos Práticos](#exemplos-práticos)
8. [Performance](#performance)

---

## Visão Geral

**Problema**: Streaming chunk-by-chunk independente causa:
- ❌ Cortes mid-word
- ❌ Latência alta
- ❌ Texto instável (muda constantemente)
- ❌ Experiência ruim para o usuário

**Solução**: LocalAgreement + StreamingBuffer
- ✅ Confirma apenas texto **estável**
- ✅ Re-transcreve com overlap
- ✅ Usa contexto (últimas 100 palavras)
- ✅ Corta em pausas naturais (VAD)

---

## LocalAgreement-n Policy

### O Que É?

**LocalAgreement-n** confirma transcrição quando **n updates consecutivos concordam no prefixo**.

### Como Funciona (n=2)

```python
# Update 1
buffer = "olá mundo"
history = ["olá mundo"]
# Não confirma ainda (precisa de 2 concordâncias)

# Update 2
buffer = "olá mundo como você"
history = ["olá mundo", "olá mundo como você"]
# Prefixo comum: "olá mundo" ✅
# CONFIRMA: "olá mundo"

# Update 3
buffer = "olá mundo como você está"
history = ["olá mundo como você", "olá mundo como você está"]
# Prefixo comum: "olá mundo como você" ✅
# CONFIRMA: "como você" (apenas o NOVO)
```

### Implementação

```python
class LocalAgreementPolicy:
    def __init__(self, n: int = 2):
        self.n = n
        self.history: list[str] = []
        self.last_confirmed: str = ""

    def check(self, new_transcript: str) -> tuple[bool, str | None]:
        # Adicionar ao histórico
        self.history.append(new_transcript)

        # Manter apenas últimas n transcrições
        if len(self.history) > self.n:
            self.history.pop(0)

        # Precisa de n concordâncias
        if len(self.history) < self.n:
            return False, None

        # Encontrar prefixo comum (palavra por palavra)
        common_prefix = self._longest_common_prefix(self.history)

        # Retornar apenas texto NOVO (incremental)
        new_confirmed = common_prefix[len(self.last_confirmed):].strip()

        if new_confirmed:
            self.last_confirmed = common_prefix
            return True, new_confirmed

        return False, None
```

### Por Que Palavra por Palavra?

```python
# ❌ ERRADO: comparar strings completas
if transcript1 == transcript2:  # Nunca vai concordar!

# ✅ CORRETO: comparar prefixo palavra por palavra
words1 = ["olá", "mundo", "como"]
words2 = ["olá", "mundo", "como", "você"]
# Prefixo comum: ["olá", "mundo", "como"]
```

---

## StreamingBuffer

### Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                 StreamingBuffer                      │
├─────────────────────────────────────────────────────┤
│  Audio Buffer: [chunk1, chunk2, chunk3, ...]        │
│  Confirmed Text: "olá mundo como"                   │
│  LocalAgreement: Policy(n=2)                        │
└─────────────────────────────────────────────────────┘
         │
         ├─> add_chunk(audio)  # Acumula áudio
         │
         └─> process()  # Transcreve + LocalAgreement
                │
                ├─> Parcial (is_final=False)
                └─> Final (is_final=True)
```

### Fluxo de Processamento

```python
# 1. Cliente envia chunk de áudio
buffer.add_chunk(audio_float32)

# 2. Processar quando buffer >= min_chunk_size
result = buffer.process()

# 3. Transcrever com CONTEXTO
context = buffer.get_context_window(max_words=100)
transcript = backend.transcribe(audio, context=context)

# 4. LocalAgreement decide
should_confirm, confirmed_text = agreement.check(transcript)

if should_confirm:
    # Texto estável! Confirmar
    return TranscriptionResult(
        text=confirmed_text,
        is_final=True  # ✅ CONFIRMADO
    )
else:
    # Ainda não estável, mostrar preview
    return TranscriptionResult(
        text=transcript,
        is_final=False  # ⏳ PARCIAL
    )
```

---

## Context Window Otimizado

### Por Que Context?

**Whisper funciona MUITO melhor com contexto!**

- ✅ -15% WER (segundo whisper_streaming)
- ✅ Coerência (nomes próprios, termos técnicos)
- ✅ Capitalização e formatação

### Problema: Context Muito Grande

```python
# ❌ ERRADO: passar TODO o texto confirmado
context = "Olá, tudo bem? Como você está hoje? E aí, o que você está fazendo? ..."
# 224+ palavras = degrada performance!

# ✅ CORRETO: últimas 100 palavras apenas
words = full_text.split()
context = " ".join(words[-100:])
# Contexto suficiente, performance ótima
```

### Implementação

```python
def _get_context_window(self, full_text: str, max_words: int = 100) -> str:
    if not full_text:
        return ""

    words = full_text.split()

    if len(words) <= max_words:
        return full_text

    # Retornar últimas N palavras
    return " ".join(words[-max_words:])
```

---

## Buffer Trimming

### Por Que Trimming?

Sem trimming, o buffer cresce infinitamente → OOM!

### Estratégias

#### 1. Segment-based (padrão)

Corta no timestamp do último segmento confirmado:

```python
# Texto confirmado: "olá mundo como você"
# Último segmento: end=2.5s

# Trim: remover áudio de 0s a 2.5s
audio_buffer = audio_buffer[2.5s:]  # Manter resto
```

#### 2. Sentence-based (conservador)

Espera fim de frase (`.`, `!`, `?`) antes de trim:

```python
# Texto: "olá mundo como você"
# Sem pontuação → NÃO trim ainda

# Texto: "olá mundo como você está."
# Tem ponto → OK, trim até "está."
```

### Configuração

```yaml
whisper:
  buffer_trimming: "segment"  # ou "sentence"
```

---

## VAD Chunking

### Silero VAD

Voice Activity Detection para cortar chunks em pausas naturais:

```python
from server.streaming.vad import VADChunker

chunker = VADChunker(
    threshold=0.5,
    min_silence_duration=0.3,  # 300ms
    min_speech_duration=0.5,   # 500ms
)

# Detectar segmentos de fala
segments = chunker.detect_speech(audio)
# [(0.0, 2.5), (3.0, 5.5)]  # start, end em segundos

# Encontrar melhor ponto de corte
split_point = chunker.find_best_split_point(audio, target_duration=5.0)
```

### Benefícios

- ✅ Evita cortes mid-word
- ✅ Melhora accuracy (chunks completos)
- ✅ Latência mais consistente

---

## Exemplos Práticos

### Exemplo 1: Streaming Básico

```python
from server.streaming import StreamingBuffer, BufferConfig

# Configurar buffer
config = BufferConfig(
    min_chunk_size=1.0,
    agreement_threshold=2,
    buffer_trimming="segment"
)

buffer = StreamingBuffer(backend=whisper_backend, config=config)

# Loop de streaming
while True:
    audio_chunk = capture_audio()  # 1s de áudio

    # Adicionar ao buffer
    await buffer.add_chunk(audio_chunk)

    # Processar
    result = await buffer.process()

    if result:
        if result.is_final:
            print(f"✅ Confirmado: {result.text}")
        else:
            print(f"⏳ Parcial: {result.text}")
```

### Exemplo 2: Integração WebSocket

```python
class WebSocketHandler:
    def __init__(self):
        # Um buffer por cliente
        self.buffers: dict[int, StreamingBuffer] = {}

    async def handle_audio(self, client_id: int, audio_data: bytes):
        # Obter buffer do cliente
        buffer = self.buffers[client_id]

        # Converter e adicionar
        audio = convert_to_float32(audio_data)
        await buffer.add_chunk(audio)

        # Processar
        result = await buffer.process()

        if result and result.text:
            # Enviar ao cliente
            await self.send_json({
                "type": "transcription",
                "text": result.text,
                "is_final": result.is_final
            })
```

---

## Performance

### Métricas Esperadas

| Métrica | Sem LocalAgreement | Com LocalAgreement |
|---------|-------------------|-------------------|
| **Latência (primeira palavra)** | 1-2s | < 2s |
| **Accuracy (WER)** | Baseline | -15% (melhor) |
| **Estabilidade** | Texto muda constantemente | Apenas quando confirmado |
| **UX** | ⚠️ Confuso | ✅ Claro |

### Benchmarks

```bash
# Rodar benchmarks (quando implementado)
python scripts/benchmark.py --model large-v3 --language pt

# Métricas:
# - WER (Word Error Rate)
# - Latência (tempo até primeira palavra)
# - RTF (Real-Time Factor)
# - Memória usada
```

---

## Configuração Recomendada

### Para Português (PT)

```yaml
whisper:
  model: "large-v3"
  language: "pt"
  backend: "auto"
  compute_type: "auto"

  # Streaming
  min_chunk_size: 1.0
  buffer_trimming: "segment"

  # VAD
  use_vad: true
  vad_threshold: 0.5
```

### Para Baixa Latência

```yaml
whisper:
  model: "distil-large-v3"  # 6x mais rápido!
  min_chunk_size: 0.5       # Chunks menores
  agreement_threshold: 2     # Mínimo para confirmar
```

### Para Máxima Accuracy

```yaml
whisper:
  model: "large-v3"
  min_chunk_size: 2.0        # Mais contexto
  agreement_threshold: 3     # Mais conservador
  buffer_trimming: "sentence"  # Espera frases completas
```

---

## Referências

- **whisper_streaming (UFAL)**: https://github.com/ufal/whisper_streaming
- **OpenAI Whisper**: https://github.com/openai/whisper
- **faster-whisper**: https://github.com/guillaumekln/faster-whisper
- **Silero VAD**: https://github.com/snakers4/silero-vad

---

## Próximos Passos (FASE 1+)

- [ ] Validar WhisperX (diarization em PT)
- [ ] Benchmarks completos (WER, latência, RTF)
- [ ] Otimizar n (agreement_threshold) dinamicamente
- [ ] Suporte a múltiplos idiomas simultâneos
- [ ] Streaming híbrido (VAD + LocalAgreement)

---

**Documentação criada em**: 2025-10-18
**Versão**: 1.0 (FASE 0)
**Autor**: Claude Code + Pedro

🤖 Generated with [Claude Code](https://claude.com/claude-code)
