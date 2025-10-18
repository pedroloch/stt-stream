# 🎯 Streaming Inteligente - LocalAgreement Policy

**Status**: ✅ Implementado e CORRIGIDO (FASE 0)

**Última atualização**: 2025-10-18 - Correção completa baseada em whisper_streaming (UFAL)

Este documento explica como funciona o algoritmo de streaming inteligente implementado no Whisper Stream.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [HypothesisBuffer - Implementação Correta](#hypothesisbuffer---implementação-correta)
3. [LocalAgreement-n Policy](#localagreement-n-policy)
4. [StreamingBuffer](#streamingbuffer)
5. [Context Window Otimizado](#context-window-otimizado)
6. [Buffer Trimming](#buffer-trimming)
7. [buffer_time_offset - Timestamps Absolutos](#buffer_time_offset---timestamps-absolutos)
8. [VAD Chunking](#vad-chunking)
9. [Exemplos Práticos](#exemplos-práticos)
10. [Performance](#performance)
11. [Troubleshooting - Bugs Comuns](#troubleshooting---bugs-comuns)

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

## HypothesisBuffer - Implementação Correta

### O Que É?

**HypothesisBuffer** é o componente central do LocalAgreement-n, baseado 100% na implementação de referência do [whisper_streaming (UFAL)](https://github.com/ufal/whisper_streaming).

**Diferenças da Implementação Anterior (BUGADA):**

| Aspecto | ❌ Anterior (Bugada) | ✅ Atual (Correta) |
|---------|---------------------|-------------------|
| **Dados** | Strings completas | Palavras com timestamps `[(start, end, word), ...]` |
| **Filtro** | Nenhum | Filtra palavras por timestamp (`> last_commited_time`) |
| **Comparação** | Prefixo de string | Palavra por palavra (word-level) |
| **Duplicatas** | Não remove | Remove via n-gram matching (1-5 palavras) |
| **Estados** | 1 (history) | 3 (commited, buffer, new) |

### Como Funciona

```python
class HypothesisBuffer:
    def __init__(self):
        self.commited_in_buffer = []  # Palavras confirmadas (ainda no buffer de áudio)
        self.buffer = []              # Última transcrição
        self.new = []                 # Nova transcrição
        self.last_commited_time = 0.0 # Timestamp da última palavra confirmada

    def insert(self, words, offset):
        """
        1. Ajusta timestamps para absolutos (+ offset)
        2. Filtra palavras > last_commited_time (CRÍTICO!)
        3. Remove duplicatas via n-gram matching
        """
        # Ajustar timestamps
        words_abs = [(s + offset, e + offset, w) for s, e, w in words]

        # FILTRO CRÍTICO: apenas palavras novas
        self.new = [w for w in words_abs if w[0] > self.last_commited_time - 0.1]

        # Remover duplicatas
        self._remove_duplicates()

    def flush(self):
        """
        LocalAgreement: compara self.buffer vs self.new palavra por palavra

        Returns:
            Lista de palavras confirmadas [(start, end, word), ...]
        """
        commit = []
        while self.new and self.buffer:
            if self.new[0][2] == self.buffer[0][2]:  # Mesmo texto?
                commit.append(self.new[0])
                self.last_commited_time = self.new[0][1]  # Atualizar timestamp
                self.new.pop(0)
                self.buffer.pop(0)
            else:
                break  # Divergência!

        self.buffer = self.new  # Nova vira anterior
        return commit
```

### Por Que Isso Corrige os Bugs?

**Bug 1: Alucinações Repetidas**
- ❌ Antes: Texto confirmado era re-processado sem filtro
- ✅ Agora: `last_commited_time` filtra palavras já confirmadas

**Bug 2: Timestamps Resetando**
- ❌ Antes: Sem `buffer_time_offset`, timestamps sempre relativos a 0.0
- ✅ Agora: `offset` parameter ajusta timestamps para absolutos

**Bug 3: Texto Cortado**
- ❌ Antes: Comparação de string falhava com alucinações
- ✅ Agora: Comparação palavra-por-palavra é robusta

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

## buffer_time_offset - Timestamps Absolutos

### O Problema

Quando transcrevemos chunks de áudio, o Whisper retorna timestamps **relativos ao chunk**:

```python
# Chunk 1 (0-2s de áudio)
words = [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]

# Chunk 2 (após trim de 1s, áudio agora é 1-3s)
# MAS timestamps são RELATIVOS ao chunk!
words = [(0.0, 0.5, "how")]  # ❌ Parece que começa em 0.0s
```

**Problema:** Não conseguimos saber se uma palavra é nova ou já foi confirmada!

### A Solução: buffer_time_offset

`buffer_time_offset` rastreia o **tempo absoluto** do início do buffer de áudio.

```python
class StreamingBuffer:
    def __init__(self):
        self.audio_buffer = np.array([])
        self.buffer_time_offset = 0.0  # Tempo absoluto do início do buffer

    async def _trim_buffer_conservative(self):
        # Remover 1.0s de áudio
        trim_samples = int(1.0 * SAMPLE_RATE)
        self.audio_buffer = self.audio_buffer[trim_samples:]

        # ⭐ CRÍTICO: Atualizar offset
        self.buffer_time_offset += 1.0  # Agora buffer começa em 1.0s (absoluto)

    async def process(self):
        result = await self.backend.transcribe(self.audio_buffer)

        # Timestamps são RELATIVOS ao buffer
        # Converter para ABSOLUTOS:
        self.hypothesis.insert(result.words, offset=self.buffer_time_offset)
        #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                     Ajusta timestamps!
```

### Fluxo Completo

```
Iteração 1:
- audio_buffer: [0-2s de áudio]
- buffer_time_offset: 0.0
- Transcrição: [(0.0, 0.5, "hello"), (0.6, 1.0, "world")]
- insert(words, offset=0.0) → timestamps absolutos: [(0.0, 0.5, "hello"), ...]

Iteração 2 (após trim de 1.0s):
- audio_buffer: [1-3s de áudio] (removeu primeiro 1s)
- buffer_time_offset: 1.0  ← Atualizado após trim
- Transcrição: [(0.0, 0.5, "how")] (relativo ao buffer)
- insert(words, offset=1.0) → timestamps absolutos: [(1.0, 1.5, "how")]
                        ^^^^
                        Ajusta timestamps!

HypothesisBuffer:
- last_commited_time = 1.0
- Filtro: w[0] > 1.0 - 0.1
- ✅ "how" (start=1.0) passa!
- ❌ "hello" (start=0.0) seria filtrado
```

### Por Que Isso É Crítico?

Sem `buffer_time_offset`:
- ❌ Timestamps sempre começam em 0.0 após trim
- ❌ HypothesisBuffer não consegue filtrar palavras antigas
- ❌ Texto confirmado é re-processado → loop infinito
- ❌ Modelo recebe contexto desalinhado → alucinações

Com `buffer_time_offset`:
- ✅ Timestamps absolutos consistentes
- ✅ HypothesisBuffer filtra corretamente
- ✅ Contexto alinhado com áudio
- ✅ Sem loops ou alucinações

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

## Troubleshooting - Bugs Comuns

### 🐛 Bug: "Muito obrigado" ou Texto Repetindo Infinitamente

**Sintomas:**
- Mesma transcrição aparece repetidamente
- Texto não relacionado ao áudio
- Probabilidades baixas (< 0.3)
- Timestamps resetando para 0.0

**Causa:**
1. Buffer trimmed agressivamente (imediato após confirmação)
2. Sobra apenas silêncio/ruído no buffer
3. Modelo alucina texto aleatório
4. LocalAgreement sempre confirma (texto sempre igual)
5. Loop infinito

**Solução:**
```yaml
# server-config.yaml
whisper:
  buffer_trimming_sec: 15.0  # ⭐ Trim conservador (não imediato!)
  vad_threshold: 0.5         # ⭐ Evita processar silêncio
```

**Validação:**
```bash
# Logs devem mostrar:
[INFO] Buffer trimmed: removido 2.5s, restante 12.5s
# ✅ Buffer NUNCA deve ficar vazio após confirmação!

# Se ver:
[INFO] Buffer completamente trimmed
# ❌ PROBLEMA! Trim agressivo demais
```

---

### 🐛 Bug: Timestamps Sempre 0.0

**Sintomas:**
- `segment.start = 0.0` em todas as transcrições
- Texto cortado ou incompleto
- Palavras confirmadas reaparecendo

**Causa:**
- `buffer_time_offset` não está sendo atualizado após trim
- Timestamps ficam relativos ao buffer (sempre começam em 0.0)

**Solução:**
```python
# StreamingBuffer._trim_buffer_conservative()
async def _trim_buffer_conservative(self):
    # ...
    self.buffer_time_offset += removed_duration  # ⭐ CRÍTICO!
    # ...
```

**Validação:**
```python
# Logs devem mostrar buffer_offset incrementando:
buffer_offset: 0.00s
buffer_offset: 2.50s  # ✅ Incrementou após trim
buffer_offset: 5.20s  # ✅ Continua incrementando
```

---

### 🐛 Bug: Texto Cortado ("to bastante" ao invés de "gosto bastante")

**Sintomas:**
- Primeiras sílabas/palavras cortadas
- Texto parece começar no meio da frase

**Causa:**
1. Backend não retornou `words` (apenas segments)
2. HypothesisBuffer não consegue fazer LocalAgreement
3. Fallback retorna transcrição sem filtragem

**Solução:**
```python
# Verificar se backend retorna words:
result = await backend.transcribe_chunk(audio)
assert result.words is not None  # ⭐ Deve ter words!

# faster_whisper_backend.py deve ter:
word_timestamps=True  # ⭐ Obrigatório!
```

**Validação:**
```bash
# Logs devem mostrar:
Extraídas 25 palavras com timestamps
✅ Confirmado (3 palavras): 'olá mundo como'

# Se ver:
Backend não retornou words! LocalAgreement não funcionará
# ❌ PROBLEMA! Backend não configurado corretamente
```

---

### 🐛 Bug: Alucinações em Silêncio

**Sintomas:**
- Texto aleatório quando não há fala
- `no_speech_prob` alto (> 0.9)
- Probabilidades baixas

**Causa:**
- Segmentos de silêncio não estão sendo filtrados
- Modelo força transcrição mesmo sem fala

**Solução:**
```python
# faster_whisper_backend.py
if segment.no_speech_prob > 0.9:
    logger.debug(f"Pulando segmento de silêncio: '{segment.text}'")
    continue  # ⭐ Não processar!
```

**Validação:**
```bash
# Logs devem mostrar:
Pulando segmento de silêncio (no_speech_prob=0.95): 'Muito obrigado'

# Se não ver esse log em silêncio:
# ❌ PROBLEMA! Filtro de no_speech não está ativo
```

---

### 🐛 Bug: Context Desalinhado (WER Alto)

**Sintomas:**
- WER (Word Error Rate) alto
- Transcrições inconsistentes
- Nomes próprios errados

**Causa:**
- Contexto inclui texto que não está no buffer de áudio
- Modelo tenta alinhar contexto com áudio inexistente

**Solução:**
```python
def _get_prompt(self):
    # ⭐ Apenas palavras FORA do buffer (scrolled away)!
    scrolled_away = [w for w in self.commited_words
                     if w[1] <= self.buffer_time_offset]

    # ❌ ERRADO: usar ALL commited_words
    # ✅ CORRETO: filtrar por buffer_time_offset
```

**Validação:**
```bash
# Logs devem mostrar:
Prompt (127 chars): 'olá mundo...' (15 palavras scrolled away)

# Se ver muitas palavras quando buffer foi trimmed recentemente:
# ❌ PROBLEMA! Contexto inclui texto ainda no buffer
```

---

### ✅ Checklist de Correção

- [ ] `buffer_trimming_sec >= 15.0` (trimming conservador)
- [ ] `vad_threshold >= 0.5` (evita silêncio)
- [ ] `word_timestamps=True` no backend
- [ ] `no_speech_prob > 0.9` filtrado
- [ ] `buffer_time_offset` atualizado após trim
- [ ] Context usa apenas `scrolled_away_words`
- [ ] HypothesisBuffer filtra por `last_commited_time`
- [ ] Tests unitários passando (`pytest tests/unit/test_hypothesis_buffer.py`)

---

**Documentação criada em**: 2025-10-18
**Versão**: 2.0 (FASE 0 - CORRIGIDO)
**Autor**: Claude Code + Pedro
**Referência**: [whisper_streaming (UFAL)](https://github.com/ufal/whisper_streaming)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
