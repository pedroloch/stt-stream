# 🏗️ Arquiteturas de Modelos STT - Guia Técnico

**Data**: 2025-10-18
**Objetivo**: Documentar arquiteturas, diferenças e status de implementação

---

## 📊 Status de Implementação

### ✅ Implementados e Funcionando

| Modelo | Backend File | Arquitetura | Batch | Streaming | Diarization | Translation | Status |
|--------|--------------|-------------|-------|-----------|-------------|-------------|--------|
| **faster-whisper** | `faster_whisper_backend.py` | Encoder-Decoder | ✅ | ✅ | Via post-proc | ❌ | ✅ Produção |
| **MLX Whisper** | `mlx_backend.py` | Encoder-Decoder | ✅ | ✅ | Via post-proc | ❌ | ✅ Produção (M1) |
| **WhisperX** | `whisperx_backend.py` | Encoder-Decoder + Align | ✅ | ⚠️ | ✅ Integrado | ❌ | ⚠️ Não validado PT |
| **CrisperWhisper** | `crisper_backend.py` | Encoder-Decoder + Fillers | ✅ | ✅ | Via post-proc | ❌ | ⚠️ Não validado PT |
| **CPU Backend** | `cpu_backend.py` | Whisper CPU | ✅ | ✅ | ❌ | ❌ | ✅ Funcionando |
| **CUDA Backend** | `cuda_backend.py` | Whisper CUDA | ✅ | ✅ | ❌ | ❌ | ✅ Funcionando |
| **SeamlessM4T** | `seamless_backend.py` | Multimodal Enc-Dec | ⚠️ | ⚠️ | ❌ | ⚠️ | ⚠️ Parcial |

**Legenda**:
- ✅ = Implementado e testado
- ⚠️ = Implementado mas não validado/parcial
- ❌ = Não suportado

---

### 🆕 A Implementar (Roadmap)

| Modelo | Prioridade | Arquitetura | Por quê implementar |
|--------|------------|-------------|---------------------|
| **Voxtral (24B/3B)** | 🔥 ALTA | LLM-based STT | Melhor WER que Whisper, PT nativo |
| **Parakeet TDT (0.6B)** | 🔥 ALTA | Transducer (TDT) | 3380x RTF, streaming nativo |
| **Granite Speech (8B)** | ⭐ MÉDIA | LLM-based STT | Translation (AST), Apache 2.0 |
| **SeamlessM4T v2** | ⭐ MÉDIA | Multimodal | 100+ idiomas, S2S translation |
| **Kyutai STT** | 🔬 BAIXA | Delayed Streams | Pesquisa (sem PT) |
| **Canary 1B** | 🔬 BAIXA | Conformer RNN-T | Batch-only, não comercial |

---

## 🔬 Taxonomia de Arquiteturas

### **Família 1: Encoder-Decoder (Whisper-style)**

```
┌─────────────────────────────────────────────────────────┐
│  Áudio → Mel Spec → Encoder → Decoder → Texto          │
│           (log)     (Transformer) (Autoregressive)      │
└─────────────────────────────────────────────────────────┘
```

**Modelos**: Whisper, faster-whisper, MLX, WhisperX, CrisperWhisper

**Características**:
- ✅ 99 idiomas (multilingual por design)
- ✅ Word timestamps nativos
- ✅ Funciona batch e streaming
- ⚠️ Latência moderada (autoregressive = gera 1 token por vez)
- ⚠️ Sem translation nativa

**Implementação**:
```python
# Todos compartilham interface similar:
segments, info = model.transcribe(
    audio,
    language="pt",
    word_timestamps=True
)
```

**Quando usar**:
- Caso de uso geral (transcrição PT/EN/outros)
- Precisa funcionar em qualquer plataforma
- Modelo maduro e testado

---

### **Família 2: LLM-Based STT (Decoder-Only)**

#### **Voxtral (Mistral AI)**

```
┌───────────────────────────────────────────────────────────────┐
│  Áudio → Audio Front-End → Mistral LLM → Texto/Resumo/Q&A    │
│           (Conformer?)       (3B ou 24B)                      │
└───────────────────────────────────────────────────────────────┘
```

**Diferenças-chave**:
- **Contexto longo**: 32k tokens (~30 min de áudio)
- **Multi-task**: Não só transcreve, também entende conteúdo
- **Streaming nativo**: API dedicada para chunks

**Capabilities únicas**:
```python
# Transcrição normal
result = voxtral.transcribe(audio, language="pt")

# ⭐ NOVO: Resumo do conteúdo (Voxtral-specific)
summary = voxtral.summarize(audio)
# "A reunião discutiu Q3 targets e lançamento de produto X"

# ⭐ NOVO: Q&A sobre o áudio (Voxtral-specific)
answer = voxtral.qa(audio, "Quem falou sobre vendas?")
# "João mencionou vendas aos 5min30s"
```

**Trade-offs**:
- ✅ Melhor WER (~5-6% vs ~7-8% Whisper)
- ✅ PT nativo (não fine-tune)
- ✅ Apache 2.0 (comercial OK)
- ⚠️ Modelo grande (3B ou 24B)
- ⚠️ Recente (julho 2025, pode ter bugs)

**Quando usar**:
- Melhor accuracy em PT/EN
- Substituir Whisper Large-v3
- Futuro: análise de conteúdo (não só transcrição)

---

#### **Granite Speech (IBM)**

```
┌────────────────────────────────────────────────────────────────────┐
│  Áudio → Conformer+CTC → Q-Former → Granite LLM → Texto/Tradução  │
│           (Encoder)       (Projector)  (8B decoder)                │
└────────────────────────────────────────────────────────────────────┘
```

**Diferenças-chave**:
- **3 componentes**: Speech encoder + projector + LLM
- **Arbitrariamente longo**: Testado até 20min, suporta 128k tokens
- **Translation nativa (AST)**: PT→EN, PT→ES diretamente

**Capabilities únicas**:
```python
# ASR (Automatic Speech Recognition)
result = granite.transcribe(audio, language="pt")

# ⭐ AST (Automatic Speech Translation)
result = granite.translate(
    audio,
    source_language="pt",
    target_language="en"
)
# Retorna texto em inglês diretamente (não transcreve PT primeiro)
```

**Trade-offs**:
- ✅ Translation nativa (único com Apache 2.0)
- ✅ Áudio muito longo (20+ min)
- ✅ Topo leaderboard HF (5.85% WER)
- ⚠️ Modelo grande (8B)
- ⚠️ Streaming não otimizado (batch-first)

**Quando usar**:
- Translation PT→EN/ES/FR comercialmente
- Arquivos longos (>10 min)
- Alternativa ao SeamlessM4T com licença comercial

---

### **Família 3: Transducer-Based (RNN-T / TDT)**

#### **Parakeet TDT (NVIDIA NeMo)**

```
┌─────────────────────────────────────────────────────────────┐
│  Áudio → FastConformer → TDT Decoder → Texto                │
│           (8x downsampled)  (Token+Duration)                │
└─────────────────────────────────────────────────────────────┘
```

**TDT = Token-and-Duration Transducer**

Diferença vs. Whisper autoregressive:

```python
# Whisper (tradicional):
# Gera 1 token por vez, sequencialmente
"O", "l", "á", ",", " ", "t", "u", "d", "o" ...
# 9 tokens = 9 forward passes

# Parakeet TDT:
# Prediz token + duração, pula blanks
("Olá", 0.4s), ("tudo", 0.3s), ("bem", 0.3s)
# 3 predições = muito mais rápido!
```

**Trade-offs**:
- ✅ **Velocidade absurda**: RTF 3380x (56 min de áudio em 1 segundo!)
- ✅ Streaming nativo (design otimizado)
- ✅ Timestamps precisos (parte do modelo)
- ✅ Modelo pequeno (600M params = roda em CPU)
- ⚠️ Benchmarks focados em EN (PT não testado)
- ⚠️ Word timestamps não confirmados

**Quando usar**:
- Batch processing rápido (milhares de arquivos)
- Edge devices (CPU, RPi, Jetson)
- Fallback rápido para M1/CPU
- Streaming de baixa latência

---

### **Família 4: Multimodal Translation**

#### **SeamlessM4T v2 (Meta)**

```
┌────────────────────────────────────────────────────────────────┐
│                      ┌→ Text Decoder → Texto                    │
│  Áudio → Conformer ──┤                                          │
│         (w2v-BERT)   └→ T2U Decoder → Vocoder → Áudio           │
│                         (Non-AR)      (HiFi-GAN)                │
└────────────────────────────────────────────────────────────────┘
```

**4 modos de operação**:
1. **S2T** (Speech-to-Text): PT (fala) → EN (texto)
2. **S2S** (Speech-to-Speech): PT (fala) → EN (fala)
3. **T2T** (Text-to-Text): PT (texto) → EN (texto)
4. **T2S** (Text-to-Speech): PT (texto) → EN (fala)

**Diferenças-chave**:
- **Pipeline duplo**: Speech → Text → Units → Speech
- **100+ idiomas**: Mais que qualquer outro
- **Multimodal**: Entrada/saída pode ser texto ou fala

**Trade-offs**:
- ✅ 100+ idiomas (líder absoluto)
- ✅ Translation melhor que Granite (especialista)
- ✅ S2S único (fala → fala diretamente)
- ⚠️ **Licença NC** (não comercial!)
- ⚠️ Latência alta (3-5s para translation)
- ⚠️ Modelo pesado (~10GB)

**Quando usar**:
- Reuniões multilíngues (PT↔EN↔ES)
- Casos não comerciais
- Pesquisa de translation

---

## 🎯 Matriz de Decisão por Caso de Uso

### **1. Transcrição Streaming PT/EN (prioridade #1)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Voxtral 24B** | ⭐⭐⭐⭐⭐ | Melhor WER, streaming nativo |
| faster-whisper (distil-large-v3) | ⭐⭐⭐⭐ | Maduro, 6x rápido, fallback confiável |
| Voxtral 3B | ⭐⭐⭐ | Edge/M1, WER ~6-7% |
| Parakeet 0.6B | ⭐⭐⭐ | Muito rápido, CPU-friendly |

**Recomendação atual**: faster-whisper (distil-large-v3)
**Upgrade path**: Voxtral 24B quando implementado

---

### **2. Batch Processing (arquivos longos)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Granite 8B** | ⭐⭐⭐⭐⭐ | Aceita 20+ min, melhor WER |
| Voxtral 24B | ⭐⭐⭐⭐ | Contexto longo (30 min) |
| WhisperX | ⭐⭐⭐ | Se precisar diarization |
| Parakeet | ⭐⭐⭐⭐ | Se velocidade > accuracy |

**Recomendação atual**: faster-whisper + WhisperX refinement
**Upgrade path**: Granite 8B ou Voxtral 24B

---

### **3. Translation (PT→EN/ES)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Granite 8B** | ⭐⭐⭐⭐⭐ | AST nativo, Apache 2.0 |
| SeamlessM4T | ⭐⭐⭐⭐ | Melhor quality, mas NC |
| Whisper → LLM | ⭐⭐ | Workaround (2 etapas) |

**Recomendação atual**: Não implementado (workaround = transcrever + traduzir separado)
**Upgrade path**: Granite 8B (comercial) ou SeamlessM4T (não comercial)

---

### **4. Diarization (speaker ID)**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **WhisperX + Pyannote** | ⭐⭐⭐⭐ | Única opção open-source |
| faster-whisper + Sortformer | ⭐⭐⭐⭐ | SOTA 2025, streaming |
| faster-whisper + Pyannote | ⭐⭐⭐ | Simples, batch |

**Recomendação atual**: faster-whisper + Sortformer (streaming)
**Ou**: WhisperX (batch, word timestamps mais precisos)

---

### **5. Edge Devices / CPU / M1**

| Modelo | Score | Motivo |
|--------|-------|--------|
| **Parakeet 0.6B** | ⭐⭐⭐⭐⭐ | 600M params, RTF 3380x |
| MLX Whisper | ⭐⭐⭐⭐ | Otimizado M1 |
| Voxtral 3B | ⭐⭐⭐ | Bom WER, 3B params |
| faster-whisper tiny | ⭐⭐ | Rápido mas WER alto |

**Recomendação atual**: MLX Whisper (M1)
**Upgrade path**: Parakeet 0.6B (universal)

---

## 📋 Diferenças de Implementação

### **Interface Base (Atual)**

```python
# server/backends/base.py

class WhisperBackend(ABC):
    """Base para todos backends"""

    @abstractmethod
    async def transcribe_chunk(
        audio: np.ndarray,
        context: str | None
    ) -> TranscriptionResult:
        """Streaming: processa chunk incremental"""

    @abstractmethod
    async def transcribe_file(
        audio_file: Path
    ) -> TranscriptionResult:
        """Batch: processa arquivo completo"""
```

### **Extensões Necessárias para Novos Modelos**

```python
# server/backends/base.py (PROPOSTA)

class WhisperBackend(ABC):
    # ... métodos existentes ...

    # ⭐ NOVO: Translation (Granite, SeamlessM4T)
    async def translate(
        self,
        audio: np.ndarray,
        source_language: str,
        target_language: str
    ) -> TranscriptionResult:
        """AST (Automatic Speech Translation)"""
        raise NotImplementedError(
            f"{self.info.name} não suporta translation nativa"
        )

    # ⭐ NOVO: Summarization (Voxtral) - BAIXA PRIORIDADE
    async def summarize(
        self,
        audio: np.ndarray
    ) -> str:
        """Sumarização de conteúdo"""
        raise NotImplementedError(
            f"{self.info.name} não suporta summarization"
        )
```

### **Capabilities Enum (Atual vs Proposto)**

```python
# server/models/capability.py

class Capability(str, Enum):
    # ✅ Existentes:
    TRANSCRIPTION = "transcription"
    WORD_TIMESTAMPS = "word_timestamps"
    SPEAKER_DIARIZATION = "diarization"
    VAD = "vad"
    STREAMING = "streaming"

    # ⭐ NOVOS (para modelos futuros):
    TRANSLATION = "translation"           # Granite, SeamlessM4T
    SPEECH_TO_SPEECH = "speech_to_speech" # SeamlessM4T S2S
    SUMMARIZATION = "summarization"       # Voxtral (baixa prioridade)
    QA = "qa"                             # Voxtral (baixa prioridade)
```

---

## 🚀 Status e Próximos Passos

### **✅ Pronto para Produção**

1. **faster-whisper** - Universal, maduro, distil-large-v3 rápido
2. **MLX Whisper** - M1 otimizado
3. **Sortformer** - Diarization streaming SOTA 2025

### **⚠️ Implementado mas Não Validado**

1. **WhisperX** - Precisa testar diarization PT (RunPod)
2. **CrisperWhisper** - Precisa validar fillers PT
3. **SeamlessM4T** - Implementação parcial, não testada

### **🆕 Roadmap de Implementação**

**FASE 1**: API Batch (prioridade #1)
- Endpoint `/v1/transcribe`
- Suporte: faster-whisper, MLX, WhisperX
- Features: transcription, diarization (post-proc), formats

**FASE 2**: Modelos Avançados
- Voxtral 24B/3B (melhor WER)
- Parakeet TDT (velocidade)
- Granite Speech (translation)

**FASE 3**: Features Avançadas
- Translation (Granite/SeamlessM4T)
- Benchmarking automático
- Comparação de modelos

---

## 📚 Referências Técnicas

### **Papers**

- **Whisper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **WhisperX**: [WhisperX: Time-Accurate Speech Transcription](https://arxiv.org/abs/2303.00747)
- **Voxtral**: [Voxtral: Speech Understanding at Scale](https://arxiv.org/abs/2507.13264)
- **Granite Speech**: [Granite-speech: Open-source Speech-Aware LLMs](https://arxiv.org/abs/2505.08699)
- **Parakeet TDT**: [Token-and-Duration Transducer](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/)
- **SeamlessM4T**: [Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)

### **Implementações**

- faster-whisper: https://github.com/guillaumekln/faster-whisper
- MLX Whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- WhisperX: https://github.com/m-bain/whisperX
- NeMo (Parakeet, Sortformer): https://github.com/NVIDIA/NeMo
- SeamlessM4T: https://github.com/facebookresearch/seamless_communication

---

**Última atualização**: 2025-10-18
**Próxima revisão**: Após implementação de Voxtral

🤖 Generated with [Claude Code](https://claude.com/claude-code)
