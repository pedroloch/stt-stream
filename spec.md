# 🎙️ Whisper Stream - Especificação Técnica v2.0

**Data**: 2025-10-17
**Status**: Arquitetura Capabilities-Based
**Target**: Empresas e reuniões com suporte a diarization e tradução

---

## 📋 Visão Geral

Sistema de transcrição em tempo real usando modelos de Speech-to-Text open-source (Whisper, WhisperX, SeamlessM4T), com arquitetura cliente-servidor via WebSocket. Suporta **múltiplos backends** com detecção automática de plataforma e **capabilities** específicas (word timestamps, speaker diarization, translation).

### Filosofia Arquitetural

**"One codebase, multiple backends, unified protocol"**

- Backend Factory Pattern com validação de plataforma
- Cada backend declara suas capabilities
- Protocolo WebSocket unificado com campos opcionais
- Fail-fast com mensagens de erro claras
- Zero dependências pagas (100% self-hosted)

---

## 🎯 Objetivos e Target

### Objetivo Principal

Sistema de transcrição profissional para **empresas e reuniões** com:

1. **Speaker Diarization**: Identificar quem está falando em reuniões
2. **Word-level Timestamps**: Precisão para captions e análise
3. **Translation**: Transcrever + traduzir simultaneamente
4. **Flexibilidade**: Rodar em Mac (dev) ou GPU Linux (prod)

### Use Cases

#### Use Case 1: Reunião Corporativa
- 3-5 pessoas em reunião
- Transcrição com identificação de speakers (SPEAKER_00, SPEAKER_01...)
- Export para ata de reunião
- **Backend**: WhisperX (CUDA)

#### Use Case 2: Call Center Internacional
- Transcrição + tradução em tempo real (PT → EN, ES)
- Alta qualidade, latência < 2s aceitável
- **Backend**: SeamlessM4T (CUDA)

#### Use Case 3: Desenvolvimento Local
- Prototipar em MacBook Pro M1
- Modelo médio, boa qualidade
- **Backend**: MLX ou faster-whisper

---

## 🏗️ Arquitetura Capabilities-Based

### Visão Geral

```
┌─────────────────────────────────────────────────────────┐
│                    CONFIG.YAML                          │
│  backend: "whisperx"  # Escolhe backend                │
│  enable_diarization: true                               │
│  target_languages: ["en", "es"]                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              BACKEND REGISTRY                           │
│                                                         │
│  - Detecta plataforma atual (Mac M1, Linux CUDA, etc)  │
│  - Lista backends disponíveis                          │
│  - Valida compatibilidade (fail-fast se incompatível)  │
│  - Cria backend escolhido                              │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│  MLX BACKEND     │      │  WHISPERX BACKEND     │
│  (Mac M1 only)   │      │  (CUDA only)          │
│                  │      │                       │
│  Capabilities:   │      │  Capabilities:        │
│  - Transcription │      │  - Transcription      │
│  - Word times    │      │  - Word timestamps    │
│  - Streaming     │      │  - Diarization ⭐     │
│                  │      │  - VAD                │
└──────────────────┘      └──────────────────────┘

        ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│           UNIFIED WEBSOCKET PROTOCOL                    │
│                                                         │
│  {                                                      │
│    "type": "transcription",                            │
│    "text": "Concordo totalmente",                     │
│    "is_final": true,                                   │
│    "speaker": "SPEAKER_01",  ← Se backend suporta      │
│    "segments": [{             ← Se backend suporta      │
│      "words": [...]                                    │
│    }],                                                 │
│    "translations": {...}      ← Se backend suporta      │
│  }                                                      │
└─────────────────────────────────────────────────────────┘

        ▼
┌─────────────────────────────────────────────────────────┐
│                  BUN CLIENT                             │
│                                                         │
│  - Renderiza UI baseado em capabilities recebidas      │
│  - Se tem speaker → mostra label de speaker            │
│  - Se tem words → mostra highlight de palavra          │
│  - Se tem translation → mostra traduções               │
└─────────────────────────────────────────────────────────┘
```

### Componentes Principais

#### 1. Platform Detection

Detecta automaticamente a plataforma de execução:

```python
class Platform(Enum):
    MACOS_APPLE_SILICON = "macos_arm64"
    MACOS_INTEL = "macos_x86_64"
    LINUX_CUDA = "linux_cuda"
    LINUX_CPU = "linux_cpu"
    WINDOWS_CUDA = "windows_cuda"

def detect_platform() -> Platform:
    """Auto-detect plataforma atual"""
    # Verifica sistema operacional + arquitetura + GPU
    # Retorna plataforma específica
```

#### 2. Capability System

Cada backend declara o que é capaz de fazer:

```python
class Capability(Enum):
    TRANSCRIPTION = "transcription"
    WORD_TIMESTAMPS = "word_timestamps"
    SPEAKER_DIARIZATION = "diarization"
    TRANSLATION = "translation"
    VAD = "vad"
    STREAMING = "streaming"

@dataclass
class BackendInfo:
    name: str
    supported_platforms: Set[Platform]
    capabilities: Set[Capability]
    supported_languages: Optional[Set[str]] = None
    model_sizes: Set[str] = None
```

#### 3. Backend Registry

Factory pattern com validação automática:

```python
class BackendRegistry:
    _backends = {
        "mlx": MLXBackend,
        "faster-whisper": FasterWhisperBackend,
        "whisperx": WhisperXBackend,
        "seamless-m4t": SeamlessM4TBackend,
    }

    @classmethod
    def create(cls, name: str, config: dict) -> TranscriptionBackend:
        """Cria backend com validação de plataforma"""
        backend = cls._backends[name]()

        # Validar plataforma
        current = detect_platform()
        if current not in backend.info.supported_platforms:
            raise PlatformNotSupportedError(
                f"Backend '{name}' não suporta {current}.\n"
                f"Backends disponíveis: {cls.list_available()}"
            )

        return backend
```

#### 4. Unified Result

Resultado normalizado independente do backend:

```python
@dataclass
class TranscriptionResult:
    text: str
    is_final: bool
    confidence: float
    language: str
    timestamp: datetime

    # Opcionais (nem todos backends preenchem)
    segments: Optional[List[Segment]] = None
    words: Optional[List[Word]] = None
    speaker: Optional[str] = None
    translation: Optional[Dict[str, str]] = None

@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float

@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None
```

---

## 🔧 Backends Implementados

### 1. MLX Backend (Development - Mac M1)

**Platform**: `MACOS_APPLE_SILICON`

**Capabilities**:
- ✅ Transcription
- ✅ Word Timestamps (se mlx-whisper suportar)
- ✅ Streaming
- ❌ Diarization (usar pyannote separadamente)
- ❌ Translation

**Use Case**: Desenvolvimento local no Mac

**Config**:
```yaml
backend:
  name: "mlx"
  model: "medium"  # tiny, base, small, medium, large
  language: "pt"
  enable_word_timestamps: true
```

**Trade-offs**:
- ✅ Rápido no Apple Silicon
- ✅ Baixo consumo de energia
- ❌ Não tem diarization nativa
- ❌ Limitado ao Mac

---

### 2. Faster-Whisper Backend (Universal)

**Platform**: Todos (CUDA, CPU, Mac)

**Capabilities**:
- ✅ Transcription
- ✅ Word Timestamps
- ✅ VAD
- ✅ Streaming
- ❌ Diarization (usar pyannote separadamente)
- ❌ Translation

**Use Case**:
- Produção CPU-only
- Fallback universal
- GPU NVIDIA (com distil-large-v3 → 6x mais rápido)

**Config**:
```yaml
backend:
  name: "faster-whisper"
  model: "distil-large-v3"  # ← RECOMENDADO (6x faster)
  language: "pt"
  device: "auto"  # auto, cuda, cpu
  compute_type: "int8_float16"  # Otimizado para distil

  # Anti-hallucination
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0
  compression_ratio_threshold: 2.4

  # VAD
  enable_vad: true
  vad_threshold: 0.75

  # Word timestamps
  enable_word_timestamps: true
```

**Trade-offs**:
- ✅ Funciona em qualquer hardware
- ✅ Muito rápido com distil-whisper
- ✅ Menos alucinações
- ❌ Sem diarization nativa

**Performance esperada**:
```
Mac M1 (medium):          ~0.25x realtime (4x faster)
Mac M1 (distil-large-v3): ~0.16x realtime (6x faster)
GPU RTX 4090:             ~0.05x realtime (20x faster)
```

---

### 3. WhisperX Backend (Production - CUDA) ⭐⭐⭐

**Platform**: `LINUX_CUDA` only

**Capabilities**:
- ✅ Transcription
- ✅ Word Timestamps (wav2vec2 alignment - mais preciso)
- ✅ **Speaker Diarization** ⭐
- ✅ VAD
- ✅ Streaming

**Use Case**:
- Reuniões com múltiplas pessoas
- Call centers
- Produção high-end

**Config**:
```yaml
backend:
  name: "whisperx"
  model: "large-v3"
  language: "pt"
  device: "cuda"

  # Diarization (requer HuggingFace token)
  enable_diarization: true
  huggingface_token: "${HUGGINGFACE_TOKEN}"

  # VAD
  enable_vad: true
  vad_threshold: 0.7
```

**Pipeline**:
```
Áudio → Whisper (transcrição) → wav2vec2 (word alignment)
                               → pyannote (diarization)
                               → assign speakers to words
```

**Trade-offs**:
- ✅ Melhor word alignment (wav2vec2)
- ✅ Speaker diarization integrada
- ✅ Tudo otimizado junto
- ❌ Requer CUDA GPU
- ❌ ~1.5-2x mais lento que faster-whisper
- ❌ Requer HuggingFace token (gratuito)

**Output Example**:
```json
{
  "text": "Concordo totalmente com essa proposta",
  "speaker": "SPEAKER_01",
  "segments": [{
    "start": 3.5,
    "end": 6.8,
    "words": [
      {"word": "Concordo", "start": 3.5, "end": 4.1, "prob": 0.98},
      {"word": "totalmente", "start": 4.1, "end": 4.9, "prob": 0.96},
      ...
    ]
  }]
}
```

---

### 4. SeamlessM4T Backend (Translation) ⭐⭐⭐

**Platform**: `LINUX_CUDA` only (modelo grande ~10GB)

**Capabilities**:
- ✅ Transcription
- ✅ **Translation** (100+ idiomas) ⭐
- ✅ Streaming
- ❌ Diarization
- ❌ Word Timestamps

**Use Case**:
- Reuniões internacionais
- Transcrever + traduzir simultaneamente
- Produto premium (feature única)

**Config**:
```yaml
backend:
  name: "seamless-m4t"
  language: "pt"  # Idioma de entrada
  target_languages:
    - "en"
    - "es"
    - "fr"
  device: "cuda"
```

**Trade-offs**:
- ✅ Transcrição + tradução em um modelo
- ✅ 100+ idiomas suportados
- ✅ Melhor em code-switching
- ❌ Modelo muito grande (10GB+)
- ❌ ~2-3x mais lento que Whisper
- ❌ Requer GPU potente

**Output Example**:
```json
{
  "text": "Olá, como você está?",
  "language": "pt",
  "translations": {
    "en": "Hello, how are you?",
    "es": "Hola, ¿cómo estás?",
    "fr": "Bonjour, comment allez-vous?"
  }
}
```

---

## 📡 Protocolo WebSocket Unificado

### Mensagens Cliente → Servidor

#### 1. Configuração de Sessão
```json
{
  "type": "configure",
  "settings": {
    "language": "pt",
    "enable_word_timestamps": true,
    "enable_diarization": true,
    "target_languages": ["en", "es"]
  }
}
```

#### 2. Áudio
```
Binary frame: Int16 PCM audio data (16kHz, mono)
```

#### 3. Controle
```json
{
  "type": "pause"  // Pausar transcrição
}

{
  "type": "resume"  // Retomar
}

{
  "type": "reset"  // Limpar contexto
}
```

### Mensagens Servidor → Cliente

#### 1. Transcrição (formato unificado)
```json
{
  "type": "transcription",
  "text": "Concordo totalmente",
  "is_final": true,
  "confidence": 0.95,
  "language": "pt",
  "timestamp": "2025-10-17T18:30:00.000Z",

  // Opcional: Word timestamps (faster-whisper, whisperx)
  "segments": [{
    "start": 3.5,
    "end": 6.8,
    "text": "Concordo totalmente",
    "words": [
      {"word": "Concordo", "start": 3.5, "end": 4.1, "probability": 0.98},
      {"word": "totalmente", "start": 4.1, "end": 4.9, "probability": 0.96}
    ]
  }],

  // Opcional: Speaker diarization (whisperx)
  "speaker": "SPEAKER_01",

  // Opcional: Translation (seamless-m4t)
  "translations": {
    "en": "I completely agree",
    "es": "Estoy completamente de acuerdo"
  },

  // Metadata
  "backend": "whisperx",
  "capabilities_used": ["diarization", "word_timestamps"]
}
```

#### 2. Erro
```json
{
  "type": "error",
  "message": "Backend 'whisperx' requires CUDA GPU",
  "code": "PLATFORM_NOT_SUPPORTED",
  "suggestions": [
    "Use 'mlx' backend on Mac",
    "Use 'faster-whisper' on CPU",
    "Deploy on GPU server (RunPod)"
  ]
}
```

#### 3. Status
```json
{
  "type": "status",
  "backend": "whisperx",
  "model": "large-v3",
  "capabilities": ["transcription", "diarization", "word_timestamps"],
  "platform": "linux_cuda",
  "gpu": "NVIDIA RTX 4090"
}
```

---

## ⚙️ Configuração

### Desenvolvimento (Mac M1)
```yaml
# config.yaml
backend:
  name: "mlx"
  model: "medium"
  language: "pt"
  enable_word_timestamps: true

server:
  host: "0.0.0.0"
  port: 9090

client:
  auto_reconnect: true
```

### Produção (GPU - Diarization)
```yaml
# config-production.yaml
backend:
  name: "whisperx"
  model: "large-v3"
  language: "pt"
  device: "cuda"

  # Diarization
  enable_diarization: true
  huggingface_token: "${HUGGINGFACE_TOKEN}"

  # Anti-hallucination
  no_speech_threshold: 0.6
  log_prob_threshold: -1.0

  # VAD
  enable_vad: true
  vad_threshold: 0.75

server:
  host: "0.0.0.0"
  port: 9090
  max_clients: 10
```

### Produção (GPU - Translation)
```yaml
# config-translation.yaml
backend:
  name: "seamless-m4t"
  language: "pt"
  target_languages:
    - "en"
    - "es"
    - "fr"
  device: "cuda"

server:
  host: "0.0.0.0"
  port: 9090
```

---

## 🚫 Error Handling

### Exemplo 1: Backend incompatível com plataforma

**Cenário**: Usuário tenta usar WhisperX no Mac

**Config**:
```yaml
backend:
  name: "whisperx"
```

**Error**:
```
❌ Backend 'whisperx' (WhisperX) is not supported on macos_arm64.

Available backends for your platform:
  - mlx: mlx-whisper (transcription, word_timestamps, streaming)
  - faster-whisper: faster-whisper (transcription, word_timestamps, vad, streaming)

💡 Suggestion: For speaker diarization on Mac, use 'faster-whisper' with external
   pyannote.audio, or deploy on GPU server (RunPod) with 'whisperx' backend.
```

### Exemplo 2: HuggingFace token faltando

**Cenário**: WhisperX com diarization sem token

**Error**:
```
❌ Speaker diarization requires HuggingFace token.

Get free token at: https://huggingface.co/settings/tokens
Add to config:
  backend:
    huggingface_token: "hf_..."

Or set environment variable:
  export HUGGINGFACE_TOKEN="hf_..."
```

---

## 📂 Estrutura de Diretórios

```
whisper-stream/
├── spec.md                     # Esta especificação
├── PLAN.md                     # Roadmap de implementação
├── README.md
├── LICENSE
│
├── config.yaml                 # Config desenvolvimento (Mac)
├── config-production.yaml      # Config produção (GPU)
├── config-translation.yaml     # Config tradução
├── .env.example                # Environment variables
│
├── package.json                # Bun dependencies
├── pyproject.toml              # Python dependencies (uv)
├── tsconfig.json
│
├── src/                        # Cliente Bun/TypeScript
│   ├── index.ts               # Entry point
│   ├── types.ts               # Type definitions
│   └── ...
│
├── server/                     # Servidor Python
│   ├── main.py                # Entry point
│   ├── config.py
│   ├── websocket_handler.py
│   │
│   ├── backends/              # ⭐ Backend implementations
│   │   ├── base.py           # Abstract base + Platform enum
│   │   ├── factory.py        # BackendRegistry
│   │   ├── mlx_backend.py
│   │   ├── faster_whisper_backend.py
│   │   ├── whisperx_backend.py     # ← NEW
│   │   └── seamless_backend.py     # ← NEW
│   │
│   ├── models/                # Shared models
│   │   ├── __init__.py
│   │   └── result.py         # TranscriptionResult, Word, Segment
│   │
│   └── utils/
│       ├── platform.py        # detect_platform()
│       └── logger.py
│
├── models/                     # Downloaded models (gitignored)
├── logs/
│
├── tests/
│   ├── test_backend_factory.py
│   ├── test_platform_detection.py
│   └── ...
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── RUNPOD.md              # RunPod deployment
│   └── API.md                 # WebSocket API
│
├── scripts/
│   └── runpod-deploy.sh       # Deploy to RunPod
│
├── Dockerfile.mac              # Mac development
├── Dockerfile.gpu              # GPU production (whisperx, seamless)
└── docker-compose.yml
```

---

## 🔐 Segurança e Privacidade

### Princípios

1. **Zero Cloud Dependencies**: Tudo self-hosted
2. **Sem telemetria**: Nenhum dado enviado para terceiros
3. **Token local**: HuggingFace token apenas para download de modelos
4. **Dados temporários**: Áudio processado não é salvo (exceto debug mode)

### HuggingFace Token

**Quando necessário**: Apenas para pyannote.audio (diarization no WhisperX)

**Como obter**:
1. Criar conta gratuita: https://huggingface.co/join
2. Gerar token: https://huggingface.co/settings/tokens
3. Aceitar termos: https://huggingface.co/pyannote/speaker-diarization

**Como usar**:
```bash
# Opção 1: Environment variable
export HUGGINGFACE_TOKEN="hf_..."

# Opção 2: Config file
# config.yaml
backend:
  huggingface_token: "hf_..."
```

**O que o token faz**:
- Faz download do modelo de diarization (primeira vez)
- Modelos ficam em cache local (~/.cache/huggingface)
- Após download, funciona offline

---

## 📊 Performance Benchmarks

### Latência Esperada (modelo large-v3)

| Backend | Hardware | Latência | Qualidade |
|---------|----------|----------|-----------|
| MLX | Mac M1 (medium) | ~0.25x RT | ⭐⭐⭐⭐ |
| faster-whisper | Mac M1 (distil-large-v3) | ~0.16x RT | ⭐⭐⭐⭐⭐ |
| faster-whisper | RTX 4090 (distil-large-v3) | ~0.05x RT | ⭐⭐⭐⭐⭐ |
| WhisperX | RTX 4090 | ~0.10x RT | ⭐⭐⭐⭐⭐ |
| SeamlessM4T | RTX 4090 | ~0.20x RT | ⭐⭐⭐⭐ |

*RT = Realtime (1.0x = mesma velocidade da fala)*

### Uso de Memória

| Backend | Modelo | VRAM (GPU) | RAM |
|---------|--------|------------|-----|
| MLX | medium | - | ~2GB |
| faster-whisper | distil-large-v3 | ~3GB | ~2GB |
| WhisperX | large-v3 | ~6GB | ~4GB |
| SeamlessM4T | large | ~12GB | ~8GB |

---

## 🎯 Critérios de Aceitação

### MVP (Semana 1-2)

✅ **Backend Factory implementado**
- [ ] Platform detection funciona
- [ ] BackendRegistry valida compatibilidade
- [ ] Erro claro se backend incompatível

✅ **Faster-Whisper com distil-large-v3**
- [ ] 6x mais rápido que large-v3
- [ ] Word timestamps funcionando
- [ ] Menos alucinações

✅ **Protocolo WebSocket unificado**
- [ ] TranscriptionResult normalizado
- [ ] Cliente renderiza baseado em capabilities

### Diarization (Semana 3-4)

✅ **WhisperX Backend**
- [ ] Funciona em CUDA
- [ ] Speaker labels corretos
- [ ] HuggingFace token configurável

✅ **UI com speakers**
- [ ] Cliente mostra speaker labels
- [ ] Cores diferentes por speaker

### Translation (Semana 5-6)

✅ **SeamlessM4T Backend**
- [ ] Transcrição + tradução simultânea
- [ ] Múltiplos idiomas alvo
- [ ] Performance aceitável

---

## 🚀 Next Steps

Ver **PLAN.md** para roadmap detalhado de implementação.

**Próxima tarefa**: Implementar Backend Factory (Sprint 1)

---

**Versão**: 2.0.0
**Data de Última Atualização**: 2025-10-17
**Aprovado para implementação**: ✅
