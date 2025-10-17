# 📋 Whisper Stream - Plano de Implementação

**Data**: 2025-10-17
**Versão**: 2.0
**Target**: Empresas e Reuniões (Diarization + Translation)

---

## 🎯 Visão Geral

Este plano detalha a implementação da arquitetura capabilities-based discutida em `spec.md`, priorizando:

1. **Abstração e Flexibilidade**: Backend factory pattern
2. **Segmentação**: Word timestamps e speaker diarization
3. **Velocidade**: distil-whisper (6x faster)
4. **Translation**: SeamlessM4T para mercado internacional

**Filosofia**: Prototipar no Mac, deployar em GPU para produção.

---

## 🏆 Sprints e Milestones

### Sprint 1: Foundation (Semana 1-2) ⭐⭐⭐⭐⭐
**Objetivo**: Arquitetura robusta que permite evolução fácil

**Prioridade**: CRÍTICA - Tudo depende disso

**Entregas**:
1. Backend abstraction layer
2. Platform detection
3. Backend factory com validação
4. Resultado normalizado (`TranscriptionResult`)
5. Refatorar MLX e faster-whisper para novo sistema

---

### Sprint 2: Word Timestamps (Semana 3) ⭐⭐⭐⭐⭐
**Objetivo**: Word-level precision para captions profissionais

**Prioridade**: ALTA - Diferencial imediato

**Entregas**:
1. Habilitar `word_timestamps=True` em faster-whisper
2. Atualizar `TranscriptionResult` com words
3. Protocolo WebSocket estendido
4. Cliente UI com word highlights
5. Distil-large-v3 como default (6x faster)

---

### Sprint 3: Speaker Diarization (Semana 4-5) ⭐⭐⭐⭐⭐
**Objetivo**: Identificar speakers em reuniões

**Prioridade**: ALTA - Use case principal (empresas)

**Entregas**:
1. WhisperX backend implementation
2. HuggingFace token configuration
3. Platform validation (CUDA only)
4. Speaker labels no protocolo
5. Cliente UI com cores por speaker

---

### Sprint 4: Translation (Semana 6-7) ⭐⭐⭐⭐
**Objetivo**: Transcrição + tradução simultânea

**Prioridade**: MÉDIA - Feature premium

**Entregas**:
1. SeamlessM4T backend
2. Multi-language config
3. Translations no protocolo
4. Cliente UI para mostrar traduções
5. Benchmarks de performance

---

### Sprint 5: Production Ready (Semana 8) ⭐⭐⭐
**Objetivo**: Deploy em produção confiável

**Prioridade**: ALTA - Viabiliza venda

**Entregas**:
1. Docker images otimizados
2. RunPod deployment guide atualizado
3. Monitoring e logs
4. Performance benchmarks
5. Documentation

---

## 📝 Sprint 1 Detalhado: Foundation

### Tarefa 1.1: Platform Detection (1 dia)

**Arquivo**: `server/utils/platform.py`

**Implementação**:
```python
from enum import Enum
import platform
import subprocess

class Platform(Enum):
    MACOS_APPLE_SILICON = "macos_arm64"
    MACOS_INTEL = "macos_x86_64"
    LINUX_CUDA = "linux_cuda"
    LINUX_CPU = "linux_cpu"
    WINDOWS_CUDA = "windows_cuda"

def detect_platform() -> Platform:
    """Detecta plataforma atual com GPU info"""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        if machine == "arm64":
            return Platform.MACOS_APPLE_SILICON
        return Platform.MACOS_INTEL

    elif system == "Linux":
        if has_nvidia_gpu():
            return Platform.LINUX_CUDA
        return Platform.LINUX_CPU

    elif system == "Windows":
        if has_nvidia_gpu():
            return Platform.WINDOWS_CUDA
        raise PlatformNotSupportedError("Windows CPU not supported")

    raise PlatformNotSupportedError(f"Unknown platform: {system} {machine}")

def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available"""
    try:
        # Try torch first (faster)
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    try:
        # Fallback to nvidia-smi
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except:
        return False
```

**Testes**:
```python
# tests/test_platform_detection.py
def test_detect_platform_mac_m1():
    # Mock platform.system() = "Darwin", platform.machine() = "arm64"
    assert detect_platform() == Platform.MACOS_APPLE_SILICON

def test_detect_platform_linux_cuda():
    # Mock torch.cuda.is_available() = True
    assert detect_platform() == Platform.LINUX_CUDA
```

**Critério de aceitação**:
- ✅ Detecta corretamente Mac M1
- ✅ Detecta CUDA se nvidia-smi existe
- ✅ Fallback para CPU se sem GPU
- ✅ Erro claro em plataforma não suportada

---

### Tarefa 1.2: Backend Base Classes (1 dia)

**Arquivo**: `server/backends/base.py`

**Implementação**:
```python
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Set, Optional, AsyncIterator
import numpy as np

class Capability(Enum):
    TRANSCRIPTION = "transcription"
    WORD_TIMESTAMPS = "word_timestamps"
    SPEAKER_DIARIZATION = "diarization"
    TRANSLATION = "translation"
    VAD = "vad"
    STREAMING = "streaming"

@dataclass
class BackendInfo:
    """Metadata sobre capabilities do backend"""
    name: str
    supported_platforms: Set[Platform]
    capabilities: Set[Capability]
    supported_languages: Optional[Set[str]] = None
    model_sizes: Optional[Set[str]] = None

class TranscriptionBackend(ABC):
    """Interface abstrata para backends de transcrição"""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Retorna metadata do backend"""
        pass

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """
        Inicializa o backend.
        Deve falhar rápido (fail-fast) se plataforma incompatível.
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio: np.ndarray
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream de transcrições"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources (opcional)"""
        pass
```

**Critério de aceitação**:
- ✅ Interface clara e bem documentada
- ✅ Type hints completos
- ✅ Enums para Platform e Capability

---

### Tarefa 1.3: Unified Result Models (1 dia)

**Arquivo**: `server/models/result.py`

**Implementação**:
```python
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime

@dataclass
class Word:
    """Palavra com timestamp"""
    word: str
    start: float  # segundos
    end: float
    probability: float

@dataclass
class Segment:
    """Segmento de transcrição"""
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None

@dataclass
class TranscriptionResult:
    """Resultado normalizado de transcrição"""

    # Campos obrigatórios (todos backends)
    text: str
    is_final: bool
    confidence: float
    language: str
    timestamp: datetime

    # Campos opcionais (dependem de capability)
    segments: Optional[List[Segment]] = None
    speaker: Optional[str] = None  # "SPEAKER_00", "SPEAKER_01", etc
    translation: Optional[Dict[str, str]] = None  # {"en": "...", "es": "..."}

    def to_websocket_dict(self) -> dict:
        """Converte para formato WebSocket"""
        result = {
            "type": "transcription",
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "language": self.language,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.segments:
            result["segments"] = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        }
                        for w in seg.words
                    ] if seg.words else None
                }
                for seg in self.segments
            ]

        if self.speaker:
            result["speaker"] = self.speaker

        if self.translation:
            result["translations"] = self.translation

        return result
```

**Critério de aceitação**:
- ✅ Suporta todos campos opcionais
- ✅ Conversão para WebSocket JSON
- ✅ Type-safe

---

### Tarefa 1.4: Backend Registry (2 dias)

**Arquivo**: `server/backends/factory.py`

**Implementação**:
```python
from typing import Dict, Type
from .base import TranscriptionBackend, BackendInfo
from ..utils.platform import detect_platform, Platform
from ..utils.exceptions import (
    BackendNotFoundError,
    PlatformNotSupportedError,
)

class BackendRegistry:
    """Registry e factory de backends"""

    _backends: Dict[str, Type[TranscriptionBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[TranscriptionBackend]):
        """Registra backend customizado"""
        cls._backends[name] = backend_class

    @classmethod
    def list_all(cls) -> Dict[str, BackendInfo]:
        """Lista todos backends (mesmo incompatíveis)"""
        return {
            name: backend_class().info
            for name, backend_class in cls._backends.items()
        }

    @classmethod
    def list_available(cls, platform: Optional[Platform] = None) -> Dict[str, BackendInfo]:
        """Lista backends disponíveis na plataforma atual"""
        if platform is None:
            platform = detect_platform()

        available = {}
        for name, backend_class in cls._backends.items():
            backend = backend_class()
            if platform in backend.info.supported_platforms:
                available[name] = backend.info

        return available

    @classmethod
    def create(cls, name: str, config: dict) -> TranscriptionBackend:
        """
        Cria backend com validação de plataforma.
        Falha rápido se incompatível.
        """
        # Verificar se backend existe
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise BackendNotFoundError(
                f"Backend '{name}' not found.\n"
                f"Available backends: {', '.join(available)}"
            )

        # Criar instância
        backend_class = cls._backends[name]
        backend = backend_class()

        # Validar plataforma
        current_platform = detect_platform()
        if current_platform not in backend.info.supported_platforms:
            # Gerar mensagem de erro útil
            available_backends = cls.list_available(current_platform)

            suggestions = "\n".join(
                f"  - {name}: {info.name} ({', '.join(c.value for c in info.capabilities)})"
                for name, info in available_backends.items()
            )

            raise PlatformNotSupportedError(
                f"Backend '{name}' ({backend.info.name}) is not supported on {current_platform.value}.\n\n"
                f"Available backends for your platform:\n{suggestions}\n\n"
                f"💡 Suggestion: Use one of the backends above, or deploy on a different platform."
            )

        return backend


# Auto-register backends quando importados
from .mlx_backend import MLXBackend
from .faster_whisper_backend import FasterWhisperBackend

BackendRegistry.register("mlx", MLXBackend)
BackendRegistry.register("faster-whisper", FasterWhisperBackend)

# WhisperX e SeamlessM4T serão registrados nos sprints futuros
# from .whisperx_backend import WhisperXBackend
# from .seamless_backend import SeamlessM4TBackend
# BackendRegistry.register("whisperx", WhisperXBackend)
# BackendRegistry.register("seamless-m4t", SeamlessM4TBackend)
```

**Testes**:
```python
# tests/test_backend_factory.py
def test_list_available_on_mac():
    backends = BackendRegistry.list_available(Platform.MACOS_APPLE_SILICON)
    assert "mlx" in backends
    assert "faster-whisper" in backends
    # whisperx não deve estar
    assert "whisperx" not in backends

def test_create_incompatible_backend_fails():
    with pytest.raises(PlatformNotSupportedError) as exc:
        # Tentar criar WhisperX no Mac
        backend = BackendRegistry.create("whisperx", {})

    assert "not supported on macos_arm64" in str(exc.value)
    assert "Available backends" in str(exc.value)
```

**Critério de aceitação**:
- ✅ Lista backends disponíveis para plataforma
- ✅ Erro claro se backend incompatível
- ✅ Sugestões de alternativas
- ✅ Permite registrar backends customizados

---

### Tarefa 1.5: Refactor MLX Backend (1 dia)

**Arquivo**: `server/backends/mlx_backend.py`

**Mudanças**:
```python
from .base import TranscriptionBackend, BackendInfo, Capability
from ..utils.platform import Platform
from ..models.result import TranscriptionResult, Segment, Word

class MLXBackend(TranscriptionBackend):
    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mlx-whisper",
            supported_platforms={Platform.MACOS_APPLE_SILICON},
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.STREAMING,
                # Capability.WORD_TIMESTAMPS,  # ← Adicionar se MLX suportar
            },
            model_sizes={"tiny", "base", "small", "medium", "large"},
        )

    async def initialize(self, config: dict) -> None:
        # Validar plataforma (fail-fast)
        from ..utils.platform import detect_platform
        current = detect_platform()

        if current != Platform.MACOS_APPLE_SILICON:
            raise PlatformNotSupportedError(
                f"MLX backend requires Apple Silicon Mac.\n"
                f"Current platform: {current}.\n"
                f"Use 'faster-whisper' backend instead."
            )

        # Verificar MLX instalado
        try:
            import mlx_whisper
        except ImportError:
            raise DependencyMissingError(
                "mlx-whisper not installed.\n"
                "Install: pip install mlx-whisper"
            )

        # Carregar modelo
        self.model = mlx_whisper.load_model(config["model"])
        self.logger.info(f"MLX model loaded: {config['model']}")

    async def transcribe_stream(
        self,
        audio: np.ndarray
    ) -> AsyncIterator[TranscriptionResult]:
        result = self.model.transcribe(audio)

        # Converter para TranscriptionResult
        for segment in result["segments"]:
            yield TranscriptionResult(
                text=segment["text"],
                is_final=True,
                confidence=1.0,  # MLX não retorna confidence
                language=result["language"],
                timestamp=datetime.now(),
                segments=[Segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"],
                    # words=None  # MLX não tem word timestamps
                )],
            )
```

**Critério de aceitação**:
- ✅ Implementa nova interface
- ✅ Declara capabilities corretas
- ✅ Retorna `TranscriptionResult` normalizado
- ✅ Fail-fast se não no Mac

---

### Tarefa 1.6: Refactor Faster-Whisper Backend (1 dia)

**Arquivo**: `server/backends/faster_whisper_backend.py`

**Mudanças**:
```python
class FasterWhisperBackend(TranscriptionBackend):
    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="faster-whisper",
            supported_platforms={
                Platform.LINUX_CUDA,
                Platform.LINUX_CPU,
                Platform.MACOS_APPLE_SILICON,
                Platform.MACOS_INTEL,
            },
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.WORD_TIMESTAMPS,  # ← NOVA
                Capability.VAD,
                Capability.STREAMING,
            },
            model_sizes={
                "tiny", "base", "small", "medium", "large-v3",
                "distil-large-v3",  # ← RECOMENDADO
            },
        )

    async def initialize(self, config: dict) -> None:
        from faster_whisper import WhisperModel

        # Auto-detect device
        device = config.get("device", "auto")
        if device == "auto":
            device = self._detect_best_device()

        # Auto-select compute type
        compute_type = config.get("compute_type", "auto")
        if compute_type == "auto":
            compute_type = self._select_compute_type(device)

        self.model = WhisperModel(
            config["model"],
            device=device,
            compute_type=compute_type,
        )

        # Salvar config para transcribe
        self.config = config
        self.logger.info(
            f"faster-whisper initialized: "
            f"model={config['model']}, device={device}, compute_type={compute_type}"
        )

    async def transcribe_stream(
        self,
        audio: np.ndarray
    ) -> AsyncIterator[TranscriptionResult]:
        segments, info = self.model.transcribe(
            audio,
            language=self.config.get("language"),

            # ⭐ WORD TIMESTAMPS
            word_timestamps=self.config.get("enable_word_timestamps", True),

            # Anti-hallucination
            no_speech_threshold=self.config.get("no_speech_threshold", 0.6),
            log_prob_threshold=self.config.get("log_prob_threshold", -1.0),
            compression_ratio_threshold=self.config.get("compression_ratio_threshold", 2.4),

            # VAD
            vad_filter=self.config.get("enable_vad", True),
            vad_parameters={
                "threshold": self.config.get("vad_threshold", 0.75),
            },
        )

        for segment in segments:
            # Construir words se existir
            words = None
            if hasattr(segment, "words") and segment.words:
                words = [
                    Word(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in segment.words
                ]

            yield TranscriptionResult(
                text=segment.text.strip(),
                is_final=True,
                confidence=segment.avg_logprob if hasattr(segment, "avg_logprob") else 1.0,
                language=info.language,
                timestamp=datetime.now(),
                segments=[Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    words=words,  # ⭐ WORD TIMESTAMPS
                )],
            )
```

**Critério de aceitação**:
- ✅ Word timestamps habilitados
- ✅ Anti-hallucination parameters
- ✅ VAD configurável
- ✅ Suporta distil-large-v3

---

### Tarefa 1.7: Update Main Server (1 dia)

**Arquivo**: `server/main.py`

**Mudanças principais**:
```python
from backends.factory import BackendRegistry
from utils.platform import detect_platform

async def main():
    # Listar backends disponíveis
    logger.info("Detecting platform...")
    platform = detect_platform()
    logger.info(f"Platform: {platform.value}")

    available = BackendRegistry.list_available(platform)
    logger.info(f"Available backends:")
    for name, info in available.items():
        caps = ", ".join(c.value for c in info.capabilities)
        logger.info(f"  - {name}: {caps}")

    # Criar backend escolhido
    backend_name = config["backend"]["name"]
    logger.info(f"Creating backend: {backend_name}")

    try:
        backend = BackendRegistry.create(backend_name, config["backend"])
        await backend.initialize(config["backend"])

        logger.info(f"✅ Backend initialized successfully")
        logger.info(f"   Capabilities: {[c.value for c in backend.info.capabilities]}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize backend: {e}")
        sys.exit(1)

    # Iniciar servidor WebSocket
    # ...
```

**Critério de aceitação**:
- ✅ Lista backends no startup
- ✅ Erro claro se backend inválido
- ✅ Mostra capabilities do backend escolhido

---

## 📝 Sprint 2 Detalhado: Word Timestamps

### Tarefa 2.1: Distil-Whisper Default (30 min)

**Arquivo**: `server-config.yaml`

**Mudança**:
```yaml
backend:
  name: "faster-whisper"
  model: "distil-large-v3"  # ← Mudar de "medium" ou "large-v3"
  compute_type: "int8_float16"  # Otimizado para distil
```

**Testar**:
```bash
# Comparar performance
time python -m server.main --model large-v3
time python -m server.main --model distil-large-v3

# Expectativa: distil-large-v3 ~6x mais rápido
```

---

### Tarefa 2.2: Cliente UI com Word Highlights (2 dias)

**Arquivo**: `src/index.ts`

**Adicionar**:
```typescript
interface TranscriptionMessage {
  // ... campos existentes
  segments?: Array<{
    start: number;
    end: number;
    text: string;
    words?: Array<{
      word: string;
      start: number;
      end: number;
      probability: number;
    }>;
  }>;
}

function renderTranscription(msg: TranscriptionMessage) {
  // Se tem words, fazer highlight palavra por palavra
  if (msg.segments?.[0]?.words) {
    for (const word of msg.segments[0].words) {
      // Highlight baseado em tempo
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

**Critério de aceitação**:
- ✅ Mostra palavras individuais se disponível
- ✅ Cor baseada em probability
- ✅ Fallback gracioso se sem words

---

## 📝 Sprint 3 Detalhado: Speaker Diarization

### Tarefa 3.1: WhisperX Backend (3 dias)

**Arquivo**: `server/backends/whisperx_backend.py`

**Implementação completa** (já detalhada no spec.md):
- Platform check (CUDA only)
- HuggingFace token validation
- pyannote.audio integration
- Speaker assignment

**Registrar**:
```python
# server/backends/factory.py
from .whisperx_backend import WhisperXBackend
BackendRegistry.register("whisperx", WhisperXBackend)
```

---

### Tarefa 3.2: Cliente UI com Speakers (1 dia)

**Arquivo**: `src/index.ts`

```typescript
const SPEAKER_COLORS = [
  chalk.blue,
  chalk.magenta,
  chalk.cyan,
  chalk.yellow,
];

function renderTranscription(msg: TranscriptionMessage) {
  if (msg.speaker) {
    // Escolher cor baseado no speaker ID
    const speakerIndex = parseInt(msg.speaker.replace("SPEAKER_", ""));
    const color = SPEAKER_COLORS[speakerIndex % SPEAKER_COLORS.length];

    console.log(color(`[${msg.speaker}] ${msg.text}`));
  } else {
    console.log(chalk.green(msg.text));
  }
}
```

---

## 🚀 Deployment Strategy

### Development (Mac)
```bash
# Local development
bun start

# Usa MLX backend automaticamente
```

### Production (GPU)
```bash
# Build Docker image
./runpod-deploy.sh all

# Deploy no RunPod com:
#  - Backend: whisperx
#  - Diarization: enabled
#  - GPU: RTX 4090
```

---

## ✅ Definition of Done

Cada sprint está completo quando:

1. ✅ Código implementado e testado
2. ✅ Testes passando (unit + integration)
3. ✅ Documentação atualizada
4. ✅ Exemplo de config funcionando
5. ✅ Commit com mensagem clara
6. ✅ Não quebra features existentes

---

## 📊 Tracking Progress

### Status Atual (2025-10-17)

**Completo** ✅:
- [x] Estrutura inicial do projeto
- [x] Cliente Bun básico
- [x] Servidor Python WebSocket
- [x] MLX backend (básico)
- [x] faster-whisper backend (básico)
- [x] RunPod deployment

**Em Andamento** 🚧:
- [ ] Backend Factory Pattern (Sprint 1)

**Pendente** ⏳:
- [ ] Word Timestamps (Sprint 2)
- [ ] WhisperX Diarization (Sprint 3)
- [ ] SeamlessM4T Translation (Sprint 4)

---

## 🎯 Success Metrics

### MVP (Sprint 1-2)
- ✅ Backend factory funciona
- ✅ distil-large-v3 6x mais rápido
- ✅ Word timestamps aparecem no cliente

### Production (Sprint 1-3)
- ✅ Diarization funciona em reunião de 3+ pessoas
- ✅ Acurácia de speaker > 85%
- ✅ Deploy no RunPod sem erros

### Full Product (Sprint 1-4)
- ✅ Translation simultânea funciona
- ✅ Suporta 3+ idiomas alvo
- ✅ Latência aceitável (< 2s)

---

## 📅 Timeline

| Semana | Sprint | Entregas |
|--------|--------|----------|
| 1-2 | Sprint 1 | Backend Factory + Platform Detection |
| 3 | Sprint 2 | Word Timestamps + distil-whisper |
| 4-5 | Sprint 3 | WhisperX + Speaker Diarization |
| 6-7 | Sprint 4 | SeamlessM4T + Translation |
| 8 | Sprint 5 | Production Ready + Docs |

**Total**: ~8 semanas para produto completo

**MVP**: ~3 semanas (Sprint 1-2)

---

## 🚧 Riscos e Mitigações

### Risco 1: MLX não suporta word timestamps
**Mitigação**: Usar faster-whisper em CPU no Mac (ainda rápido)

### Risco 2: WhisperX muito lento em GPU média
**Mitigação**: Oferecer faster-whisper com pyannote separado como alternativa

### Risco 3: SeamlessM4T requer GPU muito potente
**Mitigação**: Documentar requisitos mínimos (RTX 4090 ou A100)

---

## 📚 Recursos

- **spec.md**: Arquitetura detalhada
- **docs/ARCHITECTURE.md**: Decisões técnicas
- **docs/RUNPOD.md**: Deploy em GPU
- **docs/API.md**: Protocolo WebSocket

---

**Próxima ação**: Começar Sprint 1, Tarefa 1.1 (Platform Detection)

**Última atualização**: 2025-10-17
