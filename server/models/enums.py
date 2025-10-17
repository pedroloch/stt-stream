"""
Enums para configuração do Whisper Stream

Define enums para valores de configuração, proporcionando type safety,
autocomplete em IDEs e validação automática.
"""

from enum import Enum


class BackendType(str, Enum):
    """
    Tipos de backend disponíveis

    Herda de str para ser JSON-serializable e permitir comparação
    direta com strings.
    """
    AUTO = "auto"
    """Auto-detecta melhor backend para plataforma atual"""

    MLX = "mlx"
    """MLX backend (Apple Silicon optimized)"""

    FASTER_WHISPER = "faster-whisper"
    """Faster-Whisper backend (universal, word timestamps)"""

    WHISPERX = "whisperx"
    """WhisperX backend (CUDA only, speaker diarization)"""

    CRISPER = "crisper-whisper"
    """CrisperWhisper backend (verbatim transcription)"""

    # Aliases para compatibilidade
    CUDA = "cuda"
    """Alias para faster-whisper com CUDA"""

    CPU = "cpu"
    """Alias para faster-whisper com CPU"""


class LogLevel(str, Enum):
    """
    Níveis de log disponíveis

    Mapeiam para logging.DEBUG, logging.INFO, etc.
    """
    DEBUG = "debug"
    """Log detalhado para debugging"""

    INFO = "info"
    """Informações gerais de operação"""

    WARNING = "warning"
    """Avisos que não impedem operação"""

    ERROR = "error"
    """Erros que requerem atenção"""

    CRITICAL = "critical"
    """Erros críticos que impedem operação"""


class LogFormat(str, Enum):
    """
    Formatos de log disponíveis
    """
    PRETTY = "pretty"
    """Formato colorido e legível para desenvolvimento"""

    JSON = "json"
    """Formato JSON estruturado para produção"""

    SIMPLE = "simple"
    """Formato simples de uma linha"""


class ComputeType(str, Enum):
    """
    Tipos de computação para Whisper

    Define a precisão numérica usada durante inferência.
    Impacto em velocidade e qualidade.
    """
    AUTO = "auto"
    """Auto-seleciona baseado em device"""

    FLOAT16 = "float16"
    """16-bit floating point (rápido em GPU, boa qualidade)"""

    FLOAT32 = "float32"
    """32-bit floating point (mais lento, máxima qualidade)"""

    INT8 = "int8"
    """8-bit integer (muito rápido em CPU, qualidade ok)"""

    INT8_FLOAT16 = "int8_float16"
    """Híbrido otimizado para distil-whisper"""

    INT8_BFLOAT16 = "int8_bfloat16"
    """Híbrido para GPUs modernas"""


class DeviceType(str, Enum):
    """
    Tipos de device para processamento
    """
    AUTO = "auto"
    """Auto-detecta melhor device disponível"""

    CPU = "cpu"
    """Forçar uso de CPU"""

    CUDA = "cuda"
    """Usar GPU NVIDIA via CUDA"""

    MPS = "mps"
    """Usar Apple Metal Performance Shaders"""


class BufferTrimming(str, Enum):
    """
    Estratégias de trimming de buffer de áudio
    """
    SEGMENT = "segment"
    """Trim por segmentos detectados"""

    VAD = "vad"
    """Trim usando Voice Activity Detection"""

    NONE = "none"
    """Sem trimming"""


class CORSMode(str, Enum):
    """
    Modos de CORS para servidor
    """
    ENABLED = "enabled"
    """CORS habilitado para todos origins"""

    DISABLED = "disabled"
    """CORS desabilitado"""

    RESTRICTED = "restricted"
    """CORS apenas para origins específicos"""
