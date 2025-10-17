"""
Capability system para backends

Define quais funcionalidades cada backend suporta:
- TRANSCRIPTION: Transcrição básica
- WORD_TIMESTAMPS: Timestamps por palavra
- SPEAKER_DIARIZATION: Identificação de speakers
- TRANSLATION: Tradução para outros idiomas
- VAD: Voice Activity Detection
- STREAMING: Processamento em streaming
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set, Optional

from ..utils.platform import Platform


class Capability(Enum):
    """Capabilities que um backend pode suportar"""

    TRANSCRIPTION = "transcription"
    WORD_TIMESTAMPS = "word_timestamps"
    SPEAKER_DIARIZATION = "diarization"
    TRANSLATION = "translation"
    VAD = "vad"
    STREAMING = "streaming"


@dataclass
class BackendInfo:
    """
    Informações sobre um backend de transcrição

    Declara as capabilities e plataformas suportadas por um backend.
    Usado pelo BackendRegistry para validar compatibilidade.

    Attributes:
        name: Nome do backend (ex: "mlx-whisper", "whisperx")
        supported_platforms: Set de Platform enums suportados
        capabilities: Set de Capability enums que o backend oferece
        supported_languages: Idiomas suportados (None = todos)
        model_sizes: Tamanhos de modelo disponíveis (None = padrão do Whisper)

    Example:
        >>> info = BackendInfo(
        ...     name="whisperx",
        ...     supported_platforms={Platform.LINUX_CUDA},
        ...     capabilities={
        ...         Capability.TRANSCRIPTION,
        ...         Capability.WORD_TIMESTAMPS,
        ...         Capability.SPEAKER_DIARIZATION,
        ...     },
        ... )
        >>> Capability.SPEAKER_DIARIZATION in info.capabilities
        True
    """

    name: str
    supported_platforms: Set[Platform]
    capabilities: Set[Capability]
    supported_languages: Optional[Set[str]] = None
    model_sizes: Optional[Set[str]] = None

    def has_capability(self, capability: Capability) -> bool:
        """
        Verifica se backend tem uma capability específica

        Args:
            capability: Capability a verificar

        Returns:
            True se o backend suporta a capability
        """
        return capability in self.capabilities

    def supports_platform(self, platform: Platform) -> bool:
        """
        Verifica se backend suporta uma plataforma

        Args:
            platform: Platform a verificar

        Returns:
            True se o backend roda nesta plataforma
        """
        return platform in self.supported_platforms
