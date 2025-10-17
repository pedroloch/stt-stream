"""
SeamlessM4T Backend (CUDA only)

Backend para traducao simultanea usando SeamlessM4T.
Requer Linux com CUDA GPU potente.

Capabilities:
- TRANSCRIPTION
- TRANSLATION (100+ idiomas!)
- STREAMING
"""

from datetime import datetime
from typing import Optional, AsyncIterator
import numpy as np
import logging

from .base import WhisperBackend, BackendError, BackendNotAvailableError
from ..models.capability import Capability, BackendInfo
from ..models.result import TranscriptionResult
from ..utils.platform import Platform, detect_platform, PlatformNotSupportedError


class SeamlessM4TBackend(WhisperBackend):
    """SeamlessM4T Backend - Traducao simultanea"""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="seamless-m4t",
            supported_platforms={Platform.LINUX_CUDA},
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.TRANSLATION,  # ⭐ Multi-language translation
                Capability.STREAMING,
            },
            supported_languages={"pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"},
        )

    async def initialize(self) -> None:
        """Inicializa SeamlessM4T (fail-fast se nao CUDA)"""
        # Validar plataforma
        current = detect_platform()
        if current != Platform.LINUX_CUDA:
            raise PlatformNotSupportedError(
                f"SeamlessM4T requires Linux with CUDA GPU (RTX 4090 or A100).\n"
                f"Current platform: {current.value}\n"
                f"This model is ~10GB and very slow without GPU."
            )

        # TODO: Implementacao completa quando em GPU
        self.logger = logging.getLogger(__name__)
        self.logger.warning("SeamlessM4T backend - implementacao completa pendente")
        self._initialized = True

    async def transcribe_chunk(
        self, audio: np.ndarray, context: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcreve com traducao"""
        # Placeholder
        return TranscriptionResult(
            text="Ola mundo",
            is_final=True,
            confidence=1.0,
            language="pt",
            timestamp=datetime.now(),
            translation={"en": "Hello world", "es": "Hola mundo"},  # ⭐ Translation
        )

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream com traducao"""
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result.text:
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        self._initialized = False
