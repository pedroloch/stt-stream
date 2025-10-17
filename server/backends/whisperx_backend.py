"""
WhisperX Backend (CUDA only)

Backend para diarization (speaker identification) usando WhisperX.
Requer Linux com CUDA GPU.

Capabilities:
- TRANSCRIPTION
- WORD_TIMESTAMPS (wav2vec2 alignment - mais preciso!)
- SPEAKER_DIARIZATION (pyannote.audio)
- VAD
- STREAMING
"""

from datetime import datetime
from typing import Optional, AsyncIterator
import numpy as np
import logging

from .base import WhisperBackend, BackendError, BackendNotAvailableError
from ..models.capability import Capability, BackendInfo
from ..models.result import TranscriptionResult, Segment, Word
from ..utils.platform import Platform, detect_platform, PlatformNotSupportedError


class WhisperXBackend(WhisperBackend):
    """WhisperX Backend - Diarization com pyannote.audio"""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="whisperx",
            supported_platforms={Platform.LINUX_CUDA},
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.WORD_TIMESTAMPS,
                Capability.SPEAKER_DIARIZATION,  # ⭐ Speaker ID
                Capability.VAD,
                Capability.STREAMING,
            },
            model_sizes={"tiny", "base", "small", "medium", "large", "large-v3"},
        )

    async def initialize(self) -> None:
        """Inicializa WhisperX (fail-fast se nao CUDA)"""
        # Validar plataforma
        current = detect_platform()
        if current != Platform.LINUX_CUDA:
            raise PlatformNotSupportedError(
                f"WhisperX requires Linux with CUDA GPU.\n"
                f"Current platform: {current.value}\n"
                f"Deploy on GPU server (RunPod, Vast.ai) or use 'faster-whisper' instead."
            )

        # TODO: Implementar inicializacao completa quando em GPU
        self.logger = logging.getLogger(__name__)
        self.logger.warning("WhisperX backend - implementacao completa pendente")
        self._initialized = True

    async def transcribe_chunk(
        self, audio: np.ndarray, context: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcreve com speaker diarization"""
        # Placeholder - implementacao completa quando em GPU
        return TranscriptionResult(
            text="WhisperX backend ativo (placeholder)",
            is_final=True,
            confidence=1.0,
            language=self.language,
            timestamp=datetime.now(),
            speaker="SPEAKER_00",  # ⭐ Speaker diarization
        )

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream com diarization"""
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result.text:
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        self._initialized = False
