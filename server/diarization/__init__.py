"""
Módulo de Diarization (Speaker Identification) para Whisper Stream

Suporta múltiplos backends:
- Sortformer (NeMo): SOTA 2025, streaming, 4 speakers simultâneos
- Pyannote: Simples e eficiente, ótimo para casos gerais

Uso:
    from server.diarization import DiarizationProcessor

    processor = DiarizationProcessor(backend="sortformer")
    words_with_speakers = await processor.assign_speakers(audio, words)
"""

from .base import DiarizationBackend
from .processor import DiarizationProcessor

__all__ = [
    "DiarizationBackend",
    "DiarizationProcessor",
]
