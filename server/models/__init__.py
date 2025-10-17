"""
Models para Whisper Stream

Dataclasses e enums usados para representar resultados de transcrição
de forma normalizada, independente do backend usado.
"""

from .capability import Capability, BackendInfo
from .result import Word, Segment, TranscriptionResult

__all__ = [
    "Capability",
    "BackendInfo",
    "Word",
    "Segment",
    "TranscriptionResult",
]
