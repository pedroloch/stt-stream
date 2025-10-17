"""
Models para Whisper Stream

Dataclasses e enums usados para representar resultados de transcrição
de forma normalizada, independente do backend usado.
"""

from .capability import BackendInfo, Capability
from .result import Segment, TranscriptionResult, Word

__all__ = [
    "Capability",
    "BackendInfo",
    "Word",
    "Segment",
    "TranscriptionResult",
]
