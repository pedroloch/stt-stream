"""
Backends para Whisper Stream

Auto-registra todos backends disponiveis no BackendRegistry.
"""

from .crisper_backend import CrisperWhisperBackend
from .faster_whisper_backend import FasterWhisperBackend
from .mlx_backend import MLXBackend
from .registry import BackendRegistry
from .whisperx_backend import WhisperXBackend

# Registrar backends
BackendRegistry.register("mlx", MLXBackend)
BackendRegistry.register("faster-whisper", FasterWhisperBackend)
BackendRegistry.register("whisperx", WhisperXBackend)
BackendRegistry.register("crisper-whisper", CrisperWhisperBackend)

__all__ = [
    "BackendRegistry",
    "MLXBackend",
    "FasterWhisperBackend",
    "WhisperXBackend",
    "CrisperWhisperBackend",
]
