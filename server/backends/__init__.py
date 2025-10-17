"""
Backends para Whisper Stream

Auto-registra todos backends disponiveis no BackendRegistry.
"""

from .registry import BackendRegistry
from .mlx_backend import MLXBackend
from .faster_whisper_backend import FasterWhisperBackend
from .whisperx_backend import WhisperXBackend
from .crisper_backend import CrisperWhisperBackend

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
