"""
Streaming module - LocalAgreement policy and intelligent buffering

This module implements real-time streaming transcription with:
- LocalAgreement-n policy (confirms stable text)
- Intelligent buffer trimming (segment/sentence based)
- Context window optimization (last N words)
- VAD-based chunking (optional)
"""

from .buffer import BufferConfig, LocalAgreementPolicy, StreamingBuffer
from .vad import VADChunker

__all__ = [
    "BufferConfig",
    "LocalAgreementPolicy",
    "StreamingBuffer",
    "VADChunker",
]
