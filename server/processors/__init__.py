"""
Processors para Whisper Stream

Contém lógica de processamento para diferentes modos:
- batch_processor: Processamento batch (arquivos completos)
- streaming_processor: Processamento streaming (chunks incrementais)
"""

from .batch_processor import BatchProcessor

__all__ = ["BatchProcessor"]
