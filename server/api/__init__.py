"""
API Handlers para Whisper Stream

Contém handlers para diferentes endpoints:
- batch: API batch para transcrição de arquivos
- streaming: WebSocket para streaming (já existe em websocket_handler.py)
"""

from .batch import handle_batch_transcribe

__all__ = ["handle_batch_transcribe"]
