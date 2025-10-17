"""
Serializers for Whisper Stream

Handles serialization of data models to transport formats (WebSocket, HTTP, etc).
"""

from .websocket import WebSocketSerializer

__all__ = ["WebSocketSerializer"]
