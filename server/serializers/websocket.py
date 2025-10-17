"""
WebSocket serializers for Whisper Stream

Serializa modelos de dados para formato WebSocket JSON.
Separação de concerns: Model layer não conhece transport layer.
"""

from typing import Dict, Any
from ..models.result import TranscriptionResult, Segment, Word


class WebSocketSerializer:
    """
    Serializer para protocolo WebSocket

    Converte modelos de dados internos para formato JSON
    do protocolo WebSocket.
    """

    @staticmethod
    def serialize_transcription(result: TranscriptionResult) -> Dict[str, Any]:
        """
        Serializa TranscriptionResult para formato WebSocket

        Args:
            result: TranscriptionResult do backend

        Returns:
            Dicionário pronto para ws.send_json()

        Example:
            >>> result = TranscriptionResult(...)
            >>> ws_message = WebSocketSerializer.serialize_transcription(result)
            >>> ws_message["type"]
            'transcription'
        """
        message = {
            "type": "transcription",
            "text": result.text,
            "is_final": result.is_final,
            "confidence": result.confidence,
            "language": result.language,
            "timestamp": result.timestamp.isoformat(),
        }

        # Adicionar segments se presente (word timestamps)
        if result.segments:
            message["segments"] = [
                WebSocketSerializer._serialize_segment(seg)
                for seg in result.segments
            ]

        # Adicionar speaker se presente (diarization)
        if result.speaker:
            message["speaker"] = result.speaker

        # Adicionar translations se presente
        if result.translation:
            message["translations"] = result.translation

        # Metadata opcional
        if result.processing_time_ms is not None:
            message["processing_time_ms"] = result.processing_time_ms

        if result.model_name:
            message["model"] = result.model_name

        return message

    @staticmethod
    def _serialize_segment(segment: Segment) -> Dict[str, Any]:
        """
        Serializa um Segment para dict

        Args:
            segment: Segment object

        Returns:
            Dicionário com dados do segment
        """
        seg_dict = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }

        # Words (word-level timestamps)
        if segment.words:
            seg_dict["words"] = [
                WebSocketSerializer._serialize_word(w)
                for w in segment.words
            ]

        # Speaker ID (segment-level diarization)
        if segment.speaker_id:
            seg_dict["speaker_id"] = segment.speaker_id

        # Confidence per frame (frame-level confidence)
        if segment.confidence_per_frame:
            seg_dict["confidence_per_frame"] = segment.confidence_per_frame

        return seg_dict

    @staticmethod
    def _serialize_word(word: Word) -> Dict[str, Any]:
        """
        Serializa uma Word para dict

        Args:
            word: Word object

        Returns:
            Dicionário com dados da word
        """
        # Usar método to_dict() do Word (é simples e OK ficar lá)
        return word.to_dict()

    @staticmethod
    def serialize_error(message: str, code: str = None) -> Dict[str, Any]:
        """
        Serializa mensagem de erro

        Args:
            message: Mensagem de erro
            code: Código de erro (opcional)

        Returns:
            Dicionário com erro formatado

        Example:
            >>> error = WebSocketSerializer.serialize_error("Invalid audio", "INVALID_AUDIO")
            >>> error["type"]
            'error'
        """
        error = {
            "type": "error",
            "message": message,
        }
        if code:
            error["code"] = code
        return error

    @staticmethod
    def serialize_connected(server_info: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Serializa mensagem de conexão

        Args:
            server_info: Informações do servidor
            session_id: ID da sessão (opcional)

        Returns:
            Dicionário com mensagem de conexão

        Example:
            >>> info = {"model": "base", "language": "pt"}
            >>> msg = WebSocketSerializer.serialize_connected(info, "session-123")
            >>> msg["type"]
            'connected'
        """
        message = {
            "type": "connected",
            "message": "Conectado ao servidor Whisper Stream",
            "server_info": server_info,
        }

        if session_id:
            message["session_id"] = session_id

        return message

    @staticmethod
    def serialize_pong() -> Dict[str, Any]:
        """
        Serializa resposta pong

        Returns:
            Dicionário com pong
        """
        return {"type": "pong"}

    @staticmethod
    def serialize_info(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serializa resposta de info

        Args:
            data: Dados de informação

        Returns:
            Dicionário com info
        """
        return {
            "type": "info",
            "data": data,
        }

    @staticmethod
    def serialize_server_shutdown(message: str = "Servidor está encerrando") -> Dict[str, Any]:
        """
        Serializa mensagem de shutdown

        Args:
            message: Mensagem de shutdown

        Returns:
            Dicionário com shutdown message
        """
        return {
            "type": "server_shutdown",
            "message": message,
        }
