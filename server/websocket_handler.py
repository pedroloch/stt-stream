"""
WebSocket Handler para o servidor Whisper Stream

Gerencia conexões WebSocket e streaming de áudio/transcrições
"""

import json
import logging

from aiohttp import WSMsgType, web

from .audio import AudioConverter
from .config import Config
from .constants import DEFAULT_SAMPLE_RATE
from .serializers import WebSocketSerializer
from .whisper_processor import WhisperProcessor


class WebSocketHandler:
    """
    Handler para conexões WebSocket

    Recebe áudio via WebSocket, processa com Whisper, retorna transcrições
    """

    def __init__(self, processor: WhisperProcessor, config: Config):
        """
        Inicializa o handler

        Args:
            processor: WhisperProcessor inicializado
            config: Configuração do servidor
        """
        self.processor = processor
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Audio converter
        self.audio_converter = AudioConverter()

        # Tracking de clientes conectados
        self.active_connections: set[web.WebSocketResponse] = set()

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connection

        Args:
            request: aiohttp Request

        Returns:
            WebSocketResponse
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Adicionar à lista de conexões ativas
        self.active_connections.add(ws)
        client_id = id(ws)

        self.logger.info(f"✅ Cliente conectado: {client_id}")
        self.logger.info(f"Clientes ativos: {len(self.active_connections)}")

        # Verificar limite de clientes
        if len(self.active_connections) > self.config.server.max_clients:
            self.logger.warning(f"Limite de clientes excedido: {self.config.server.max_clients}")
            await ws.send_json({
                "type": "error",
                "message": "Limite de clientes conectados excedido"
            })
            await ws.close()
            self.active_connections.remove(ws)
            return ws

        # Enviar mensagem de boas-vindas
        welcome_msg = WebSocketSerializer.serialize_connected(
            server_info=self.processor.get_info(),
            session_id=str(client_id)
        )
        await ws.send_json(welcome_msg)

        try:
            # Loop principal de mensagens
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    # Recebeu áudio binário
                    await self._handle_audio(ws, msg.data, client_id)

                elif msg.type == WSMsgType.TEXT:
                    # Recebeu mensagem de texto (comandos JSON)
                    await self._handle_text(ws, msg.data, client_id)

                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(
                        f"WebSocket error [{client_id}]: {ws.exception()}"
                    )

        except Exception as e:
            self.logger.error(f"Erro no handler [{client_id}]: {e}", exc_info=True)

        finally:
            # Remover da lista de conexões ativas
            self.active_connections.discard(ws)
            self.logger.info(f"❌ Cliente desconectado: {client_id}")
            self.logger.info(f"Clientes ativos: {len(self.active_connections)}")

        return ws

    async def _handle_audio(
        self,
        ws: web.WebSocketResponse,
        audio_data: bytes,
        client_id: int
    ) -> None:
        """
        Processa áudio recebido

        Args:
            ws: WebSocket connection
            audio_data: Áudio em bytes (raw PCM int16)
            client_id: ID do cliente
        """
        try:
            # Converter bytes para float32 usando AudioConverter
            audio_float = self.audio_converter.pcm_int16_to_float32(audio_data)

            # Debug: log de info do áudio
            if self.config.logging.log_audio_stats:
                stats = self.audio_converter.get_audio_stats(audio_float)
                duration = self.audio_converter.get_duration(audio_float, DEFAULT_SAMPLE_RATE)
                self.logger.debug(
                    f"[{client_id}] Áudio recebido: "
                    f"{len(audio_data)} bytes, "
                    f"{duration:.2f}s, "
                    f"range: [{stats['min']:.3f}, {stats['max']:.3f}], "
                    f"rms: {stats['rms']:.3f}"
                )

            # Processar com Whisper (retorna TranscriptionResult)
            result = await self.processor.process_audio(audio_float)

            # Enviar resultado se houver texto
            if result.text:
                # Serializar para WebSocket
                ws_message = WebSocketSerializer.serialize_transcription(result)
                await ws.send_json(ws_message)

                self.logger.debug(
                    f"[{client_id}] Transcrição: '{result.text[:50]}...' "
                    f"(conf: {result.confidence:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Erro ao processar áudio [{client_id}]: {e}")
            error_msg = WebSocketSerializer.serialize_error(
                f"Erro ao processar áudio: {str(e)}",
                code="PROCESSING_ERROR"
            )
            await ws.send_json(error_msg)

    async def _handle_text(
        self,
        ws: web.WebSocketResponse,
        text_data: str,
        client_id: int
    ) -> None:
        """
        Processa mensagem de texto (comandos JSON)

        Args:
            ws: WebSocket connection
            text_data: Texto JSON
            client_id: ID do cliente
        """
        try:
            message = json.loads(text_data)
            msg_type = message.get("type")

            self.logger.debug(f"[{client_id}] Mensagem recebida: {msg_type}")

            if msg_type == "ping":
                # Responder pong
                pong_msg = WebSocketSerializer.serialize_pong()
                await ws.send_json(pong_msg)

            elif msg_type == "get_info":
                # Enviar informações do servidor
                info_msg = WebSocketSerializer.serialize_info(self.processor.get_info())
                await ws.send_json(info_msg)

            elif msg_type == "reset_context":
                # Reset de contexto (futuro)
                await ws.send_json({
                    "type": "ack",
                    "message": "Context reset"
                })

            else:
                self.logger.warning(f"[{client_id}] Tipo de mensagem desconhecido: {msg_type}")
                error_msg = WebSocketSerializer.serialize_error(
                    f"Tipo de mensagem desconhecido: {msg_type}",
                    code="UNKNOWN_MESSAGE_TYPE"
                )
                await ws.send_json(error_msg)

        except json.JSONDecodeError as e:
            self.logger.error(f"[{client_id}] JSON inválido: {e}")
            error_msg = WebSocketSerializer.serialize_error("JSON inválido", code="INVALID_JSON")
            await ws.send_json(error_msg)

        except Exception as e:
            self.logger.error(f"[{client_id}] Erro ao processar mensagem: {e}")
            error_msg = WebSocketSerializer.serialize_error(str(e), code="MESSAGE_ERROR")
            await ws.send_json(error_msg)

    def _get_timestamp(self) -> str:
        """Retorna timestamp atual em formato ISO"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast mensagem para todos os clientes conectados

        Args:
            message: Mensagem para enviar (será convertida para JSON)
        """
        if not self.active_connections:
            return

        # Enviar para todos os clientes
        disconnected = set()
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                self.logger.error(f"Erro ao enviar broadcast: {e}")
                disconnected.add(ws)

        # Remover conexões que falharam
        self.active_connections -= disconnected

    def get_stats(self) -> dict:
        """
        Retorna estatísticas do handler

        Returns:
            Dicionário com estatísticas
        """
        return {
            "active_connections": len(self.active_connections),
            "max_clients": self.config.server.max_clients,
        }
