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
from .streaming.buffer import BufferConfig, StreamingBuffer
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

        # Streaming buffers por cliente (client_id -> StreamingBuffer)
        self.streaming_buffers: dict[int, StreamingBuffer] = {}

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

        # Criar StreamingBuffer para este cliente
        if not self.processor.backend:
            self.logger.error("Backend não inicializado!")
            await ws.close()
            self.active_connections.remove(ws)
            return ws

        buffer_config = BufferConfig(
            min_chunk_size=self.config.whisper.min_chunk_size,
            buffer_trimming=self.config.whisper.buffer_trimming,
            buffer_trimming_sec=self.config.whisper.buffer_trimming_sec,
            max_buffer_size=self.config.whisper.max_buffer_size,
            agreement_threshold=2,  # Deprecated mas mantido por compatibilidade
        )

        self.logger.info(
            f"BufferConfig para cliente {client_id}: "
            f"min_chunk={buffer_config.min_chunk_size}s, "
            f"buffer_trimming={buffer_config.buffer_trimming}, "
            f"buffer_trimming_sec={buffer_config.buffer_trimming_sec}s, "
            f"max_buffer_size={buffer_config.max_buffer_size}s"
        )

        self.streaming_buffers[client_id] = StreamingBuffer(
            backend=self.processor.backend,
            config=buffer_config
        )

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
            # Cleanup: remover buffer do cliente
            if client_id in self.streaming_buffers:
                self.streaming_buffers[client_id].reset()
                del self.streaming_buffers[client_id]

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
        Processa áudio recebido com streaming inteligente

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

            # Obter buffer do cliente
            buffer = self.streaming_buffers.get(client_id)
            if not buffer:
                if not self.processor.backend:
                    raise RuntimeError("Backend não inicializado")

                self.logger.warning(f"[{client_id}] Buffer não encontrado, criando novo")
                buffer_config = BufferConfig(
                    min_chunk_size=self.config.whisper.min_chunk_size,
                    buffer_trimming=self.config.whisper.buffer_trimming,
                    buffer_trimming_sec=self.config.whisper.buffer_trimming_sec,
                    max_buffer_size=self.config.whisper.max_buffer_size,
                    agreement_threshold=2,  # Deprecated
                )
                buffer = StreamingBuffer(
                    backend=self.processor.backend,
                    config=buffer_config
                )
                self.streaming_buffers[client_id] = buffer

            # Adicionar chunk ao buffer
            await buffer.add_chunk(audio_float)

            # Processar buffer (retorna parcial ou final)
            result = await buffer.process()

            # Enviar resultado se houver texto
            if result and result.text:
                # Serializar para WebSocket
                ws_message = WebSocketSerializer.serialize_transcription(result)
                await ws.send_json(ws_message)

                # Log diferente para parcial vs final
                status = "✅ FINAL" if result.is_final else "⏳ PARCIAL"
                self.logger.debug(
                    f"[{client_id}] {status}: '{result.text[:50]}...' "
                    f"(conf: {result.confidence:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Erro ao processar áudio [{client_id}]: {e}", exc_info=True)
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
                # Reset de contexto do buffer
                if client_id in self.streaming_buffers:
                    self.streaming_buffers[client_id].reset()
                    self.logger.info(f"[{client_id}] Buffer resetado")

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
        # Stats dos buffers
        buffer_stats = {}
        for client_id, buffer in self.streaming_buffers.items():
            buffer_stats[client_id] = buffer.get_stats()

        return {
            "active_connections": len(self.active_connections),
            "max_clients": self.config.server.max_clients,
            "streaming_buffers": len(self.streaming_buffers),
            "buffer_stats": buffer_stats,
        }
