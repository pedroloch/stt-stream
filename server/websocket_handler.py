"""
WebSocket Handler para o servidor Whisper Stream

Gerencia conexões WebSocket e streaming de áudio/transcrições
"""

import logging
import json
import asyncio
from typing import Optional, Set
from aiohttp import web, WSMsgType

from .whisper_processor import WhisperProcessor
from .config import Config
from .audio import AudioConverter
from .constants import DEFAULT_SAMPLE_RATE


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
        self.active_connections: Set[web.WebSocketResponse] = set()

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
        await ws.send_json({
            "type": "connected",
            "message": "Conectado ao servidor Whisper Stream",
            "server_info": self.processor.get_info()
        })

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

            # Processar com Whisper
            result = await self.processor.process_audio(audio_float)

            # Enviar resultado se houver texto
            if result["text"]:
                # Enviar resultado completo (já vem do to_websocket_dict() com todos os campos)
                await ws.send_json(result)

                self.logger.debug(
                    f"[{client_id}] Transcrição: '{result['text'][:50]}...' "
                    f"(conf: {result['confidence']:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Erro ao processar áudio [{client_id}]: {e}")
            await ws.send_json({
                "type": "error",
                "message": f"Erro ao processar áudio: {str(e)}"
            })

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
                await ws.send_json({"type": "pong"})

            elif msg_type == "get_info":
                # Enviar informações do servidor
                await ws.send_json({
                    "type": "info",
                    "data": self.processor.get_info()
                })

            elif msg_type == "reset_context":
                # Reset de contexto (futuro)
                await ws.send_json({
                    "type": "ack",
                    "message": "Context reset"
                })

            else:
                self.logger.warning(f"[{client_id}] Tipo de mensagem desconhecido: {msg_type}")
                await ws.send_json({
                    "type": "error",
                    "message": f"Tipo de mensagem desconhecido: {msg_type}"
                })

        except json.JSONDecodeError as e:
            self.logger.error(f"[{client_id}] JSON inválido: {e}")
            await ws.send_json({
                "type": "error",
                "message": "JSON inválido"
            })

        except Exception as e:
            self.logger.error(f"[{client_id}] Erro ao processar mensagem: {e}")
            await ws.send_json({
                "type": "error",
                "message": str(e)
            })

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
