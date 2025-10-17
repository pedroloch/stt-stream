#!/usr/bin/env python
"""
Servidor WebSocket simples com distil-whisper

Para testar com o cliente Bun
"""

import asyncio
import json
import logging
from datetime import datetime
from aiohttp import web
import numpy as np

from server.backends import BackendRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend global
backend = None

async def init_backend():
    """Inicializa o backend faster-whisper"""
    global backend

    logger.info("=" * 60)
    logger.info("🎙️  Iniciando servidor com Faster-Whisper")
    logger.info("=" * 60)

    # Criar backend
    logger.info("Criando backend faster-whisper...")
    backend = BackendRegistry.create("faster-whisper", {
        "model": "base",  # Mude para "distil-large-v3" depois
        "language": "pt",
        "enable_word_timestamps": True,
        "enable_vad": True,
        "vad_threshold": 0.75,
        "no_speech_threshold": 0.6,
    })

    logger.info("Inicializando backend (pode demorar na primeira vez)...")
    await backend.initialize()

    logger.info("✅ Backend pronto!")
    logger.info("   Modelo: base")
    logger.info("   Capabilities: word_timestamps, vad, streaming")
    logger.info("=" * 60)

async def websocket_handler(request):
    """Handler do WebSocket"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info("🔌 Cliente conectado")

    # Enviar mensagem de boas-vindas
    await ws.send_json({
        "type": "status",
        "message": "Conectado ao servidor Whisper",
        "backend": "faster-whisper",
        "model": "base",
        "capabilities": ["transcription", "word_timestamps", "vad", "streaming"]
    })

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                # Recebeu audio binary
                audio_data = msg.data

                # Converter bytes para numpy array (int16 PCM)
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

                # Converter para float32 normalizado
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                logger.info(f"📥 Recebeu {len(audio_float32)} samples")

                # Transcrever
                try:
                    result = await backend.transcribe_chunk(audio_float32)

                    # Converter para WebSocket dict
                    ws_dict = result.to_websocket_dict()

                    # Enviar ao cliente
                    await ws.send_json(ws_dict)

                    if result.text:
                        logger.info(f"📝 Transcrito: '{result.text}'")

                        # Log word timestamps se disponivel
                        if result.segments:
                            for seg in result.segments:
                                if seg.words:
                                    logger.info(f"   Words: {len(seg.words)}")

                except Exception as e:
                    logger.error(f"Erro na transcricao: {e}")
                    await ws.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif msg.type == web.WSMsgType.TEXT:
                # Mensagem de controle (pause, resume, etc)
                data = json.loads(msg.data)
                logger.info(f"📨 Mensagem: {data.get('type')}")

            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")

    except Exception as e:
        logger.error(f"Erro no handler: {e}")
    finally:
        logger.info("🔌 Cliente desconectado")

    return ws

async def health_check(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "backend": "faster-whisper",
        "model": "base",
        "initialized": backend is not None
    })

async def start_background_services(app):
    """Inicializa backend no startup"""
    await init_backend()

async def cleanup_background_services(app):
    """Cleanup no shutdown"""
    if backend:
        await backend.cleanup()
        logger.info("✅ Backend limpo")

async def main():
    """Main function"""
    app = web.Application()

    # Startup/cleanup
    app.on_startup.append(start_background_services)
    app.on_cleanup.append(cleanup_background_services)

    # Rotas
    app.router.add_get('/health', health_check)
    app.router.add_get('/ws', websocket_handler)

    # Iniciar servidor
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, '0.0.0.0', 9090)
    await site.start()

    logger.info("=" * 60)
    logger.info("✅ Servidor rodando em ws://localhost:9090/ws")
    logger.info("   Health check: http://localhost:9090/health")
    logger.info("=" * 60)
    logger.info("Pressione Ctrl+C para parar")
    logger.info("=" * 60)

    # Manter rodando
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("\n🛑 Parando servidor...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
