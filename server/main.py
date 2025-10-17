"""
Main entry point para o servidor Whisper Stream

Servidor WebSocket independente que pode ser deployado separadamente
com GPU para processamento Whisper.

Uso:
    python -m server.main --config server-config.yaml
    ou
    python -m server.main --host 0.0.0.0 --port 9090 --model base
"""

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from aiohttp import web

from .config import Config
from .serializers import WebSocketSerializer
from .utils.logger import setup_logger
from .websocket_handler import WebSocketHandler
from .whisper_processor import WhisperProcessor



class WhisperServer:
    """
    Servidor Whisper Stream

    Servidor WebSocket que processa áudio e retorna transcrições
    """

    def __init__(self, config: Config):
        """
        Inicializa o servidor

        Args:
            config: Configuração do servidor
        """
        self.config = config
        self.logger = setup_logger(
            name="whisper_server",
            level=config.logging.level.upper(),
            save_to_file=config.logging.save_to_file,
            log_dir=Path(config.logging.log_dir) if config.logging.save_to_file else None,
            format_type=config.logging.format
        )

        self.processor: WhisperProcessor | None = None
        self.handler: WebSocketHandler | None = None
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def initialize(self) -> None:
        """Inicializa o servidor"""
        self.logger.info("="*60)
        self.logger.info("🎤 WHISPER STREAM SERVER")
        self.logger.info("="*60)

        # Inicializar processor
        self.processor = WhisperProcessor(self.config)
        await self.processor.initialize()

        # Criar handler
        self.handler = WebSocketHandler(self.processor, self.config)

        # Criar app aiohttp
        self.app = web.Application()

        # Adicionar CORS se habilitado
        if self.config.server.cors_enabled:
            self._setup_cors()

        # Rotas
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/info', self.get_info)
        self.app.router.add_get('/stats', self.get_stats)
        self.app.router.add_get('/ws', self.handler.handle_websocket)

        self.logger.info("Servidor inicializado")

    def _setup_cors(self) -> None:
        """Configura CORS"""
        import aiohttp_cors

        assert self.app is not None
        cors = aiohttp_cors.setup(self.app, defaults={
            origin: aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
            for origin in self.config.server.allowed_origins
        })

        # Adicionar CORS a todas as rotas
        for route in list(self.app.router.routes()):
            cors.add(route)

    async def health_check(self, request: web.Request) -> web.Response:
        """
        Health check endpoint

        Returns:
            200 OK se servidor está saudável
        """
        if self.processor and self.processor.is_ready():
            return web.json_response({
                "status": "healthy",
                "processor_ready": True
            })
        return web.json_response({
            "status": "initializing",
            "processor_ready": False
        }, status=503)

    async def get_info(self, request: web.Request) -> web.Response:
        """
        Retorna informações sobre o servidor

        Returns:
            JSON com informações do servidor
        """
        info = {
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
                "max_clients": self.config.server.max_clients,
            },
            "processor": self.processor.get_info() if self.processor else None,
        }
        return web.json_response(info)

    async def get_stats(self, request: web.Request) -> web.Response:
        """
        Retorna estatísticas do servidor

        Returns:
            JSON com estatísticas
        """
        stats = {
            "connections": self.handler.get_stats() if self.handler else {},
        }
        return web.json_response(stats)

    async def start(self) -> None:
        """Inicia o servidor"""
        assert self.app is not None
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(
            self.runner,
            self.config.server.host,
            self.config.server.port
        )
        await self.site.start()

        self.logger.info("="*60)
        self.logger.info(f"✅ Servidor rodando em ws://{self.config.server.host}:{self.config.server.port}")
        self.logger.info(f"Health check: http://{self.config.server.host}:{self.config.server.port}/health")
        self.logger.info(f"WebSocket: ws://{self.config.server.host}:{self.config.server.port}/ws")
        self.logger.info("="*60)

        # Mostrar info do processor
        if self.processor:
            info = self.processor.get_info()
            self.logger.info(f"Modelo: {info.get('model', 'unknown')}")
            self.logger.info(f"Idioma: {info.get('language', 'unknown')}")
            self.logger.info(f"Backend: {info.get('backend_type', 'unknown')}")
            if 'hardware' in info:
                hw = info['hardware']
                self.logger.info(f"Hardware: {hw.get('type', 'unknown')} - {hw.get('device_name', 'unknown')}")

        self.logger.info("="*60)
        self.logger.info("Servidor pronto para receber conexões!")
        self.logger.info("Pressione Ctrl+C para encerrar")
        self.logger.info("="*60)

    async def stop(self) -> None:
        """Para o servidor"""
        self.logger.info("\nEncerrando servidor...")

        # Fechar conexões WebSocket
        if self.handler:
            shutdown_msg = WebSocketSerializer.serialize_server_shutdown()
            await self.handler.broadcast(shutdown_msg)

        # Parar site e runner
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        # Limpar processor
        if self.processor:
            await self.processor.cleanup()

        self.logger.info("✅ Servidor encerrado")

    async def run(self) -> None:
        """Run server loop"""
        with contextlib.suppress(asyncio.CancelledError):
            # Manter servidor rodando
            await asyncio.Future()


async def main_async(config: Config) -> None:
    """
    Main async function

    Args:
        config: Server configuration
    """
    server = WhisperServer(config)

    # Setup signal handlers para graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(server.stop())
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Inicializar e iniciar servidor
        await server.initialize()
        await server.start()

        # Manter rodando
        await server.run()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        server.logger.error(f"Erro fatal: {e}", exc_info=True)
    finally:
        await server.stop()


def cli() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Whisper Stream Server - Servidor de transcrição em tempo real"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Caminho para arquivo de configuração YAML"
    )

    parser.add_argument("--host", type=str, help="Host do servidor (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="Porta do servidor (padrão: 9090)")
    parser.add_argument("--model", type=str, help="Modelo Whisper (padrão: base)")
    parser.add_argument("--language", type=str, help="Idioma (padrão: pt)")
    parser.add_argument("--backend", type=str, help="Backend (auto, mlx, cuda, cpu)")
    parser.add_argument("--log-level", type=str, help="Nível de log (debug, info, warning, error)")
    parser.add_argument("--verbose", action="store_true", help="Modo verbose")

    args = parser.parse_args()

    # Carregar configuração
    if args.config:
        try:
            config = Config.from_file(args.config)
            print(f"✅ Configuração carregada de: {args.config}")
        except Exception as e:
            print(f"❌ Erro ao carregar config: {e}")
            sys.exit(1)
    else:
        # Usar argumentos de linha de comando
        config_kwargs = {}
        if args.host:
            config_kwargs["host"] = args.host
        if args.port:
            config_kwargs["port"] = args.port
        if args.model:
            config_kwargs["model"] = args.model
        if args.language:
            config_kwargs["language"] = args.language
        if args.backend:
            config_kwargs["backend"] = args.backend
        if args.log_level:
            config_kwargs["log_level"] = args.log_level
        if args.verbose:
            config_kwargs["verbose"] = True

        config = Config.from_args(**config_kwargs)

    # Rodar servidor
    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        print("\nServidor encerrado")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
