"""
Whisper Processor - wrapper de alto nível para backends

Gerencia o ciclo de vida do backend e fornece API simples para transcrição
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, AsyncIterator

from .config import Config
from .backends import BackendRegistry  # Novo sistema!
from .backends.base import WhisperBackend
from .utils.platform import detect_platform
from .utils.hardware_detector import HardwareInfo


class WhisperProcessor:
    """
    Processador Whisper de alto nível

    Gerencia backend, configuração e fornece API simples para transcrição.

    Exemplo de uso:
        config = Config.from_file("server-config.yaml")
        processor = WhisperProcessor(config)
        await processor.initialize()

        result = await processor.process_audio(audio_data)
        print(result["text"])

        await processor.cleanup()
    """

    def __init__(self, config: Config):
        """
        Inicializa o processor

        Args:
            config: Configuração do servidor
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backend: Optional[WhisperBackend] = None
        self.hardware_info: Optional[HardwareInfo] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Inicializa o processor

        - Detecta hardware (se necessário)
        - Cria backend apropriado
        - Carrega modelo
        """
        if self._initialized:
            self.logger.warning("Processor já inicializado")
            return

        self.logger.info("Inicializando Whisper Processor...")

        # Detectar plataforma
        platform = detect_platform()
        self.logger.info(f"Plataforma: {platform.value}")

        # Listar backends disponíveis
        available = BackendRegistry.list_available(platform)
        self.logger.info("Backends disponíveis:")
        for name, info in available.items():
            caps = [c.value for c in info.capabilities]
            self.logger.info(f"  - {name}: {', '.join(caps)}")

        # Expandir paths
        self.config.expand_paths()

        # Criar backend usando novo Registry
        backend_name = self.config.whisper.backend

        # Se "auto", escolher o melhor disponível
        if backend_name == "auto":
            if "faster-whisper" in available:
                backend_name = "faster-whisper"  # Preferir faster-whisper (universal + word timestamps)
                self.logger.info("Auto-selecionado: faster-whisper (universal + word timestamps)")
            elif "mlx" in available:
                backend_name = "mlx"
                self.logger.info("Auto-selecionado: mlx (Apple Silicon)")
            else:
                backend_name = list(available.keys())[0]
                self.logger.info(f"Auto-selecionado: {backend_name}")

        self.logger.info(f"Criando backend: {backend_name}")

        # Preparar config do backend
        backend_config = {
            "model": self.config.whisper.model,
            "language": self.config.whisper.language,
            "compute_type": self.config.whisper.compute_type,
            "device": self.config.whisper.device,
            "models_dir": self.config.paths.models_dir,
            "cache_dir": self.config.paths.cache_dir,
            # Configurações adicionais
            "beam_size": self.config.whisper.beam_size,
            "best_of": self.config.whisper.best_of,
            "temperature": self.config.whisper.temperature,
            "condition_on_previous_text": self.config.whisper.condition_on_previous_text,
            "use_vad": self.config.whisper.use_vad,
            "enable_word_timestamps": True,  # Habilitar word timestamps!
        }

        # Criar backend via Registry
        self.backend = BackendRegistry.create(backend_name, backend_config)

        # Inicializar backend (carrega modelo)
        self.logger.info("Carregando modelo Whisper...")
        await self.backend.initialize()

        # Log de informações do backend
        backend_info = self.backend.get_backend_info()
        self.logger.info(f"Backend: {backend_info.get('name', 'Unknown')}")
        self.logger.info(f"Modelo: {backend_info.get('model', 'Unknown')}")
        self.logger.info(f"Device: {backend_info.get('device', 'Unknown')}")

        self._initialized = True
        self.logger.info("✅ Whisper Processor inicializado")

    async def process_audio(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processa um chunk de áudio

        Args:
            audio: Array numpy (16kHz, mono, float32)
            context: Contexto da transcrição anterior

        Returns:
            Resultado da transcrição:
            {
                "text": str,
                "is_final": bool,
                "language": str,
                "confidence": float,
                "segments": list
            }

        Raises:
            RuntimeError: Se processor não foi inicializado
        """
        if not self._initialized or not self.backend:
            raise RuntimeError(
                "Processor não inicializado. Chame initialize() primeiro."
            )

        # Backend retorna TranscriptionResult, converter para dict WebSocket
        result = await self.backend.transcribe_chunk(audio, context)
        return result.to_websocket_dict()

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Processa stream de áudio

        Args:
            audio_stream: Iterator assíncrono de chunks de áudio

        Yields:
            Resultados de transcrição
        """
        if not self._initialized or not self.backend:
            raise RuntimeError(
                "Processor não inicializado. Chame initialize() primeiro."
            )

        async for result in self.backend.transcribe_stream(audio_stream):
            yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        if self.backend:
            await self.backend.cleanup()
            self.backend = None

        self._initialized = False
        self.logger.info("Whisper Processor limpo")

    def get_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o processor

        Returns:
            Dicionário com informações do processor e backend
        """
        info = {
            "initialized": self._initialized,
            "model": self.config.whisper.model,
            "language": self.config.whisper.language,
            "backend_type": self.config.whisper.backend,
        }

        if self.hardware_info:
            info["hardware"] = {
                "type": self.hardware_info.type,
                "device_name": self.hardware_info.device_name,
                "memory": self.hardware_info.memory,
            }

        if self.backend:
            info["backend"] = self.backend.get_backend_info()

        return info

    def is_ready(self) -> bool:
        """Verifica se processor está pronto para processar áudio"""
        return self._initialized and self.backend is not None and self.backend.is_initialized()
