"""
Base interface para backends de Whisper

Define a interface comum que todos os backends devem implementar:
- MLX (Apple Silicon)
- CUDA (NVIDIA GPU)
- CPU (fallback)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator
import numpy as np


class WhisperBackend(ABC):
    """
    Interface abstrata para backends de Whisper

    Cada backend (MLX, CUDA, CPU) deve implementar esta interface
    """

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "float16",
        **kwargs
    ):
        """
        Inicializa o backend

        Args:
            model: Modelo do Whisper (tiny, base, small, medium, large)
            language: Código do idioma (pt, en, es, etc)
            compute_type: Tipo de computação (float16, int8, etc)
            **kwargs: Argumentos adicionais específicos do backend
        """
        self.model = model
        self.language = language
        self.compute_type = compute_type
        self.kwargs = kwargs
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Inicializa o backend (carrega modelo, etc)

        Esta operação pode ser demorada (download/load do modelo)
        Deve ser chamada antes de processar áudio
        """
        pass

    @abstractmethod
    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcreve um chunk de áudio

        Args:
            audio: Array numpy com áudio (sample_rate=16000, mono)
            context: Contexto da transcrição anterior (para continuidade)

        Returns:
            Dicionário com resultado:
            {
                "text": str,              # Texto transcrito
                "is_final": bool,         # Se é transcrição final ou parcial
                "language": str,          # Idioma detectado
                "confidence": float,      # Confiança (0-1)
                "segments": list,         # Segmentos detalhados (opcional)
            }
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Transcreve stream de áudio (chunks contínuos)

        Args:
            audio_stream: Iterator assíncrono de chunks de áudio

        Yields:
            Dicionários com resultados parciais e finais
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Limpa recursos (libera memória do modelo, etc)
        """
        pass

    def is_initialized(self) -> bool:
        """Verifica se o backend está inicializado"""
        return self._initialized

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o backend

        Returns:
            Dicionário com informações:
            {
                "name": str,           # Nome do backend
                "version": str,        # Versão
                "device": str,         # Device usado
                "model": str,          # Modelo carregado
                "language": str,       # Idioma configurado
                "memory_usage": str,   # Uso de memória (se disponível)
            }
        """
        pass


# Constantes para separar parâmetros de modelo vs transcrição
# Parâmetros aceitos pelo construtor WhisperModel (ctranslate2)
MODEL_INIT_PARAMS = {
    'inter_threads',
    'intra_threads',
    'max_queued_batches',
    'flash_attention',
    'tensor_parallel',
    'files'
}

# Parâmetros aceitos pelo método transcribe()
TRANSCRIBE_PARAMS = {
    'beam_size',
    'best_of',
    'temperature',
    'condition_on_previous_text',
    'use_vad'
}


class BackendError(Exception):
    """Erro genérico de backend"""
    pass


class BackendNotAvailableError(BackendError):
    """Backend não está disponível neste sistema"""
    pass


class ModelNotFoundError(BackendError):
    """Modelo não encontrado"""
    pass


class TranscriptionError(BackendError):
    """Erro durante transcrição"""
    pass
