"""
Diarization Processor - Factory e Gerenciador de Backends

Simplifica o uso de diarization selecionando e gerenciando o backend apropriado.
"""

import logging
from typing import Optional

import numpy as np

from ..models.result import Word

logger = logging.getLogger(__name__)


class DiarizationProcessor:
    """
    Processor de diarization com suporte a múltiplos backends

    Gerencia seleção, inicialização e uso de backends de diarization.

    Backends suportados:
    - "sortformer": NeMo Sortformer (SOTA 2025, 4 speakers simultâneos)
    - "pyannote": Pyannote Audio (simples e eficiente)
    - "none": Diarization desabilitado

    Args:
        backend: Nome do backend ("sortformer", "pyannote", "none")
        sample_rate: Sample rate do áudio (default: 16000)
        num_speakers: Número de speakers (None = auto-detect)
        device: Device para inferência ("auto", "cpu", "cuda", "mps")
        **backend_kwargs: Argumentos específicos do backend

    Example:
        # Sortformer com 4 speakers
        processor = DiarizationProcessor(
            backend="sortformer",
            num_speakers=4,
            device="cuda"
        )

        # Pyannote com auto-detect
        processor = DiarizationProcessor(
            backend="pyannote",
            min_speakers=2,
            max_speakers=8
        )

        # Processar áudio
        segments = await processor.diarize(audio_chunk)
        words_with_speakers = await processor.assign_speakers_to_words(words)
    """

    def __init__(
        self,
        backend: str = "none",
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        device: str = "auto",
        **backend_kwargs,
    ):
        self.backend_name = backend.lower()
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.device = device
        self.backend_kwargs = backend_kwargs

        # Backend instance
        self.backend: Optional[object] = None  # DiarizationBackend

        # Inicializar backend
        if self.backend_name != "none":
            self._init_backend()

        logger.info(
            f"DiarizationProcessor inicializado "
            f"(backend={self.backend_name}, speakers={num_speakers or 'auto'})"
        )

    def _init_backend(self):
        """Inicializa o backend selecionado"""
        if self.backend_name == "sortformer":
            self._init_sortformer()
        elif self.backend_name == "pyannote":
            self._init_pyannote()
        else:
            raise ValueError(
                f"Backend '{self.backend_name}' desconhecido. "
                "Use 'sortformer', 'pyannote' ou 'none'"
            )

    def _init_sortformer(self):
        """Inicializa backend Sortformer"""
        try:
            from .sortformer_backend import SortformerBackend

            # Sortformer só aceita: sample_rate, num_speakers, device, chunk_duration
            sortformer_kwargs = {
                'sample_rate': self.sample_rate,
                'num_speakers': self.num_speakers,
                'device': self.device,
            }

            # Adicionar chunk_duration se presente em backend_kwargs
            if 'chunk_duration' in self.backend_kwargs:
                sortformer_kwargs['chunk_duration'] = self.backend_kwargs['chunk_duration']

            # model_name também é aceito
            if 'model_name' in self.backend_kwargs:
                sortformer_kwargs['model_name'] = self.backend_kwargs['model_name']

            self.backend = SortformerBackend(**sortformer_kwargs)
            logger.info("Backend Sortformer inicializado")
        except ImportError as e:
            logger.error(
                f"Erro ao carregar Sortformer: {e}. "
                "Instale com: pip install nemo_toolkit[asr]"
            )
            raise

    def _init_pyannote(self):
        """Inicializa backend Pyannote"""
        try:
            from .pyannote_backend import PyannoteBackend

            # Pyannote aceita: sample_rate, num_speakers, min_speakers, max_speakers, auth_token, device, model_name
            pyannote_kwargs = {
                'sample_rate': self.sample_rate,
                'num_speakers': self.num_speakers,
                'device': self.device,
            }

            # Adicionar parâmetros específicos do Pyannote se presentes
            pyannote_specific_params = ['min_speakers', 'max_speakers', 'auth_token', 'model_name']
            for param in pyannote_specific_params:
                if param in self.backend_kwargs:
                    pyannote_kwargs[param] = self.backend_kwargs[param]

            self.backend = PyannoteBackend(**pyannote_kwargs)
            logger.info("Backend Pyannote inicializado")
        except ImportError as e:
            logger.error(
                f"Erro ao carregar Pyannote: {e}. "
                "Instale com: pip install pyannote.audio"
            )
            raise

    @property
    def enabled(self) -> bool:
        """Verifica se diarization está habilitado"""
        return self.backend is not None

    async def diarize(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
    ) -> list[tuple[float, float, str]]:
        """
        Processa áudio e retorna segmentos com speaker IDs

        Args:
            audio: Áudio em formato float32 mono
            offset: Offset temporal (para streaming)

        Returns:
            Lista de segmentos [(start, end, speaker_id)]
            Retorna lista vazia se diarization desabilitado
        """
        if not self.enabled:
            return []

        return await self.backend.diarize(audio, offset)  # type: ignore

    async def assign_speakers_to_words(
        self,
        words: list[Word],
        diarization_segments: list[tuple[float, float, str]] | None = None,
    ) -> list[Word]:
        """
        Atribui speaker IDs às palavras

        Args:
            words: Lista de Word objects
            diarization_segments: Segmentos de diarization (opcional)

        Returns:
            Lista de Word objects com speaker_id atribuído
            Retorna words inalterado se diarization desabilitado
        """
        if not self.enabled:
            return words

        return await self.backend.assign_speakers_to_words(  # type: ignore
            words, diarization_segments
        )

    def reset(self):
        """Reseta estado do backend (importante para streaming)"""
        if self.enabled:
            self.backend.reset()  # type: ignore
            logger.debug(f"Backend {self.backend_name} resetado")

    def get_info(self) -> dict:
        """
        Retorna informações sobre o processor e backend

        Returns:
            Dicionário com informações
        """
        info = {
            "backend": self.backend_name,
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "num_speakers": self.num_speakers,
            "device": self.device,
        }

        if self.enabled:
            backend_info = self.backend.get_info()  # type: ignore
            info["backend_info"] = backend_info

        return info
