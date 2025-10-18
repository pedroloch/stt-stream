"""
Interface abstrata para backends de diarization

Define o contrato que todos os backends devem implementar.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..models.result import Word


class DiarizationBackend(ABC):
    """
    Interface abstrata para backends de diarization

    Todos os backends (Sortformer, Pyannote, etc) devem implementar esta interface.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        device: str = "auto",
        **kwargs,
    ):
        """
        Inicializa o backend de diarization

        Args:
            sample_rate: Sample rate do áudio (default: 16000)
            num_speakers: Número de speakers (None = auto-detect)
            device: Device para inferência ("auto", "cpu", "cuda", "mps")
            **kwargs: Argumentos específicos do backend
        """
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.device = device

    @abstractmethod
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
            Lista de tuplas (start_time, end_time, speaker_id)
            Speaker IDs são strings no formato "SPEAKER_00", "SPEAKER_01", etc.
            Exemplo: [(0.0, 2.5, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01")]
        """
        pass

    @abstractmethod
    async def assign_speakers_to_words(
        self,
        words: list[Word],
        diarization_segments: list[tuple[float, float, str]],
    ) -> list[Word]:
        """
        Atribui speaker IDs às palavras baseado em overlap temporal

        Args:
            words: Lista de Word objects com timestamps
            diarization_segments: Segmentos de diarization [(start, end, speaker_id)]
                                  Speaker IDs são strings "SPEAKER_XX"

        Returns:
            Lista de Word objects com speaker_id atribuído
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reseta estado interno do backend (importante para streaming)
        """
        pass

    def get_info(self) -> dict:
        """
        Retorna informações sobre o backend

        Returns:
            Dicionário com informações do backend
        """
        return {
            "backend": self.__class__.__name__,
            "sample_rate": self.sample_rate,
            "num_speakers": self.num_speakers,
            "device": self.device,
        }
