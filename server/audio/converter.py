"""
Audio conversion utilities for Whisper Stream

Handles conversion between different audio formats and provides
utilities for audio processing.
"""

import logging
from typing import Any

import numpy as np

from ..constants import DEFAULT_SAMPLE_RATE, PCM_INT16_MAX


class AudioConverter:
    """
    Conversor de formatos de áudio

    Responsável por converter áudio entre diferentes formatos
    e normalizar para processamento com Whisper.

    Example:
        >>> converter = AudioConverter()
        >>> audio_float = converter.pcm_int16_to_float32(pcm_bytes)
        >>> stats = converter.get_audio_stats(audio_float)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def pcm_int16_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """
        Converte PCM int16 para float32 normalizado

        Args:
            pcm_bytes: Raw PCM audio bytes (16-bit signed integer, little-endian)

        Returns:
            numpy array float32 normalizado para range [-1.0, 1.0]

        Raises:
            ValueError: Se pcm_bytes está vazio

        Example:
            >>> pcm_bytes = b'\\x00\\x00\\xff\\x7f'  # [0, 32767]
            >>> audio = AudioConverter.pcm_int16_to_float32(pcm_bytes)
            >>> audio.shape
            (2,)
            >>> audio.dtype
            dtype('float32')
        """
        if not pcm_bytes:
            raise ValueError("PCM bytes cannot be empty")

        # Converter bytes para numpy array int16
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Converter para float32 e normalizar usando constante
        return audio_int16.astype(np.float32) / PCM_INT16_MAX


    @staticmethod
    def validate_audio(
        audio: np.ndarray,
        expected_sample_rate: int | None = None,
        min_duration: float = 0.1,
        max_duration: float = 30.0,
    ) -> bool:
        """
        Valida array de áudio

        Args:
            audio: Array numpy de áudio
            expected_sample_rate: Taxa de amostragem esperada (opcional)
            min_duration: Duração mínima em segundos
            max_duration: Duração máxima em segundos

        Returns:
            True se válido

        Raises:
            ValueError: Se áudio é inválido

        Example:
            >>> audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
            >>> AudioConverter.validate_audio(audio)
            True
        """
        if audio.size == 0:
            raise ValueError("Audio array is empty")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values")

        # Verificar range [-1.0, 1.0]
        max_abs = np.abs(audio).max()
        if max_abs > 1.0:
            raise ValueError(f"Audio values exceed [-1.0, 1.0]: max={max_abs:.3f}")

        # Verificar duração se sample_rate fornecido
        if expected_sample_rate:
            duration = len(audio) / float(expected_sample_rate)

            if duration < min_duration:
                raise ValueError(f"Audio too short: {duration:.2f}s (min: {min_duration}s)")

            if duration > max_duration:
                raise ValueError(f"Audio too long: {duration:.2f}s (max: {max_duration}s)")

        return True

    @staticmethod
    def get_duration(audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
        """
        Calcula duração do áudio em segundos

        Args:
            audio: Array numpy de áudio
            sample_rate: Taxa de amostragem em Hz

        Returns:
            Duração em segundos

        Example:
            >>> audio = np.zeros(16000, dtype=np.float32)  # 1 segundo a 16kHz
            >>> AudioConverter.get_duration(audio, 16000)
            1.0
        """
        return len(audio) / float(sample_rate)

    @staticmethod
    def get_audio_stats(audio: np.ndarray) -> dict[str, Any]:
        """
        Retorna estatísticas do áudio para debugging

        Args:
            audio: Array numpy de áudio

        Returns:
            Dicionário com estatísticas:
            - length: Número de samples
            - min: Valor mínimo
            - max: Valor máximo
            - mean: Média
            - std: Desvio padrão
            - has_silence: Se áudio parece silêncio

        Example:
            >>> audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
            >>> stats = AudioConverter.get_audio_stats(audio)
            >>> stats['length']
            3
            >>> stats['has_silence']
            False
        """
        return {
            "length": len(audio),
            "min": float(audio.min()),
            "max": float(audio.max()),
            "mean": float(audio.mean()),
            "std": float(audio.std()),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "has_silence": bool(np.abs(audio).max() < 0.01),
        }

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        """
        Normaliza áudio para um nível target

        Args:
            audio: Array numpy de áudio
            target_level: Nível target (0.0 - 1.0)

        Returns:
            Áudio normalizado

        Example:
            >>> audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)
            >>> normalized = AudioConverter.normalize_audio(audio, 0.95)
            >>> np.abs(normalized).max()
            0.95
        """
        max_abs = np.abs(audio).max()

        if max_abs > 0:
            audio = audio * (target_level / max_abs)

        return audio
