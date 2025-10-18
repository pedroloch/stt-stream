"""
Voice Activity Detection para chunking inteligente

Usa Silero VAD para detectar pausas e cortar chunks em pausas naturais de fala.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger(__name__)


class VADChunker:
    """
    Chunker baseado em VAD (Voice Activity Detection)

    Usa Silero VAD para detectar pausas e cortar chunks naturalmente.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration: float = 0.3,  # segundos
        min_speech_duration: float = 0.5,  # segundos
    ):
        """
        Args:
            threshold: Threshold de confiança VAD (0-1)
            min_silence_duration: Mínimo de silêncio para considerar pausa
            min_speech_duration: Mínimo de fala para considerar speech
        """
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration

        # Lazy load do modelo
        self._model = None
        self._utils = None

    def _load_model(self):
        """Load Silero VAD model"""
        if self._model is None:
            try:
                import torch

                # Silero VAD model
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                self._model = model
                self._utils = utils
                logger.info("Silero VAD model loaded")
            except ImportError as e:
                logger.error(
                    f"PyTorch not available for VAD. Install with: poetry install -E cuda\n{e}"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                raise

    def detect_speech(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[tuple[float, float]]:
        """
        Detecta segmentos de fala no áudio

        Args:
            audio: Array numpy (float32, mono)
            sample_rate: Sample rate

        Returns:
            Lista de tuplas (start_time, end_time) em segundos
        """
        self._load_model()

        import torch

        # Converter para torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get speech timestamps
        speech_timestamps = self._utils[0](
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
        )

        # Converter para segundos
        segments = [
            (ts["start"] / sample_rate, ts["end"] / sample_rate)
            for ts in speech_timestamps
        ]

        logger.debug(f"VAD detected {len(segments)} speech segments")
        return segments

    def find_best_split_point(
        self, audio: np.ndarray, target_duration: float = 5.0, sample_rate: int = 16000
    ) -> int:
        """
        Encontra melhor ponto de corte (em samples) baseado em pausas

        Args:
            audio: Áudio completo
            target_duration: Duração alvo do chunk (segundos)
            sample_rate: Sample rate

        Returns:
            Index (em samples) do melhor ponto de corte
        """
        # Detectar segmentos de fala
        segments = self.detect_speech(audio, sample_rate)

        if not segments:
            # Sem fala detectada, retornar target padrão
            logger.debug("VAD: sem fala detectada, usando target padrão")
            return int(target_duration * sample_rate)

        # Converter target para samples
        target_samples = int(target_duration * sample_rate)

        # Encontrar segmento mais próximo do target
        best_split = target_samples
        min_diff = float("inf")

        for _start, end in segments:
            end_samples = int(end * sample_rate)

            # Se este segmento termina perto do target
            diff = abs(end_samples - target_samples)
            if diff < min_diff and end_samples <= len(audio):
                min_diff = diff
                best_split = end_samples

        logger.debug(
            f"VAD best split: {best_split/sample_rate:.2f}s "
            f"(target: {target_duration:.2f}s)"
        )

        return best_split

    def has_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Verifica se há fala no áudio

        Args:
            audio: Array numpy (float32, mono)
            sample_rate: Sample rate

        Returns:
            True se detectou fala
        """
        segments = self.detect_speech(audio, sample_rate)
        return len(segments) > 0
