"""
Faster-Whisper Backend (Universal)

Backend universal usando faster-whisper que funciona em:
- Apple Silicon (Mac M1/M2/M3)
- Linux com CUDA GPU
- Linux CPU
- Windows com CUDA

Capabilities:
- TRANSCRIPTION
- WORD_TIMESTAMPS (word-level precision!)
- VAD (Voice Activity Detection)
- STREAMING

Recomendado: usar modelo "distil-large-v3" (6x mais rapido)
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..models.capability import BackendInfo, Capability
from ..models.result import Segment, TranscriptionResult, Word
from ..utils.platform import Platform
from .base import (
    BackendError,
    BackendNotAvailableError,
    ModelNotFoundError,
    TranscriptionError,
    WhisperBackend,
)


class FasterWhisperBackend(WhisperBackend):
    """
    Backend Faster-Whisper (Universal)

    Funciona em qualquer plataforma e suporta word-level timestamps!
    Recomendado usar distil-large-v3 para performance 6x melhor.
    """

    # Class-level constant (não cria objeto toda vez)
    INFO = BackendInfo(
        name="faster-whisper",
        supported_platforms={
            Platform.MACOS_APPLE_SILICON,
            Platform.MACOS_INTEL,
            Platform.LINUX_CUDA,
            Platform.LINUX_CPU,
            Platform.WINDOWS_CUDA,
        },
        capabilities={
            Capability.TRANSCRIPTION,
            Capability.WORD_TIMESTAMPS,  # ⭐ Word-level precision
            Capability.VAD,
            Capability.STREAMING,
        },
        model_sizes={
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "distil-large-v3",  # ⭐ Recomendado - 6x mais rapido
        },
    )

    @property
    def info(self) -> BackendInfo:
        """Retorna metadata e capabilities do Faster-Whisper backend (cached)"""
        return self.__class__.INFO

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "auto",
        device: str = "auto",
        models_dir: str | None = None,
        cache_dir: str | None = None,
        **kwargs
    ):
        super().__init__(model, language, compute_type, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whisper-stream"
        )
        self.model_instance = None
        self.current_context = ""

    async def initialize(self) -> None:
        """Inicializa o modelo faster-whisper"""
        try:
            from faster_whisper import WhisperModel

            # Auto-detect device
            if self.device == "auto":
                self.device = self._detect_best_device()

            # Auto-select compute type
            if self.compute_type == "auto":
                self.compute_type = self._select_compute_type()

            self.logger.info(
                f"Carregando modelo {self.model} com faster-whisper..."
            )
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Compute type: {self.compute_type}")

            # Criar diretorios
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Carregar modelo
            self.model_instance = WhisperModel(
                self.model,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.models_dir),
            )

            self._initialized = True
            self.logger.info("✅ Modelo faster-whisper carregado")

            # Logar se distil-large-v3 (recomendado)
            if "distil" in self.model:
                self.logger.info("   ⚡ Usando distil-whisper (6x mais rapido!)")

        except ImportError:
            raise BackendNotAvailableError(
                "faster-whisper nao instalado.\n"
                "Instale com: pip install faster-whisper"
            ) from None
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelNotFoundError(f"Falha ao carregar modelo {self.model}: {e}") from e

    def _detect_best_device(self) -> str:
        """Detecta melhor device disponivel"""
        try:
            import torch

            if torch.cuda.is_available():
                self.logger.info("GPU CUDA detectada")
                return "cuda"
        except ImportError:
            pass

        self.logger.info("Usando CPU")
        return "cpu"

    def _select_compute_type(self) -> str:
        """Seleciona compute type baseado no device"""
        if self.device == "cuda":
            # Para distil-whisper, int8_float16 eh otimo
            if "distil" in self.model:
                return "int8_float16"
            return "float16"
        # CPU: int8 eh mais rapido
        return "int8"

    async def transcribe_chunk(
        self, audio: np.ndarray, context: str | None = None
    ) -> TranscriptionResult:
        """
        Transcreve chunk de audio com word-level timestamps

        Returns:
            TranscriptionResult com segments e words
        """
        if not self._initialized:
            raise BackendError("Backend nao inicializado. Chame initialize() primeiro.")

        try:
            # Preparar audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Transcricao com WORD TIMESTAMPS! ⭐
            segments, info = self.model_instance.transcribe(
                audio,
                language=self.language if self.language != "auto" else None,
                # ⭐ WORD TIMESTAMPS
                word_timestamps=self.kwargs.get("enable_word_timestamps", True),
                # Anti-hallucination
                no_speech_threshold=self.kwargs.get("no_speech_threshold", 0.6),
                log_prob_threshold=self.kwargs.get("log_prob_threshold", -1.0),
                compression_ratio_threshold=self.kwargs.get(
                    "compression_ratio_threshold", 2.4
                ),
                # VAD (fix: use "use_vad" to match config key)
                vad_filter=self.kwargs.get("use_vad", False),
                vad_parameters={
                    "threshold": self.kwargs.get("vad_threshold", 0.5),
                },
                # Context
                initial_prompt=context or self.current_context or None,
                condition_on_previous_text=self.kwargs.get(
                    "condition_on_previous_text", True
                ),
            )

            # Converter segments para lista
            segments_list = list(segments)

            if not segments_list:
                # Sem transcricao
                return TranscriptionResult(
                    text="",
                    is_final=False,
                    confidence=0.0,
                    language=info.language,
                    timestamp=datetime.now(),
                )

            # Concatenar texto
            full_text = " ".join(seg.text.strip() for seg in segments_list).strip()

            if not full_text:
                return TranscriptionResult(
                    text="",
                    is_final=False,
                    confidence=0.0,
                    language=info.language,
                    timestamp=datetime.now(),
                )

            # Atualizar contexto
            self.current_context = full_text[-500:] if len(full_text) > 500 else full_text

            # Calcular confianca media
            avg_confidence = (
                sum(seg.avg_logprob for seg in segments_list) / len(segments_list)
                if segments_list
                else 0.0
            )

            # Converter segmentos para formato normalizado
            normalized_segments = []
            for seg in segments_list:
                # Converter words se existirem
                words = None
                if hasattr(seg, "words") and seg.words:
                    words = [
                        Word(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                        for w in seg.words
                    ]

                normalized_segments.append(
                    Segment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        words=words,  # ⭐ Word-level timestamps
                    )
                )

            # Retornar TranscriptionResult com word timestamps! ⭐
            return TranscriptionResult(
                text=full_text,
                is_final=True,
                confidence=float(avg_confidence),
                language=info.language,
                timestamp=datetime.now(),
                segments=normalized_segments,
            )


        except Exception as e:
            self.logger.error(f"Erro na transcricao: {e}")
            raise TranscriptionError(f"Falha na transcricao: {e}") from e

    async def transcribe_stream(  # type: ignore[override]
        self, audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcreve stream de audio"""
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result.text:
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        if self.model_instance:
            del self.model_instance
            self.model_instance = None
            self._initialized = False
            self.logger.info("Backend faster-whisper limpo")

    def get_backend_info(self) -> dict[str, Any]:
        """Retorna informacoes sobre o backend"""
        info_dict = {
            "name": "faster-whisper (Universal)",
            "device": self.device,
            "model": self.model,
            "language": self.language,
            "compute_type": self.compute_type,
            "initialized": self._initialized,
            "capabilities": [c.value for c in self.info.capabilities],
        }

        # Info da GPU se CUDA
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    info_dict["gpu"] = torch.cuda.get_device_name(0)
                    info_dict["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            except Exception:
                pass

        return info_dict
