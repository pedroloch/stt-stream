"""
MLX Backend para Whisper

Usa mlx-whisper para Apple Silicon (M1/M2/M3).
Otimizado para chips Apple com unified memory.
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..models.capability import BackendInfo, Capability
from ..models.result import Segment, TranscriptionResult
from ..utils.platform import Platform
from .base import (
    BackendError,
    BackendNotAvailableError,
    ModelNotFoundError,
    TranscriptionError,
    WhisperBackend,
)


class MLXBackend(WhisperBackend):
    """
    Backend MLX para Apple Silicon

    Requer macOS com M1/M2/M3 chip

    Capabilities:
    - TRANSCRIPTION: Transcrição básica
    - STREAMING: Processamento em streaming
    - WORD_TIMESTAMPS: Word-level timestamps (para LocalAgreement)
    """

    # Class-level constant
    INFO = BackendInfo(
        name="mlx-whisper",
        supported_platforms={Platform.MACOS_APPLE_SILICON},
        capabilities={
            Capability.TRANSCRIPTION,
            Capability.STREAMING,
            Capability.WORD_TIMESTAMPS,  # ⭐ Agora suporta word timestamps!
        },
        model_sizes={"tiny", "small", "medium", "large", "large-v3"},
    )

    @property
    def info(self) -> BackendInfo:
        """Retorna metadata e capabilities do MLX backend (cached)"""
        return self.__class__.INFO

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "float16",
        models_dir: str | None = None,
        cache_dir: str | None = None,
        **kwargs
    ):
        super().__init__(model, language, compute_type, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whisper-stream"
        self.model_instance = None
        self.current_context = ""

    async def initialize(self) -> None:
        """Inicializa o modelo mlx-whisper"""
        try:
            # Verificar se mlx-whisper está disponível
            try:
                import mlx_whisper  # type: ignore[import-not-found] # noqa: F401
            except ImportError:
                raise BackendNotAvailableError(
                    "mlx-whisper não instalado. Instale com: pip install mlx-whisper\n"
                    "Nota: mlx-whisper só funciona em Apple Silicon (M1/M2/M3)"
                ) from None

            self.logger.info(f"Carregando modelo {self.model} com MLX (Apple Silicon)...")

            # Criar diretórios
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # MLX Whisper usa formato HuggingFace Hub
            # IMPORTANTE: Nem todos modelos Whisper estão disponíveis em MLX
            # Modelos disponíveis em mlx-community: tiny, small, medium, large
            # Verificar: https://huggingface.co/mlx-community
            available_models = ["tiny", "small", "medium", "large", "large-v3"]
            if self.model not in available_models:
                self.logger.warning(
                    f"Modelo '{self.model}' pode não estar disponível em MLX. "
                    f"Modelos conhecidos: {available_models}"
                )
                self.logger.info("Tentando usar 'tiny' como fallback...")
                self.model = "tiny"

            self.model_path = f"mlx-community/whisper-{self.model}"

            # Testar que o mlx_whisper funciona (importação já foi feita)
            self.logger.info(f"Usando modelo MLX: {self.model_path}")

            self._initialized = True
            self.logger.info("✅ Modelo MLX configurado (será baixado no primeiro uso)")

        except BackendNotAvailableError:
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo MLX: {e}")
            raise ModelNotFoundError(f"Falha ao carregar modelo {self.model}: {e}") from e

    async def transcribe_chunk(  # type: ignore[override]
        self,
        audio: np.ndarray,
        context: str | None = None
    ) -> dict[str, Any]:
        """
        Transcreve um chunk de áudio usando MLX

        Args:
            audio: Array numpy (16kHz, mono, float32)
            context: Texto anterior para contexto

        Returns:
            Resultado da transcrição
        """
        if not self._initialized:
            raise BackendError("Backend não inicializado. Chame initialize() primeiro.")

        try:
            import mlx_whisper  # type: ignore[import-not-found]

            # Preparar áudio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Preparar opções de transcrição
            transcribe_options = {
                "language": self.language if self.language != "auto" else None,
                "task": "transcribe",
                "initial_prompt": context or self.current_context or None,
                # ⭐ WORD TIMESTAMPS! (necessário para LocalAgreement)
                "word_timestamps": True,
            }

            # Adicionar opções adicionais se fornecidas
            if "temperature" in self.kwargs:
                transcribe_options["temperature"] = self.kwargs["temperature"]
            if "condition_on_previous_text" in self.kwargs:
                transcribe_options["condition_on_previous_text"] = self.kwargs["condition_on_previous_text"]

            # Transcrever usando MLX (rápido em Apple Silicon!)
            # Nota: mlx_whisper.transcribe baixa o modelo automaticamente na primeira vez
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_path,
                verbose=False,  # Não mostrar logs de progresso
                **transcribe_options
            )

            # MLX Whisper retorna formato similar ao Whisper original
            text = result.get("text", "").strip()

            if not text:
                # Retornar resultado vazio
                return TranscriptionResult(  # type: ignore[return-value]
                    text="",
                    is_final=False,
                    confidence=0.0,
                    language=result.get("language", self.language),
                    timestamp=datetime.now(),
                )

            # Atualizar contexto
            self.current_context = text[-500:] if len(text) > 500 else text

            # Processar segmentos
            segments_raw = result.get("segments", [])

            # Calcular confiança média (se disponível)
            confidence = 0.0
            if segments_raw:
                # MLX whisper pode não ter logprobs em todos os casos
                # Usar no_speech_prob invertido como aproximação
                no_speech_probs = [seg.get("no_speech_prob", 0.5) for seg in segments_raw]
                confidence = 1.0 - (sum(no_speech_probs) / len(no_speech_probs))

            # Converter segmentos para formato normalizado
            from ..models.result import Word

            segments = []
            all_words = []  # Lista flat de todas as palavras (para HypothesisBuffer)

            for seg in segments_raw:
                # Extrair words se existirem
                words = None
                if "words" in seg and seg["words"]:
                    words = [
                        Word(
                            word=w.get("word", ""),
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            probability=w.get("probability", 0.0),
                        )
                        for w in seg["words"]
                    ]
                    all_words.extend(words)
                else:
                    self.logger.debug(f"Segmento MLX sem words: '{seg.get('text', '')[:30]}...'")

                segments.append(
                    Segment(
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", "").strip(),
                        words=words,
                    )
                )

            if not all_words:
                self.logger.warning("MLX não retornou word timestamps! LocalAgreement não funcionará.")

            segments_list = segments if segments else None

            # Retornar TranscriptionResult normalizado com words! ⭐
            return TranscriptionResult(  # type: ignore[return-value]
                text=text,
                is_final=True,
                confidence=float(confidence),
                language=result.get("language", self.language),
                timestamp=datetime.now(),
                segments=segments_list,
                words=all_words if all_words else None,  # ⭐ Lista flat para HypothesisBuffer
            )

        except Exception as e:
            self.logger.error(f"Erro na transcrição MLX: {e}")
            raise TranscriptionError(f"Falha na transcrição: {e}") from e

    async def transcribe_stream(  # type: ignore[override]
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[dict[str, Any]]:
        """Transcreve stream de áudio"""
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result["text"]:
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        if self.model_instance:
            del self.model_instance
            self.model_instance = None
            self._initialized = False
            self.logger.info("Backend MLX limpo")

    def get_backend_info(self) -> dict[str, Any]:
        """Retorna informações sobre o backend"""
        info = {
            "name": "MLX Backend (Apple Silicon)",
            "version": self._get_mlx_whisper_version(),
            "device": "Apple Silicon (Unified Memory)",
            "model": self.model,
            "language": self.language,
            "compute_type": self.compute_type,
            "initialized": self._initialized,
        }

        # Tentar obter info do chip Apple
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                info["chip"] = result.stdout.strip()
        except Exception:
            pass

        return info

    def _get_mlx_whisper_version(self) -> str:
        """Obtém versão do mlx-whisper"""
        try:
            import mlx_whisper  # type: ignore[import-not-found]
            return mlx_whisper.__version__
        except (ImportError, AttributeError):
            return "unknown"
