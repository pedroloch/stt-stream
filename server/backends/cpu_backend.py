"""
CPU Backend para Whisper

Usa faster-whisper como base, rodando em CPU.
Fallback quando não há GPU disponível.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, AsyncIterator
from pathlib import Path

from .base import (
    WhisperBackend,
    BackendError,
    BackendNotAvailableError,
    ModelNotFoundError,
    TranscriptionError,
    MODEL_INIT_PARAMS
)


class CPUBackend(WhisperBackend):
    """
    Backend CPU usando faster-whisper

    Compatível com qualquer sistema, mas mais lento que GPU
    """

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "int8",  # int8 é melhor para CPU
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, language, compute_type, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whisper-stream"
        self.model_instance = None
        self.current_context = ""

    async def initialize(self) -> None:
        """Inicializa o modelo faster-whisper em CPU"""
        try:
            from faster_whisper import WhisperModel

            self.logger.info(f"Carregando modelo {self.model} em CPU...")
            self.logger.info(f"Compute type: {self.compute_type}")

            # Criar diretórios se não existirem
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Filtrar kwargs: apenas parâmetros válidos para o construtor WhisperModel
            # Parâmetros de transcrição (beam_size, temperature, etc) são usados em transcribe_chunk()
            model_init_kwargs = {
                k: v for k, v in self.kwargs.items()
                if k in MODEL_INIT_PARAMS
            }

            # Carregar modelo
            # faster-whisper baixa automaticamente se necessário
            self.model_instance = WhisperModel(
                self.model,
                device="cpu",
                compute_type=self.compute_type,
                download_root=str(self.models_dir),
                **model_init_kwargs
            )

            self._initialized = True
            self.logger.info("✅ Modelo CPU carregado com sucesso")

        except ImportError:
            raise BackendNotAvailableError(
                "faster-whisper não instalado. Instale com: pip install faster-whisper"
            )
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelNotFoundError(f"Falha ao carregar modelo {self.model}: {e}")

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcreve um chunk de áudio

        Args:
            audio: Array numpy (16kHz, mono, float32)
            context: Texto anterior para contexto

        Returns:
            Resultado da transcrição
        """
        if not self._initialized:
            raise BackendError("Backend não inicializado. Chame initialize() primeiro.")

        try:
            # Garantir que áudio está no formato correto
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar áudio se necessário
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Usar contexto se fornecido
            initial_prompt = context or self.current_context

            # Transcrever
            segments, info = self.model_instance.transcribe(
                audio,
                language=self.language if self.language != "auto" else None,
                initial_prompt=initial_prompt,
                beam_size=self.kwargs.get("beam_size", 1),
                best_of=self.kwargs.get("best_of", 1),
                temperature=self.kwargs.get("temperature", 0.0),
                condition_on_previous_text=self.kwargs.get("condition_on_previous_text", True),
                vad_filter=self.kwargs.get("use_vad", True),
            )

            # Coletar segmentos
            segments_list = list(segments)

            if not segments_list:
                return {
                    "text": "",
                    "is_final": False,
                    "language": info.language,
                    "confidence": 0.0,
                    "segments": []
                }

            # Combinar texto dos segmentos
            text = " ".join(segment.text.strip() for segment in segments_list)

            # Atualizar contexto
            self.current_context = text[-500:] if len(text) > 500 else text

            # Calcular confiança média
            avg_confidence = sum(
                segment.avg_logprob for segment in segments_list
            ) / len(segments_list) if segments_list else 0.0

            # Converter para probabilidade aproximada (logprob → prob)
            confidence = np.exp(avg_confidence)

            return {
                "text": text.strip(),
                "is_final": True,
                "language": info.language,
                "confidence": float(confidence),
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    }
                    for segment in segments_list
                ]
            }

        except Exception as e:
            self.logger.error(f"Erro na transcrição: {e}")
            raise TranscriptionError(f"Falha na transcrição: {e}")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Transcreve stream de áudio

        Para CPU, processamos chunk por chunk
        """
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result["text"]:  # Só yield se houver texto
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        if self.model_instance:
            del self.model_instance
            self.model_instance = None
            self._initialized = False
            self.logger.info("Backend CPU limpo")

    def get_backend_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o backend"""
        return {
            "name": "CPU Backend (faster-whisper)",
            "version": self._get_faster_whisper_version(),
            "device": "cpu",
            "model": self.model,
            "language": self.language,
            "compute_type": self.compute_type,
            "initialized": self._initialized,
        }

    def _get_faster_whisper_version(self) -> str:
        """Obtém versão do faster-whisper"""
        try:
            import faster_whisper
            return faster_whisper.__version__
        except (ImportError, AttributeError):
            return "unknown"
