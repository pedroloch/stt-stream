"""
MLX Backend para Whisper

Usa mlx-whisper para Apple Silicon (M1/M2/M3).
Otimizado para chips Apple com unified memory.
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
    TranscriptionError
)


class MLXBackend(WhisperBackend):
    """
    Backend MLX para Apple Silicon

    Requer macOS com M1/M2/M3 chip
    """

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "float16",
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
        """Inicializa o modelo mlx-whisper"""
        try:
            # Verificar se mlx-whisper está disponível
            try:
                import mlx_whisper
            except ImportError:
                raise BackendNotAvailableError(
                    "mlx-whisper não instalado. Instale com: pip install mlx-whisper\n"
                    "Nota: mlx-whisper só funciona em Apple Silicon (M1/M2/M3)"
                )

            self.logger.info(f"Carregando modelo {self.model} com MLX (Apple Silicon)...")

            # Criar diretórios
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # MLX Whisper carrega modelos de forma diferente
            # Ele usa a API do Whisper original mas otimizado para Apple Silicon
            self.model_instance = mlx_whisper.load_model(
                self.model,
                # mlx-whisper baixa automaticamente se necessário
            )

            self._initialized = True
            self.logger.info("✅ Modelo MLX carregado com sucesso (Apple Silicon)")

        except BackendNotAvailableError:
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo MLX: {e}")
            raise ModelNotFoundError(f"Falha ao carregar modelo {self.model}: {e}")

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
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
            import mlx_whisper

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
            }

            # Adicionar opções adicionais se fornecidas
            if "temperature" in self.kwargs:
                transcribe_options["temperature"] = self.kwargs["temperature"]
            if "condition_on_previous_text" in self.kwargs:
                transcribe_options["condition_on_previous_text"] = self.kwargs["condition_on_previous_text"]

            # Transcrever usando MLX (rápido em Apple Silicon!)
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_instance,
                **transcribe_options
            )

            # MLX Whisper retorna formato similar ao Whisper original
            text = result.get("text", "").strip()

            if not text:
                return {
                    "text": "",
                    "is_final": False,
                    "language": result.get("language", self.language),
                    "confidence": 0.0,
                    "segments": []
                }

            # Atualizar contexto
            self.current_context = text[-500:] if len(text) > 500 else text

            # Processar segmentos
            segments = result.get("segments", [])

            # Calcular confiança média (se disponível)
            confidence = 0.0
            if segments:
                # MLX whisper pode não ter logprobs em todos os casos
                # Usar no_speech_prob invertido como aproximação
                no_speech_probs = [seg.get("no_speech_prob", 0.5) for seg in segments]
                confidence = 1.0 - (sum(no_speech_probs) / len(no_speech_probs))

            return {
                "text": text,
                "is_final": True,
                "language": result.get("language", self.language),
                "confidence": float(confidence),
                "segments": [
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in segments
                ]
            }

        except Exception as e:
            self.logger.error(f"Erro na transcrição MLX: {e}")
            raise TranscriptionError(f"Falha na transcrição: {e}")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[Dict[str, Any]]:
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

    def get_backend_info(self) -> Dict[str, Any]:
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
            import mlx_whisper
            return mlx_whisper.__version__
        except (ImportError, AttributeError):
            return "unknown"
