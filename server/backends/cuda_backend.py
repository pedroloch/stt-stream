"""
CUDA Backend para Whisper

Usa faster-whisper com CUDA para NVIDIA GPUs.
Significativamente mais rápido que CPU.
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


class CUDABackend(WhisperBackend):
    """
    Backend CUDA usando faster-whisper

    Requer NVIDIA GPU com CUDA instalado
    """

    def __init__(
        self,
        model: str = "base",
        language: str = "pt",
        compute_type: str = "float16",
        device: str = "cuda:0",
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, language, compute_type, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whisper-stream"
        self.model_instance = None
        self.current_context = ""

    async def initialize(self) -> None:
        """Inicializa o modelo faster-whisper em CUDA"""
        try:
            from faster_whisper import WhisperModel

            # Verificar se CUDA está disponível
            if not self._check_cuda_available():
                raise BackendNotAvailableError(
                    "CUDA não disponível. Verifique se os drivers NVIDIA estão instalados."
                )

            self.logger.info(f"Carregando modelo {self.model} em CUDA ({self.device})...")
            self.logger.info(f"Compute type: {self.compute_type}")

            # Criar diretórios
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Extrair device_index do device string (cuda:0 → 0)
            device_index = int(self.device.split(":")[-1]) if ":" in self.device else 0

            # Filtrar kwargs: apenas parâmetros válidos para o construtor WhisperModel
            # Parâmetros de transcrição (beam_size, temperature, etc) são usados em transcribe_chunk()
            model_init_kwargs = {
                k: v for k, v in self.kwargs.items()
                if k in MODEL_INIT_PARAMS
            }

            # Carregar modelo
            self.model_instance = WhisperModel(
                self.model,
                device="cuda",
                device_index=device_index,
                compute_type=self.compute_type,
                download_root=str(self.models_dir),
                **model_init_kwargs
            )

            self._initialized = True
            self.logger.info(f"✅ Modelo CUDA carregado em {self.device}")
            self._log_gpu_memory()

        except ImportError:
            raise BackendNotAvailableError(
                "faster-whisper não instalado. Instale com: pip install faster-whisper"
            )
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            raise ModelNotFoundError(f"Falha ao carregar modelo {self.model}: {e}")

    def _check_cuda_available(self) -> bool:
        """Verifica se CUDA está disponível"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            self.logger.warning("PyTorch não instalado, não é possível verificar CUDA")
            # Tentar de outra forma
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            except Exception:
                return False

    def _log_gpu_memory(self) -> None:
        """Log de uso de memória GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_idx) / 1024**3    # GB
                self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        except Exception as e:
            self.logger.debug(f"Não foi possível obter info de memória GPU: {e}")

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcreve um chunk de áudio usando GPU

        Args:
            audio: Array numpy (16kHz, mono, float32)
            context: Texto anterior para contexto

        Returns:
            Resultado da transcrição
        """
        if not self._initialized:
            raise BackendError("Backend não inicializado. Chame initialize() primeiro.")

        try:
            # Preparar áudio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Contexto
            initial_prompt = context or self.current_context

            # Transcrever (rápido em GPU!)
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

            # Processar segmentos
            segments_list = list(segments)

            if not segments_list:
                return {
                    "text": "",
                    "is_final": False,
                    "language": info.language,
                    "confidence": 0.0,
                    "segments": []
                }

            # Texto completo
            text = " ".join(segment.text.strip() for segment in segments_list)

            # Atualizar contexto
            self.current_context = text[-500:] if len(text) > 500 else text

            # Confiança
            avg_confidence = sum(
                segment.avg_logprob for segment in segments_list
            ) / len(segments_list) if segments_list else 0.0
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
            self.logger.error(f"Erro na transcrição CUDA: {e}")
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

            # Limpar cache CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("Cache CUDA limpo")
            except ImportError:
                pass

            self._initialized = False
            self.logger.info("Backend CUDA limpo")

    def get_backend_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o backend"""
        info = {
            "name": "CUDA Backend (faster-whisper)",
            "version": self._get_faster_whisper_version(),
            "device": self.device,
            "model": self.model,
            "language": self.language,
            "compute_type": self.compute_type,
            "initialized": self._initialized,
        }

        # Adicionar info da GPU
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(device_idx).total_memory / 1024**3:.2f} GB"
        except Exception:
            pass

        return info

    def _get_faster_whisper_version(self) -> str:
        """Obtém versão do faster-whisper"""
        try:
            import faster_whisper
            return faster_whisper.__version__
        except (ImportError, AttributeError):
            return "unknown"
