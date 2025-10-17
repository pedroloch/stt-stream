"""
Factory para criar o backend correto de Whisper

Seleciona automaticamente o melhor backend baseado no hardware:
1. Apple Silicon → MLX
2. NVIDIA GPU → CUDA
3. CPU → faster-whisper CPU
"""

import logging
from typing import Optional

from .base import WhisperBackend, BackendNotAvailableError
from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .mlx_backend import MLXBackend
from ..utils.hardware_detector import HardwareDetector, HardwareInfo


class BackendFactory:
    """
    Factory para criar backends de Whisper

    Exemplo de uso:
        factory = BackendFactory(logger)
        backend = factory.create_backend(
            backend_type="auto",
            model="base",
            language="pt"
        )
        await backend.initialize()
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.detector = HardwareDetector(self.logger)

    def create_backend(
        self,
        backend_type: str = "auto",
        model: str = "base",
        language: str = "pt",
        compute_type: str = "float16",
        device: str = "auto",
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_hardware_type: Optional[str] = None,
        **kwargs
    ) -> WhisperBackend:
        """
        Cria o backend apropriado

        Args:
            backend_type: "auto", "mlx", "cuda", ou "cpu"
            model: Modelo do Whisper
            language: Idioma
            compute_type: Tipo de computação
            device: Device específico (para CUDA)
            models_dir: Diretório de modelos
            cache_dir: Diretório de cache
            force_hardware_type: Força tipo de hardware (para testing)
            **kwargs: Argumentos adicionais para o backend

        Returns:
            Instância do backend apropriado

        Raises:
            BackendNotAvailableError: Se o backend solicitado não está disponível
        """

        # Se backend_type é "auto", detectar hardware
        if backend_type == "auto":
            hardware = self.detector.detect(force_type=force_hardware_type)
            backend_type = self._hardware_to_backend(hardware)
            self.logger.info(f"Auto-selecionado backend: {backend_type}")

        # Criar backend apropriado
        backend_type = backend_type.lower()

        try:
            if backend_type == "mlx":
                return self._create_mlx_backend(
                    model, language, compute_type, models_dir, cache_dir, **kwargs
                )
            elif backend_type == "cuda":
                return self._create_cuda_backend(
                    model, language, compute_type, device, models_dir, cache_dir, **kwargs
                )
            elif backend_type == "cpu":
                return self._create_cpu_backend(
                    model, language, compute_type, models_dir, cache_dir, **kwargs
                )
            else:
                raise ValueError(
                    f"Backend inválido: {backend_type}. "
                    f"Use 'auto', 'mlx', 'cuda' ou 'cpu'"
                )

        except BackendNotAvailableError as e:
            self.logger.warning(f"Backend {backend_type} não disponível: {e}")
            self.logger.info("Tentando fallback para CPU...")
            return self._create_cpu_backend(
                model, language, "int8", models_dir, cache_dir, **kwargs
            )

    def _hardware_to_backend(self, hardware: HardwareInfo) -> str:
        """Converte tipo de hardware para tipo de backend"""
        mapping = {
            "apple_silicon": "mlx",
            "cuda": "cuda",
            "cpu": "cpu"
        }
        return mapping.get(hardware.type, "cpu")

    def _create_mlx_backend(
        self,
        model: str,
        language: str,
        compute_type: str,
        models_dir: Optional[str],
        cache_dir: Optional[str],
        **kwargs
    ) -> MLXBackend:
        """Cria backend MLX"""
        self.logger.info("Criando MLX backend para Apple Silicon...")
        return MLXBackend(
            model=model,
            language=language,
            compute_type=compute_type,
            models_dir=models_dir,
            cache_dir=cache_dir,
            **kwargs
        )

    def _create_cuda_backend(
        self,
        model: str,
        language: str,
        compute_type: str,
        device: str,
        models_dir: Optional[str],
        cache_dir: Optional[str],
        **kwargs
    ) -> CUDABackend:
        """Cria backend CUDA"""
        self.logger.info(f"Criando CUDA backend para GPU ({device})...")
        return CUDABackend(
            model=model,
            language=language,
            compute_type=compute_type,
            device=device,
            models_dir=models_dir,
            cache_dir=cache_dir,
            **kwargs
        )

    def _create_cpu_backend(
        self,
        model: str,
        language: str,
        compute_type: str,
        models_dir: Optional[str],
        cache_dir: Optional[str],
        **kwargs
    ) -> CPUBackend:
        """Cria backend CPU"""
        self.logger.info("Criando CPU backend...")
        # Para CPU, int8 geralmente é melhor
        if compute_type == "float16":
            compute_type = "int8"
            self.logger.info("Usando int8 para CPU (melhor performance)")

        return CPUBackend(
            model=model,
            language=language,
            compute_type=compute_type,
            models_dir=models_dir,
            cache_dir=cache_dir,
            **kwargs
        )

    def detect_hardware(self, force_type: Optional[str] = None) -> HardwareInfo:
        """
        Detecta hardware disponível

        Args:
            force_type: Força um tipo específico

        Returns:
            Informações sobre o hardware
        """
        return self.detector.detect(force_type=force_type)
