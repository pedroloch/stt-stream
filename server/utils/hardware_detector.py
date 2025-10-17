"""
Hardware detection para o servidor Whisper Stream

Detecta automaticamente o hardware disponível:
- Apple Silicon (M1/M2/M3) → MLX
- NVIDIA GPU → CUDA
- CPU fallback

Esta detecção roda APENAS no servidor Python, não no cliente Bun.
"""

import platform
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class HardwareInfo:
    """Informações sobre o hardware detectado"""
    type: str  # "apple_silicon", "cuda", or "cpu"
    device_name: Optional[str] = None
    memory: Optional[str] = None
    compute_capability: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HardwareDetector:
    """
    Detector de hardware para escolher o backend correto do Whisper

    Ordem de preferência:
    1. Apple Silicon (se macOS + ARM64) → MLX
    2. NVIDIA GPU (se nvidia-smi encontrado) → CUDA
    3. CPU (fallback)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def detect(self, force_type: Optional[str] = None) -> HardwareInfo:
        """
        Detecta o hardware disponível

        Args:
            force_type: Se especificado, força um tipo específico
                       (útil para testing ou override)

        Returns:
            HardwareInfo com detalhes do hardware
        """
        if force_type:
            self.logger.info(f"Hardware forçado: {force_type}")
            return HardwareInfo(type=force_type)

        # 1. Tentar detectar Apple Silicon
        if self._is_apple_silicon():
            self.logger.info("Apple Silicon detectado")
            return self._get_apple_silicon_info()

        # 2. Tentar detectar CUDA
        if self._has_cuda():
            self.logger.info("NVIDIA GPU detectada")
            return self._get_cuda_info()

        # 3. Fallback para CPU
        self.logger.info("Usando CPU (nenhuma GPU detectada)")
        return self._get_cpu_info()

    def _is_apple_silicon(self) -> bool:
        """Verifica se está rodando em Apple Silicon"""
        try:
            # Checa se é macOS e ARM64
            if platform.system() != "Darwin":
                return False

            machine = platform.machine().lower()
            return machine == "arm64"
        except Exception as e:
            self.logger.debug(f"Erro ao detectar Apple Silicon: {e}")
            return False

    def _has_cuda(self) -> bool:
        """Verifica se CUDA está disponível"""
        try:
            # Tenta executar nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.logger.debug(f"CUDA não detectado: {e}")
            return False

    def _get_apple_silicon_info(self) -> HardwareInfo:
        """Obtém informações sobre Apple Silicon"""
        try:
            # Pega o modelo do chip
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )

            device_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

            # Pega memória total
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )

            memory_bytes = int(result.stdout.strip()) if result.returncode == 0 else 0
            memory_gb = f"{memory_bytes / (1024**3):.1f} GB" if memory_bytes > 0 else "Unknown"

            return HardwareInfo(
                type="apple_silicon",
                device_name=device_name,
                memory=memory_gb,
                details={
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                }
            )
        except Exception as e:
            self.logger.warning(f"Erro ao obter info Apple Silicon: {e}")
            return HardwareInfo(type="apple_silicon", device_name="Apple Silicon")

    def _get_cuda_info(self) -> HardwareInfo:
        """Obtém informações sobre GPU NVIDIA"""
        try:
            # Pega nome da GPU e memória
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                device_name = parts[0].strip() if len(parts) > 0 else "NVIDIA GPU"
                memory = parts[1].strip() if len(parts) > 1 else "Unknown"
                compute_cap = parts[2].strip() if len(parts) > 2 else "Unknown"

                return HardwareInfo(
                    type="cuda",
                    device_name=device_name,
                    memory=memory,
                    compute_capability=compute_cap,
                    details={
                        "driver_version": self._get_cuda_driver_version(),
                    }
                )
        except Exception as e:
            self.logger.warning(f"Erro ao obter info CUDA: {e}")

        return HardwareInfo(type="cuda", device_name="NVIDIA GPU")

    def _get_cpu_info(self) -> HardwareInfo:
        """Obtém informações sobre CPU"""
        try:
            import os
            cpu_count = os.cpu_count() or 1

            return HardwareInfo(
                type="cpu",
                device_name=platform.processor() or "CPU",
                details={
                    "cores": cpu_count,
                    "platform": platform.platform(),
                    "architecture": platform.machine(),
                }
            )
        except Exception as e:
            self.logger.warning(f"Erro ao obter info CPU: {e}")
            return HardwareInfo(type="cpu", device_name="CPU")

    def _get_cuda_driver_version(self) -> Optional[str]:
        """Obtém versão do driver CUDA"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def print_hardware_info(self, hardware: HardwareInfo) -> None:
        """Imprime informações sobre o hardware de forma formatada"""
        print("\n" + "="*60)
        print("🖥️  HARDWARE DETECTADO")
        print("="*60)
        print(f"Tipo: {hardware.type.upper()}")
        if hardware.device_name:
            print(f"Dispositivo: {hardware.device_name}")
        if hardware.memory:
            print(f"Memória: {hardware.memory}")
        if hardware.compute_capability:
            print(f"Compute Capability: {hardware.compute_capability}")
        if hardware.details:
            print("\nDetalhes:")
            for key, value in hardware.details.items():
                print(f"  {key}: {value}")
        print("="*60 + "\n")
