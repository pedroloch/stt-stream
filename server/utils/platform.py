"""
Platform detection para Whisper Stream

Detecta automaticamente a plataforma de execução:
- macOS Apple Silicon (M1/M2/M3)
- macOS Intel
- Linux com CUDA GPU
- Linux CPU only
- Windows com CUDA GPU

Esta detecção acontece APENAS no servidor Python.
"""

import platform
import subprocess
from enum import Enum


class Platform(Enum):
    """Enumeração de plataformas suportadas"""
    MACOS_APPLE_SILICON = "macos_arm64"
    MACOS_INTEL = "macos_x86_64"
    LINUX_CUDA = "linux_cuda"
    LINUX_CPU = "linux_cpu"
    WINDOWS_CUDA = "windows_cuda"


class PlatformNotSupportedError(Exception):
    """Exceção levantada quando plataforma não é suportada"""
    pass


def detect_platform() -> Platform:
    """
    Detecta automaticamente a plataforma atual

    Retorna a plataforma detectada baseado em:
    - Sistema operacional (Darwin, Linux, Windows)
    - Arquitetura do processador (arm64, x86_64)
    - Disponibilidade de GPU NVIDIA

    Returns:
        Platform enum correspondente à plataforma atual

    Raises:
        PlatformNotSupportedError: Se a plataforma não é suportada

    Example:
        >>> platform = detect_platform()
        >>> if platform == Platform.MACOS_APPLE_SILICON:
        ...     print("Rodando em Apple Silicon!")
    """
    system = platform.system()
    machine = platform.machine().lower()

    # macOS
    if system == "Darwin":
        if machine == "arm64":
            return Platform.MACOS_APPLE_SILICON
        return Platform.MACOS_INTEL

    # Linux
    if system == "Linux":
        if has_nvidia_gpu():
            return Platform.LINUX_CUDA
        return Platform.LINUX_CPU

    # Windows
    if system == "Windows":
        if has_nvidia_gpu():
            return Platform.WINDOWS_CUDA
        raise PlatformNotSupportedError(
            "Windows CPU not supported. "
            "Please use a machine with NVIDIA GPU or deploy on Linux/macOS."
        )

    # Plataforma desconhecida
    raise PlatformNotSupportedError(
        f"Unknown platform: {system} {machine}. "
        f"Supported platforms: macOS, Linux, Windows (with NVIDIA GPU)"
    )


def has_nvidia_gpu() -> bool:
    """
    Verifica se uma GPU NVIDIA está disponível

    Tenta duas abordagens:
    1. torch.cuda.is_available() - mais rápido se torch está instalado
    2. nvidia-smi command - fallback se torch não disponível

    Returns:
        True se GPU NVIDIA detectada, False caso contrário

    Example:
        >>> if has_nvidia_gpu():
        ...     print("GPU NVIDIA disponível!")
    """
    # Tentativa 1: Usar torch.cuda se disponível
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass  # torch não instalado, tentar nvidia-smi
    except Exception:
        pass  # outro erro ao importar torch

    # Tentativa 2: Executar nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=2,  # timeout curto
            check=False
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def get_platform_name(plat: Platform | None = None) -> str:
    """
    Retorna nome humanizado da plataforma

    Args:
        plat: Platform enum (se None, detecta automaticamente)

    Returns:
        Nome humanizado da plataforma

    Example:
        >>> name = get_platform_name()
        >>> print(f"Plataforma: {name}")
        Plataforma: macOS Apple Silicon
    """
    if plat is None:
        plat = detect_platform()

    names = {
        Platform.MACOS_APPLE_SILICON: "macOS Apple Silicon",
        Platform.MACOS_INTEL: "macOS Intel",
        Platform.LINUX_CUDA: "Linux with CUDA GPU",
        Platform.LINUX_CPU: "Linux (CPU only)",
        Platform.WINDOWS_CUDA: "Windows with CUDA GPU",
    }

    return names.get(plat, str(plat.value))
