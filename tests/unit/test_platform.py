"""
Tests para platform detection

Testa detecção automática de plataforma e GPU
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

# Importar do código que vamos criar
from server.utils.platform import (
    Platform,
    detect_platform,
    has_nvidia_gpu,
    PlatformNotSupportedError,
)


class TestPlatform:
    """Tests para Platform enum e detecção"""

    def test_platform_enum_values(self):
        """Enum Platform tem todos os valores esperados"""
        assert Platform.MACOS_APPLE_SILICON.value == "macos_arm64"
        assert Platform.MACOS_INTEL.value == "macos_x86_64"
        assert Platform.LINUX_CUDA.value == "linux_cuda"
        assert Platform.LINUX_CPU.value == "linux_cpu"
        assert Platform.WINDOWS_CUDA.value == "windows_cuda"


class TestDetectPlatform:
    """Tests para detect_platform()"""

    @patch("platform.system")
    @patch("platform.machine")
    def test_detect_macos_apple_silicon(self, mock_machine, mock_system):
        """Detecta macOS com Apple Silicon (M1/M2/M3)"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"

        result = detect_platform()

        assert result == Platform.MACOS_APPLE_SILICON

    @patch("platform.system")
    @patch("platform.machine")
    def test_detect_macos_intel(self, mock_machine, mock_system):
        """Detecta macOS com Intel"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"

        result = detect_platform()

        assert result == Platform.MACOS_INTEL

    @patch("platform.system")
    @patch("server.utils.platform.has_nvidia_gpu")
    def test_detect_linux_cuda(self, mock_has_gpu, mock_system):
        """Detecta Linux com CUDA GPU"""
        mock_system.return_value = "Linux"
        mock_has_gpu.return_value = True

        result = detect_platform()

        assert result == Platform.LINUX_CUDA

    @patch("platform.system")
    @patch("server.utils.platform.has_nvidia_gpu")
    def test_detect_linux_cpu(self, mock_has_gpu, mock_system):
        """Detecta Linux sem GPU (CPU only)"""
        mock_system.return_value = "Linux"
        mock_has_gpu.return_value = False

        result = detect_platform()

        assert result == Platform.LINUX_CPU

    @patch("platform.system")
    @patch("server.utils.platform.has_nvidia_gpu")
    def test_detect_windows_cuda(self, mock_has_gpu, mock_system):
        """Detecta Windows com CUDA GPU"""
        mock_system.return_value = "Windows"
        mock_has_gpu.return_value = True

        result = detect_platform()

        assert result == Platform.WINDOWS_CUDA

    @patch("platform.system")
    @patch("server.utils.platform.has_nvidia_gpu")
    def test_detect_windows_cpu_raises_error(self, mock_has_gpu, mock_system):
        """Windows sem GPU não é suportado"""
        mock_system.return_value = "Windows"
        mock_has_gpu.return_value = False

        with pytest.raises(PlatformNotSupportedError) as exc_info:
            detect_platform()

        assert "Windows CPU not supported" in str(exc_info.value)

    @patch("platform.system")
    def test_detect_unknown_platform_raises_error(self, mock_system):
        """Plataforma desconhecida lança erro"""
        mock_system.return_value = "FreeBSD"

        with pytest.raises(PlatformNotSupportedError) as exc_info:
            detect_platform()

        assert "Unknown platform" in str(exc_info.value)


class TestHasNvidiaGPU:
    """Tests para has_nvidia_gpu()"""

    @patch("torch.cuda.is_available")
    def test_has_gpu_via_torch(self, mock_torch_cuda):
        """Detecta GPU via torch.cuda.is_available()"""
        mock_torch_cuda.return_value = True

        result = has_nvidia_gpu()

        assert result is True
        mock_torch_cuda.assert_called_once()

    @patch("torch.cuda.is_available")
    @patch("subprocess.run")
    def test_has_gpu_via_nvidia_smi(self, mock_run, mock_torch_cuda):
        """Fallback para nvidia-smi se torch não disponível"""
        # Torch não instalado ou falha
        mock_torch_cuda.side_effect = ImportError("No module named 'torch'")

        # nvidia-smi retorna sucesso
        mock_run.return_value = MagicMock(returncode=0)

        result = has_nvidia_gpu()

        assert result is True
        mock_run.assert_called_once()
        assert "nvidia-smi" in mock_run.call_args[0][0]

    @patch("torch.cuda.is_available")
    @patch("subprocess.run")
    def test_no_gpu_nvidia_smi_fails(self, mock_run, mock_torch_cuda):
        """Sem GPU quando nvidia-smi falha"""
        mock_torch_cuda.side_effect = ImportError("No module named 'torch'")
        mock_run.return_value = MagicMock(returncode=1)

        result = has_nvidia_gpu()

        assert result is False

    @patch("torch.cuda.is_available")
    @patch("subprocess.run")
    def test_no_gpu_nvidia_smi_not_found(self, mock_run, mock_torch_cuda):
        """Sem GPU quando nvidia-smi não existe"""
        mock_torch_cuda.side_effect = ImportError("No module named 'torch'")
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        result = has_nvidia_gpu()

        assert result is False

    @patch("torch.cuda.is_available")
    @patch("subprocess.run")
    def test_no_gpu_nvidia_smi_timeout(self, mock_run, mock_torch_cuda):
        """Sem GPU quando nvidia-smi timeout"""
        mock_torch_cuda.side_effect = ImportError("No module named 'torch'")
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)

        result = has_nvidia_gpu()

        assert result is False


class TestPlatformNotSupportedError:
    """Tests para PlatformNotSupportedError"""

    def test_exception_creation(self):
        """Pode criar exceção com mensagem"""
        error = PlatformNotSupportedError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_exception_can_be_raised(self):
        """Pode levantar e capturar exceção"""
        with pytest.raises(PlatformNotSupportedError) as exc_info:
            raise PlatformNotSupportedError("Platform not supported")

        assert "Platform not supported" in str(exc_info.value)
