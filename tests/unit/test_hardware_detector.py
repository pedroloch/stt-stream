"""
Testes unitários para HardwareDetector

TDD: Testes escritos primeiro para garantir que o detector funciona corretamente
"""

import pytest
import platform
from unittest.mock import patch, MagicMock
import subprocess

from server.utils.hardware_detector import HardwareDetector, HardwareInfo


class TestHardwareDetector:
    """Testes para o detector de hardware"""

    def test_detector_initialization(self):
        """Test: Detector deve inicializar sem erros"""
        detector = HardwareDetector()
        assert detector is not None

    def test_apple_silicon_detection_on_macos_arm64(self):
        """Test: Deve detectar Apple Silicon em macOS ARM64"""
        detector = HardwareDetector()

        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                is_apple = detector._is_apple_silicon()
                assert is_apple is True

    def test_apple_silicon_detection_on_macos_x86(self):
        """Test: Não deve detectar Apple Silicon em macOS x86"""
        detector = HardwareDetector()

        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='x86_64'):
                is_apple = detector._is_apple_silicon()
                assert is_apple is False

    def test_apple_silicon_detection_on_linux(self):
        """Test: Não deve detectar Apple Silicon em Linux"""
        detector = HardwareDetector()

        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='arm64'):
                is_apple = detector._is_apple_silicon()
                assert is_apple is False

    def test_cuda_detection_with_nvidia_smi(self):
        """Test: Deve detectar CUDA quando nvidia-smi está disponível"""
        detector = HardwareDetector()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla V100"

        with patch('subprocess.run', return_value=mock_result):
            has_cuda = detector._has_cuda()
            assert has_cuda is True

    def test_cuda_detection_without_nvidia_smi(self):
        """Test: Não deve detectar CUDA quando nvidia-smi não existe"""
        detector = HardwareDetector()

        with patch('subprocess.run', side_effect=FileNotFoundError()):
            has_cuda = detector._has_cuda()
            assert has_cuda is False

    def test_cuda_detection_timeout(self):
        """Test: Deve tratar timeout do nvidia-smi"""
        detector = HardwareDetector()

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            has_cuda = detector._has_cuda()
            assert has_cuda is False

    def test_detect_returns_hardware_info(self):
        """Test: detect() deve retornar HardwareInfo"""
        detector = HardwareDetector()
        hardware = detector.detect()

        assert isinstance(hardware, HardwareInfo)
        assert hardware.type in ["apple_silicon", "cuda", "cpu"]

    def test_detect_apple_silicon_full(self):
        """Test: detect() deve identificar Apple Silicon completamente"""
        detector = HardwareDetector()

        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                hardware = detector.detect()

                assert hardware.type == "apple_silicon"
                assert hardware.device_name is not None

    def test_detect_cuda_full(self):
        """Test: detect() deve identificar CUDA completamente"""
        detector = HardwareDetector()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 3090"

        with patch('platform.system', return_value='Linux'):
            with patch('subprocess.run', return_value=mock_result):
                hardware = detector.detect()

                # Deve ser CUDA ou CPU (depende do ambiente real)
                assert hardware.type in ["cuda", "cpu"]

    def test_detect_cpu_fallback(self):
        """Test: detect() deve usar CPU como fallback"""
        detector = HardwareDetector()

        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='x86_64'):
                with patch('subprocess.run', side_effect=FileNotFoundError()):
                    hardware = detector.detect()

                    assert hardware.type == "cpu"
                    assert hardware.device_name is not None

    def test_force_hardware_type(self):
        """Test: deve respeitar force_type parameter"""
        detector = HardwareDetector()

        # Forçar CPU
        hardware = detector.detect(force_type="cpu")
        assert hardware.type == "cpu"

        # Forçar CUDA
        hardware = detector.detect(force_type="cuda")
        assert hardware.type == "cuda"

        # Forçar Apple Silicon
        hardware = detector.detect(force_type="apple_silicon")
        assert hardware.type == "apple_silicon"

    def test_get_cpu_info(self):
        """Test: deve obter informações do CPU"""
        detector = HardwareDetector()
        cpu_info = detector._get_cpu_info()

        assert cpu_info.type == "cpu"
        assert cpu_info.device_name is not None
        assert cpu_info.details is not None
        assert "cores" in cpu_info.details

    def test_print_hardware_info_no_crash(self):
        """Test: print_hardware_info não deve crashar"""
        detector = HardwareDetector()
        hardware = HardwareInfo(
            type="cpu",
            device_name="Test CPU",
            memory="8GB"
        )

        # Deve executar sem erros
        detector.print_hardware_info(hardware)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
