"""
Tests para BackendRegistry

Testa o sistema de registro e criação de backends com validação
de plataforma e capabilities.
"""

import pytest
from unittest.mock import MagicMock, patch

from server.backends.registry import BackendRegistry, BackendNotFoundError
from server.models.capability import Capability, BackendInfo
from server.utils.platform import Platform, PlatformNotSupportedError


# Mock backends para testing
class MockMLXBackend:
    """Mock MLX backend para testes"""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock-mlx",
            supported_platforms={Platform.MACOS_APPLE_SILICON},
            capabilities={Capability.TRANSCRIPTION, Capability.STREAMING},
        )

    async def initialize(self, config: dict):
        pass


class MockWhisperXBackend:
    """Mock WhisperX backend para testes"""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock-whisperx",
            supported_platforms={Platform.LINUX_CUDA},
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.WORD_TIMESTAMPS,
                Capability.SPEAKER_DIARIZATION,
            },
        )

    async def initialize(self, config: dict):
        pass


class MockUniversalBackend:
    """Mock backend que funciona em todas plataformas"""

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mock-universal",
            supported_platforms={
                Platform.MACOS_APPLE_SILICON,
                Platform.MACOS_INTEL,
                Platform.LINUX_CUDA,
                Platform.LINUX_CPU,
            },
            capabilities={Capability.TRANSCRIPTION},
        )

    async def initialize(self, config: dict):
        pass


@pytest.fixture
def clean_registry():
    """Limpa o registry antes de cada teste"""
    # Salvar backends originais
    original_backends = BackendRegistry._backends.copy()

    # Limpar registry
    BackendRegistry._backends.clear()

    yield BackendRegistry

    # Restaurar backends originais
    BackendRegistry._backends = original_backends


class TestBackendRegistry:
    """Tests para BackendRegistry"""

    def test_register_backend(self, clean_registry):
        """Pode registrar um backend customizado"""
        clean_registry.register("mock-mlx", MockMLXBackend)

        assert "mock-mlx" in clean_registry._backends
        assert clean_registry._backends["mock-mlx"] == MockMLXBackend

    def test_list_all_backends(self, clean_registry):
        """Lista todos backends registrados"""
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)

        all_backends = clean_registry.list_all()

        assert len(all_backends) == 2
        assert "mock-mlx" in all_backends
        assert "mock-whisperx" in all_backends
        assert all_backends["mock-mlx"].name == "mock-mlx"
        assert all_backends["mock-whisperx"].name == "mock-whisperx"

    def test_list_available_on_macos(self, clean_registry):
        """Lista apenas backends disponíveis no macOS"""
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)
        clean_registry.register("mock-universal", MockUniversalBackend)

        available = clean_registry.list_available(Platform.MACOS_APPLE_SILICON)

        # mock-mlx e mock-universal devem estar disponíveis
        # mock-whisperx NÃO deve estar (requer CUDA)
        assert "mock-mlx" in available
        assert "mock-universal" in available
        assert "mock-whisperx" not in available

    def test_list_available_on_linux_cuda(self, clean_registry):
        """Lista backends disponíveis no Linux com CUDA"""
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)
        clean_registry.register("mock-universal", MockUniversalBackend)

        available = clean_registry.list_available(Platform.LINUX_CUDA)

        # mock-whisperx e mock-universal devem estar disponíveis
        # mock-mlx NÃO deve estar (requer Apple Silicon)
        assert "mock-whisperx" in available
        assert "mock-universal" in available
        assert "mock-mlx" not in available

    @patch("server.backends.registry.detect_platform")
    def test_list_available_auto_detect(self, mock_detect, clean_registry):
        """Lista backends detectando plataforma automaticamente"""
        mock_detect.return_value = Platform.MACOS_APPLE_SILICON
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)

        # Não passar plataforma - deve detectar automaticamente
        available = clean_registry.list_available()

        assert "mock-mlx" in available
        assert "mock-whisperx" not in available
        mock_detect.assert_called_once()

    @patch("server.backends.registry.detect_platform")
    def test_create_backend_success(self, mock_detect, clean_registry):
        """Cria backend com sucesso quando plataforma compatível"""
        mock_detect.return_value = Platform.MACOS_APPLE_SILICON
        clean_registry.register("mock-mlx", MockMLXBackend)

        backend = clean_registry.create("mock-mlx", {})

        assert isinstance(backend, MockMLXBackend)
        assert backend.info.name == "mock-mlx"

    @patch("server.backends.registry.detect_platform")
    def test_create_backend_platform_mismatch(self, mock_detect, clean_registry):
        """Erro ao criar backend incompatível com plataforma"""
        mock_detect.return_value = Platform.MACOS_APPLE_SILICON
        clean_registry.register("mock-whisperx", MockWhisperXBackend)

        with pytest.raises(PlatformNotSupportedError) as exc_info:
            clean_registry.create("mock-whisperx", {})

        # Verificar mensagem de erro
        error_msg = str(exc_info.value)
        assert "mock-whisperx" in error_msg
        assert "not supported" in error_msg
        assert "macos_arm64" in error_msg

    @patch("server.backends.registry.detect_platform")
    def test_error_message_has_suggestions(self, mock_detect, clean_registry):
        """Mensagem de erro lista backends disponíveis"""
        mock_detect.return_value = Platform.MACOS_APPLE_SILICON
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)

        with pytest.raises(PlatformNotSupportedError) as exc_info:
            clean_registry.create("mock-whisperx", {})

        error_msg = str(exc_info.value)
        # Deve sugerir mock-mlx como alternativa
        assert "Available backends" in error_msg
        assert "mock-mlx" in error_msg

    def test_backend_not_found(self, clean_registry):
        """Erro quando backend não existe"""
        with pytest.raises(BackendNotFoundError) as exc_info:
            clean_registry.create("nonexistent-backend", {})

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "nonexistent-backend" in error_msg

    def test_backend_not_found_shows_available(self, clean_registry):
        """Erro de backend não encontrado lista backends disponíveis"""
        clean_registry.register("mock-mlx", MockMLXBackend)
        clean_registry.register("mock-whisperx", MockWhisperXBackend)

        with pytest.raises(BackendNotFoundError) as exc_info:
            clean_registry.create("typo-backend", {})

        error_msg = str(exc_info.value)
        # Deve listar backends disponíveis
        assert "mock-mlx" in error_msg
        assert "mock-whisperx" in error_msg


class TestBackendRegistryIntegration:
    """Tests de integração para BackendRegistry"""

    @patch("server.backends.registry.detect_platform")
    def test_registry_workflow(self, mock_detect, clean_registry):
        """Workflow completo de registro, listagem e criação"""
        mock_detect.return_value = Platform.MACOS_APPLE_SILICON

        # 1. Registrar backends
        clean_registry.register("backend-a", MockMLXBackend)
        clean_registry.register("backend-b", MockWhisperXBackend)
        clean_registry.register("backend-c", MockUniversalBackend)

        # 2. Listar todos
        all_backends = clean_registry.list_all()
        assert len(all_backends) == 3

        # 3. Listar disponíveis para plataforma atual
        available = clean_registry.list_available()
        assert len(available) == 2  # backend-a e backend-c

        # 4. Criar backend compatível
        backend = clean_registry.create("backend-a", {})
        assert backend.info.name == "mock-mlx"

        # 5. Tentar criar incompatível → erro
        with pytest.raises(PlatformNotSupportedError):
            clean_registry.create("backend-b", {})
