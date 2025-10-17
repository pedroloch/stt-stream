"""
Testes unitários para Config

TDD: Testar carregamento e validação de configurações
"""

import pytest
import tempfile
from pathlib import Path

from server.config import (
    Config,
    ServerConfig,
    WhisperConfig,
    PerformanceConfig,
    PathsConfig,
    LoggingConfig,
    HardwareConfig,
    DebugConfig
)


class TestServerConfig:
    """Testes para ServerConfig"""

    def test_default_values(self):
        """Test: ServerConfig deve ter valores padrão sensatos"""
        config = ServerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.max_clients == 5
        assert config.idle_timeout == 300
        assert config.cors_enabled is True


class TestWhisperConfig:
    """Testes para WhisperConfig"""

    def test_default_values(self):
        """Test: WhisperConfig deve ter defaults corretos"""
        config = WhisperConfig()

        assert config.model == "base"
        assert config.language == "pt"
        assert config.backend == "auto"
        assert config.use_vad is True

    def test_custom_values(self):
        """Test: WhisperConfig deve aceitar valores customizados"""
        config = WhisperConfig(
            model="large",
            language="en",
            backend="cuda"
        )

        assert config.model == "large"
        assert config.language == "en"
        assert config.backend == "cuda"


class TestConfig:
    """Testes para Config principal"""

    def test_default_initialization(self):
        """Test: Config deve inicializar com defaults"""
        config = Config()

        assert config.server is not None
        assert config.whisper is not None
        assert config.performance is not None
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.whisper, WhisperConfig)

    def test_from_dict(self):
        """Test: Config deve carregar de dicionário"""
        data = {
            "server": {
                "host": "127.0.0.1",
                "port": 8080
            },
            "whisper": {
                "model": "small",
                "language": "en"
            }
        }

        config = Config.from_dict(data)

        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8080
        assert config.whisper.model == "small"
        assert config.whisper.language == "en"

    def test_from_dict_partial(self):
        """Test: Config deve usar defaults para campos não especificados"""
        data = {
            "whisper": {
                "model": "large"
            }
        }

        config = Config.from_dict(data)

        # Deve ter o valor especificado
        assert config.whisper.model == "large"

        # Deve ter defaults para o resto
        assert config.server.port == 9090
        assert config.whisper.language == "pt"

    def test_from_file(self):
        """Test: Config deve carregar de arquivo YAML"""
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
server:
  host: "localhost"
  port: 7070

whisper:
  model: "tiny"
  language: "es"
""")
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)

            assert config.server.host == "localhost"
            assert config.server.port == 7070
            assert config.whisper.model == "tiny"
            assert config.whisper.language == "es"
        finally:
            # Cleanup
            Path(temp_path).unlink()

    def test_from_file_not_found(self):
        """Test: Config deve lançar erro se arquivo não existe"""
        with pytest.raises(FileNotFoundError):
            Config.from_file("/path/that/does/not/exist.yaml")

    def test_from_file_empty(self):
        """Test: Config deve lançar erro se arquivo vazio"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Arquivo vazio
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Empty config file"):
                Config.from_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_args(self):
        """Test: Config deve aceitar argumentos de linha de comando"""
        config = Config.from_args(
            host="192.168.1.1",
            port=9999,
            model="medium",
            language="fr",
            backend="cpu"
        )

        assert config.server.host == "192.168.1.1"
        assert config.server.port == 9999
        assert config.whisper.model == "medium"
        assert config.whisper.language == "fr"
        assert config.whisper.backend == "cpu"

    def test_get_threads_auto(self):
        """Test: get_threads() deve detectar CPU cores quando auto"""
        config = Config()
        config.performance.threads = "auto"

        threads = config.get_threads()

        assert isinstance(threads, int)
        assert threads > 0

    def test_get_threads_specific(self):
        """Test: get_threads() deve retornar valor especificado"""
        config = Config()
        config.performance.threads = "8"

        threads = config.get_threads()

        assert threads == 8

    def test_expand_paths(self):
        """Test: expand_paths() deve expandir ~ em paths"""
        config = Config()
        config.paths.models_dir = "~/models"
        config.paths.cache_dir = "~/.cache/test"

        config.expand_paths()

        assert "~" not in config.paths.models_dir
        assert "~" not in config.paths.cache_dir
        assert Path(config.paths.models_dir).is_absolute()


class TestPerformanceConfig:
    """Testes para PerformanceConfig"""

    def test_default_values(self):
        """Test: PerformanceConfig defaults"""
        config = PerformanceConfig()

        assert config.threads == "auto"
        assert config.batch_size == 1
        assert config.prefer_gpu is True


class TestLoggingConfig:
    """Testes para LoggingConfig"""

    def test_default_values(self):
        """Test: LoggingConfig defaults"""
        config = LoggingConfig()

        assert config.level == "info"
        assert config.save_to_file is True
        assert config.format == "pretty"


class TestHardwareConfig:
    """Testes para HardwareConfig"""

    def test_default_values(self):
        """Test: HardwareConfig defaults"""
        config = HardwareConfig()

        assert config.auto_detect is True
        assert config.force_type is None


class TestDebugConfig:
    """Testes para DebugConfig"""

    def test_default_values(self):
        """Test: DebugConfig defaults"""
        config = DebugConfig()

        assert config.verbose is False
        assert config.save_received_audio is False
        assert config.show_hardware_info is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
