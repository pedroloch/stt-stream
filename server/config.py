"""
Configuration management para o servidor Whisper Stream

Carrega e valida configurações do arquivo YAML
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

from .models.enums import BackendType, LogLevel, LogFormat, ComputeType, DeviceType, BufferTrimming


@dataclass
class ServerConfig:
    """Configurações do servidor WebSocket"""
    host: str = "0.0.0.0"
    port: int = 9090
    max_clients: int = 5
    idle_timeout: int = 300
    cors_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class WhisperConfig:
    """Configurações do Whisper"""
    model: str = "base"
    language: str = "pt"
    backend: Union[BackendType, str] = BackendType.AUTO
    device: Union[DeviceType, str] = DeviceType.AUTO
    compute_type: Union[ComputeType, str] = ComputeType.FLOAT16
    use_vad: bool = True
    vad_threshold: float = 0.5
    min_chunk_size: float = 1.0
    buffer_trimming: Union[BufferTrimming, str] = BufferTrimming.SEGMENT
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = True


@dataclass
class PerformanceConfig:
    """Configurações de performance"""
    threads: str = "auto"  # "auto" ou número
    batch_size: int = 1
    prefer_gpu: bool = True


@dataclass
class PathsConfig:
    """Configurações de paths"""
    models_dir: str = "./models"
    cache_dir: str = "~/.cache/whisper-stream"


@dataclass
class LoggingConfig:
    """Configurações de logging"""
    level: Union[LogLevel, str] = LogLevel.INFO
    save_to_file: bool = True
    log_dir: str = "./logs"
    format: Union[LogFormat, str] = LogFormat.PRETTY
    log_audio_stats: bool = False


@dataclass
class HardwareConfig:
    """Configurações de hardware"""
    auto_detect: bool = True
    force_type: Optional[str] = None  # apple_silicon, cuda, cpu


@dataclass
class DebugConfig:
    """Configurações de debug"""
    verbose: bool = False
    save_received_audio: bool = False
    audio_output_dir: str = "./debug/audio"
    profile: bool = False
    profile_output_dir: str = "./debug/profiles"
    show_hardware_info: bool = True


@dataclass
class Config:
    """
    Configuração completa do servidor

    Exemplo de uso:
        config = Config.from_file("server-config.yaml")
        print(config.whisper.model)
    """
    server: ServerConfig = field(default_factory=ServerConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """
        Carrega configuração de arquivo YAML

        Args:
            config_path: Caminho para o arquivo de configuração

        Returns:
            Config object com configurações carregadas
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty config file: {config_path}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Cria Config a partir de dicionário

        Args:
            data: Dicionário com configurações

        Returns:
            Config object
        """
        # Converter strings para enums se necessário
        whisper_data = data.get("whisper", {})
        if "backend" in whisper_data and isinstance(whisper_data["backend"], str):
            try:
                whisper_data["backend"] = BackendType(whisper_data["backend"])
            except ValueError:
                pass  # Manter string se não for um enum válido

        if "compute_type" in whisper_data and isinstance(whisper_data["compute_type"], str):
            try:
                whisper_data["compute_type"] = ComputeType(whisper_data["compute_type"])
            except ValueError:
                pass

        if "device" in whisper_data and isinstance(whisper_data["device"], str):
            try:
                whisper_data["device"] = DeviceType(whisper_data["device"])
            except ValueError:
                pass

        if "buffer_trimming" in whisper_data and isinstance(whisper_data["buffer_trimming"], str):
            try:
                whisper_data["buffer_trimming"] = BufferTrimming(whisper_data["buffer_trimming"])
            except ValueError:
                pass

        logging_data = data.get("logging", {})
        if "level" in logging_data and isinstance(logging_data["level"], str):
            try:
                logging_data["level"] = LogLevel(logging_data["level"])
            except ValueError:
                pass

        if "format" in logging_data and isinstance(logging_data["format"], str):
            try:
                logging_data["format"] = LogFormat(logging_data["format"])
            except ValueError:
                pass

        return cls(
            server=ServerConfig(**data.get("server", {})),
            whisper=WhisperConfig(**whisper_data),
            performance=PerformanceConfig(**data.get("performance", {})),
            paths=PathsConfig(**data.get("paths", {})),
            logging=LoggingConfig(**logging_data),
            hardware=HardwareConfig(**data.get("hardware", {})),
            debug=DebugConfig(**data.get("debug", {})),
        )

    @classmethod
    def from_args(cls, **kwargs) -> "Config":
        """
        Cria Config a partir de argumentos de linha de comando

        Args:
            **kwargs: Argumentos para sobrescrever valores padrão

        Returns:
            Config object
        """
        config = cls()

        # Mapear argumentos flat para estrutura aninhada
        if "host" in kwargs:
            config.server.host = kwargs["host"]
        if "port" in kwargs:
            config.server.port = kwargs["port"]
        if "model" in kwargs:
            config.whisper.model = kwargs["model"]
        if "language" in kwargs:
            config.whisper.language = kwargs["language"]
        if "backend" in kwargs:
            # Tentar converter para enum
            try:
                config.whisper.backend = BackendType(kwargs["backend"])
            except ValueError:
                config.whisper.backend = kwargs["backend"]
        if "use_vad" in kwargs:
            config.whisper.use_vad = kwargs["use_vad"]
        if "log_level" in kwargs:
            # Tentar converter para enum
            try:
                config.logging.level = LogLevel(kwargs["log_level"])
            except ValueError:
                config.logging.level = kwargs["log_level"]
        if "verbose" in kwargs:
            config.debug.verbose = kwargs["verbose"]

        return config

    def get_threads(self) -> int:
        """
        Retorna número de threads a usar

        Returns:
            Número de threads (auto-detectado se "auto")
        """
        if self.performance.threads == "auto":
            import os
            return os.cpu_count() or 4
        return int(self.performance.threads)

    def expand_paths(self) -> None:
        """Expande ~ em paths para home directory"""
        self.paths.models_dir = str(Path(self.paths.models_dir).expanduser())
        self.paths.cache_dir = str(Path(self.paths.cache_dir).expanduser())
        self.logging.log_dir = str(Path(self.logging.log_dir).expanduser())
        if self.debug.save_received_audio:
            self.debug.audio_output_dir = str(Path(self.debug.audio_output_dir).expanduser())
        if self.debug.profile:
            self.debug.profile_output_dir = str(Path(self.debug.profile_output_dir).expanduser())
