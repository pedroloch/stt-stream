"""
Backend Registry para Whisper Stream

Sistema de registro e criação de backends com validação de plataforma
e capabilities. Substitui o BackendFactory com uma API mais declarativa.
"""

from typing import Dict, Type, Optional
import logging

from .base import WhisperBackend
from ..models.capability import BackendInfo
from ..utils.platform import Platform, detect_platform, PlatformNotSupportedError


class BackendNotFoundError(Exception):
    """Exceção levantada quando backend não existe"""
    pass


class BackendRegistry:
    """
    Registry de backends com validação de plataforma

    Mantém registro de todos backends disponíveis e cria instâncias
    validando compatibilidade de plataforma (fail-fast).

    Example:
        >>> # Registrar backends
        >>> BackendRegistry.register("mlx", MLXBackend)
        >>> BackendRegistry.register("whisperx", WhisperXBackend)
        >>>
        >>> # Listar disponíveis
        >>> available = BackendRegistry.list_available()
        >>> print(available)  # {'mlx': BackendInfo(...)}
        >>>
        >>> # Criar backend
        >>> backend = BackendRegistry.create("mlx", {"model": "base"})
    """

    # Registry interno: {nome: classe}
    _backends: Dict[str, Type[WhisperBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[WhisperBackend]) -> None:
        """
        Registra um backend customizado

        Args:
            name: Nome do backend (ex: "mlx", "whisperx")
            backend_class: Classe do backend

        Example:
            >>> BackendRegistry.register("custom-backend", MyBackend)
        """
        cls._backends[name] = backend_class

    @classmethod
    def list_all(cls) -> Dict[str, BackendInfo]:
        """
        Lista todos backends registrados (mesmo incompatíveis)

        Returns:
            Dicionário {nome: BackendInfo}

        Example:
            >>> all_backends = BackendRegistry.list_all()
            >>> for name, info in all_backends.items():
            ...     print(f"{name}: {info.capabilities}")
        """
        result = {}
        for name, backend_class in cls._backends.items():
            # Instanciar temporariamente para pegar info
            # (backends devem ter __init__ sem argumentos obrigatórios ou info como class property)
            try:
                backend = backend_class()
                result[name] = backend.info
            except Exception:
                # Se falhar, criar dummy info
                result[name] = BackendInfo(
                    name=name,
                    supported_platforms=set(),
                    capabilities=set(),
                )
        return result

    @classmethod
    def list_available(cls, platform: Optional[Platform] = None) -> Dict[str, BackendInfo]:
        """
        Lista backends disponíveis para uma plataforma

        Args:
            platform: Plataforma (se None, detecta automaticamente)

        Returns:
            Dicionário {nome: BackendInfo} filtrado por plataforma

        Example:
            >>> # Listar para plataforma atual
            >>> available = BackendRegistry.list_available()
            >>>
            >>> # Listar para plataforma específica
            >>> cuda_backends = BackendRegistry.list_available(Platform.LINUX_CUDA)
        """
        if platform is None:
            platform = detect_platform()

        available = {}
        for name, backend_class in cls._backends.items():
            try:
                backend = backend_class()
                if platform in backend.info.supported_platforms:
                    available[name] = backend.info
            except Exception:
                # Skip backends que falham ao instanciar
                pass

        return available

    @classmethod
    def create(cls, name: str, config: dict) -> WhisperBackend:
        """
        Cria backend com validação de plataforma (fail-fast)

        Args:
            name: Nome do backend
            config: Configuração do backend

        Returns:
            Instância do backend

        Raises:
            BackendNotFoundError: Se backend não existe
            PlatformNotSupportedError: Se backend incompatível com plataforma

        Example:
            >>> try:
            ...     backend = BackendRegistry.create("whisperx", {"model": "large-v3"})
            ... except PlatformNotSupportedError as e:
            ...     print(f"Erro: {e}")
        """
        # Verificar se backend existe
        if name not in cls._backends:
            available_names = list(cls._backends.keys())
            raise BackendNotFoundError(
                f"Backend '{name}' not found.\n"
                f"Available backends: {', '.join(available_names)}"
            )

        # Criar instância
        backend_class = cls._backends[name]
        backend = backend_class()

        # Detectar plataforma atual
        current_platform = detect_platform()

        # Validar compatibilidade
        if current_platform not in backend.info.supported_platforms:
            # Gerar mensagem de erro útil com sugestões
            available_backends = cls.list_available(current_platform)

            suggestions = "\n".join(
                f"  - {name}: {info.name} ({', '.join(c.value for c in info.capabilities)})"
                for name, info in available_backends.items()
            )

            raise PlatformNotSupportedError(
                f"Backend '{name}' ({backend.info.name}) is not supported on {current_platform.value}.\n\n"
                f"Available backends for your platform:\n{suggestions}\n\n"
                f"💡 Suggestion: Use one of the backends above, or deploy on a different platform."
            )

        return backend


# Função utilitária para migração gradual do factory
def get_logger() -> logging.Logger:
    """Helper para obter logger"""
    return logging.getLogger(__name__)
