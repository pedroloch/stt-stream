"""
Testes de integração para MLX Backend

Estes testes verificam que o backend MLX funciona corretamente em Apple Silicon.
Skip automaticamente se não estiver em Apple Silicon ou mlx-whisper não estiver instalado.
"""

import pytest
import platform
import numpy as np

# Skip se não for Apple Silicon
pytestmark = pytest.mark.skipif(
    not (platform.system() == "Darwin" and platform.machine() == "arm64"),
    reason="MLX backend só funciona em Apple Silicon"
)


def test_mlx_import():
    """Test: mlx-whisper deve estar instalado"""
    try:
        import mlx_whisper
        assert mlx_whisper is not None
    except ImportError:
        pytest.skip("mlx-whisper não instalado. Instale com: uv pip install mlx-whisper")


def test_mlx_backend_creation():
    """Test: MLX backend deve ser criado com sucesso"""
    from server.backends.mlx_backend import MLXBackend

    backend = MLXBackend(model="tiny", language="pt")

    assert backend is not None
    assert backend.model == "tiny"
    assert backend.language == "pt"


def test_mlx_backend_info():
    """Test: MLX backend deve retornar informações corretas"""
    from server.backends.mlx_backend import MLXBackend

    backend = MLXBackend(model="tiny")
    info = backend.get_backend_info()

    assert info["name"] == "MLX Backend (Apple Silicon)"
    assert info["model"] == "tiny"
    assert info["device"] == "Apple Silicon (Unified Memory)"
    assert "chip" in info  # Deve incluir info do chip (M1/M2/M3)


def test_factory_creates_mlx_on_apple_silicon():
    """Test: Factory deve criar MLX backend em Apple Silicon"""
    from server.backends.factory import BackendFactory
    import logging

    factory = BackendFactory(logging.getLogger("test"))
    backend = factory.create_backend(backend_type="mlx", model="tiny")

    assert backend is not None
    assert backend.model == "tiny"


@pytest.mark.asyncio
@pytest.mark.slow  # Marca como slow pois vai baixar modelo
async def test_mlx_backend_initialize():
    """
    Test: MLX backend deve inicializar (baixa modelo se necessário)

    NOTA: Este teste pode demorar na primeira vez pois baixa o modelo.
    Marque com @pytest.mark.slow e rode apenas quando necessário.
    """
    from server.backends.mlx_backend import MLXBackend

    try:
        import mlx_whisper
    except ImportError:
        pytest.skip("mlx-whisper não instalado")

    backend = MLXBackend(model="tiny", language="en")

    # Inicializar (pode baixar modelo)
    await backend.initialize()

    assert backend.is_initialized()

    # Cleanup
    await backend.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
