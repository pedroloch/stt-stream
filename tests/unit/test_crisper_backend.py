"""
Tests para CrisperWhisperBackend

Testa o backend CrisperWhisper (Nyra Health) com verbatim transcription
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from server.backends.crisper_backend import CrisperWhisperBackend
from server.models.capability import Capability, Platform
from server.models.result import TranscriptionResult, Segment, Word


class TestCrisperBackendInfo:
    """Tests para backend info e metadata"""

    def test_backend_info(self):
        """Backend info deve declarar capabilities corretas"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            backend = CrisperWhisperBackend({"language": "en"})
            info = backend.info

            assert info.name == "crisper-whisper"
            assert Capability.VERBATIM in info.capabilities
            assert Capability.WORD_TIMESTAMPS in info.capabilities
            assert Capability.TRANSCRIPTION in info.capabilities
            assert Capability.STREAMING in info.capabilities

    def test_supported_platforms(self):
        """Backend deve suportar CUDA e Apple Silicon"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            backend = CrisperWhisperBackend({"language": "en"})
            info = backend.info

            assert Platform.LINUX_CUDA in info.supported_platforms
            assert Platform.MACOS_APPLE_SILICON in info.supported_platforms
            # Não suporta CPU puro (muito pesado)
            assert Platform.LINUX_CPU not in info.supported_platforms

    def test_supported_languages(self):
        """Backend deve suportar English e German"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            backend = CrisperWhisperBackend({"language": "en"})
            info = backend.info

            assert "en" in info.supported_languages
            assert "de" in info.supported_languages
            assert len(info.supported_languages) == 2

    def test_model_sizes(self):
        """Backend usa Whisper Large v3 fine-tuned"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            backend = CrisperWhisperBackend({"language": "en"})
            info = backend.info

            assert "large-v3" in info.model_sizes


class TestCrisperBackendInitialization:
    """Tests para inicialização do backend"""

    def test_import_error_when_not_installed(self):
        """Deve lançar ImportError se CrisperWhisper não instalado"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                CrisperWhisperBackend({"language": "en"})

            assert "CrisperWhisper não está instalado" in str(exc_info.value)
            assert "pip install git+https://github.com/nyrahealth/transformers.git" in str(exc_info.value)

    def test_language_validation(self):
        """Deve validar idioma suportado"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with pytest.raises(ValueError) as exc_info:
                CrisperWhisperBackend({"language": "pt"})

            assert "Idioma 'pt' não suportado" in str(exc_info.value)
            assert "en" in str(exc_info.value)
            assert "de" in str(exc_info.value)

    def test_platform_validation_on_unsupported_platform(self):
        """Deve fail-fast em plataforma não suportada"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.LINUX_CPU

                with pytest.raises(Exception) as exc_info:  # PlatformNotSupportedError
                    CrisperWhisperBackend({"language": "en"})

                error_msg = str(exc_info.value)
                assert "CrisperWhisper" in error_msg
                assert "não suporta" in error_msg or "not support" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Deve inicializar corretamente com plataforma suportada"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                with patch('server.backends.crisper_backend.pipeline') as mock_pipeline:
                    mock_pipeline.return_value = Mock()

                    backend = CrisperWhisperBackend({"language": "en"})
                    await backend.initialize()

                    assert backend.is_initialized()
                    assert backend.model is not None


class TestCrisperFillerDetection:
    """Tests para detecção de fillers"""

    def test_is_filler_word_english(self):
        """Deve detectar fillers em inglês"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON
                backend = CrisperWhisperBackend({"language": "en"})

                assert backend._is_filler_word("um") is True
                assert backend._is_filler_word("uh") is True
                assert backend._is_filler_word("er") is True
                assert backend._is_filler_word("ah") is True
                assert backend._is_filler_word("like") is True
                assert backend._is_filler_word("you know") is True

    def test_is_filler_word_german(self):
        """Deve detectar fillers em alemão"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON
                backend = CrisperWhisperBackend({"language": "de"})

                assert backend._is_filler_word("äh") is True
                assert backend._is_filler_word("ähm") is True
                assert backend._is_filler_word("öh") is True
                assert backend._is_filler_word("ehm") is True

    def test_is_not_filler_word(self):
        """Não deve marcar palavras normais como fillers"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON
                backend = CrisperWhisperBackend({"language": "en"})

                assert backend._is_filler_word("hello") is False
                assert backend._is_filler_word("world") is False
                assert backend._is_filler_word("test") is False

    def test_filler_detection_case_insensitive(self):
        """Detecção de fillers deve ser case-insensitive"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON
                backend = CrisperWhisperBackend({"language": "en"})

                assert backend._is_filler_word("UM") is True
                assert backend._is_filler_word("Um") is True
                assert backend._is_filler_word("uM") is True

    def test_filler_detection_with_punctuation(self):
        """Deve remover pontuação antes de verificar"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON
                backend = CrisperWhisperBackend({"language": "en"})

                assert backend._is_filler_word("um,") is True
                assert backend._is_filler_word("uh.") is True
                assert backend._is_filler_word("er!") is True


class TestCrisperTranscription:
    """Tests para transcrição com CrisperWhisper"""

    @pytest.mark.asyncio
    async def test_transcribe_chunk_with_fillers(self):
        """Deve transcrever e marcar fillers corretamente"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                with patch('server.backends.crisper_backend.pipeline') as mock_pipeline:
                    # Mock do resultado do CrisperWhisper
                    mock_model = Mock()
                    mock_model.return_value = {
                        "text": "So um I uh think",
                        "chunks": [
                            {"text": "So", "timestamp": (0.0, 0.3)},
                            {"text": "um", "timestamp": (0.4, 0.6)},
                            {"text": "I", "timestamp": (0.7, 0.9)},
                            {"text": "uh", "timestamp": (1.0, 1.2)},
                            {"text": "think", "timestamp": (1.3, 1.8)},
                        ],
                    }
                    mock_pipeline.return_value = mock_model

                    backend = CrisperWhisperBackend({"language": "en"})
                    await backend.initialize()

                    # Transcrever
                    audio = np.random.randn(16000).astype(np.float32)
                    result = await backend.transcribe_chunk(audio)

                    # Verificar resultado
                    assert isinstance(result, TranscriptionResult)
                    assert result.text == "So um I uh think"
                    assert result.is_final is True
                    assert result.language == "en"
                    assert result.model_name == "crisper-whisper"

                    # Verificar segments
                    assert len(result.segments) == 1
                    segment = result.segments[0]
                    assert len(segment.words) == 5

                    # Verificar que fillers foram marcados
                    assert segment.words[0].word == "So"
                    assert segment.words[0].is_filler is False

                    assert segment.words[1].word == "um"
                    assert segment.words[1].is_filler is True  # ⭐

                    assert segment.words[2].word == "I"
                    assert segment.words[2].is_filler is False

                    assert segment.words[3].word == "uh"
                    assert segment.words[3].is_filler is True  # ⭐

                    assert segment.words[4].word == "think"
                    assert segment.words[4].is_filler is False

    @pytest.mark.asyncio
    async def test_transcribe_chunk_includes_processing_time(self):
        """Deve incluir processing_time_ms no resultado"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                with patch('server.backends.crisper_backend.pipeline') as mock_pipeline:
                    mock_model = Mock()
                    mock_model.return_value = {
                        "text": "Test",
                        "chunks": [
                            {"text": "Test", "timestamp": (0.0, 0.5)},
                        ],
                    }
                    mock_pipeline.return_value = mock_model

                    backend = CrisperWhisperBackend({"language": "en"})
                    await backend.initialize()

                    audio = np.random.randn(16000).astype(np.float32)
                    result = await backend.transcribe_chunk(audio)

                    assert result.processing_time_ms is not None
                    assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_transcribe_chunk_without_initialized(self):
        """Deve lançar erro se backend não inicializado"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                backend = CrisperWhisperBackend({"language": "en"})

                audio = np.random.randn(16000).astype(np.float32)

                with pytest.raises(RuntimeError) as exc_info:
                    await backend.transcribe_chunk(audio)

                assert "não inicializado" in str(exc_info.value).lower()


class TestCrisperBackendInfo:
    """Tests para get_backend_info()"""

    @pytest.mark.asyncio
    async def test_get_backend_info_after_init(self):
        """Deve retornar informações corretas após inicialização"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                with patch('server.backends.crisper_backend.pipeline') as mock_pipeline:
                    mock_pipeline.return_value = Mock()

                    backend = CrisperWhisperBackend({"language": "en", "device": "mps"})
                    await backend.initialize()

                    info = backend.get_backend_info()

                    assert info["name"] == "crisper-whisper"
                    assert info["model"] == "nyrahealth/CrisperWhisper"
                    assert info["device"] == "mps"
                    assert info["language"] == "en"
                    assert info["initialized"] is True


class TestCrisperCleanup:
    """Tests para cleanup de recursos"""

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Deve limpar recursos corretamente"""
        with patch('server.backends.crisper_backend.CRISPER_AVAILABLE', True):
            with patch('server.backends.crisper_backend.detect_platform') as mock_detect:
                mock_detect.return_value = Platform.MACOS_APPLE_SILICON

                with patch('server.backends.crisper_backend.pipeline') as mock_pipeline:
                    mock_pipeline.return_value = Mock()

                    backend = CrisperWhisperBackend({"language": "en"})
                    await backend.initialize()

                    assert backend.is_initialized() is True

                    await backend.cleanup()

                    assert backend.is_initialized() is False
                    assert backend.model is None
