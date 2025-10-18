"""
Testes para VADChunker

Nota: Estes testes fazem mock do torch já que VAD requer PyTorch.
Em produção, o VAD só é usado se PyTorch estiver disponível.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

from server.streaming.vad import VADChunker
from server.constants import DEFAULT_SAMPLE_RATE


@pytest.fixture(autouse=True)
def mock_torch():
    """Mock torch module para todos os testes"""
    # Create mock torch
    mock = MagicMock()
    mock.hub.load.return_value = (MagicMock(), [MagicMock()])
    mock.from_numpy.return_value = MagicMock()

    # Inject into sys.modules
    old_torch = sys.modules.get('torch')
    sys.modules['torch'] = mock

    yield mock

    # Cleanup
    if old_torch is None:
        sys.modules.pop('torch', None)
    else:
        sys.modules['torch'] = old_torch


class TestVADChunker:
    """Testes para VADChunker"""

    @pytest.fixture
    def vad_chunker(self):
        """Fixture de VADChunker"""
        return VADChunker(
            threshold=0.5,
            min_silence_duration=0.3,
            min_speech_duration=0.5,
        )

    def test_init_default_params(self):
        """Test: inicialização com parâmetros padrão"""
        chunker = VADChunker()

        assert chunker.threshold == 0.5
        assert chunker.min_silence_duration == 0.3
        assert chunker.min_speech_duration == 0.5
        assert chunker._model is None  # Lazy loading

    def test_init_custom_params(self):
        """Test: inicialização com parâmetros customizados"""
        chunker = VADChunker(
            threshold=0.7,
            min_silence_duration=0.5,
            min_speech_duration=1.0,
        )

        assert chunker.threshold == 0.7
        assert chunker.min_silence_duration == 0.5
        assert chunker.min_speech_duration == 1.0

    def test_load_model_success(self, vad_chunker, mock_torch):
        """Test: carregar modelo Silero VAD com sucesso"""
        # Load model
        vad_chunker._load_model()

        # Verificar que foi carregado
        assert vad_chunker._model is not None
        assert vad_chunker._utils is not None
        mock_torch.hub.load.assert_called_once()

    def test_load_model_no_torch(self, vad_chunker):
        """Test: erro se PyTorch não disponível"""
        # Remove torch from sys.modules
        torch_backup = sys.modules.get('torch')
        if 'torch' in sys.modules:
            del sys.modules['torch']

        try:
            with pytest.raises(ImportError):
                vad_chunker._load_model()
        finally:
            # Restore
            if torch_backup:
                sys.modules['torch'] = torch_backup

    def test_detect_speech_returns_segments(self, vad_chunker, mock_torch):
        """Test: detect_speech retorna segmentos de fala"""
        # Mock get_speech_timestamps to return speech segments
        mock_get_speech_ts = MagicMock(
            return_value=[
                {"start": 0, "end": 8000},  # 0.5s
                {"start": 12000, "end": 20000},  # 0.75s - 1.25s
            ]
        )
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        # Áudio de 2 segundos
        audio = np.random.randn(2 * DEFAULT_SAMPLE_RATE).astype(np.float32)

        # Detect speech
        segments = vad_chunker.detect_speech(audio)

        # Verificar segmentos
        assert len(segments) == 2
        assert segments[0] == (0.0, 0.5)  # 8000 / 16000
        assert segments[1] == (0.75, 1.25)  # 12000/16000, 20000/16000

    def test_detect_speech_empty_audio(self, vad_chunker, mock_torch):
        """Test: detect_speech com áudio sem fala"""
        # Mock returns empty list (no speech)
        mock_get_speech_ts = MagicMock(return_value=[])
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        # Áudio de 1 segundo
        audio = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)

        # Detect speech
        segments = vad_chunker.detect_speech(audio)

        # Não deve encontrar nada
        assert len(segments) == 0

    def test_find_best_split_point_with_speech(self, vad_chunker, mock_torch):
        """Test: encontrar melhor ponto de corte com fala detectada"""
        # Mock speech segments
        mock_get_speech_ts = MagicMock(
            return_value=[
                {"start": 0, "end": 32000},  # 2s de fala
                {"start": 40000, "end": 64000},  # 4s de fala (2.5s - 4s)
            ]
        )
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        # Áudio de 5 segundos
        audio = np.random.randn(5 * DEFAULT_SAMPLE_RATE).astype(np.float32)

        # Target: 3 segundos (48000 samples)
        # Segmentos terminam em 2s (32000) e 4s (64000)
        # Mais próximo de 3s é 2s (diff=1s) vs 4s (diff=1s)
        # Deve escolher o primeiro: 2s (32000 samples)
        split_point = vad_chunker.find_best_split_point(audio, target_duration=3.0)

        # Deve cortar no fim do primeiro segmento (2s)
        assert split_point == 32000

    def test_find_best_split_point_no_speech(self, vad_chunker, mock_torch):
        """Test: sem fala detectada, usar target padrão"""
        # Mock no speech
        mock_get_speech_ts = MagicMock(return_value=[])
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        # Áudio de 5 segundos
        audio = np.zeros(5 * DEFAULT_SAMPLE_RATE, dtype=np.float32)

        # Target: 3 segundos
        split_point = vad_chunker.find_best_split_point(audio, target_duration=3.0)

        # Deve usar target padrão
        assert split_point == 3 * DEFAULT_SAMPLE_RATE

    def test_find_best_split_point_custom_sample_rate(self, vad_chunker, mock_torch):
        """Test: find_best_split_point com sample rate customizado"""
        # Mock speech segment
        mock_get_speech_ts = MagicMock(
            return_value=[
                {"start": 0, "end": 22050},  # 1s @ 22050 Hz
            ]
        )
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        # Áudio @ 22050 Hz
        sample_rate = 22050
        audio = np.random.randn(sample_rate).astype(np.float32)

        split_point = vad_chunker.find_best_split_point(
            audio, target_duration=1.0, sample_rate=sample_rate
        )

        # Deve cortar em 1s
        assert split_point == 22050

    def test_has_speech_true(self, vad_chunker, mock_torch):
        """Test: has_speech retorna True quando há fala"""
        # Mock with speech
        mock_get_speech_ts = MagicMock(
            return_value=[{"start": 0, "end": DEFAULT_SAMPLE_RATE}]
        )
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        audio = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)

        assert vad_chunker.has_speech(audio) is True

    def test_has_speech_false(self, vad_chunker, mock_torch):
        """Test: has_speech retorna False quando não há fala"""
        # Mock no speech
        mock_get_speech_ts = MagicMock(return_value=[])
        mock_torch.hub.load.return_value = (MagicMock(), [mock_get_speech_ts])

        audio = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)

        assert vad_chunker.has_speech(audio) is False

    def test_lazy_loading_model(self, vad_chunker, mock_torch):
        """Test: modelo é carregado apenas quando necessário (lazy)"""
        # Modelo não deve estar carregado inicialmente
        assert vad_chunker._model is None

        # Primeira chamada carrega modelo
        audio = np.random.randn(DEFAULT_SAMPLE_RATE).astype(np.float32)
        vad_chunker.detect_speech(audio)

        assert vad_chunker._model is not None

        # Segunda chamada não recarrega
        mock_torch.hub.load.reset_mock()
        vad_chunker.detect_speech(audio)

        mock_torch.hub.load.assert_not_called()
