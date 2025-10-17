"""
Tests para models (TranscriptionResult, Word, Segment, Capability, BackendInfo)

Testa as dataclasses e enums usados para resultados normalizados
"""

import pytest
from datetime import datetime

from server.models.capability import Capability, BackendInfo
from server.models.result import Word, Segment, TranscriptionResult
from server.utils.platform import Platform


class TestCapability:
    """Tests para Capability enum"""

    def test_capability_enum_values(self):
        """Enum Capability tem todos os valores esperados"""
        assert Capability.TRANSCRIPTION.value == "transcription"
        assert Capability.WORD_TIMESTAMPS.value == "word_timestamps"
        assert Capability.SPEAKER_DIARIZATION.value == "diarization"
        assert Capability.TRANSLATION.value == "translation"
        assert Capability.VAD.value == "vad"
        assert Capability.STREAMING.value == "streaming"


class TestBackendInfo:
    """Tests para BackendInfo dataclass"""

    def test_backend_info_minimal(self):
        """BackendInfo com campos mínimos"""
        info = BackendInfo(
            name="test-backend",
            supported_platforms={Platform.MACOS_APPLE_SILICON},
            capabilities={Capability.TRANSCRIPTION},
        )

        assert info.name == "test-backend"
        assert Platform.MACOS_APPLE_SILICON in info.supported_platforms
        assert Capability.TRANSCRIPTION in info.capabilities
        assert info.supported_languages is None
        assert info.model_sizes is None

    def test_backend_info_full(self):
        """BackendInfo com todos os campos"""
        info = BackendInfo(
            name="whisperx",
            supported_platforms={Platform.LINUX_CUDA},
            capabilities={
                Capability.TRANSCRIPTION,
                Capability.WORD_TIMESTAMPS,
                Capability.SPEAKER_DIARIZATION,
            },
            supported_languages={"pt", "en", "es"},
            model_sizes={"tiny", "base", "small", "medium", "large-v3"},
        )

        assert info.name == "whisperx"
        assert len(info.capabilities) == 3
        assert Capability.SPEAKER_DIARIZATION in info.capabilities
        assert "pt" in info.supported_languages
        assert "large-v3" in info.model_sizes


class TestWord:
    """Tests para Word dataclass"""

    def test_word_creation(self):
        """Criar Word com todos os campos"""
        word = Word(
            word="hello",
            start=1.5,
            end=2.0,
            probability=0.95,
        )

        assert word.word == "hello"
        assert word.start == 1.5
        assert word.end == 2.0
        assert word.probability == 0.95

    def test_word_duration(self):
        """Word tem duração correta"""
        word = Word(word="test", start=0.0, end=0.5, probability=1.0)

        duration = word.end - word.start
        assert duration == 0.5


class TestSegment:
    """Tests para Segment dataclass"""

    def test_segment_without_words(self):
        """Segment sem words (word timestamps opcionais)"""
        segment = Segment(
            start=0.0,
            end=5.0,
            text="Hello world",
        )

        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Hello world"
        assert segment.words is None

    def test_segment_with_words(self):
        """Segment com word-level timestamps"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.99),
            Word(word="world", start=0.6, end=1.2, probability=0.98),
        ]

        segment = Segment(
            start=0.0,
            end=1.2,
            text="Hello world",
            words=words,
        )

        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"
        assert segment.words[1].word == "world"

    def test_segment_duration(self):
        """Segment tem duração correta"""
        segment = Segment(start=1.0, end=5.5, text="Test segment")

        duration = segment.end - segment.start
        assert duration == 4.5


class TestTranscriptionResult:
    """Tests para TranscriptionResult dataclass"""

    def test_transcription_result_minimal(self):
        """TranscriptionResult com campos obrigatórios apenas"""
        result = TranscriptionResult(
            text="Hello world",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime(2025, 1, 17, 10, 30, 0),
        )

        assert result.text == "Hello world"
        assert result.is_final is True
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.segments is None
        assert result.speaker is None
        assert result.translation is None

    def test_transcription_result_with_segments(self):
        """TranscriptionResult com segments (word timestamps)"""
        segment = Segment(
            start=0.0,
            end=1.0,
            text="Hello",
            words=[
                Word(word="Hello", start=0.0, end=1.0, probability=0.99),
            ],
        )

        result = TranscriptionResult(
            text="Hello",
            is_final=True,
            confidence=0.99,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello"
        assert len(result.segments[0].words) == 1

    def test_transcription_result_with_speaker(self):
        """TranscriptionResult com speaker diarization"""
        result = TranscriptionResult(
            text="I agree with that",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            speaker="SPEAKER_01",
        )

        assert result.speaker == "SPEAKER_01"

    def test_transcription_result_with_translation(self):
        """TranscriptionResult com translation"""
        result = TranscriptionResult(
            text="Olá mundo",
            is_final=True,
            confidence=0.98,
            language="pt",
            timestamp=datetime.now(),
            translation={
                "en": "Hello world",
                "es": "Hola mundo",
                "fr": "Bonjour monde",
            },
        )

        assert len(result.translation) == 3
        assert result.translation["en"] == "Hello world"
        assert result.translation["es"] == "Hola mundo"

    def test_transcription_result_full(self):
        """TranscriptionResult com todos os campos"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.99),
            Word(word="world", start=0.6, end=1.2, probability=0.98),
        ]

        segment = Segment(
            start=0.0,
            end=1.2,
            text="Hello world",
            words=words,
        )

        result = TranscriptionResult(
            text="Hello world",
            is_final=True,
            confidence=0.97,
            language="en",
            timestamp=datetime(2025, 1, 17, 10, 30, 0),
            segments=[segment],
            speaker="SPEAKER_00",
            translation={"pt": "Olá mundo"},
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.speaker == "SPEAKER_00"
        assert result.translation["pt"] == "Olá mundo"


class TestTranscriptionResultSerialization:
    """Tests para serialização TranscriptionResult.to_websocket_dict()"""

    def test_to_websocket_dict_minimal(self):
        """Serializar resultado mínimo para WebSocket JSON"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime(2025, 1, 17, 10, 30, 0),
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["type"] == "transcription"
        assert ws_dict["text"] == "Test"
        assert ws_dict["is_final"] is True
        assert ws_dict["confidence"] == 0.95
        assert ws_dict["language"] == "en"
        assert ws_dict["timestamp"] == "2025-01-17T10:30:00"
        assert "segments" not in ws_dict
        assert "speaker" not in ws_dict
        assert "translations" not in ws_dict

    def test_to_websocket_dict_with_segments(self):
        """Serializar com segments e words"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.99),
            Word(word="world", start=0.6, end=1.2, probability=0.98),
        ]

        segment = Segment(start=0.0, end=1.2, text="Hello world", words=words)

        result = TranscriptionResult(
            text="Hello world",
            is_final=True,
            confidence=0.97,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        ws_dict = result.to_websocket_dict()

        assert "segments" in ws_dict
        assert len(ws_dict["segments"]) == 1
        assert ws_dict["segments"][0]["text"] == "Hello world"
        assert len(ws_dict["segments"][0]["words"]) == 2
        assert ws_dict["segments"][0]["words"][0]["word"] == "Hello"
        assert ws_dict["segments"][0]["words"][1]["word"] == "world"

    def test_to_websocket_dict_with_speaker(self):
        """Serializar com speaker"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            speaker="SPEAKER_02",
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["speaker"] == "SPEAKER_02"

    def test_to_websocket_dict_with_translations(self):
        """Serializar com translations"""
        result = TranscriptionResult(
            text="Olá",
            is_final=True,
            confidence=0.98,
            language="pt",
            timestamp=datetime.now(),
            translation={"en": "Hello", "es": "Hola"},
        )

        ws_dict = result.to_websocket_dict()

        assert "translations" in ws_dict
        assert ws_dict["translations"]["en"] == "Hello"
        assert ws_dict["translations"]["es"] == "Hola"

    def test_to_websocket_dict_full(self):
        """Serializar com todos os campos"""
        words = [Word(word="Test", start=0.0, end=0.5, probability=1.0)]
        segment = Segment(start=0.0, end=0.5, text="Test", words=words)

        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.99,
            language="en",
            timestamp=datetime(2025, 1, 17, 12, 0, 0),
            segments=[segment],
            speaker="SPEAKER_00",
            translation={"pt": "Teste"},
        )

        ws_dict = result.to_websocket_dict()

        # Verificar todos os campos
        assert ws_dict["type"] == "transcription"
        assert ws_dict["text"] == "Test"
        assert "segments" in ws_dict
        assert ws_dict["speaker"] == "SPEAKER_00"
        assert "translations" in ws_dict
        assert ws_dict["translations"]["pt"] == "Teste"
