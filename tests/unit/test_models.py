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

    def test_new_capability_enum_values(self):
        """Novas capabilities (verbatim, punctuation, etc)"""
        assert Capability.VERBATIM.value == "verbatim"
        assert Capability.PUNCTUATION.value == "punctuation"
        assert Capability.MULTILINGUAL_WORD.value == "multilingual_word"
        assert Capability.CONFIDENCE_FRAMES.value == "confidence_frames"


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

    def test_word_with_filler_flag(self):
        """Word com flag is_filler (para 'um', 'uh', etc)"""
        filler = Word(
            word="um",
            start=1.0,
            end=1.2,
            probability=0.99,
            is_filler=True,
        )

        assert filler.is_filler is True
        assert filler.word == "um"

    def test_word_with_punctuation_flag(self):
        """Word com flag is_punctuation"""
        punct = Word(
            word=".",
            start=2.0,
            end=2.05,
            probability=1.0,
            is_punctuation=True,
        )

        assert punct.is_punctuation is True
        assert punct.word == "."

    def test_word_with_speaker_id(self):
        """Word com speaker_id (word-level diarization)"""
        word = Word(
            word="hello",
            start=0.0,
            end=0.5,
            probability=0.98,
            speaker_id="SPEAKER_01",
        )

        assert word.speaker_id == "SPEAKER_01"

    def test_word_with_language(self):
        """Word com language (code-switching)"""
        word = Word(
            word="bonjour",
            start=1.0,
            end=1.5,
            probability=0.95,
            language="fr",
        )

        assert word.language == "fr"

    def test_word_duration_property(self):
        """Word tem propriedade duration calculada"""
        word = Word(word="test", start=1.0, end=2.5, probability=1.0)

        assert word.duration == 1.5

    def test_word_to_dict_minimal(self):
        """Word.to_dict() com campos mínimos"""
        word = Word(word="test", start=0.0, end=0.5, probability=0.99)

        d = word.to_dict()

        assert d["word"] == "test"
        assert d["start"] == 0.0
        assert d["end"] == 0.5
        assert d["probability"] == 0.99
        assert "is_filler" not in d
        assert "is_punctuation" not in d
        assert "speaker_id" not in d
        assert "language" not in d

    def test_word_to_dict_with_filler(self):
        """Word.to_dict() inclui is_filler quando True"""
        filler = Word(
            word="uh",
            start=0.0,
            end=0.2,
            probability=1.0,
            is_filler=True,
        )

        d = filler.to_dict()

        assert d["is_filler"] is True

    def test_word_to_dict_full(self):
        """Word.to_dict() com todos os campos opcionais"""
        word = Word(
            word="hello",
            start=0.0,
            end=0.5,
            probability=0.98,
            is_filler=False,  # Não deve aparecer
            speaker_id="SPEAKER_02",
            language="en",
        )

        d = word.to_dict()

        assert d["word"] == "hello"
        assert d["speaker_id"] == "SPEAKER_02"
        assert d["language"] == "en"
        assert "is_filler" not in d  # False não serializa


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

    def test_segment_with_confidence_per_frame(self):
        """Segment com confidence_per_frame"""
        confidence_frames = [0.95, 0.96, 0.97, 0.98, 0.99]

        segment = Segment(
            start=0.0,
            end=0.5,
            text="Test",
            confidence_per_frame=confidence_frames,
        )

        assert segment.confidence_per_frame == confidence_frames
        assert len(segment.confidence_per_frame) == 5

    def test_segment_with_speaker_id(self):
        """Segment com speaker_id"""
        segment = Segment(
            start=0.0,
            end=2.0,
            text="Hello there",
            speaker_id="SPEAKER_03",
        )

        assert segment.speaker_id == "SPEAKER_03"

    def test_segment_duration_property(self):
        """Segment tem propriedade duration calculada"""
        segment = Segment(start=1.5, end=4.0, text="Test")

        assert segment.duration == 2.5

    def test_segment_words_count_property(self):
        """Segment tem propriedade words_count"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.99),
            Word(word="world", start=0.6, end=1.2, probability=0.98),
            Word(word="test", start=1.3, end=1.8, probability=0.97),
        ]

        segment = Segment(start=0.0, end=1.8, text="Hello world test", words=words)

        assert segment.words_count == 3

    def test_segment_words_count_without_words(self):
        """Segment.words_count retorna 0 se sem words"""
        segment = Segment(start=0.0, end=1.0, text="Test")

        assert segment.words_count == 0

    def test_segment_average_confidence_from_frames(self):
        """Segment.average_confidence calculado de confidence_per_frame"""
        segment = Segment(
            start=0.0,
            end=0.5,
            text="Test",
            confidence_per_frame=[0.90, 0.95, 1.0],
        )

        expected_avg = (0.90 + 0.95 + 1.0) / 3
        assert abs(segment.average_confidence - expected_avg) < 0.001

    def test_segment_average_confidence_from_words(self):
        """Segment.average_confidence calculado de words probability (fallback)"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.90),
            Word(word="world", start=0.6, end=1.2, probability=0.95),
        ]

        segment = Segment(start=0.0, end=1.2, text="Hello world", words=words)

        expected_avg = (0.90 + 0.95) / 2
        assert abs(segment.average_confidence - expected_avg) < 0.001

    def test_segment_average_confidence_no_data(self):
        """Segment.average_confidence retorna 0.0 se sem dados"""
        segment = Segment(start=0.0, end=1.0, text="Test")

        assert segment.average_confidence == 0.0


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

    def test_transcription_result_with_metadata(self):
        """TranscriptionResult com processing_time_ms e model_name"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            processing_time_ms=125.5,
            model_name="crisper-whisper",
        )

        assert result.processing_time_ms == 125.5
        assert result.model_name == "crisper-whisper"

    def test_transcription_result_total_words(self):
        """TranscriptionResult.total_words exclui fillers"""
        words = [
            Word(word="Hello", start=0.0, end=0.5, probability=0.99),
            Word(word="um", start=0.6, end=0.8, probability=0.95, is_filler=True),
            Word(word="world", start=0.9, end=1.5, probability=0.98),
        ]

        segment = Segment(start=0.0, end=1.5, text="Hello um world", words=words)

        result = TranscriptionResult(
            text="Hello world",  # Fillers removidos
            is_final=True,
            confidence=0.97,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        assert result.total_words == 2  # "Hello" + "world" (sem "um")

    def test_transcription_result_fillers_count(self):
        """TranscriptionResult.fillers_count conta apenas fillers"""
        words = [
            Word(word="So", start=0.0, end=0.3, probability=0.99),
            Word(word="um", start=0.4, end=0.6, probability=0.95, is_filler=True),
            Word(word="I", start=0.7, end=0.9, probability=0.99),
            Word(word="uh", start=1.0, end=1.2, probability=0.97, is_filler=True),
            Word(word="think", start=1.3, end=1.8, probability=0.98),
        ]

        segment = Segment(start=0.0, end=1.8, text="So um I uh think", words=words)

        result = TranscriptionResult(
            text="So I think",
            is_final=True,
            confidence=0.96,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        assert result.fillers_count == 2  # "um" + "uh"

    def test_transcription_result_duration(self):
        """TranscriptionResult.duration é o max end dos segments"""
        segment1 = Segment(start=0.0, end=2.5, text="First segment")
        segment2 = Segment(start=2.5, end=5.0, text="Second segment")
        segment3 = Segment(start=5.0, end=7.2, text="Third segment")

        result = TranscriptionResult(
            text="First segment Second segment Third segment",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            segments=[segment1, segment2, segment3],
        )

        assert result.duration == 7.2

    def test_transcription_result_duration_no_segments(self):
        """TranscriptionResult.duration retorna 0.0 se sem segments"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
        )

        assert result.duration == 0.0

    def test_transcription_result_total_words_no_segments(self):
        """TranscriptionResult.total_words retorna 0 se sem segments"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
        )

        assert result.total_words == 0

    def test_transcription_result_fillers_count_no_segments(self):
        """TranscriptionResult.fillers_count retorna 0 se sem segments"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
        )

        assert result.fillers_count == 0


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

    def test_to_websocket_dict_with_metadata(self):
        """Serializar com processing_time_ms e model_name"""
        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            processing_time_ms=150.0,
            model_name="crisper-whisper",
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["processing_time_ms"] == 150.0
        assert ws_dict["model"] == "crisper-whisper"

    def test_to_websocket_dict_word_with_filler(self):
        """Serializar word com is_filler marcado"""
        filler = Word(
            word="um",
            start=0.0,
            end=0.2,
            probability=0.99,
            is_filler=True,
        )

        segment = Segment(start=0.0, end=0.2, text="um", words=[filler])

        result = TranscriptionResult(
            text="",
            is_final=True,
            confidence=0.99,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["segments"][0]["words"][0]["is_filler"] is True

    def test_to_websocket_dict_word_with_speaker(self):
        """Serializar word com speaker_id"""
        word = Word(
            word="hello",
            start=0.0,
            end=0.5,
            probability=0.98,
            speaker_id="SPEAKER_02",
        )

        segment = Segment(start=0.0, end=0.5, text="hello", words=[word])

        result = TranscriptionResult(
            text="hello",
            is_final=True,
            confidence=0.98,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["segments"][0]["words"][0]["speaker_id"] == "SPEAKER_02"

    def test_to_websocket_dict_segment_with_confidence_frames(self):
        """Serializar segment com confidence_per_frame"""
        segment = Segment(
            start=0.0,
            end=0.5,
            text="Test",
            confidence_per_frame=[0.95, 0.96, 0.97],
        )

        result = TranscriptionResult(
            text="Test",
            is_final=True,
            confidence=0.96,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        ws_dict = result.to_websocket_dict()

        assert "confidence_per_frame" in ws_dict["segments"][0]
        assert ws_dict["segments"][0]["confidence_per_frame"] == [0.95, 0.96, 0.97]

    def test_to_websocket_dict_segment_with_speaker_id(self):
        """Serializar segment com speaker_id"""
        segment = Segment(
            start=0.0,
            end=1.0,
            text="Hello",
            speaker_id="SPEAKER_01",
        )

        result = TranscriptionResult(
            text="Hello",
            is_final=True,
            confidence=0.95,
            language="en",
            timestamp=datetime.now(),
            segments=[segment],
        )

        ws_dict = result.to_websocket_dict()

        assert ws_dict["segments"][0]["speaker_id"] == "SPEAKER_01"
