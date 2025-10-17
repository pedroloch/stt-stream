"""
Resultado normalizado de transcrição

Define estruturas de dados para representar resultados de transcrição
de forma uniforme, independente do backend usado (MLX, WhisperX, etc).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class Word:
    """
    Uma palavra com timestamp

    Usado para word-level timestamps, permitindo highlight
    palavra por palavra e análise precisa.

    Attributes:
        word: Texto da palavra
        start: Timestamp de início (segundos)
        end: Timestamp de fim (segundos)
        probability: Confiança da transcrição (0-1)

    Example:
        >>> word = Word(word="hello", start=1.5, end=2.0, probability=0.95)
        >>> word.word
        'hello'
        >>> word.end - word.start  # duração
        0.5
    """

    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    """
    Um segmento de transcrição

    Representa um trecho contínuo de fala com timestamps.
    Opcionalmente contém word-level timestamps.

    Attributes:
        start: Timestamp de início (segundos)
        end: Timestamp de fim (segundos)
        text: Texto transcrito do segmento
        words: Lista de palavras com timestamps (opcional)

    Example:
        >>> segment = Segment(
        ...     start=0.0,
        ...     end=2.0,
        ...     text="Hello world",
        ...     words=[
        ...         Word(word="Hello", start=0.0, end=0.5, probability=0.99),
        ...         Word(word="world", start=0.6, end=1.2, probability=0.98),
        ...     ],
        ... )
        >>> len(segment.words)
        2
    """

    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None


@dataclass
class TranscriptionResult:
    """
    Resultado normalizado de transcrição

    Formato unificado usado por todos os backends. Campos obrigatórios
    são garantidos, campos opcionais dependem das capabilities do backend.

    Attributes:
        text: Texto transcrito (obrigatório)
        is_final: Se é transcrição final ou parcial (obrigatório)
        confidence: Confiança geral (0-1) (obrigatório)
        language: Código do idioma detectado (obrigatório)
        timestamp: Momento da transcrição (obrigatório)
        segments: Lista de segmentos com timestamps (opcional - WORD_TIMESTAMPS)
        speaker: ID do speaker (opcional - SPEAKER_DIARIZATION)
        translation: Traduções para outros idiomas (opcional - TRANSLATION)

    Example:
        >>> result = TranscriptionResult(
        ...     text="Hello world",
        ...     is_final=True,
        ...     confidence=0.95,
        ...     language="en",
        ...     timestamp=datetime.now(),
        ...     speaker="SPEAKER_01",  # opcional
        ... )
        >>> result.text
        'Hello world'
    """

    # Campos obrigatórios (todos backends)
    text: str
    is_final: bool
    confidence: float
    language: str
    timestamp: datetime

    # Campos opcionais (dependem de capability)
    segments: Optional[List[Segment]] = None
    speaker: Optional[str] = None  # "SPEAKER_00", "SPEAKER_01", etc
    translation: Optional[Dict[str, str]] = None  # {"en": "...", "es": "..."}

    def to_websocket_dict(self) -> dict:
        """
        Converte resultado para formato WebSocket JSON

        Serializa o resultado para um dicionário que pode ser enviado
        via WebSocket para o cliente. Campos opcionais só são incluídos
        se presentes.

        Returns:
            Dicionário com formato WebSocket

        Example:
            >>> result = TranscriptionResult(
            ...     text="Test",
            ...     is_final=True,
            ...     confidence=0.95,
            ...     language="en",
            ...     timestamp=datetime(2025, 1, 17, 10, 30, 0),
            ... )
            >>> ws_dict = result.to_websocket_dict()
            >>> ws_dict["type"]
            'transcription'
            >>> ws_dict["text"]
            'Test'
        """
        result = {
            "type": "transcription",
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "language": self.language,
            "timestamp": self.timestamp.isoformat(),
        }

        # Adicionar segments se presente (word timestamps)
        if self.segments:
            result["segments"] = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": (
                        [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in seg.words
                        ]
                        if seg.words
                        else None
                    ),
                }
                for seg in self.segments
            ]

        # Adicionar speaker se presente (diarization)
        if self.speaker:
            result["speaker"] = self.speaker

        # Adicionar translations se presente (translation)
        if self.translation:
            result["translations"] = self.translation

        return result
