"""
Resultado normalizado de transcrição

Define estruturas de dados para representar resultados de transcrição
de forma uniforme, independente do backend usado (MLX, WhisperX, etc).
"""

from dataclasses import dataclass
from datetime import datetime


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
        is_filler: Se é um filler ("um", "uh", "er") (opcional)
        is_punctuation: Se é pontuação (".", ",", "?") (opcional)
        speaker_id: ID do speaker (word-level diarization) (opcional)
        language: Idioma da palavra (code-switching) (opcional)

    Example:
        >>> word = Word(word="hello", start=1.5, end=2.0, probability=0.95)
        >>> word.word
        'hello'
        >>> word.duration
        0.5
        >>> filler = Word(word="um", start=0.0, end=0.2, probability=0.99, is_filler=True)
        >>> filler.is_filler
        True
    """

    word: str
    start: float
    end: float
    probability: float

    # Campos opcionais (verbatim transcription, diarization, multilingual)
    is_filler: bool = False
    is_punctuation: bool = False
    speaker_id: str | None = None
    language: str | None = None

    @property
    def duration(self) -> float:
        """Duração da palavra em segundos"""
        return self.end - self.start

    def to_dict(self) -> dict:
        """
        Serializa palavra para dicionário (apenas campos presentes)

        Retorna dicionário com campos obrigatórios sempre presentes,
        e campos opcionais apenas se tiverem valores relevantes.

        Returns:
            Dicionário com representação da palavra

        Example:
            >>> word = Word(word="test", start=0.0, end=0.5, probability=0.99)
            >>> d = word.to_dict()
            >>> d["word"]
            'test'
            >>> "is_filler" in d
            False
        """
        result = {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }

        # Adicionar campos opcionais apenas se presentes
        if self.is_filler:
            result["is_filler"] = True

        if self.is_punctuation:
            result["is_punctuation"] = True

        if self.speaker_id:
            result["speaker_id"] = self.speaker_id

        if self.language:
            result["language"] = self.language

        return result


@dataclass
class Segment:
    """
    Um segmento de transcrição

    Representa um trecho contínuo de fala com timestamps.
    Opcionalmente contém word-level timestamps e metadata adicional.

    Attributes:
        start: Timestamp de início (segundos)
        end: Timestamp de fim (segundos)
        text: Texto transcrito do segmento
        words: Lista de palavras com timestamps (opcional)
        confidence_per_frame: Confiança por frame (10ms) (opcional)
        speaker_id: ID do speaker dominante no segmento (opcional)

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
        >>> segment.duration
        2.0
        >>> segment.words_count
        2
    """

    start: float
    end: float
    text: str
    words: list[Word] | None = None

    # Campos opcionais (frame-level confidence, diarization)
    confidence_per_frame: list[float] | None = None
    speaker_id: str | None = None

    @property
    def duration(self) -> float:
        """Duração do segmento em segundos"""
        return self.end - self.start

    @property
    def words_count(self) -> int:
        """Número de palavras no segmento"""
        return len(self.words) if self.words else 0

    @property
    def average_confidence(self) -> float:
        """
        Confiança média do segmento

        Calcula a partir de confidence_per_frame se disponível,
        senão usa a média das probabilidades das palavras.
        Retorna 0.0 se nenhum dado disponível.

        Returns:
            Confiança média (0-1)
        """
        # Preferir confidence_per_frame se disponível
        if self.confidence_per_frame:
            return sum(self.confidence_per_frame) / len(self.confidence_per_frame)

        # Fallback: média das probabilidades das palavras
        if self.words:
            return sum(w.probability for w in self.words) / len(self.words)

        # Sem dados
        return 0.0


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
        processing_time_ms: Tempo de processamento em milissegundos (opcional)
        model_name: Nome do modelo usado (opcional)

    Example:
        >>> result = TranscriptionResult(
        ...     text="Hello world",
        ...     is_final=True,
        ...     confidence=0.95,
        ...     language="en",
        ...     timestamp=datetime.now(),
        ...     speaker="SPEAKER_01",  # opcional
        ...     model_name="crisper-whisper",
        ... )
        >>> result.text
        'Hello world'
        >>> result.total_words  # propriedade calculada
        2
    """

    # Campos obrigatórios (todos backends)
    text: str
    is_final: bool
    confidence: float
    language: str
    timestamp: datetime

    # Campos opcionais (dependem de capability)
    segments: list[Segment] | None = None
    speaker: str | None = None  # "SPEAKER_00", "SPEAKER_01", etc
    translation: dict[str, str] | None = None  # {"en": "...", "es": "..."}

    # Metadata opcional
    processing_time_ms: float | None = None
    model_name: str | None = None

    @property
    def total_words(self) -> int:
        """
        Total de palavras (excluindo fillers)

        Conta apenas palavras reais, excluindo fillers como "um", "uh".

        Returns:
            Número de palavras não-filler
        """
        if not self.segments:
            return 0

        total = 0
        for seg in self.segments:
            if seg.words:
                total += sum(1 for w in seg.words if not w.is_filler)

        return total

    @property
    def fillers_count(self) -> int:
        """
        Total de fillers detectados

        Conta fillers ("um", "uh", "er") marcados com is_filler=True.

        Returns:
            Número de fillers detectados
        """
        if not self.segments:
            return 0

        total = 0
        for seg in self.segments:
            if seg.words:
                total += sum(1 for w in seg.words if w.is_filler)

        return total

    @property
    def duration(self) -> float:
        """
        Duração total do áudio transcrito

        Calcula como o max end time dos segmentos.

        Returns:
            Duração em segundos (0.0 se sem segments)
        """
        if not self.segments:
            return 0.0

        return max(seg.end for seg in self.segments)
