"""
Pydantic Models para API Batch

Definições de request/response para endpoint `/v1/transcribe`
Compatible com OpenAI Whisper-1 API + extensões
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# REQUEST MODELS
# ============================================================

class TranscribeRequest(BaseModel):
    """
    Request para transcrição batch

    Compatible com OpenAI Whisper-1 API
    """

    # Backend selection
    model: str = Field(
        default="faster-whisper",
        description="Backend de transcrição (faster-whisper, mlx, whisperx, auto)",
        examples=["faster-whisper", "mlx", "whisperx", "auto"]
    )

    model_size: str = Field(
        default="base",
        description="Tamanho do modelo Whisper",
        examples=["tiny", "base", "small", "medium", "large", "large-v3", "distil-large-v3"]
    )

    # Language settings
    language: str | None = Field(
        default="pt",
        description="Idioma do áudio (auto-detect se None)",
        examples=["pt", "en", "es", "fr", None]
    )

    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Tarefa: transcrever ou traduzir para inglês"
    )

    # Output format
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"] = Field(
        default="verbose_json",
        description="Formato da resposta"
    )

    # Transcription options
    initial_prompt: str | None = Field(
        default=None,
        description="Prompt/contexto inicial para melhorar transcrição",
        examples=["Esta é uma reunião sobre vendas Q3"]
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperatura de sampling (0.0 = determinístico)"
    )

    # Extensions (além do Whisper-1)
    enable_diarization: bool = Field(
        default=False,
        description="⭐ Habilitar identificação de speakers (SPEAKER_00, SPEAKER_01...)"
    )

    diarization_backend: Literal["sortformer", "pyannote", "auto"] | None = Field(
        default="auto",
        description="Backend de diarization (se enable_diarization=True)"
    )

    num_speakers: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Número de speakers (None = auto-detect, requer pyannote)"
    )

    # Advanced options
    no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold para detectar silêncio/não-fala"
    )

    compression_ratio_threshold: float = Field(
        default=2.4,
        description="Threshold para detectar repetições (anti-hallucination)"
    )

    # Metadata
    return_metrics: bool = Field(
        default=True,
        description="Retornar métricas de processamento (tempo, RTF, etc)"
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class Word(BaseModel):
    """Palavra com timestamp"""
    word: str = Field(description="Texto da palavra")
    start: float = Field(description="Tempo de início (segundos)")
    end: float = Field(description="Tempo de fim (segundos)")
    probability: float | None = Field(default=None, description="Confiança (0.0-1.0)")
    speaker_id: str | None = Field(default=None, description="Speaker ID (se diarization habilitado)")


class Segment(BaseModel):
    """Segmento de transcrição"""
    id: int = Field(description="ID do segmento (sequencial)")
    start: float = Field(description="Tempo de início (segundos)")
    end: float = Field(description="Tempo de fim (segundos)")
    text: str = Field(description="Texto transcrito")
    words: list[Word] | None = Field(default=None, description="Palavras com timestamps")
    speaker_id: str | None = Field(default=None, description="Speaker principal do segmento")
    confidence: float | None = Field(default=None, description="Confiança média (0.0-1.0)")


class ProcessingMetrics(BaseModel):
    """Métricas de processamento"""
    processing_time_sec: float = Field(description="Tempo total de processamento")
    audio_duration_sec: float = Field(description="Duração do áudio")
    real_time_factor: float = Field(description="RTF (processing_time / audio_duration)")

    model_load_time_sec: float | None = Field(default=None, description="Tempo de carregamento do modelo")
    transcription_time_sec: float | None = Field(default=None, description="Tempo de transcrição")
    diarization_time_sec: float | None = Field(default=None, description="Tempo de diarization")
    formatting_time_sec: float | None = Field(default=None, description="Tempo de formatação")


class BackendInfo(BaseModel):
    """Informações do backend usado"""
    model: str = Field(description="Nome do backend")
    model_size: str = Field(description="Tamanho do modelo")
    device: str = Field(description="Device usado (cpu, cuda, mps)")
    language: str | None = Field(default=None, description="Idioma detectado")
    diarization_backend: str | None = Field(default=None, description="Backend de diarization")


class TranscribeResponse(BaseModel):
    """
    Resposta completa de transcrição (verbose_json)

    Compatible com OpenAI Whisper-1 + extensões
    """

    # Texto completo
    text: str = Field(description="Transcrição completa")

    # Metadata
    task: str = Field(description="Tarefa executada (transcribe ou translate)")
    language: str = Field(description="Idioma detectado/especificado")
    duration: float = Field(description="Duração do áudio (segundos)")

    # Segments detalhados
    segments: list[Segment] = Field(description="Segmentos com timestamps")

    # Backend info
    backend_info: BackendInfo = Field(description="Informações do backend usado")

    # Metrics (opcional)
    metrics: ProcessingMetrics | None = Field(default=None, description="Métricas de processamento")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da transcrição")


class SimpleTranscribeResponse(BaseModel):
    """Resposta simples (formato 'json')"""
    text: str = Field(description="Transcrição completa")


class ErrorResponse(BaseModel):
    """Resposta de erro estruturada"""
    error: str = Field(description="Tipo de erro")
    message: str = Field(description="Mensagem de erro")
    details: dict | None = Field(default=None, description="Detalhes adicionais")
