"""
CrisperWhisper Backend

Backend para Cris

perWhisper (Nyra Health) com verbatim transcription.

CrisperWhisper é um Whisper Large v3 fine-tuned para:
- Verbatim transcription (fillers, stutters, false starts)
- ±50ms timestamp precision (4x melhor que Whisper)
- Word-level timestamps
- English + German

Instalação:
    pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper

Licença: CC-BY-NC-4.0 (não comercial)
"""

import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import time

try:
    from transformers import pipeline
    CRISPER_AVAILABLE = True
except ImportError:
    CRISPER_AVAILABLE = False

from .base import WhisperBackend
from ..models.result import TranscriptionResult, Segment, Word
from ..models.capability import BackendInfo, Capability, Platform
from ..utils.platform import detect_platform, PlatformNotSupportedError

logger = logging.getLogger(__name__)


class CrisperWhisperBackend(WhisperBackend):
    """
    CrisperWhisper Backend - Verbatim transcription com ±50ms precision

    Recursos:
    - Verbatim transcription (fillers, stutters, false starts)
    - ±50ms timestamp precision (4x melhor que Whisper)
    - Word-level timestamps
    - English + German

    Instalação:
        pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper

    Example:
        >>> backend = BackendRegistry.create("crisper-whisper", {
        ...     "language": "en",
        ...     "device": "cuda",
        ... })
        >>> await backend.initialize()
        >>> result = await backend.transcribe_chunk(audio)
        >>> result.fillers_count  # Fillers detectados!
        2
    """

    # Class-level constant
    INFO = BackendInfo(
        name="crisper-whisper",
        supported_platforms={
            Platform.LINUX_CUDA,
            Platform.MACOS_APPLE_SILICON,
        },
        capabilities={
            Capability.TRANSCRIPTION,
            Capability.WORD_TIMESTAMPS,
            Capability.VERBATIM,  # ⭐ Fillers!
            Capability.STREAMING,
        },
        supported_languages={"en", "de"},
        model_sizes={"large-v3"},  # Base é Whisper Large v3 fine-tuned
    )

    @property
    def info(self) -> BackendInfo:
        """Retorna metadata e capabilities do CrisperWhisper backend (cached)"""
        return self.__class__.INFO

    def __init__(self, config: dict):
        """
        Inicializa o backend CrisperWhisper

        Args:
            config: Configuração do backend
                - language: Idioma ("en" ou "de")
                - device: Device ("cuda", "mps", "cpu", ou "auto")
                - model: Modelo (sempre "nyrahealth/CrisperWhisper")

        Raises:
            ImportError: Se CrisperWhisper não está instalado
            ValueError: Se idioma não é suportado
            PlatformNotSupportedError: Se plataforma não é suportada
        """
        super().__init__(config)

        if not CRISPER_AVAILABLE:
            raise ImportError(
                "CrisperWhisper não está instalado. "
                "Instale com: pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper"
            )

        self.model = None
        self.device = config.get("device", "auto")
        self.language = config.get("language", "en")

        # Validar idioma
        if self.language not in self.info.supported_languages:
            raise ValueError(
                f"Idioma '{self.language}' não suportado. "
                f"Suportados: {', '.join(self.info.supported_languages)}"
            )

        # Validar plataforma
        current_platform = detect_platform()
        if current_platform not in self.info.supported_platforms:
            supported = ", ".join(p.value for p in self.info.supported_platforms)
            raise PlatformNotSupportedError(
                f"CrisperWhisper não suporta {current_platform.value}. "
                f"Plataformas suportadas: {supported}. "
                f"Sugestão: Use 'faster-whisper' no Mac ou deploy no RunPod com GPU."
            )

        logger.info("CrisperWhisperBackend criado (language=%s, device=%s)", self.language, self.device)

    async def initialize(self) -> None:
        """
        Inicializa o modelo CrisperWhisper

        Carrega o modelo do HuggingFace Hub. Pode demorar na primeira vez
        (download do modelo 2B params).

        Raises:
            RuntimeError: Se inicialização falhar
        """
        if self._initialized:
            logger.warning("Backend já inicializado")
            return

        logger.info("Carregando CrisperWhisper...")
        logger.info("  Device: %s", self.device)
        logger.info("  Language: %s", self.language)
        logger.info("  Model: nyrahealth/CrisperWhisper")

        try:
            # Carregar pipeline do HuggingFace
            self.model = pipeline(
                "automatic-speech-recognition",
                model="nyrahealth/CrisperWhisper",
                device=self.device if self.device != "auto" else None,
                return_timestamps="word",  # ⭐ Word-level timestamps
            )

            self._initialized = True
            logger.info("✅ CrisperWhisper inicializado")

        except Exception as e:
            logger.error("Erro ao inicializar CrisperWhisper: %s", e)
            raise RuntimeError(f"Falha ao inicializar CrisperWhisper: {e}") from e

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        context: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcreve chunk de áudio com detecção de fillers

        Args:
            audio: Array numpy (16kHz, mono, float32)
            context: Contexto da transcrição anterior (opcional)

        Returns:
            TranscriptionResult com is_filler marcado

        Raises:
            RuntimeError: Se backend não foi inicializado

        Example:
            >>> audio = np.random.randn(16000).astype(np.float32)
            >>> result = await backend.transcribe_chunk(audio)
            >>> result.fillers_count
            2
            >>> result.total_words  # Exclui fillers
            8
        """
        if not self._initialized or not self.model:
            raise RuntimeError("Backend não inicializado. Chame initialize() primeiro.")

        start_time = time.time()

        # Transcrever com CrisperWhisper
        result = self.model(
            audio,
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
            },
            return_timestamps="word",
        )

        processing_time = (time.time() - start_time) * 1000

        # Converter para nosso formato
        segments = []

        if result.get("chunks"):
            current_segment_words = []
            segment_start = None
            segment_end = None
            segment_text = ""

            for chunk in result["chunks"]:
                word_text = chunk["text"].strip()
                word_start = chunk["timestamp"][0]
                word_end = chunk["timestamp"][1]

                # Detectar fillers
                is_filler = self._is_filler_word(word_text)

                word = Word(
                    word=word_text,
                    start=word_start,
                    end=word_end,
                    probability=1.0,  # CrisperWhisper não retorna prob por palavra
                    is_filler=is_filler,  # ⭐ Marcado!
                )

                current_segment_words.append(word)

                if segment_start is None:
                    segment_start = word_start
                segment_end = word_end
                segment_text += word_text + " "

            # Criar segmento
            if current_segment_words:
                segments.append(Segment(
                    start=segment_start,
                    end=segment_end,
                    text=segment_text.strip(),
                    words=current_segment_words,
                ))

        # Criar resultado
        return TranscriptionResult(
            text=result["text"],
            is_final=True,
            confidence=1.0,  # CrisperWhisper não retorna confidence global
            language=self.language,
            timestamp=datetime.now(),
            segments=segments,
            processing_time_ms=processing_time,
            model_name="crisper-whisper",
        )

    def _is_filler_word(self, word: str) -> bool:
        """
        Detecta se palavra é filler

        Fillers são palavras de preenchimento como "um", "uh", "er", etc.

        Args:
            word: Palavra a verificar

        Returns:
            True se é filler, False caso contrário

        Example:
            >>> backend._is_filler_word("um")
            True
            >>> backend._is_filler_word("hello")
            False
        """
        # Lista de fillers comuns em inglês e alemão
        fillers_en = {"um", "uh", "er", "ah", "like", "you know"}
        fillers_de = {"äh", "ähm", "öh", "ehm"}

        # Normalizar: lowercase e remover pontuação
        word_lower = word.lower().strip(".,!?;:\"'")

        return word_lower in fillers_en or word_lower in fillers_de

    async def transcribe_stream(
        self,
        audio_stream: Any  # AsyncIterator[np.ndarray]
    ):
        """
        Transcreve stream de áudio (chunks contínuos)

        Args:
            audio_stream: Iterator assíncrono de chunks de áudio

        Yields:
            TranscriptionResult para cada chunk

        Example:
            >>> async for result in backend.transcribe_stream(audio_stream):
            ...     print(result.text)
        """
        if not self._initialized or not self.model:
            raise RuntimeError("Backend não inicializado. Chame initialize() primeiro.")

        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            yield result

    async def cleanup(self) -> None:
        """
        Limpa recursos do backend

        Libera memória do modelo.
        """
        if self.model:
            del self.model
            self.model = None

        self._initialized = False
        logger.info("CrisperWhisper limpo")

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o backend

        Returns:
            Dicionário com informações do backend

        Example:
            >>> info = backend.get_backend_info()
            >>> info["name"]
            'crisper-whisper'
        """
        return {
            "name": "crisper-whisper",
            "model": "nyrahealth/CrisperWhisper",
            "device": self.device,
            "language": self.language,
            "initialized": self._initialized,
        }

    def is_initialized(self) -> bool:
        """
        Verifica se backend está inicializado

        Returns:
            True se inicializado, False caso contrário
        """
        return self._initialized
