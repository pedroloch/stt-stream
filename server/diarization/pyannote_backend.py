"""
Pyannote Diarization Backend

Backend simples e eficiente usando pyannote.audio.
Ótimo para casos gerais, fácil de usar, VAD integrado.

Baseado em: pyannote/speaker-diarization-3.1

Dependências:
    pip install pyannote.audio
"""

import logging
from typing import Optional

import numpy as np

try:
    import torch
    from pyannote.audio import Pipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    torch = None

from ..models.result import Word
from .base import DiarizationBackend

logger = logging.getLogger(__name__)


class PyannoteBackend(DiarizationBackend):
    """
    Backend de diarization usando Pyannote

    Características:
    - Simples e eficiente
    - VAD integrado
    - Auto-detect de número de speakers
    - Ótima accuracy
    - Menos recursos que Sortformer

    Args:
        model_name: Nome do modelo Pyannote
        sample_rate: Sample rate do áudio (deve ser 16000)
        num_speakers: Número de speakers (None = auto-detect)
        device: Device para inferência
        min_speakers: Número mínimo de speakers (para auto-detect)
        max_speakers: Número máximo de speakers (para auto-detect)
        auth_token: Token HuggingFace (necessário para alguns modelos)
    """

    # Singleton para compartilhar pipeline entre instâncias
    _shared_pipeline = None
    _shared_config = None

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        device: str = "auto",
        min_speakers: int = 1,
        max_speakers: int = 8,
        auth_token: Optional[str] = None,
    ):
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "Pyannote não disponível. "
                "Instale com: pip install pyannote.audio"
            )

        super().__init__(
            sample_rate=sample_rate,
            num_speakers=num_speakers,
            device=device,
        )

        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.auth_token = auth_token

        # Carregar pipeline compartilhado (singleton)
        config_key = (model_name, device, auth_token)
        if PyannoteBackend._shared_pipeline is None or PyannoteBackend._shared_config != config_key:
            logger.info(f"⏳ Carregando pipeline Pyannote: {model_name}")
            PyannoteBackend._shared_pipeline = self._load_pipeline()
            PyannoteBackend._shared_config = config_key
            logger.info(f"✅ Pipeline Pyannote carregado")

        self.pipeline = PyannoteBackend._shared_pipeline

        # Buffer para acumular áudio
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0.0

        # Cache de segmentos de diarization (start, end, speaker_id)
        self.diarization_cache: list[tuple[float, float, str]] = []

        # Duração mínima para processar (em segundos)
        # Pyannote funciona melhor com chunks maiores
        self.min_process_duration = 5.0

        logger.info(
            f"PyannoteBackend inicializado "
            f"(model={model_name}, device={self.device}, "
            f"speakers={num_speakers or f'{min_speakers}-{max_speakers}'})"
        )

    def _load_pipeline(self) -> "Pipeline":
        """Carrega o pipeline Pyannote"""
        # Pyannote 3.1+ usa 'token' em vez de 'use_auth_token'
        try:
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.auth_token,
            )
        except TypeError:
            # Fallback para versões antigas
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token,
            )

        # Mover para device apropriado
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.device)

        pipeline = pipeline.to(device)

        logger.info(f"Pipeline Pyannote carregado no device: {device}")
        return pipeline

    def reset(self):
        """Reseta estado de streaming"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0.0
        self.diarization_cache.clear()
        logger.debug("Pyannote state resetado")

    async def diarize(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
    ) -> list[tuple[float, float, str]]:
        """
        Processa áudio e retorna segmentos com speaker IDs

        Estratégia para streaming:
        1. Acumula áudio em buffer
        2. Quando buffer >= min_process_duration, processa
        3. Retorna segmentos detectados
        4. Mantém overlap para continuidade

        Args:
            audio: Áudio em formato float32 mono (16kHz)
            offset: Offset temporal em segundos

        Returns:
            Lista de segmentos [(start, end, speaker_id)]
        """
        if audio.size == 0:
            return []

        # Adicionar ao buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

        # Verificar se temos áudio suficiente
        buffer_duration = len(self.audio_buffer) / self.sample_rate

        if buffer_duration < self.min_process_duration:
            # Ainda não temos áudio suficiente
            return []

        # Processar buffer acumulado
        new_segments = await self._process_buffer()

        # Adicionar ao cache
        self.diarization_cache.extend(new_segments)

        # Manter apenas últimos 2 segundos no buffer (overlap)
        overlap_samples = int(2.0 * self.sample_rate)
        if len(self.audio_buffer) > overlap_samples:
            trimmed_samples = len(self.audio_buffer) - overlap_samples
            self.audio_buffer = self.audio_buffer[-overlap_samples:]
            self.buffer_offset += trimmed_samples / self.sample_rate

        return new_segments

    async def _process_buffer(self) -> list[tuple[float, float, str]]:
        """
        Processa buffer acumulado através do Pyannote pipeline

        Returns:
            Segmentos detectados
        """
        if len(self.audio_buffer) == 0:
            return []

        if torch is None:
            raise RuntimeError("PyTorch não disponível")

        # Criar dicionário de áudio (formato esperado pelo Pyannote)
        audio_dict = {
            "waveform": torch.from_numpy(self.audio_buffer).unsqueeze(0),  # [1, samples]
            "sample_rate": self.sample_rate,
        }

        # Configurar parâmetros de diarization
        if self.num_speakers is not None:
            # Número fixo de speakers
            diarization = self.pipeline(
                audio_dict,
                num_speakers=self.num_speakers,
            )
        else:
            # Auto-detect número de speakers
            diarization = self.pipeline(
                audio_dict,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

        # Converter resultado para lista de segmentos
        segments: list[tuple[float, float, str]] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # turn: Segment object com start e end
            # speaker: string como "SPEAKER_00", "SPEAKER_01", etc
            start_time = turn.start + self.buffer_offset
            end_time = turn.end + self.buffer_offset

            # speaker já é string no formato correto
            segments.append((start_time, end_time, speaker))

        return segments

    async def assign_speakers_to_words(
        self,
        words: list[Word],
        diarization_segments: list[tuple[float, float, str]] | None = None,
    ) -> list[Word]:
        """
        Atribui speaker IDs às palavras baseado em overlap temporal

        Args:
            words: Lista de Word objects
            diarization_segments: Segmentos de diarization (usa cache se None)

        Returns:
            Lista de Word objects com speaker_id atribuído
        """
        # Usar cache se não fornecido
        if diarization_segments is None:
            diarization_segments = self.diarization_cache

        if not diarization_segments:
            logger.warning("Nenhum segmento de diarization disponível")
            return words

        # Criar cópias dos Word objects com speaker_id atribuído
        words_with_speakers: list[Word] = []

        for word in words:
            # Encontrar segmento com maior overlap
            best_speaker_id: str | None = None
            best_overlap = 0.0

            for seg_start, seg_end, speaker_id in diarization_segments:
                # Calcular overlap temporal
                overlap_start = max(word.start, seg_start)
                overlap_end = min(word.end, seg_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker_id = speaker_id

            # Criar novo Word com speaker_id (já é string)
            word_with_speaker = Word(
                word=word.word,
                start=word.start,
                end=word.end,
                probability=word.probability,
                is_filler=word.is_filler,
                is_punctuation=word.is_punctuation,
                speaker_id=best_speaker_id,  # Já é string "SPEAKER_XX"
                language=word.language,
            )

            words_with_speakers.append(word_with_speaker)

        return words_with_speakers

    def finish(self) -> list[tuple[float, float, str]]:
        """
        Processa buffer restante ao finalizar stream

        Returns:
            Últimos segmentos detectados
        """
        if len(self.audio_buffer) == 0:
            return []

        # Processar buffer final (mesmo se menor que min_process_duration)
        import asyncio

        new_segments = asyncio.run(self._process_buffer())

        self.diarization_cache.extend(new_segments)
        self.audio_buffer = np.array([], dtype=np.float32)

        return new_segments

    def get_info(self) -> dict:
        """Retorna informações do backend"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "min_process_duration": self.min_process_duration,
            "pyannote_available": PYANNOTE_AVAILABLE,
        })
        return info
