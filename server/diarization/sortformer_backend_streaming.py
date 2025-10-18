"""
Sortformer Diarization Backend (NVIDIA NeMo)

SOTA 2025 para diarization em tempo real com streaming.
Suporta até 4 speakers simultâneos.

Baseado em: nvidia/diar_streaming_sortformer_4spk-v2
Paper: https://arxiv.org/abs/2505.14345

Dependências:
    pip install nemo_toolkit[asr]
"""

import logging
from typing import Optional

import numpy as np

try:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.streaming_utils import StreamingSortformerState

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    torch = None
    SortformerEncLabelModel = None
    StreamingSortformerState = None

from ..models.result import Word
from .base import DiarizationBackend

logger = logging.getLogger(__name__)


class SortformerBackend(DiarizationBackend):
    """
    Backend de diarization usando Sortformer (NeMo)

    Características:
    - Streaming: Mantém estado entre chunks
    - 4 speakers simultâneos
    - Latência baixa (processa em chunks de 1s)
    - SOTA accuracy (2025)

    Args:
        model_name: Nome do modelo NeMo
        sample_rate: Sample rate do áudio (deve ser 16000)
        device: Device para inferência
        chunk_duration: Duração dos chunks em segundos (default: 1.0)
    """

    # Singleton para compartilhar modelo entre instâncias
    _shared_model = None

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        sample_rate: int = 16000,
        num_speakers: Optional[int] = 4,
        device: str = "auto",
        chunk_duration: float = 1.0,
    ):
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit não disponível. "
                "Instale com: pip install nemo_toolkit[asr]"
            )

        super().__init__(
            sample_rate=sample_rate,
            num_speakers=num_speakers or 4,  # Sortformer é fixo em 4
            device=device,
        )

        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        # Carregar modelo compartilhado (singleton)
        if SortformerBackend._shared_model is None:
            logger.info(f"⏳ Carregando modelo Sortformer: {model_name}")
            SortformerBackend._shared_model = self._load_model()
            logger.info(f"✅ Modelo Sortformer carregado")

        self.model = SortformerBackend._shared_model

        # Streaming state (único por instância - cada cliente tem o seu)
        self.streaming_state: StreamingSortformerState | None = None
        self.reset()

        # Buffer para acumular áudio
        self.audio_buffer = np.array([], dtype=np.float32)

        # Cache de segmentos de diarization (start, end, speaker_id)
        self.diarization_cache: list[tuple[float, float, str]] = []

        logger.info(
            f"SortformerBackend inicializado "
            f"(model={model_name}, device={self.device}, speakers={self.num_speakers})"
        )

    def _load_model(self) -> "SortformerEncLabelModel":
        """Carrega o modelo Sortformer do NeMo"""
        model = SortformerEncLabelModel.from_pretrained(self.model_name)

        # Mover para device apropriado
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        model = model.to(device)
        model.eval()  # Modo de inferência

        logger.info(f"Modelo Sortformer carregado no device: {device}")
        return model

    def reset(self):
        """Reseta estado de streaming"""
        self.streaming_state = StreamingSortformerState()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.diarization_cache.clear()
        logger.debug("Sortformer state resetado")

    async def diarize(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
    ) -> list[tuple[float, float, str]]:
        """
        Processa áudio e retorna segmentos com speaker IDs

        Algoritmo:
        1. Adiciona áudio ao buffer
        2. Processa em chunks de 1 segundo
        3. Para cada chunk:
           - Converte para mel spectrogram
           - Passa pelo modelo Sortformer
           - Atualiza streaming state
           - Gera predições de speaker
        4. Retorna segmentos acumulados

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

        # Processar chunks completos de 1s
        new_segments: list[tuple[float, float, int]] = []

        while len(self.audio_buffer) >= self.chunk_samples:
            # Extrair chunk
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]

            # Processar chunk
            segments = await self._process_chunk(chunk, offset)
            new_segments.extend(segments)

            # Atualizar offset para próximo chunk
            offset += self.chunk_duration

        # Adicionar ao cache
        self.diarization_cache.extend(new_segments)

        return new_segments

    async def _process_chunk(
        self,
        chunk: np.ndarray,
        offset: float,
    ) -> list[tuple[float, float, str]]:
        """
        Processa um chunk de áudio através do Sortformer

        Args:
            chunk: Chunk de áudio (1 segundo)
            offset: Offset temporal

        Returns:
            Segmentos detectados neste chunk
        """
        if torch is None:
            raise RuntimeError("PyTorch não disponível")

        # Converter para tensor PyTorch
        audio_tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, samples]

        # Mover para device
        device = next(self.model.parameters()).device
        audio_tensor = audio_tensor.to(device)

        # Processar com modelo
        with torch.no_grad():
            # O modelo retorna predições de speaker frame a frame
            # Frame duration: ~0.08s (80ms)
            diar_logits = self.model.forward_streaming(
                audio_signal=audio_tensor,
                audio_signal_length=torch.tensor([len(chunk)]),
                streaming_state=self.streaming_state,
            )

            # diar_logits shape: [1, num_frames, num_speakers]
            # Converter para speaker predictions (argmax)
            speaker_preds = torch.argmax(diar_logits, dim=-1)  # [1, num_frames]
            speaker_preds = speaker_preds.cpu().numpy()[0]  # [num_frames]

        # Converter frame-level predictions para segmentos
        return self._frames_to_segments(speaker_preds, offset)

    def _frames_to_segments(
        self,
        speaker_frames: np.ndarray,
        offset: float,
    ) -> list[tuple[float, float, str]]:
        """
        Converte predições frame-level para segmentos contínuos

        Args:
            speaker_frames: Array de speaker IDs por frame
            offset: Offset temporal

        Returns:
            Lista de segmentos [(start, end, speaker_id)]
        """
        if len(speaker_frames) == 0:
            return []

        segments: list[tuple[float, float, str]] = []

        # Frame duration: chunk_duration / num_frames
        frame_duration = self.chunk_duration / len(speaker_frames)

        # Agrupar frames consecutivos com mesmo speaker
        current_speaker_num = int(speaker_frames[0])
        segment_start = offset

        for i, speaker_id in enumerate(speaker_frames[1:], start=1):
            speaker_id_num = int(speaker_id)

            if speaker_id_num != current_speaker_num:
                # Mudança de speaker - fechar segmento
                segment_end = offset + (i * frame_duration)
                # Converter para formato string "SPEAKER_XX"
                speaker_str = f"SPEAKER_{current_speaker_num:02d}"
                segments.append((segment_start, segment_end, speaker_str))

                # Novo segmento
                current_speaker_num = speaker_id_num
                segment_start = segment_end

        # Fechar último segmento
        segment_end = offset + self.chunk_duration
        speaker_str = f"SPEAKER_{current_speaker_num:02d}"
        segments.append((segment_start, segment_end, speaker_str))

        return segments

    async def assign_speakers_to_words(
        self,
        words: list[Word],
        diarization_segments: list[tuple[float, float, str]] | None = None,
    ) -> list[Word]:
        """
        Atribui speaker IDs às palavras baseado em overlap temporal

        Algoritmo:
        1. Para cada palavra, encontrar segmento de diarization com maior overlap
        2. Atribuir speaker_id do segmento à palavra
        3. Se não houver overlap, manter speaker_id=None

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

    def get_info(self) -> dict:
        """Retorna informações do backend"""
        info = super().get_info()
        info.update({
            "model_name": self.model_name,
            "chunk_duration": self.chunk_duration,
            "nemo_available": NEMO_AVAILABLE,
        })
        return info
