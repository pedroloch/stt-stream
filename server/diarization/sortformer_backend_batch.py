"""
Sortformer Diarization - Batch/Offline Mode

Baseado em WhisperLiveKit sortformer_backend_offline.py
Processa áudio completo (não incremental) para API Batch.

Referência:
https://github.com/collabora/whisperlivekit/blob/main/whisperlivekit/diarization/sortformer_backend_offline.py
"""

import logging
from typing import Optional

import numpy as np

try:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    torch = None
    SortformerEncLabelModel = None
    AudioToMelSpectrogramPreprocessor = None

from ..models.result import Word
from .base import DiarizationBackend

logger = logging.getLogger(__name__)


class StreamingSortformerState:
    """
    Streaming state criado MANUALMENTE (não vem do NeMo).
    Baseado em WhisperLiveKit implementation.

    Attributes:
        spkcache: Speaker cache to store embeddings from start
        spkcache_lengths: Lengths of the speaker cache
        spkcache_preds: Speaker predictions for speaker cache
        fifo: FIFO queue to save embeddings from latest chunks
        fifo_lengths: Lengths of FIFO queue
        mean_sil_emb: Mean silence embedding
        n_sil_frames: Number of silence frames
    """

    def __init__(self):
        self.spkcache: "torch.Tensor | None" = None  # type: ignore
        self.spkcache_lengths: "torch.Tensor | None" = None  # type: ignore
        self.spkcache_preds: "torch.Tensor | None" = None  # type: ignore
        self.fifo: "torch.Tensor | None" = None  # type: ignore
        self.fifo_lengths: "torch.Tensor | None" = None  # type: ignore
        self.fifo_preds: "torch.Tensor | None" = None  # type: ignore
        self.spk_perm: "torch.Tensor | None" = None  # type: ignore
        self.mean_sil_emb: "torch.Tensor | None" = None  # type: ignore
        self.n_sil_frames: "torch.Tensor | None" = None  # type: ignore


class SortformerBatchBackend(DiarizationBackend):
    """
    Backend Sortformer para processamento BATCH (áudio completo).

    Fluxo:
    1. Carrega modelo Sortformer
    2. Configura parâmetros de streaming
    3. Divide áudio em chunks de ~1s
    4. Processa chunks com forward_streaming_step()
    5. Retorna segmentos [(start, end, speaker_id)]

    Args:
        model_name: Nome do modelo NeMo
        sample_rate: Sample rate do áudio (deve ser 16000)
        num_speakers: Número de speakers (4 para este modelo)
        device: Device para inferência ("auto", "cpu", "cuda", "mps")
    """

    # Singleton para compartilhar modelo entre instâncias
    _shared_model = None
    _shared_audio2mel = None

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        sample_rate: int = 16000,
        num_speakers: Optional[int] = 4,
        device: str = "auto",
    ):
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit não disponível. "
                "Instale com: pip install nemo_toolkit[asr]"
            )

        super().__init__(
            sample_rate=sample_rate,
            num_speakers=num_speakers or 4,
            device=device,  # Manter como string na classe base
        )

        self.model_name = model_name

        # IMPORTANTE: Detectar torch.device SEMPRE no __init__()
        # para evitar problema com singleton pattern
        self.torch_device = self._detect_device(device)

        # Carregar modelo compartilhado (singleton)
        if SortformerBatchBackend._shared_model is None:
            logger.info(f"⏳ Carregando modelo Sortformer: {model_name}")
            self._load_model()
            logger.info("✅ Modelo Sortformer carregado")

        self.model = SortformerBatchBackend._shared_model
        self.audio2mel = SortformerBatchBackend._shared_audio2mel

        # Chunk duration (configurado pelo modelo)
        self.chunk_duration_seconds = (
            self.model.sortformer_modules.chunk_len
            * self.model.sortformer_modules.subsampling_factor
            * self.model.preprocessor._cfg.window_stride
        )

        logger.info(
            f"SortformerBatchBackend inicializado "
            f"(chunk_duration={self.chunk_duration_seconds:.2f}s, device={self.torch_device})"
        )

    def _detect_device(self, device: str) -> torch.device:
        """
        Detecta o device apropriado

        Args:
            device: "auto", "cpu", "cuda", ou "mps"

        Returns:
            torch.device object
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _load_model(self):
        """Carrega modelo Sortformer e configura parâmetros"""
        # Usar torch_device (já detectado no __init__)
        device = self.torch_device

        # 1. Carregar modelo
        # Tentar carregar com strict=False para evitar problemas de versão
        try:
            model = SortformerEncLabelModel.from_pretrained(
                self.model_name,
                strict=False  # Permite incompatibilidades de config
            )
        except TypeError:
            # Se strict não for aceito, tentar sem
            logger.warning("Tentando carregar modelo sem strict=False")
            model = SortformerEncLabelModel.from_pretrained(self.model_name)

        model.eval()
        model.to(device)

        # 2. Configurar parâmetros (baseado em WhisperLiveKit)
        # Estes valores otimizam para ~1s latência
        # IMPORTANTE: Setar APÓS carregar o modelo, não durante __init__
        model.sortformer_modules.chunk_len = 10
        model.sortformer_modules.subsampling_factor = 10
        model.sortformer_modules.chunk_right_context = 0
        model.sortformer_modules.chunk_left_context = 10
        model.sortformer_modules.spkcache_len = 188
        model.sortformer_modules.fifo_len = 188
        model.sortformer_modules.spkcache_update_period = 144
        model.sortformer_modules.log = False
        model.sortformer_modules._check_streaming_parameters()

        # 3. Audio2Mel preprocessor
        audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025, normalize="NA", n_fft=512, features=128, pad_to=0
        )
        audio2mel.to(device)  # type: ignore

        SortformerBatchBackend._shared_model = model
        SortformerBatchBackend._shared_audio2mel = audio2mel

        logger.info(f"Modelo Sortformer carregado no device: {device}")

    async def diarize(
        self, audio: np.ndarray, offset: float = 0.0
    ) -> list[tuple[float, float, str]]:
        """
        Processa áudio completo e retorna segmentos com speaker IDs

        Args:
            audio: Áudio completo em float32 mono (16kHz)
            offset: Offset temporal (não usado em batch mode)

        Returns:
            Lista de segmentos [(start, end, speaker_id)]
            speaker_id é string "SPEAKER_00", "SPEAKER_01", etc
        """
        # 1. Dividir em chunks de ~1 segundo
        chunk_size = int(self.chunk_duration_seconds * self.sample_rate)
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) == chunk_size:  # Ignorar último chunk parcial
                chunks.append(chunk)

        if not chunks:
            logger.warning("Áudio muito curto para diarization")
            return []

        logger.debug(f"Dividido em {len(chunks)} chunks de {self.chunk_duration_seconds:.2f}s")

        # 2. Processar chunks
        segments = await self._process_chunks(chunks)

        return segments

    async def _process_chunks(
        self, chunks: list[np.ndarray]
    ) -> list[tuple[float, float, str]]:
        """
        Processa lista de chunks de áudio através do Sortformer.
        Baseado em WhisperLiveKit process_diarization()
        """
        # 1. Preprocessar chunks para features mel
        previous_chunk = None
        chunk_features = []

        for chunk in chunks:
            # Converter para tensor
            audio_signal = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(self.torch_device)
            audio_signal_length = torch.tensor([audio_signal.shape[1]]).to(self.torch_device)

            # Extrair features mel
            processed_signal, _ = self.audio2mel.get_features(
                audio_signal, audio_signal_length
            )

            # Adicionar overlap com chunk anterior (99 frames)
            if previous_chunk is not None:
                to_add = previous_chunk[:, :, -99:]
                total = torch.cat([to_add, processed_signal], dim=2)
            else:
                total = processed_signal

            previous_chunk = processed_signal

            # Transpose para formato esperado
            chunk_feat_seq_t = torch.transpose(total, 1, 2)
            chunk_features.append(chunk_feat_seq_t)

        # 2. Inicializar streaming state
        batch_size = 1
        streaming_state = self._init_streaming_state(batch_size)
        total_preds = torch.zeros(
            (batch_size, 0, self.model.sortformer_modules.n_spk), device=self.torch_device
        )

        # 3. Processar cada chunk
        left_offset = 0
        right_offset = 8
        len_prediction = None

        speaker_segments = []

        for i, chunk_feat in enumerate(chunk_features):
            with torch.inference_mode():
                # Forward streaming step (método que EXISTE no NeMo)
                streaming_state, total_preds = self.model.forward_streaming_step(
                    processed_signal=chunk_feat,
                    processed_signal_length=torch.tensor([chunk_feat.shape[1]]).to(
                        self.torch_device
                    ),
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
                left_offset = 8  # Após primeiro chunk

                # Extrair predições
                preds_np = total_preds[0].cpu().numpy()
                active_speakers = np.argmax(preds_np, axis=1)

                if len_prediction is None:
                    len_prediction = len(active_speakers)

                # Processar predições do chunk atual
                frame_duration = self.chunk_duration_seconds / len_prediction
                current_chunk_preds = active_speakers[-len_prediction:]

                # Converter para segmentos
                for idx, spk in enumerate(current_chunk_preds):
                    start_time = i * self.chunk_duration_seconds + idx * frame_duration
                    end_time = i * self.chunk_duration_seconds + (idx + 1) * frame_duration
                    speaker_id_str = f"SPEAKER_{spk:02d}"

                    # Merge com segmento anterior se mesmo speaker
                    if (
                        speaker_segments
                        and speaker_segments[-1][2] == speaker_id_str
                        and abs(speaker_segments[-1][1] - start_time) < frame_duration * 0.5
                    ):
                        # Estender segmento existente
                        speaker_segments[-1] = (
                            speaker_segments[-1][0],
                            end_time,
                            speaker_id_str,
                        )
                    else:
                        # Novo segmento
                        speaker_segments.append((start_time, end_time, speaker_id_str))

        logger.debug(f"Detectados {len(speaker_segments)} segmentos")
        return speaker_segments

    def _init_streaming_state(self, batch_size: int) -> StreamingSortformerState:
        """
        Inicializa StreamingSortformerState (criado manualmente, não do NeMo)
        Baseado em WhisperLiveKit init_streaming_state()
        """
        streaming_state = StreamingSortformerState()

        # Async streaming mode (conforme WhisperLiveKit)
        streaming_state.spkcache = torch.zeros(
            (
                batch_size,
                self.model.sortformer_modules.spkcache_len,
                self.model.sortformer_modules.fc_d_model,
            ),
            device=self.torch_device,
        )
        streaming_state.spkcache_preds = torch.zeros(
            (
                batch_size,
                self.model.sortformer_modules.spkcache_len,
                self.model.sortformer_modules.n_spk,
            ),
            device=self.torch_device,
        )
        streaming_state.spkcache_lengths = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.torch_device
        )
        streaming_state.fifo = torch.zeros(
            (
                batch_size,
                self.model.sortformer_modules.fifo_len,
                self.model.sortformer_modules.fc_d_model,
            ),
            device=self.torch_device,
        )
        streaming_state.fifo_lengths = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.torch_device
        )
        streaming_state.mean_sil_emb = torch.zeros(
            (batch_size, self.model.sortformer_modules.fc_d_model), device=self.torch_device
        )
        streaming_state.n_sil_frames = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.torch_device
        )

        return streaming_state

    async def assign_speakers_to_words(
        self,
        words: list[Word],
        diarization_segments: list[tuple[float, float, str]] | None = None,
    ) -> list[Word]:
        """
        Atribui speaker IDs às palavras baseado em overlap temporal.
        Mesmo algoritmo do PyannoteBackend.

        Args:
            words: Lista de Word objects
            diarization_segments: Segmentos de diarization (não usado, usa cache)

        Returns:
            Lista de Word objects com speaker_id atribuído
        """
        if not diarization_segments:
            logger.warning("Nenhum segmento de diarization disponível")
            return words

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

            # Criar novo Word com speaker_id
            word_with_speaker = Word(
                word=word.word,
                start=word.start,
                end=word.end,
                probability=word.probability,
                is_filler=word.is_filler,
                is_punctuation=word.is_punctuation,
                speaker_id=best_speaker_id,
                language=word.language,
            )

            words_with_speakers.append(word_with_speaker)

        return words_with_speakers

    def reset(self):
        """Reset não é necessário em batch mode"""
        pass

    def get_info(self) -> dict:
        """Retorna informações do backend"""
        info = super().get_info()
        info.update(
            {
                "model_name": self.model_name,
                "chunk_duration": self.chunk_duration_seconds,
                "nemo_available": NEMO_AVAILABLE,
            }
        )
        return info
