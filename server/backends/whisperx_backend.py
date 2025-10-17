"""
WhisperX Backend (CUDA only)

Backend para diarization (speaker identification) usando WhisperX.
Requer Linux com CUDA GPU.

Capabilities:
- TRANSCRIPTION
- WORD_TIMESTAMPS (wav2vec2 alignment - mais preciso!)
- SPEAKER_DIARIZATION (pyannote.audio)
- VAD
- STREAMING
"""

import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import numpy as np

from ..models.capability import BackendInfo, Capability
from ..models.result import Segment, TranscriptionResult, Word
from ..utils.platform import Platform, PlatformNotSupportedError, detect_platform
from .base import BackendError, BackendNotAvailableError, TranscriptionError, WhisperBackend


class WhisperXBackend(WhisperBackend):
    """WhisperX Backend - Diarization com pyannote.audio"""

    # Class-level constant
    INFO = BackendInfo(
        name="whisperx",
        supported_platforms={Platform.LINUX_CUDA},
        capabilities={
            Capability.TRANSCRIPTION,
            Capability.WORD_TIMESTAMPS,
            Capability.SPEAKER_DIARIZATION,  # ⭐ Speaker ID
            Capability.VAD,
            Capability.STREAMING,
        },
        model_sizes={"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"},
    )

    @property
    def info(self) -> BackendInfo:
        """Retorna metadata e capabilities do WhisperX backend (cached)"""
        return self.__class__.INFO

    async def initialize(self) -> None:
        """Inicializa WhisperX (fail-fast se nao CUDA)"""
        # Validar plataforma
        current = detect_platform()
        if current != Platform.LINUX_CUDA:
            raise PlatformNotSupportedError(
                f"WhisperX requires Linux with CUDA GPU.\n"
                f"Current platform: {current.value}\n"
                f"Deploy on GPU server (RunPod, Vast.ai) or use 'faster-whisper' instead."
            )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Inicializando WhisperX backend (modelo: {self.model}, idioma: {self.language})")

        try:
            # Import WhisperX (fail-fast se não instalado)
            import whisperx  # type: ignore
            self.whisperx = whisperx

            # Device
            self.device = self.kwargs.get("device", "cuda")
            self.compute_type = self.kwargs.get("compute_type", "float16")

            # 1. Carregar modelo Whisper
            self.logger.info(f"Carregando modelo Whisper: {self.model}")
            self.model_instance = whisperx.load_model(
                self.model,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language if self.language != "auto" else None,
            )

            # 2. Carregar modelo de alinhamento (word timestamps precisos)
            self.logger.info("Carregando modelo de alinhamento (wav2vec2)")
            self.align_model = None
            self.align_metadata = None
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=self.language if self.language != "auto" else "en",
                    device=self.device,
                )
                self.logger.info("✅ Modelo de alinhamento carregado")
            except Exception as e:
                self.logger.warning(f"Não foi possível carregar modelo de alinhamento: {e}")
                self.logger.warning("Word timestamps estarão disponíveis mas menos precisos")

            # 3. Carregar modelo de diarization (speaker identification)
            self.logger.info("Carregando modelo de diarization (pyannote.audio)")
            self.diarize_model = None
            try:
                # Nota: Requer HuggingFace token com acesso ao pyannote/speaker-diarization
                # Token pode ser passado via HUGGING_FACE_HUB_TOKEN env var
                hf_token = self.kwargs.get("hf_token", None)
                if hf_token:
                    self.diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=hf_token, device=self.device
                    )
                    self.logger.info("✅ Modelo de diarization carregado")
                else:
                    self.logger.warning("HF token não fornecido - diarization desabilitado")
                    self.logger.warning("Para habilitar, passe 'hf_token' no config")
            except Exception as e:
                self.logger.warning(f"Não foi possível carregar diarization: {e}")
                self.logger.warning("Speaker identification não estará disponível")

            # Configurações
            self.batch_size = self.kwargs.get("batch_size", 16)
            self.min_speakers = self.kwargs.get("min_speakers", None)
            self.max_speakers = self.kwargs.get("max_speakers", None)

            self._initialized = True
            self.logger.info("✅ WhisperX backend inicializado com sucesso")

        except ImportError as e:
            raise BackendNotAvailableError(
                f"WhisperX não está instalado.\n"
                f"Instale com: pip install git+https://github.com/m-bain/whisperx.git\n"
                f"Erro: {e}"
            ) from None
        except Exception as e:
            raise BackendError(f"Erro ao inicializar WhisperX: {e}") from e

    async def transcribe_chunk(
        self, audio: np.ndarray, context: str | None = None
    ) -> TranscriptionResult:
        """Transcreve com speaker diarization"""
        if not self._initialized or not self.model_instance:
            raise RuntimeError("Backend não inicializado. Chame initialize() primeiro.")

        try:
            # Normalizar áudio para float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalizar para [-1, 1]
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # 1. Transcrição com Whisper
            self.logger.debug(f"Transcrevendo {len(audio)/16000:.2f}s de áudio")
            result = self.model_instance.transcribe(
                audio,
                batch_size=self.batch_size,
                language=self.language if self.language != "auto" else None,
            )

            # Se vazio
            if not result["segments"]:
                return TranscriptionResult(
                    text="",
                    is_final=False,
                    confidence=0.0,
                    language=result.get("language", self.language),
                    timestamp=datetime.now(),
                )

            # 2. Alinhar palavras (word timestamps precisos)
            if self.align_model and self.align_metadata:
                self.logger.debug("Alinhando palavras com wav2vec2")
                result = self.whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )

            # 3. Diarization (speaker identification)
            speaker_segments = None
            if self.diarize_model:
                self.logger.debug("Executando diarization")
                diarize_result = self.diarize_model(
                    audio,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )
                speaker_segments = self.whisperx.assign_word_speakers(
                    diarize_result, result["segments"]
                )

            # Converter para TranscriptionResult
            segments_list = speaker_segments if speaker_segments else result["segments"]

            # Concatenar texto
            full_text = " ".join(seg["text"].strip() for seg in segments_list).strip()

            if not full_text:
                return TranscriptionResult(
                    text="",
                    is_final=False,
                    confidence=0.0,
                    language=result.get("language", self.language),
                    timestamp=datetime.now(),
                )

            # Converter segmentos
            normalized_segments = []
            for seg in segments_list:
                # Words
                words = None
                if "words" in seg:
                    words = [
                        Word(
                            word=w["word"],
                            start=w["start"],
                            end=w["end"],
                            probability=w.get("score", 1.0),
                            speaker_id=w.get("speaker", None),  # ⭐ Speaker ID!
                        )
                        for w in seg["words"]
                    ]

                normalized_segments.append(
                    Segment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"].strip(),
                        words=words,
                        speaker_id=seg.get("speaker", None),  # ⭐ Speaker ID no segment!
                    )
                )

            # Confiança média (se disponível)
            avg_confidence = 0.0
            if segments_list and "score" in segments_list[0]:
                avg_confidence = sum(s.get("score", 0.0) for s in segments_list) / len(
                    segments_list
                )

            # Determinar speaker principal (mais comum)
            main_speaker = None
            if speaker_segments:
                speakers = [s.get("speaker") for s in speaker_segments if s.get("speaker")]
                if speakers:
                    main_speaker = max(set(speakers), key=speakers.count)

            return TranscriptionResult(
                text=full_text,
                is_final=True,
                confidence=float(avg_confidence),
                language=result.get("language", self.language),
                timestamp=datetime.now(),
                segments=normalized_segments,
                speaker=main_speaker,  # ⭐ Speaker principal
            )

        except Exception as e:
            self.logger.error(f"Erro na transcrição: {e}")
            raise TranscriptionError(f"Falha na transcrição: {e}") from e

    async def transcribe_stream(  # type: ignore[override]
        self, audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream com diarization"""
        async for audio_chunk in audio_stream:
            result = await self.transcribe_chunk(audio_chunk)
            if result.text:
                yield result

    async def cleanup(self) -> None:
        """Limpa recursos"""
        self.logger.info("Limpando WhisperX backend")
        self.model_instance = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self._initialized = False

    def get_backend_info(self) -> dict[str, Any]:
        """
        Retorna informações sobre o backend WhisperX

        Returns:
            Dicionário com informações do backend
        """
        info_dict = {
            "name": "WhisperX (Speaker Diarization)",
            "device": self.device if hasattr(self, 'device') else "cuda",
            "model": self.model,
            "language": self.language,
            "compute_type": self.compute_type if hasattr(self, 'compute_type') else "float16",
            "initialized": self._initialized,
            "capabilities": [c.value for c in self.info.capabilities],
        }

        # Adicionar info de diarization
        if hasattr(self, 'diarize_model') and self.diarize_model:
            info_dict["diarization_enabled"] = True
        else:
            info_dict["diarization_enabled"] = False

        # Info de alinhamento
        if hasattr(self, 'align_model') and self.align_model:
            info_dict["alignment_enabled"] = True
        else:
            info_dict["alignment_enabled"] = False

        # Info da GPU se disponível
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                info_dict["gpu"] = torch.cuda.get_device_name(0)
                info_dict["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                info_dict["gpu_memory_used"] = f"{torch.cuda.memory_allocated(0) / 1e9:.1f} GB"
        except Exception:
            pass

        return info_dict
