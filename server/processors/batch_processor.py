"""
Batch Processor para API `/v1/transcribe`

Pipeline completo:
1. Preparação de áudio (converter para PCM 16kHz mono)
2. Validação (tamanho, duração)
3. Transcrição (backend selecionado)
4. Post-processing (diarization, translation)
5. Formatting (JSON, SRT, VTT)
6. Metrics
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env
load_dotenv()

from ..backends import BackendRegistry
from ..backends.base import WhisperBackend
from ..config import Config
from ..models.batch import (
    BackendInfo,
    ProcessingMetrics,
    Segment,
    TranscribeRequest,
    TranscribeResponse,
    Word,
)
from ..models.result import TranscriptionResult
from ..diarization.processor import DiarizationProcessor


class BatchProcessorError(Exception):
    """Erro de processamento batch"""
    pass


class AudioTooLargeError(BatchProcessorError):
    """Áudio excede limite de duração"""
    pass


class BatchProcessor:
    """
    Processador de transcrição batch

    Gerencia o pipeline completo de transcrição de arquivos de áudio
    """

    # Limites (configuráveis)
    MAX_DURATION_SEC = 3600  # 1 hora
    MAX_FILE_SIZE_MB = 100

    def __init__(self, config: Config):
        """
        Inicializa o batch processor

        Args:
            config: Configuração do servidor
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def process(
        self,
        audio_file: Path,
        request: TranscribeRequest
    ) -> TranscribeResponse:
        """
        Processa arquivo de áudio completo

        Args:
            audio_file: Caminho para arquivo de áudio
            request: Parâmetros de transcrição

        Returns:
            Resposta completa com transcrição

        Raises:
            AudioTooLargeError: Se áudio excede limites
            BatchProcessorError: Outros erros de processamento
        """
        start_time = time.time()

        self.logger.info(f"Processando arquivo: {audio_file.name}")
        self.logger.info(f"  Backend: {request.model}")
        self.logger.info(f"  Idioma: {request.language}")
        self.logger.info(f"  Diarization: {request.enable_diarization}")

        # Métricas
        metrics = {
            "model_load_time": 0.0,
            "transcription_time": 0.0,
            "diarization_time": 0.0,
            "formatting_time": 0.0,
        }

        try:
            # 1. Preparar áudio
            self.logger.debug("Convertendo áudio para PCM 16kHz mono...")
            audio_data, sample_rate, duration = await self._prepare_audio(audio_file)

            # Validar duração
            if duration > self.MAX_DURATION_SEC:
                raise AudioTooLargeError(
                    f"Áudio muito longo: {duration:.1f}s (max: {self.MAX_DURATION_SEC}s)"
                )

            self.logger.info(f"  Duração: {duration:.1f}s")

            # 2. Criar backend
            model_load_start = time.time()
            backend = await self._create_backend(request)
            metrics["model_load_time"] = time.time() - model_load_start

            # 3. Transcrever
            transcription_start = time.time()
            result = await self._transcribe(backend, audio_data, request)
            metrics["transcription_time"] = time.time() - transcription_start

            self.logger.info(f"  Transcrição completa: {len(result.text)} caracteres")

            # 4. Post-processing (diarization)
            if request.enable_diarization:
                self.logger.info("Aplicando diarization...")
                diarization_start = time.time()
                result = await self._apply_diarization(result, audio_data, request)
                metrics["diarization_time"] = time.time() - diarization_start

            # 5. Formatar resposta
            formatting_start = time.time()
            response = await self._format_response(
                result=result,
                request=request,
                backend=backend,
                duration=duration,
                metrics=metrics,
                total_time=time.time() - start_time
            )
            metrics["formatting_time"] = time.time() - formatting_start

            self.logger.info(
                f"Processamento completo em {time.time() - start_time:.2f}s "
                f"(RTF: {(time.time() - start_time) / duration:.3f}x)"
            )

            return response

        except AudioTooLargeError:
            raise
        except Exception as e:
            self.logger.error(f"Erro ao processar áudio: {e}", exc_info=True)
            raise BatchProcessorError(f"Falha no processamento: {e}") from e

    async def _prepare_audio(
        self,
        audio_file: Path
    ) -> tuple[np.ndarray, int, float]:
        """
        Converte áudio para formato esperado (PCM 16kHz mono)

        Args:
            audio_file: Arquivo de áudio

        Returns:
            Tuple: (audio_data, sample_rate, duration)
        """
        sample_rate = 16000

        # Usar ffmpeg para converter
        cmd = [
            "ffmpeg",
            "-i", str(audio_file),
            "-ar", str(sample_rate),  # 16kHz
            "-ac", "1",               # Mono
            "-f", "s16le",            # PCM signed 16-bit little-endian
            "-acodec", "pcm_s16le",
            "-",                      # Output para stdout
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise BatchProcessorError(
                    f"ffmpeg falhou: {stderr.decode()}"
                )

            # Converter bytes para numpy array
            audio_data = np.frombuffer(stdout, dtype=np.int16)

            # Converter para float32 [-1.0, 1.0]
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Calcular duração
            duration = len(audio_data) / sample_rate

            return audio_data, sample_rate, duration

        except FileNotFoundError:
            raise BatchProcessorError(
                "ffmpeg não encontrado. Instale com: brew install ffmpeg (Mac) "
                "ou apt-get install ffmpeg (Linux)"
            ) from None

    async def _create_backend(self, request: TranscribeRequest) -> WhisperBackend:
        """
        Cria backend apropriado

        Args:
            request: Request com parâmetros

        Returns:
            Backend inicializado
        """
        # Mapear model para backend name
        backend_name = request.model

        # Se "auto", detectar melhor backend
        if backend_name == "auto":
            from ..utils.platform import detect_platform
            platform = detect_platform()
            available = BackendRegistry.list_available(platform)

            if not available:
                raise BatchProcessorError(f"Nenhum backend disponível para {platform}")

            # Usar primeiro disponível (por enquanto)
            backend_name = list(available.keys())[0]
            self.logger.info(f"Auto-select backend: {backend_name}")

        # Criar backend
        try:
            backend = BackendRegistry.create(
                backend_name,
                {
                    "model": request.model_size,
                    "language": request.language,
                }
            )

            # Inicializar
            await backend.initialize()

            return backend

        except Exception as e:
            raise BatchProcessorError(
                f"Falha ao criar backend '{backend_name}': {e}"
            ) from e

    async def _transcribe(
        self,
        backend: WhisperBackend,
        audio: np.ndarray,
        request: TranscribeRequest
    ) -> TranscriptionResult:
        """
        Transcreve áudio usando backend

        Args:
            backend: Backend inicializado
            audio: Áudio em float32
            request: Request com parâmetros

        Returns:
            Resultado da transcrição
        """
        # Transcrever usando transcribe_chunk (MVP: sem transcribe_file ainda)
        # Passar áudio completo como um único chunk
        result = await backend.transcribe_chunk(
            audio=audio,
            context=request.initial_prompt
        )

        return result

    async def _apply_diarization(
        self,
        result: TranscriptionResult,
        audio: np.ndarray,
        request: TranscribeRequest
    ) -> TranscriptionResult:
        """
        Aplica diarization ao resultado da transcrição

        Args:
            result: Resultado com words e segments
            audio: Áudio completo (float32, 16kHz mono)
            request: Request com parâmetros de diarization

        Returns:
            TranscriptionResult atualizado com speaker_id
        """
        # Auto-select backend
        backend_name = request.diarization_backend or "auto"
        if backend_name == "auto":
            # Mac: pyannote (funciona em CPU/MPS)
            # Linux CUDA: sortformer (melhor performance)
            try:
                import torch
                if torch.cuda.is_available():
                    backend_name = "sortformer"
                else:
                    backend_name = "pyannote"
            except ImportError:
                backend_name = "pyannote"

        self.logger.info(f"Diarization backend selecionado: {backend_name}")

        # Obter HuggingFace token do ambiente
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            self.logger.debug("HuggingFace token encontrado no ambiente")
        else:
            self.logger.warning("HuggingFace token não encontrado - modelos gated podem falhar")

        # Criar processor
        processor = DiarizationProcessor(
            backend=backend_name,
            sample_rate=16000,
            num_speakers=request.num_speakers,
            device="auto",
            auth_token=hf_token,  # Passar token para Pyannote
        )

        # Processar áudio completo
        diarization_segments = await processor.diarize(audio)

        if not diarization_segments:
            self.logger.warning("Nenhum speaker detectado")
            return result

        self.logger.info(f"Detectados {len(set(s[2] for s in diarization_segments))} speakers")

        # Atribuir speaker_id às words
        if result.segments:
            updated_segments = []
            for segment in result.segments:
                if segment.words:
                    # Atribuir speakers às words
                    words_with_speakers = await processor.assign_speakers_to_words(
                        segment.words, diarization_segments
                    )

                    # Determinar speaker principal do segment (mais comum)
                    speaker_counts = {}
                    for word in words_with_speakers:
                        if word.speaker_id:
                            speaker_counts[word.speaker_id] = speaker_counts.get(word.speaker_id, 0) + 1

                    main_speaker = max(speaker_counts, key=speaker_counts.get) if speaker_counts else None

                    # Criar segment atualizado
                    from ..models.result import Segment as ResultSegment
                    updated_segment = ResultSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words_with_speakers,
                        speaker_id=main_speaker,
                    )
                    updated_segments.append(updated_segment)
                else:
                    # Segment sem words, manter como está
                    updated_segments.append(segment)

            # Retornar resultado atualizado
            return TranscriptionResult(
                text=result.text,
                is_final=result.is_final,
                confidence=result.confidence,
                language=result.language,
                timestamp=result.timestamp,
                segments=updated_segments,
                speaker=result.speaker,
            )

        return result

    async def _format_response(
        self,
        result: TranscriptionResult,
        request: TranscribeRequest,
        backend: WhisperBackend,
        duration: float,
        metrics: dict,
        total_time: float
    ) -> TranscribeResponse:
        """
        Formata resultado em resposta final

        Args:
            result: Resultado da transcrição
            request: Request original
            backend: Backend usado
            duration: Duração do áudio
            metrics: Métricas coletadas
            total_time: Tempo total

        Returns:
            Resposta formatada
        """
        # Converter segments
        segments_list: list[Segment] = []

        if result.segments:
            for i, seg in enumerate(result.segments):
                # Converter words
                words_list: list[Word] | None = None
                if seg.words:
                    words_list = [
                        Word(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                            speaker_id=w.speaker_id
                        )
                        for w in seg.words
                    ]

                segment = Segment(
                    id=i,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=words_list,
                    speaker_id=getattr(seg, 'speaker_id', None),
                    confidence=getattr(seg, 'confidence', None)
                )
                segments_list.append(segment)

        # Backend info
        backend_info = BackendInfo(
            model=backend.info.name,
            model_size=request.model_size,
            device=getattr(backend, 'device', 'unknown'),
            language=result.language or request.language,
            diarization_backend=request.diarization_backend if request.enable_diarization else None
        )

        # Processing metrics
        processing_metrics: ProcessingMetrics | None = None
        if request.return_metrics:
            processing_metrics = ProcessingMetrics(
                processing_time_sec=total_time,
                audio_duration_sec=duration,
                real_time_factor=total_time / duration if duration > 0 else 0.0,
                model_load_time_sec=metrics.get("model_load_time"),
                transcription_time_sec=metrics.get("transcription_time"),
                diarization_time_sec=metrics.get("diarization_time") if request.enable_diarization else None,
                formatting_time_sec=metrics.get("formatting_time")
            )

        # Resposta completa
        return TranscribeResponse(
            text=result.text,
            task=request.task,
            language=result.language or request.language or "unknown",
            duration=duration,
            segments=segments_list,
            backend_info=backend_info,
            metrics=processing_metrics
        )
