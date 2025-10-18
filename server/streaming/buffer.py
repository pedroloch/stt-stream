"""
Streaming Buffer com LocalAgreement Policy

Inspirado no whisper_streaming (UFAL) mas adaptado para nossa arquitetura.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..backends.base import WhisperBackend
from ..constants import DEFAULT_SAMPLE_RATE
from ..models.result import TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class BufferConfig:
    """Configuração do buffer de streaming"""

    min_chunk_size: float = 1.0  # segundos
    buffer_trimming: str = "segment"  # "segment" ou "sentence"
    agreement_threshold: int = 2  # número de concordâncias necessárias
    max_buffer_size: float = 30.0  # segundos (limite de memória)


class LocalAgreementPolicy:
    """
    LocalAgreement-n Policy

    Confirma transcrição quando n updates consecutivos concordam no prefixo.

    Exemplo (n=2):
        Update 1: "olá mundo como você"
        Update 2: "olá mundo como você está"
        → Confirma: "olá mundo como você" (prefixo comum)

        Update 3: "olá mundo como você está hoje"
        → Confirma: "olá mundo como você está" (novo prefixo)
    """

    def __init__(self, n: int = 2):
        """
        Args:
            n: Número de concordâncias consecutivas necessárias
        """
        self.n = n
        self.history: list[str] = []
        self.last_confirmed: str = ""

    def check(self, new_transcript: str) -> tuple[bool, str | None]:
        """
        Verifica se deve confirmar transcrição

        Args:
            new_transcript: Nova transcrição do buffer atual

        Returns:
            (should_confirm, confirmed_text)
            - should_confirm: True se deve confirmar
            - confirmed_text: Texto confirmado (apenas o novo, não repetir)
        """
        # Adicionar ao histórico
        self.history.append(new_transcript)

        # Manter apenas últimas n transcrições
        if len(self.history) > self.n:
            self.history.pop(0)

        # Precisa de n transcrições para confirmar
        if len(self.history) < self.n:
            logger.debug(f"LocalAgreement: {len(self.history)}/{self.n} histórico")
            return False, None

        # Extrair prefixo comum
        common_prefix = self._longest_common_prefix(self.history)

        # Se prefixo vazio, não confirmar
        if not common_prefix:
            logger.debug("LocalAgreement: sem prefixo comum")
            return False, None

        # Remover texto já confirmado anteriormente
        new_confirmed = common_prefix[len(self.last_confirmed) :].strip()

        # Se não há texto novo, não confirmar
        if not new_confirmed:
            logger.debug("LocalAgreement: sem texto novo")
            return False, None

        # Confirmar!
        self.last_confirmed = common_prefix
        logger.info(f"LocalAgreement: confirmado '{new_confirmed}'")
        return True, new_confirmed

    def _longest_common_prefix(self, transcripts: list[str]) -> str:
        """
        Encontra maior prefixo em comum entre transcrições

        Estratégia: comparar palavra por palavra

        Example:
            ["olá mundo como você", "olá mundo como você está"]
            → "olá mundo como você"
        """
        if not transcripts:
            return ""

        # Dividir em palavras
        words_lists = [t.split() for t in transcripts]

        # Encontrar menor lista
        min_len = min(len(wl) for wl in words_lists)

        common_words = []
        for i in range(min_len):
            word = words_lists[0][i]

            # Verificar se todas concordam nessa palavra
            if all(wl[i] == word for wl in words_lists):
                common_words.append(word)
            else:
                # Primeira divergência, parar
                break

        return " ".join(common_words)

    def reset(self):
        """Reset para nova sessão"""
        self.history.clear()
        self.last_confirmed = ""


class StreamingBuffer:
    """
    Buffer inteligente para streaming de áudio

    Acumula chunks, processa com re-transcrição, e usa LocalAgreement
    para confirmar texto estável.
    """

    def __init__(self, backend: WhisperBackend, config: BufferConfig | None = None):
        """
        Args:
            backend: Backend de transcrição (faster-whisper, MLX, etc)
            config: Configuração do buffer
        """
        self.backend = backend
        self.config = config or BufferConfig()

        # Buffer de áudio
        self.audio_buffer = np.array([], dtype=np.float32)

        # LocalAgreement policy
        self.agreement = LocalAgreementPolicy(n=self.config.agreement_threshold)

        # Estado
        self.confirmed_text = ""
        self.total_duration = 0.0  # segundos de áudio processado
        self.chunk_count = 0

        logger.info(
            f"StreamingBuffer criado: "
            f"min_chunk={self.config.min_chunk_size}s, "
            f"agreement_n={self.config.agreement_threshold}, "
            f"trimming={self.config.buffer_trimming}"
        )

    async def add_chunk(self, audio_chunk: np.ndarray):
        """
        Adiciona chunk de áudio ao buffer

        Args:
            audio_chunk: Array numpy (float32, mono, 16kHz)
        """
        # Validar formato
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalizar se necessário
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()

        # Adicionar ao buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        self.chunk_count += 1

        logger.debug(
            f"Chunk #{self.chunk_count} adicionado: "
            f"{len(audio_chunk)/DEFAULT_SAMPLE_RATE:.2f}s, "
            f"buffer total: {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s"
        )

        # Verificar limite de buffer
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if buffer_duration > self.config.max_buffer_size:
            logger.warning(f"Buffer muito grande ({buffer_duration:.1f}s), " f"forçando trim")
            await self._force_trim()

    def _get_context_window(self, full_text: str, max_words: int = 100) -> str:
        """
        Retorna últimas N palavras para contexto

        TÉCNICA CRÍTICA: Passar texto confirmado como initial_prompt melhora:
        - Coerência (nomes próprios, termos técnicos)
        - Accuracy (-15% WER segundo whisper_streaming)
        - Capitalização e formatação

        Whisper tem limite de ~224 tokens no initial_prompt.
        Passar muito texto pode degradar performance.

        Args:
            full_text: Texto completo confirmado
            max_words: Máximo de palavras no contexto (default: 100)

        Returns:
            Últimas max_words palavras
        """
        if not full_text:
            return ""

        words = full_text.split()

        if len(words) <= max_words:
            return full_text

        # Retornar últimas N palavras
        context = " ".join(words[-max_words:])

        logger.debug(
            f"Context window: {len(words)} total → {max_words} palavras usadas"
        )

        return context

    async def process(self) -> TranscriptionResult | None:
        """
        Processa buffer atual e retorna transcrição (parcial ou final)

        Returns:
            TranscriptionResult se tiver resultado, None se buffer muito pequeno
        """
        # Verificar se temos áudio suficiente
        duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        if duration < self.config.min_chunk_size:
            logger.debug(
                f"Buffer pequeno ({duration:.2f}s < {self.config.min_chunk_size}s), aguardando"
            )
            return None

        # Obter contexto otimizado (últimas N palavras apenas)
        # IMPORTANTE: Não passar TODO o texto - degrada performance!
        context = self._get_context_window(self.confirmed_text, max_words=100)

        # Transcrever buffer completo com contexto
        logger.debug(f"Transcrevendo buffer: {duration:.2f}s")
        logger.debug(
            f"Context: '{context[:80]}...' ({len(context.split())} palavras)"
        )

        result = await self.backend.transcribe_chunk(
            self.audio_buffer,
            context=context,  # ⭐ Context otimizado (últimas 100 palavras)
        )

        # Se vazio, retornar None
        if not result.text:
            logger.debug("Transcrição vazia")
            return None

        # LocalAgreement policy
        should_confirm, confirmed = self.agreement.check(result.text)

        if should_confirm and confirmed:
            # Texto confirmado!
            logger.info(f"✅ Confirmado: '{confirmed}'")

            # Atualizar contexto
            self.confirmed_text += " " + confirmed
            self.confirmed_text = self.confirmed_text.strip()

            # Trim buffer (remover áudio confirmado)
            await self._trim_buffer(result)

            # Retornar resultado FINAL
            return TranscriptionResult(
                text=confirmed,
                is_final=True,
                confidence=result.confidence,
                language=result.language,
                timestamp=datetime.now(),
                segments=result.segments,  # Manter segmentos originais
            )
        # Texto ainda não estável, retornar PARCIAL
        logger.debug(f"⏳ Parcial: '{result.text}'")

        return TranscriptionResult(
            text=result.text,
            is_final=False,
            confidence=result.confidence,
            language=result.language,
            timestamp=datetime.now(),
            segments=result.segments,
        )

    async def _trim_buffer(self, result: TranscriptionResult):
        """
        Remove áudio confirmado do buffer

        Estratégias:
        - "segment": corta no timestamp do último segmento
        - "sentence": corta no fim de frase (mais conservador)
        """
        if self.config.buffer_trimming == "segment":
            await self._trim_buffer_segment(result)
        elif self.config.buffer_trimming == "sentence":
            await self._trim_buffer_sentence(result)
        else:
            logger.warning(
                f"Trimming strategy '{self.config.buffer_trimming}' desconhecida"
            )

    async def _trim_buffer_segment(self, result: TranscriptionResult):
        """
        Corta buffer no timestamp do último segmento confirmado
        """
        if not result.segments:
            logger.debug("Sem segmentos, não é possível trim")
            return

        # Último segmento
        last_segment = result.segments[-1]
        trim_time = last_segment.end

        # Converter para samples
        trim_samples = int(trim_time * DEFAULT_SAMPLE_RATE)

        # Trim (manter apenas áudio após trim_time)
        if trim_samples < len(self.audio_buffer):
            removed_duration = trim_samples / DEFAULT_SAMPLE_RATE
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.total_duration += removed_duration

            logger.info(
                f"Buffer trimmed: removido {removed_duration:.2f}s, "
                f"restante {len(self.audio_buffer)/DEFAULT_SAMPLE_RATE:.2f}s"
            )
        else:
            # Trim completo
            self.total_duration += len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
            self.audio_buffer = np.array([], dtype=np.float32)
            logger.info("Buffer completamente trimmed")

    async def _trim_buffer_sentence(self, result: TranscriptionResult):
        """
        Corta buffer no fim de frase (ponto, interrogação, exclamação)

        Mais conservador: espera frase completa antes de trim.
        Latência maior mas menos risco de cortar mid-sentence.
        """
        confirmed = self.agreement.last_confirmed

        # Encontrar último caractere de pontuação
        last_punct_idx = max(
            confirmed.rfind("."),
            confirmed.rfind("?"),
            confirmed.rfind("!"),
        )

        if last_punct_idx < 0:
            logger.debug("Sem pontuação, não é possível trim conservador")
            return

        # Texto até pontuação
        confirmed_until_punct = confirmed[: last_punct_idx + 1]

        # Estimar timestamp (aproximado via proporção)
        # TODO: melhorar com word-level timestamps
        proportion = (
            len(confirmed_until_punct) / len(result.text) if result.text else 0
        )
        buffer_duration = len(self.audio_buffer) / DEFAULT_SAMPLE_RATE
        trim_time = buffer_duration * proportion

        # Trim
        trim_samples = int(trim_time * DEFAULT_SAMPLE_RATE)
        if trim_samples < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[trim_samples:]
            logger.info(f"Buffer trimmed (sentence): removido {trim_time:.2f}s")

    async def _force_trim(self):
        """
        Forçar trim quando buffer muito grande (evitar OOM)
        """
        # Manter apenas últimos 15s
        max_samples = int(15.0 * DEFAULT_SAMPLE_RATE)
        if len(self.audio_buffer) > max_samples:
            removed = len(self.audio_buffer) - max_samples
            self.audio_buffer = self.audio_buffer[-max_samples:]
            logger.warning(
                f"Buffer forçadamente trimmed: "
                f"removido {removed/DEFAULT_SAMPLE_RATE:.2f}s"
            )

    def reset(self):
        """Reset para nova sessão"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.agreement.reset()
        self.confirmed_text = ""
        self.total_duration = 0.0
        self.chunk_count = 0
        logger.info("StreamingBuffer reset")

    def get_stats(self) -> dict:
        """Estatísticas do buffer"""
        return {
            "buffer_duration_s": len(self.audio_buffer) / DEFAULT_SAMPLE_RATE,
            "total_processed_s": self.total_duration,
            "chunks_received": self.chunk_count,
            "confirmed_text_len": len(self.confirmed_text),
        }
